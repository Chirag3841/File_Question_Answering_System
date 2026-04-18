# ── main.py ───────────────────────────────────────────────────────────────────
# Entry point — loads models, indexes file, runs interactive REPL

import os
import commands
import embeddings as emb
from model import load_model
from embeddings import load_embedder_and_reranker, build_index
from parser import load_chunks_from_file

# ── On Kaggle: get token from secrets ─────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("hugging_face")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")


def main():
    # 1. Load models
    load_model(hf_token)
    load_embedder_and_reranker()

    # 2. Load file
    print("\nEnter the path to your file (any language):")
    file_path = input(" File path: ").strip()

    all_chunks, detected_lang = load_chunks_from_file(file_path)
    if not all_chunks:
        print("No chunks loaded. Exiting.")
        return

    # Deduplicate by name
    seen = set()
    all_chunks = [c for c in all_chunks if not (c["name"] in seen or seen.add(c["name"]))]
    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    for c in all_chunks[:5]:
        print(f"  [{c['type']}] {c['name']}")

    # 3. Build FAISS index
    build_index(all_chunks)

    # 4. Share detected_lang with commands module
    commands.detected_lang = detected_lang

    # 5. REPL
    print("\n RAG Code Assistant — Powered by Mistral-7B")
    print("="*65)
    print("Commands:")
    print("  'exit'                     -> Quit")
    print("  'list'                     -> Show all functions/classes")
    print("  'summary'                  -> Summarize the file")
    print("  'debug:<function_name>'    -> Deep debug a function")
    print("  'explain:<name>:<line>'    -> Explain a specific line")
    print("  'generate:<description>'   -> Generate new code")
    print("  or just ask any question!")
    print("="*65 + "\n")

    while True:
        user_query = input(" You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ("exit", "quit", "bye"):
            print(" Goodbye!")
            break

        elif user_query.lower() == "list":
            print("\n Available functions/classes:")
            for c in emb.all_chunks:
                if c["type"] not in ("import", "text_chunk"):
                    print(f"  [{c['type']}] {c['name']}")

        elif user_query.lower() == "summary":
            print("\n File Summary:\n")
            print(commands.summarize_file())

        elif user_query.lower().startswith("debug:"):
            func_name = user_query.split("debug:", 1)[1].strip()
            commands.debug_function(func_name)

        elif user_query.lower().startswith("explain:"):
            # Split on first 2 colons only — handles function names with special chars
            parts = user_query.split(":", 2)
            if len(parts) == 3:
                func_name = parts[1].strip()
                try:
                    line_num = int(parts[2].strip())
                    commands.explain_line(func_name, line_num)
                except ValueError:
                    print(" Format: explain:function_name:line_number")
            else:
                print(" Format: explain:function_name:line_number")

        elif user_query.lower().startswith("generate:"):
            description = user_query.split("generate:", 1)[1].strip()
            print(f"\n Generating {detected_lang} code...\n")
            code = commands.generate_code(description)
            print(f"\n{'='*60}")
            print(" Generated Code:")
            print(f"{'='*60}\n")
            print(code)
            print(f"\n{'='*60}\n")

        else:
            answer, retrieved = commands.ask_question(user_query)
            print("\n Retrieved Chunks (after reranking):")
            for r in retrieved:
                print(f"  [{r['type']}] {r['name']}")
            print(f"\n Answer:\n{answer}")

        print("\n" + "-"*65 + "\n")


if __name__ == "__main__":
    main()
