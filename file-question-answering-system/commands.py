# ── commands.py ───────────────────────────────────────────────────────────────
# All REPL command handlers: ask, debug, explain, generate, summarize

import embeddings as emb
from retrieval import retrieve_and_rerank
from model import call_mistral

# detected_lang is set by main.py after file loading
detected_lang = "python"


def ask_question(query: str, show_context: bool = False):
    """Answer any question about the loaded codebase."""
    top_chunks = retrieve_and_rerank(query, top_k=10, final_k=5)

    context = "\n\n".join([
        f"[{c['type'].upper()}] {c['name']}:\n{c['code'][:2000]}"
        for c in top_chunks
    ])

    if show_context:
        print(" Retrieved Context:\n")
        print(context)
        print("\n" + "="*50 + "\n")

    system = """You are an expert code analyst.
Answer questions ONLY based on the provided code.
Be specific — reference exact function names, class names, and variable names.
If the user asks for the code of a function or class, reproduce it COMPLETELY and EXACTLY as given — do NOT omit, summarize, or say 'code omitted for brevity'.
If the answer is not in the code, say 'This is not covered in the provided code.'
Never hallucinate or invent information."""

    user = f"""Here is the relevant code from the project:

{context}

Question: {query}

Important: If the question asks for code of a function, copy the FULL function code from above — word for word, do not skip any lines, do not write '...' or 'omitted'.

Answer clearly and specifically, referencing the actual code above."""

    answer = call_mistral(system, user, max_tokens=800)
    return answer, top_chunks


def generate_code(description: str) -> str:
    """Generate new code matching the existing codebase style."""
    top_chunks = retrieve_and_rerank(description, top_k=8, final_k=3)

    context = "\n\n".join([
        f"[{c['type'].upper()}] {c['name']}:\n{c['code']}"
        for c in top_chunks
    ])

    system = f"""You are an expert {detected_lang} developer.
Generate clean, working code that matches the style and patterns of the existing codebase.
Always include docstrings/comments and type hints where appropriate.
Only generate what is asked — no extra explanation outside code comments."""

    user = f"""Here is the existing codebase for context and style reference:

{context}

Generate code for: {description}

Match the coding style, patterns, and conventions of the existing code above."""

    return call_mistral(system, user, max_tokens=1000)


def debug_function(function_name: str):
    """Deep debug a specific function by name."""
    chunk = next((c for c in emb.all_chunks if c["name"] == function_name), None)

    if not chunk:
        print(f"'{function_name}' not found.")
        print("\n Available functions/classes:")
        for c in emb.all_chunks:
            if c["type"] not in ("import", "text_chunk"):
                print(f"  [{c['type']}] {c['name']}")
        return

    system = """You are a senior code debugger and reviewer.
Analyze code deeply and provide specific, actionable fixes.
Always provide the corrected version of the code."""

    user = f"""Debug and review this {chunk['type']}:

{chunk['code']}

Provide:
1. Bugs & Logic Errors
2. Edge Cases & Missing Checks
3. Missing Error Handling
4. Performance Issues
5. Fixed & Improved Version of the code"""

    print(f"\n{'='*60}")
    print(f"Debug Report: `{function_name}`")
    print(f"{'='*60}\n")
    print(call_mistral(system, user, max_tokens=1200))
    print(f"\n{'='*60}\n")


def explain_line(function_name: str, line_number: int):
    """Explain a specific line within a named function."""
    chunk = next((c for c in emb.all_chunks if c["name"] == function_name), None)

    if not chunk:
        print(f"'{function_name}' not found.")
        return

    lines = chunk['code'].split('\n')
    if line_number < 1 or line_number > len(lines):
        print(f"Line {line_number} out of range. Function has {len(lines)} lines.")
        return

    target_line = lines[line_number - 1]

    system = "You are an expert code teacher. Explain code clearly and simply."

    user = f"""In the function/class `{function_name}`:

Full code:
{chunk['code']}

Explain line {line_number} specifically:
{target_line}

Explain what this line does, why it's there, and how it fits in the overall context."""

    print(f"\n{'='*60}")
    print(f"Line {line_number} in `{function_name}`:")
    print(f"   {target_line.strip()}")
    print(f"{'='*60}\n")
    print(call_mistral(system, user, max_tokens=500))
    print(f"\n{'='*60}\n")


def summarize_file() -> str:
    """Generate a one-paragraph summary of the loaded file."""
    named = sorted(
        [c for c in emb.all_chunks if c["type"] not in ("import", "text_chunk")],
        key=lambda x: x["start"][0]
    )[:20]
    if not named:
        named = emb.all_chunks[:20]

    snippets = "\n\n".join([
        f"[{c['type'].upper()}] {c['name']}:\n{c['code'][:300]}"
        for c in named
    ])

    system = "You are an expert code engineer. Summarize code accurately. Do not guess."

    user = f"""Here are the functions and classes from a {detected_lang} file:

{snippets}

Write a clear, accurate one-paragraph summary of what this file does.
Only describe what you actually see in the code."""

    return call_mistral(system, user, max_tokens=250)
