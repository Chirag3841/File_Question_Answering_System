# File Question Answering System
### RAG Code Assistant — Powered by Mistral-7B

A conversational AI tool that lets you interact with any code file using a proper RAG (Retrieval-Augmented Generation) pipeline. Instead of feeding the entire file to the model, it semantically retrieves only the most relevant functions and classes for each query — making answers accurate, grounded, and hallucination-free.

Ask questions about the code, debug functions, explain specific lines, and generate new code — all running locally on GPU with zero API costs.


## How RAG Works Here

```
Your Query
    │
    ▼
FAISS Vector Search  ──►  Top-10 candidate chunks retrieved
    │
    ▼
Cross-Encoder Reranker  ──►  Best 5 chunks selected by relevance
    │
    ▼
Mistral-7B  ──►  Answer generated strictly from retrieved code
```

1. **Retrieval** — BAAI/bge embeddings semantically search and fetch the top-10 most relevant functions/classes for your query
2. **Reranking** — Cross-encoder reranker scores each (query, chunk) pair together and selects the best 5
3. **Augmented Generation** — Retrieved code chunks are injected into the Mistral prompt as context
4. **Generation** — Mistral generates answers strictly based on the retrieved code — nothing else


## Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral-7B-Instruct-v0.2 (4-bit NF4) |
| Code Parsing | Tree-sitter (13 languages) |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Search | FAISS (IndexFlatIP cosine) |
| Quantization | BitsAndBytes NF4 4-bit |
| Platform | Kaggle T4 GPU |


## Supported Languages

| Extension | Language |
|---|---|
| `.py` | Python |
| `.js` / `.jsx` | JavaScript |
| `.ts` / `.tsx` | TypeScript |
| `.java` | Java |
| `.cpp` / `.cc` | C++ |
| `.c` | C |
| `.go` | Go |
| `.rs` | Rust |
| `.rb` | Ruby |
| `.cs` | C# |
| Any other | Plain-text sliding window fallback |


## Features

| Command | Description |
|---|---|
| Ask anything | Q&A about the code — direct and indirect questions both work |
| `summary` | High-level file overview |
| `list` | Show all indexed functions and classes |
| `debug:<function_name>` | Deep bug analysis + fixed version |
| `explain:<function_name>:<line>` | Explain a specific line in a function |
| `generate:<description>` | Generate new code matching the codebase style |
| `exit` / `quit` | Quit |


## Project Structure

```
file-question-answering-system/
file-question-answering-system
│   ├── main.py            # Entry point — loads models, indexes file, runs REPL
│   ├── config.py          # All constants — LANG_MAP, node types, model names
│   ├── model.py           # Mistral-7B loading and inference
│   ├── parser.py          # Tree-sitter chunking for 13 languages + plain-text fallback
│   ├── embeddings.py      # BAAI embedder + cross-encoder reranker + FAISS index
│   ├── retrieval.py       # Two-stage retrieve + rerank pipeline
│   ├── commands.py        # ask, debug, explain, generate, summarize handlers  
├── .gitignore
├── pyproject.toml         # project configuration
├── requirements.txt       # dependencies
├── file-q-a-system.ipynb  
└── README.md              # project documentation
```


## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/file-question-answering-system.git
cd file-question-answering-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set HuggingFace token

**On Kaggle** — add your HuggingFace token as a secret named `hugging_face`.

**Locally** — set as environment variable:
```bash
export HF_TOKEN=your_token_here
```

### 4. Run
```bash
python main.py
```

Then enter the path to any code file when prompted.


## Example Usage

```
Enter the path to your file: /path/to/sentiment_analysis.py

Language: python | Code chunks: 12 | Imports: 8
Total chunks loaded: 20
FAISS index ready with 20 chunks

RAG Code Assistant — Powered by Mistral-7B
=================================================================

You: Write the name of the model used in this project?
Answer: The model used is DistilBERT, initialized in the SentimentAnalyzer class.

You: Provide me the code for run_epoch() function?
Answer: [reproduces full function code exactly — no omissions]

You: generate: function to evaluate model on test data and return f1 score
Answer: [generates new function matching codebase style]

You: debug:run_epoch
Answer: [full debug report — bugs, edge cases, fixed version]

You: explain:predict_sentiment:5
Answer: [explains exactly what line 5 does and why]
```

---

## Improvements Over Basic RAG

- **Cross-encoder reranker** — handles indirect and paraphrased questions that pure embedding search misses
- **Multi-language support** — works on 13 languages, not just Python
- **Exact import node matching** — no false positives from broad string matching
- **Nested function indexing** — functions inside classes are also indexed
- **Duplicate deduplication** — same chunk never indexed twice
- **8192 token context window** — uses more of Mistral's capacity
- **Cosine similarity** — FAISS IndexFlatIP on normalized embeddings instead of L2
- **Modular codebase** — split into clean, single-responsibility Python modules


## Limitations

- Mistral-7B debug output should be used as reference, not copied blindly — small models can introduce new bugs while fixing old ones
- Very large files (1000+ functions) may need `top_k` tuning
- Requires GPU — CPU inference will be very slow
- HuggingFace token required to download Mistral-7B (gated model)
