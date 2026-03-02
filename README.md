# File_Question_Answering_System
🤖 RAG Code Assistant — Powered by Mistral-7B
RAG Code Assistant is an AI-powered tool that analyzes any Python file and lets you interact with it conversationally. It uses Tree-sitter to parse code, FAISS to find relevant functions, and Mistral-7B to generate accurate answers. You can ask questions, debug functions, explain specific lines, and generate new code .
🛠️ Tech Stack

LLM — Mistral-7B-Instruct-v0.2 (4-bit quantized)
Parser — Tree-sitter
Embeddings — Sentence Transformers (MiniLM-L6-v2)
Vector Search — FAISS
Platform — Kaggle T4 GPU

## 💬 Commands

| Command | Description |
|---|---|
| Ask anything | Q&A about the code |
| `summary` | File overview |
| `list` | Show all functions/classes |
| `debug:<function_name>` | Bug analysis + fixed version |
| `explain:<function>:<line>` | Explain a specific line |
| `generate:<description>` | Generate new code |
| `exit` | Quit |

