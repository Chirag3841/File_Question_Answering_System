# File_Question_Answering_System

 RAG Code Assistant — Powered by Mistral-7B
A conversational AI tool that lets you interact with any Python file using a proper RAG (Retrieval-Augmented Generation) pipeline. Instead of feeding the entire file to the model, it semantically retrieves only the most relevant functions and classes for each query — making answers accurate, grounded, and hallucination-free.
You can ask questions about the code, debug functions, explain specific lines, and generate new code — all running locally on GPU with zero API costs.

 How RAG Works Here

Retrieval — FAISS semantically searches and fetches the top-3 most relevant functions/classes for your query
Augmented — The retrieved code chunks are injected into the Mistral prompt as context
Generation — Mistral generates answers strictly based on the retrieved code — nothing else


Tech Stack
| Component     | Technology                              |
|---------------|-----------------------------------------|
| LLM           | Mistral-7B-Instruct-v0.2 (4-bit)        |
| Code Parsing  | Tree-sitter                             |
| Embeddings    | Sentence Transformers (all-MiniLM-L6-v2)|
| Vector Search | FAISS                                   |
| Quantization  | BitsAndBytes NF4 4-bit                  |
| Platform      | Kaggle T4 GPU                           |

Features
| Command                    | Description                    |
|----------------------------|--------------------------------|
| Ask anything               | Q&A about the code             |
| summary                    | High-level file overview       |
| list                       | Show all functions and classes |
| debug:<function_name>      | Bug analysis + fixed version   |
| explain:<function>:<line>  | Explain a specific line        |
| generate:<description>     | Generate new code in same style|
| exit                       | Quit                           |
