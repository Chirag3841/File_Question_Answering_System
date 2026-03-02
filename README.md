# File_Question_Answering_System

 RAG Code Assistant — Powered by Mistral-7B
A conversational AI tool that lets you interact with any Python file using a proper RAG (Retrieval-Augmented Generation) pipeline. Instead of feeding the entire file to the model, it semantically retrieves only the most relevant functions and classes for each query — making answers accurate, grounded, and hallucination-free.
You can ask questions about the code, debug functions, explain specific lines, and generate new code — all running locally on GPU with zero API costs.

 How RAG Works Here

Retrieval — FAISS semantically searches and fetches the top-3 most relevant functions/classes for your query
Augmented — The retrieved code chunks are injected into the Mistral prompt as context
Generation — Mistral generates answers strictly based on the retrieved code — nothing else


Tech Stack
ComponentTechnologyLLMMistral-7B-Instruct-v0.2 (4-bit quantized)Code ParsingTree-sitterEmbeddingsSentence Transformers — all-MiniLM-L6-v2Vector SearchFAISSQuantizationBitsAndBytes NF4 4-bitPlatformKaggle T4 GPU

Features
CommandDescriptionAsk anythingQ&A about the codesummaryHigh-level file overviewlistShow all functions and classesdebug:<function_name>Bug analysis + fixed versionexplain:<function>:<line>Explain a specific linegenerate:<description>Generate new code in same styleexitQuit
