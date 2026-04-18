# ── config.py ─────────────────────────────────────────────────────────────────
# All constants — language maps, node types, model names

LANG_MAP = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".java": "java",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".c":    "c",
    ".go":   "go",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".cs":   "c_sharp",
}

CHUNK_NODE_TYPES = {
    "python":     ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration",
                   "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "class_declaration",
                   "arrow_function", "method_definition",
                   "interface_declaration", "type_alias_declaration"],
    "java":       ["method_declaration", "class_declaration",
                   "interface_declaration"],
    "cpp":        ["function_definition", "class_specifier"],
    "c":          ["function_definition"],
    "go":         ["function_declaration", "method_declaration",
                   "type_declaration"],
    "rust":       ["function_item", "impl_item", "struct_item",
                   "enum_item", "trait_item"],
    "ruby":       ["method", "class", "module"],
    "c_sharp":    ["method_declaration", "class_declaration",
                   "interface_declaration"],
}

# Exact node types for import detection — avoids broad string matching bugs
IMPORT_NODE_TYPES = {
    "import_statement",       # Python
    "import_from_statement",  # Python
    "import_declaration",     # JS/TS
    "using_directive",        # C#
    "include_statement",      # C/C++
}

# Model names
LLM_MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDER_NAME    = "BAAI/bge-base-en-v1.5"
RERANKER_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
