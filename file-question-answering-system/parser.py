# ── parser.py ─────────────────────────────────────────────────────────────────
# Tree-sitter based code chunking for 13 languages + plain-text fallback

import os
from tree_sitter import Parser
from tree_sitter_languages import get_language
from config import LANG_MAP, CHUNK_NODE_TYPES, IMPORT_NODE_TYPES

parser = Parser()


def get_node_text(src_bytes: bytes, node) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def extract_name(src_bytes: bytes, node, lang: str) -> str:
    """Best-effort function/class name extraction across languages."""
    for field in ("name", "declarator"):
        n = node.child_by_field_name(field)
        if n:
            return get_node_text(src_bytes, n).split("(")[0].strip()
    for child in node.children:
        if child.is_named and child.type not in ("comment", "block"):
            txt = get_node_text(src_bytes, child).split("(")[0].strip()
            if txt:
                return txt
    return "<anonymous>"


def extract_chunks_treesitter(src_code: str, lang_name: str):
    """Parse source with Tree-sitter and extract named code units."""
    target_types = set(CHUNK_NODE_TYPES.get(lang_name, []))
    src_bytes = src_code.encode("utf-8", errors="replace")
    tree = parser.parse(src_bytes)

    chunks = []
    imports = []

    def traverse(node):
        t = node.type

        # Exact import node matching + early return to skip import children
        if t in IMPORT_NODE_TYPES:
            imports.append({
                "name": get_node_text(src_bytes, node).strip()[:120],
                "type": "import",
                "code": get_node_text(src_bytes, node),
                "start": node.start_point,
                "end": node.end_point,
            })
            return

        if t in target_types:
            name = extract_name(src_bytes, node, lang_name)
            chunks.append({
                "name": name,
                "type": t,
                "code": get_node_text(src_bytes, node),
                "start": node.start_point,
                "end": node.end_point,
            })
            # No return — allows nested functions inside classes to be indexed

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return chunks, imports


def chunk_by_sliding_window(text: str, chunk_size: int = 40, overlap: int = 10):
    """Split plain text into overlapping line-based windows."""
    lines = text.splitlines()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(lines) - overlap), step):
        window = lines[i: i + chunk_size]
        if not window:
            break
        chunks.append({
            "name": f"lines_{i+1}_{i+len(window)}",
            "type": "text_chunk",
            "code": "\n".join(window),
            "start": (i, 0),
            "end": (i + len(window), 0),
        })
    return chunks


def load_chunks_from_file(file_path: str):
    """Auto-detect language and extract chunks from any supported file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return [], "unknown"

    ext = os.path.splitext(file_path)[1].lower()
    lang_name = LANG_MAP.get(ext)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    if lang_name:
        try:
            lang = get_language(lang_name)
            parser.set_language(lang)
            code_chunks, imports = extract_chunks_treesitter(source, lang_name)
            all_chunks = code_chunks + imports
            print(f"Language: {lang_name} | Code chunks: {len(code_chunks)} | Imports: {len(imports)}")
            return all_chunks, lang_name
        except Exception as e:
            print(f"Tree-sitter failed for {lang_name}: {e} — falling back to text chunking")

    chunks = chunk_by_sliding_window(source)
    print(f"Plain-text fallback | Chunks: {len(chunks)}")
    return chunks, "plaintext"
