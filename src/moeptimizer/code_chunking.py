"""Code chunking with Tree-Sitter for language-aware text splitting."""

from __future__ import annotations

import hashlib
from typing import Any

from moeptimizer.config import get_config

LANG_MAP: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "cpp": "cpp",
    "java": "java",
    "csharp": "c_sharp",
    "php": "php",
    "ruby": "ruby",
    "html": "html",
    "css": "css",
    "json": "json",
}

_parser_cache: dict[str, Any] = {}
_lang_cache: dict[str, str] = {}
_LANG_CACHE_MAX = 256


def _get_cached_parser(lang_id: str) -> Any | None:
    """Get a cached tree-sitter parser, or create and cache one."""
    if lang_id in _parser_cache:
        return _parser_cache[lang_id]
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang_id)
        _parser_cache[lang_id] = parser
        return parser
    except ImportError:
        # tree-sitter-language-pack not installed; fall back to line-based chunking
        return None
    except Exception:
        return None


def detect_language_and_id(code: str) -> str:
    """Detect programming language from code content.

    Uses pygments first, with heuristics to disambiguate common confusions
    (e.g., pygments confuses JavaScript with GDScript). Falls back to
    tree-sitter-language-pack detection if available.

    Results are cached to avoid repeated pygments invocations.
    """
    # Quick check: short code always generic (no cache needed)
    if len(code) < 40:
        return "generic"

    # Check cache first (content hash as key)
    code_hash = hashlib.md5(code.encode()).hexdigest()
    if code_hash in _lang_cache:
        return _lang_cache[code_hash]

    # Try pygments first
    pygments_result: str | None = None
    try:
        from pygments.lexers import guess_lexer

        lexer = guess_lexer(code)
        lexer_name = lexer.name.lower()
        for key, value in LANG_MAP.items():
            if key in lexer_name:
                result = value
                _lang_cache_put(code_hash, result)
                return result
        pygments_result = lexer_name
    except Exception:
        pass

    # Heuristic: pygments confuses JavaScript with GDScript.
    # Check for JS-specific patterns when GDScript was detected.
    if pygments_result and "gdscript" in pygments_result:
        js_indicators = [
            "require(", "console.log", "process.env", "import ", "export ",
            "=>", "let ", "const ", "async ", "await ", "fetch(",
        ]
        if any(indicator in code for indicator in js_indicators):
            result = "javascript"
            _lang_cache_put(code_hash, result)
            return result

    # Fallback to tree-sitter-language-pack detection
    try:
        from tree_sitter_language_pack import detect_language

        lang = detect_language(code)
        if lang and lang in LANG_MAP:
            result = LANG_MAP[lang]
            _lang_cache_put(code_hash, result)
            return result
    except Exception:
        pass

    result = "generic"
    _lang_cache_put(code_hash, result)
    return result


def _lang_cache_put(code_hash: str, result: str) -> None:
    """Put a language detection result in cache, evicting oldest if full."""
    _lang_cache[code_hash] = result
    while len(_lang_cache) > _LANG_CACHE_MAX:
        _lang_cache.pop(next(iter(_lang_cache)))


def chunk_text_fallback(text: str, max_chars: int) -> list[str]:
    """Split text into chunks by lines when Tree-Sitter is unavailable."""
    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for line in lines:
        if current_length + len(line) > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += len(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def chunk_code_with_treesitter(
    code: str,
    lang_id: str,
    max_chars: int | None = None,
) -> list[str]:
    """
    Split code into language-aware chunks using Tree-Sitter AST.

    Preserves file-level imports as a header prefix in each chunk.
    Falls back to line-based chunking for unknown languages or parse failures.

    Uses tree-sitter >= 0.25 API where:
    - tree.root_node() is callable
    - node.kind() returns the node type string
    - node.byte_range() returns ByteRange with .start and .end
    - node.child_count() is callable
    - node.child(i) returns the i-th child
    """
    config = get_config().code_chunking
    max_chars = max_chars or config.chunk_max_chars

    if lang_id == "generic":
        return chunk_text_fallback(code, max_chars)

    parser = _get_cached_parser(lang_id)
    if parser is None:
        return chunk_text_fallback(code, max_chars)

    try:
        tree = parser.parse(code)
        root_node = tree.root_node()
    except Exception:
        return chunk_text_fallback(code, max_chars)

    def _get_text(node: Any) -> str:
        """Extract text from a node using byte_range."""
        br = node.byte_range()
        return code[br.start : br.end]

    def _get_kind(node: Any) -> str:
        """Get the kind (type) of a node."""
        return node.kind()

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    file_headers = []
    header_types = {
        "import_statement",
        "import_declaration",
        "package_clause",
        "namespace_declaration",
        "comment",
        "module_declaration",
        "use_declaration",
        "require_statement",
    }
    for i in range(min(root_node.child_count(), 5)):
        child = root_node.child(i)
        if _get_kind(child) in header_types:
            file_headers.append(_get_text(child))
    header_prefix = "\n".join(file_headers) + "\n" if file_headers else ""

    def _add(text: str) -> None:
        nonlocal current_chunk, current_length
        current_chunk.append(text)
        current_length += len(text)

    def _flush() -> None:
        nonlocal current_chunk, current_length
        if current_chunk:
            chunks.append(header_prefix + "\n".join(current_chunk))
            current_chunk = []
            current_length = 0

    for i in range(root_node.child_count()):
        child = root_node.child(i)
        node_text = _get_text(child)
        node_len = len(node_text)

        if node_len > max_chars:
            if child.child_count() > 0:
                for j in range(child.child_count()):
                    sub = child.child(j)
                    sub_text = _get_text(sub)
                    sub_len = len(sub_text)
                    if current_length + sub_len > max_chars and current_chunk:
                        _flush()
                    _add(sub_text)
            else:
                _flush()
                for start in range(0, len(node_text), max_chars):
                    chunks.append(header_prefix + node_text[start : start + max_chars])
        else:
            if current_length + node_len > max_chars and current_chunk:
                _flush()
            _add(node_text)

    _flush()
    return chunks


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    """Remove duplicate code chunks by content hash."""
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        h = hashlib.md5(c.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique
