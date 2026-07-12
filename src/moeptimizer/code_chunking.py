"""Code chunking with Tree-Sitter for language-aware text splitting."""

from __future__ import annotations

import hashlib
from typing import Any

from moeptimizer.config import get_config

# Curated fence-tag aliases -> tree-sitter-language-pack language id.
# Only targets that exist in the installed pack are kept (see _build_lang_map).
_LANG_ALIASES: dict[str, str] = {
    "py": "python",
    "py3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "sh": "bash",
    "shell": "bash",
    "zsh": "zsh",
    "yml": "yaml",
    "golang": "go",
    "cs": "csharp",
    "c#": "csharp",
    "kt": "kotlin",
    "rs": "rust",
    "rb": "ruby",
    "rake": "ruby",
    "pl": "perl",
    "pm": "perl",
    "hs": "haskell",
    "lisp": "commonlisp",
    "el": "elisp",
    "erl": "erlang",
    "ex": "elixir",
    "exs": "elixir",
    "ml": "ocaml",
    "sc": "scala",
    "scala": "scala",
    "md": "markdown",
    "docker": "dockerfile",
    "tf": "terraform",
    "sol": "solidity",
    "gql": "graphql",
    "graphql": "graphql",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "hxx": "cpp",
    "h": "c",
    "objc": "objc",
    "objectivec": "objc",
    "clj": "clojure",
    "jl": "julia",
    "nim": "nim",
    "dart": "dart",
    "swift": "swift",
    "groovy": "groovy",
    "r": "r",
    "lua": "lua",
    "sql": "sql",
    "zig": "zig",
    "vue": "vue",
    "svelte": "svelte",
    "proto": "proto",
    "make": "make",
    "cmake": "cmake",
    "toml": "toml",
    "ini": "ini",
    "json5": "json5",
    "jsonc": "json",
    "xml": "xml",
    "htm": "html",
    "sass": "scss",
    "less": "less",
    "php3": "php",
    "php4": "php",
    "php5": "php",
    "php7": "php",
    "php8": "php",
    "asm": "asm",
    "nasm": "nasm",
    "x86asm": "x86asm",
    "bat": "batch",
    "ps1": "powershell",
}


def _build_lang_map() -> dict[str, str]:
    """Build the fence-tag -> grammar-id map.

    The base is every grammar shipped by tree-sitter-language-pack (so the map
    can never reference a non-existent grammar), then curated aliases are
    layered on top. Falls back to a static subset if the pack is unavailable.
    """
    try:
        from tree_sitter_language_pack import manifest_languages

        base = {lang: lang for lang in manifest_languages()}
    except Exception:
        # Fallback subset: languages known to be valid without pack metadata.
        base = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "go": "go",
            "rust": "rust",
            "cpp": "cpp",
            "c": "c",
            "java": "java",
            "csharp": "csharp",
            "php": "php",
            "ruby": "ruby",
            "html": "html",
            "css": "css",
            "json": "json",
        }
    # Keep only aliases whose target grammar actually exists in the pack.
    base.update({k: v for k, v in _LANG_ALIASES.items() if v in base})
    return base


LANG_MAP: dict[str, str] = _build_lang_map()

_parser_cache: dict[str, Any] = {}
_lang_cache: dict[str, str] = {}
_LANG_CACHE_MAX = 256


def _get_cached_parser(lang_id: str) -> Any | None:
    """Get a cached tree-sitter parser, or create and cache one."""
    if lang_id in _parser_cache:
        return _parser_cache[lang_id]
    from tree_sitter_language_pack import get_parser

    parser = get_parser(lang_id)
    _parser_cache[lang_id] = parser
    return parser


def detect_language_and_id(code: str) -> str:
    """Detect programming language from code content.

    Uses tree-sitter-language-pack for accurate language detection (core dependency).
    Falls back to pygments if tree-sitter detection fails.

    Results are cached to avoid repeated invocations.
    """
    # Quick check: short code always generic (no cache needed)
    if len(code) < 40:
        return "generic"

    # Check cache first (content hash as key)
    code_hash = hashlib.md5(code.encode()).hexdigest()
    if code_hash in _lang_cache:
        return _lang_cache[code_hash]

    # Primary: tree-sitter-language-pack (core dependency)
    from tree_sitter_language_pack import detect_language

    lang = detect_language(code)
    if lang and lang in LANG_MAP:
        _lang_cache_put(code_hash, LANG_MAP[lang])
        return LANG_MAP[lang]

    # Fallback: pygments (core dependency)
    from pygments.lexers import guess_lexer

    lexer = guess_lexer(code)
    lexer_name = lexer.name.lower()
    for key, value in LANG_MAP.items():
        if key in lexer_name:
            _lang_cache_put(code_hash, value)
            return value

    _lang_cache_put(code_hash, "generic")
    return "generic"


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
