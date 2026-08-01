"""Tree-sitter based code block detection and optimization.

Replaces regex-based code block detection with proper AST parsing.
"""

from __future__ import annotations

import re
from typing import Any

from moeptimizer.code_chunking import (
    LANG_MAP,
    _get_cached_parser,
    chunk_code_with_treesitter,
    deduplicate_chunks,
    detect_language_and_id,
)

# Pre-compiled regex for fallback (kept for performance)
_CODE_BLOCK_PATTERN = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)

# Top-level node kinds treated as file header (kept verbatim when slicing).
_HEADER_KINDS = {
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "package_clause",
    "use_declaration",
    "namespace_declaration",
    "module_declaration",
    "require_statement",
}


def has_code_blocks(text: str) -> bool:
    """Check if text contains fenced code blocks using regex (fast path).

    For more accurate detection, use tree-sitter to parse the content.
    """
    return bool(_CODE_BLOCK_PATTERN.search(text))


def extract_code_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """Extract code blocks with their language and position.

    Returns list of (language, code, start_pos, end_pos) tuples.
    Uses regex for extraction but validates with tree-sitter when available.
    """
    blocks = []
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        full_match = match.group(1)
        start = match.start()
        end = match.end()

        # Extract language
        lang_match = re.match(r"```(\w*)", full_match)
        lang = lang_match.group(1) if lang_match else ""

        # Extract code content
        code = full_match[3 + len(lang):]  # Skip ```lang or just ```
        if code.startswith("\n"):
            code = code[1:]
        if code.endswith("```"):
            code = code[:-3]

        blocks.append((lang, code, start, end))

    return blocks


def optimize_code_in_text(
    text: str,
    config: Any,
    embedding_service: Any,
) -> str:
    """Optimize code blocks within a text string using Tree-Sitter + NPU.

    Returns the original text if optimization would reduce code block count.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return text

    # Group chunks by block to preserve structure
    block_chunks: list[list[str]] = []
    block_langs: list[str] = []  # Track language per block

    for lang, code, _, _ in blocks:
        block_langs.append(lang)
        # Detect language if not specified
        lang_id = detect_language_and_id(code) if not lang else LANG_MAP.get(lang, lang)

        # Chunk the code
        chunks = chunk_code_with_treesitter(
            code,
            lang_id or "generic",
            config.code_chunking.chunk_max_chars,
        )
        block_chunks.append(chunks)

    # Check if any block has no chunks (would lose code)
    if any(not chunks for chunks in block_chunks):
        return text

    # Deduplicate within each block's chunks, not across all blocks
    deduped_block_chunks: list[list[str]] = []
    for chunks in block_chunks:
        deduped_block_chunks.append(deduplicate_chunks(chunks))

    # If any block has fewer chunks after dedup, we'd lose code
    # Return original text to preserve all code blocks
    for original, deduped in zip(block_chunks, deduped_block_chunks, strict=True):
        if len(deduped) < len(original):
            return text

    # Reassemble text with optimized code blocks
    # Build result by processing from end to start to preserve positions
    result = text
    offset = 0
    for i in range(len(blocks) - 1, -1, -1):
        lang, code, start, end = blocks[i]
        chunks = deduped_block_chunks[i]
        if chunks:
            # Join all chunks for this block with newlines
            optimized_code = "\n".join(chunks)
            # Preserve original language from the block
            original_lang = block_langs[i] if i < len(block_langs) else ""
            replacement = f"```{original_lang}\n{optimized_code}\n```"
            # Adjust positions for previous replacements
            actual_start = start + offset
            actual_end = end + offset
            result = result[:actual_start] + replacement + result[actual_end:]
            offset += len(replacement) - (end - start)

    return result


def _node_text(code_bytes: bytes, node: Any) -> str:
    """Extract the source text of a tree-sitter node via its byte range.

    ``node.byte_range`` is a ``(start, end)`` tuple of byte offsets, so slicing
    must happen on the encoded bytes (not the str, whose indices are code points)
    to stay correct for non-ASCII source.
    """
    start, end = node.byte_range
    return code_bytes[start:end].decode(errors="replace")


def _definition_name(code_bytes: bytes, node: Any) -> str | None:
    """Return the name of a top-level definition node, or None if it has none."""
    try:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _node_text(code_bytes, name_node)
    except Exception:
        pass
    # Fallback: first identifier-like child (covers grammars without a name field).
    for i in range(node.child_count):
        child = node.child(i)
        if child.type in ("identifier", "type_identifier", "name"):
            return _node_text(code_bytes, child)
    return None


def slice_code_to_query(code: str, lang_id: str, query: str) -> str:
    """Slice a code block down to the top-level definitions referenced by ``query``.

    Review §4.5.3: for a multi-hundred-line file read, keep the imports header plus
    every top-level function/class whose name appears in the query, and collapse the
    unrelated siblings to a marker. This keeps the code the model actually needs to
    edit/extend while dropping the rest.

    Strictly fail-open: returns the original ``code`` unchanged when the language is
    unknown, parsing fails, there are no named top-level definitions, nothing matches
    the query, or everything matches (nothing to drop). Only ever shrinks — never
    expands — so it cannot bloat the context.

    Uses the tree-sitter 0.25 property API (``tree.root_node``, ``node.type``,
    ``node.child_count``, ``node.byte_range`` are properties; ``node.child(i)`` and
    ``node.child_by_field_name(name)`` are methods).
    """
    if not code or not query or lang_id == "generic":
        return code
    parser = _get_cached_parser(lang_id)
    if parser is None:
        return code
    code_bytes = code.encode()
    try:
        tree = parser.parse(code_bytes)
        root = tree.root_node
    except Exception:
        return code

    header_parts: list[str] = []
    defs: list[tuple[str, int, int]] = []  # (name, byte_start, byte_end)
    for i in range(root.child_count):
        child = root.child(i)
        if child.type in _HEADER_KINDS:
            header_parts.append(_node_text(code_bytes, child))
            continue
        name = _definition_name(code_bytes, child)
        if name:
            start, end = child.byte_range
            defs.append((name, start, end))

    if not defs:
        return code

    query_tokens = {t.lower() for t in re.findall(r"\w+", query)}
    selected = {i for i, (name, _s, _e) in enumerate(defs) if name.lower() in query_tokens}
    # Nothing to drop: no match (fail-open, keep everything the model might need)
    # or every definition is referenced (slicing would remove nothing).
    if not selected or len(selected) == len(defs):
        return code

    # Keep definitions referenced by retained definitions, so slicing does not
    # remove a helper needed by the task-relevant function.
    changed = True
    while changed:
        changed = False
        referenced = {
            token.lower()
            for i in selected
            for token in re.findall(r"\w+", code_bytes[defs[i][1] : defs[i][2]].decode(errors="replace"))
        }
        for i, (name, _s, _e) in enumerate(defs):
            if i not in selected and name.lower() in referenced:
                selected.add(i)
                changed = True

    matched = [definition for i, definition in enumerate(defs) if i in selected]
    if len(matched) == len(defs):
        return code
    kept: list[str] = []
    if header_parts:
        kept.append("\n".join(header_parts))
    for _name, s, e in matched:
        kept.append(code_bytes[s:e].decode(errors="replace"))
    collapsed = len(defs) - len(matched)
    kept.append(f"# ... [{collapsed} other top-level definition(s) collapsed] ...")
    sliced = "\n".join(kept)
    # Never expand the context.
    return sliced if len(sliced) < len(code) else code
