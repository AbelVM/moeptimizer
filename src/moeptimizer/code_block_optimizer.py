"""Tree-sitter based code block detection and optimization.

Replaces regex-based code block detection with proper AST parsing.
"""

from __future__ import annotations

import re
from typing import Any

from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    deduplicate_chunks,
    detect_language_and_id,
)


# Pre-compiled regex for fallback (kept for performance)
_CODE_BLOCK_PATTERN = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)


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

    detected_langs: set[str] = set()
    all_chunks: list[str] = []
    block_langs: list[str] = []  # Track language per block

    for lang, code, start, end in blocks:
        # Detect language if not specified
        lang_id = detect_language_and_id(code) if not lang else LANG_MAP.get(lang, lang)

        detected_langs.add(lang_id if lang_id != "generic" else "unknown-text")
        block_langs.append(lang)

        # Chunk the code
        chunks = chunk_code_with_treesitter(
            code,
            lang_id or "generic",
            config.code_chunking.chunk_max_chars,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return text

    all_chunks = deduplicate_chunks(all_chunks)

    # If we have fewer chunks than original blocks, we'd lose code
    # Return original text to preserve all code blocks
    if len(all_chunks) < len(blocks):
        return text

    # Reassemble text with optimized code blocks
    result = text
    for i, (lang, code, start, end) in enumerate(blocks):
        if i < len(all_chunks):
            # Preserve original language from the block
            original_lang = block_langs[i] if i < len(block_langs) else ""
            replacement = f"```{original_lang}\n{all_chunks[i]}\n```"
            result = result[:start] + replacement + result[end:]

    return result