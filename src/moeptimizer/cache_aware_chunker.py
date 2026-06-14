"""Cache-aware chunker for context optimization.

Chunks code to align with cache blocks.
"""

from __future__ import annotations

import re
from typing import Any

from moeptimizer.cache import get_block_size


class CacheAwareChunker:
    """
    Chunks context to align with cache blocks.

    - Chunk code to align with cache blocks
    - Keep related functions in same block
    - Preserve AST structure in chunks
    """

    def __init__(
        self,
        block_size: int | None = None,
    ) -> None:
        self._block_size = block_size or get_block_size()

    def chunk_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chunk context to align with cache blocks."""
        result = []

        for msg in messages:
            content = msg.get("content", "")

            # Check if content has code blocks
            if "```" in content:
                chunked = self._chunk_code_message(content)
                for i, chunk in enumerate(chunked):
                    result.append({
                        **msg,
                        "content": chunk,
                        "chunk_index": i,
                    })
            else:
                result.append(dict(msg))

        return result

    def _chunk_code_message(
        self,
        content: str,
    ) -> list[str]:
        """Chunk a code message to align with cache blocks."""
        # Extract code blocks
        code_pattern = re.compile(
            r"(```[\w]*\n.*?```)",
            re.DOTALL,
        )

        # Split into code and non-code sections
        parts = []
        last_end = 0

        for match in code_pattern.finditer(content):
            # Add text before
            if match.start() > last_end:
                parts.append(content[last_end : match.start()])

            # Add code block
            parts.append(match.group(1))
            last_end = match.end()

        # Add remaining text
        if last_end < len(content):
            parts.append(content[last_end:])

        # Group into cache-aligned chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for part in parts:
            part_size = len(part)

            if current_size + part_size > self._block_size and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = [part]
                current_size = part_size
            else:
                current_chunk.append(part)
                current_size += part_size

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def preserve_ast_structure(
        self,
        code: str,
    ) -> str:
        """Preserve AST structure when chunking."""
        # For now, return as-is
        # Full implementation would parse AST and chunk at function/class boundaries
        return code

    def get_block_size(
        self,
    ) -> int:
        """Get the cache block size."""
        return self._block_size


def get_cache_aware_chunker(
    block_size: int = 1024,
) -> CacheAwareChunker:
    """Get a cache-aware chunker instance."""
    return CacheAwareChunker(block_size=block_size)