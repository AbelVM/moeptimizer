"""Context aligner for cache block optimization.

Aligns context to cache block boundaries to maximize prefix cache hits.
"""

from __future__ import annotations

from typing import Any

# Cache block size for Qwen models
CACHE_BLOCK_SIZE = 1024


class ContextAligner:
    """
    Aligns context to cache block boundaries.

    For MoE models, KV-cache fill is extremely slow. This module
    optimizes context structure to maximize cache hit rates.
    """

    def __init__(
        self,
        block_size: int = CACHE_BLOCK_SIZE,
    ) -> None:
        self._block_size = block_size

    def align_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Align context to cache block boundaries.

        Returns optimized message list with aligned static layer.
        """
        if not messages:
            return messages

        # Find static layer end
        static_end = self._find_static_layer_end(messages)
        if static_end == 0:
            return messages

        # Calculate current static layer size
        static_chars = sum(
            len(m.get("content", "")) for m in messages[:static_end]
        )

        # Check if alignment is needed
        remainder = static_chars % self._block_size
        if remainder == 0:
            return messages

        # Add padding to align to next block boundary
        padding_needed = self._block_size - remainder
        if padding_needed > 100:  # Don't add excessive padding
            return messages

        # Create aligned copy
        result = [dict(m) for m in messages]
        result[static_end - 1] = {
            **result[static_end - 1],
            "content": result[static_end - 1].get("content", "") + "\n" * padding_needed,
        }

        return result

    def _find_static_layer_end(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Find the end of the static layer (system + first user)."""
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif msg.get("role") == "user" and static_end > 0:
                static_end = i + 1
                break
            elif msg.get("role") == "user" and static_end == 0:
                static_end = i + 1
                break
        return static_end

    def optimize_block_boundaries(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Optimize message boundaries to align with cache blocks.

        Groups related messages to minimize cache fragmentation.
        """
        # For now, just return the aligned context
        return self.align_context(messages)


def get_context_aligner(
    block_size: int = CACHE_BLOCK_SIZE,
) -> ContextAligner:
    """Get a context aligner instance."""
    return ContextAligner(block_size=block_size)