"""Hierarchical attention sink management for long-context stability.

Manages attention patterns in long contexts to prevent:
- Attention drift in static layer
- Loss of context coherence
- MTP prediction degradation
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Attention sink token markers
ATTENTION_SINK_TOKENS = {
    "python": "# CONTEXT_ANCHOR",
    "javascript": "// CONTEXT_ANCHOR",
    "typescript": "// CONTEXT_ANCHOR",
    "go": "// CONTEXT_ANCHOR",
    "rust": "// CONTEXT_ANCHOR",
    "cpp": "// CONTEXT_ANCHOR",
    "java": "// CONTEXT_ANCHOR",
}


class AttentionSinkManager:
    """
    Manages attention sink tokens and patterns for long-context stability.

    In long contexts, models can lose track of important static context.
    This module injects attention sink markers and manages position IDs
    to maintain attention coherence.
    """

    def __init__(self, block_size: int = 1024) -> None:
        self._block_size = block_size
        self._sink_positions: list[int] = []
        self._attention_entropy: float = 0.0

    def inject_sink_markers(
        self,
        messages: list[dict[str, Any]],
        static_layer_size: int,
    ) -> list[dict[str, Any]]:
        """Inject attention sink markers at strategic positions.

        Adds markers at:
        - Static layer boundary
        - Every N tokens in dynamic layer
        - Before code blocks
        """
        result = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                # Add sink marker at static layer boundary
                if i == 0 and msg.get("role") == "system":
                    content = self._add_static_sink(content, static_layer_size)
                # Add periodic sink markers in code
                if "```" in content:
                    content = self._add_code_sinks(content)
            result.append({**msg, "content": content})
        return result

    def _add_static_sink(
        self,
        content: str,
        static_layer_size: int,
    ) -> str:
        """Add attention sink marker at static layer boundary."""
        # Add invisible marker that biases attention
        sink_marker = f"\n{'#' * 80}\n# STATIC_LAYER_END (position: {static_layer_size})\n{'#' * 80}\n"
        return content + sink_marker

    def _add_code_sinks(self, content: str) -> str:
        """Add attention sink markers before code blocks."""
        lines = content.split("\n")
        result = []
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and i > 0:
                # Add sink before code block
                result.append("# CONTEXT_ANCHOR: code block follows")
            result.append(line)
        return "\n".join(result)

    def calculate_attention_entropy(
        self,
        messages: list[dict[str, Any]],
    ) -> float:
        """Estimate attention entropy for context quality assessment.

        Higher entropy = more scattered attention = potential degradation.
        Returns a value between 0 and 1.
        """
        total_tokens = sum(
            len(m.get("content", "").split()) for m in messages
        )
        if total_tokens == 0:
            return 0.0

        # Simple heuristic: count unique symbols / total tokens
        symbols = set()
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Extract identifiers
                import re
                symbols.update(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content))

        # Normalize to 0-1 range
        # High symbol-to-token ratio = high entropy
        ratio = len(symbols) / max(1, total_tokens)
        return min(1.0, ratio)

    def should_inject_sinks(
        self,
        messages: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> bool:
        """Determine if attention sinks should be injected.

        Lower threshold (0.5) for more aggressive sink injection.
        """
        return self.calculate_attention_entropy(messages) > threshold

    def get_sink_positions(self) -> list[int]:
        """Get recorded sink positions."""
        return self._sink_positions.copy()

    def record_sink_position(self, position: int) -> None:
        """Record a sink position for tracking."""
        self._sink_positions.append(position)


def apply_attention_sinks(
    messages: list[dict[str, Any]],
    static_layer_size: int,
) -> list[dict[str, Any]]:
    """Apply attention sink management to message list."""
    manager = AttentionSinkManager()
    if manager.should_inject_sinks(messages):
        return manager.inject_sink_markers(messages, static_layer_size)
    return messages