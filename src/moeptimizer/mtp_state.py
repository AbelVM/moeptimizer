"""MTP state management for context switching.

Serializes and restores MTP hidden states to maintain prediction quality
across context evictions.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class MTPStateManager:
    """
    Manages MTP hidden state serialization for context switching.

    When context is evicted and restored, MTP internal state is lost.
    This module preserves that state to maintain prediction quality.
    """

    def __init__(self, max_states: int = 100) -> None:
        self._states: OrderedDict[str, bytes] = OrderedDict()
        self._max_states = max_states
        self._stats: dict[str, int] = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
        }

    def save_state(
        self,
        context_hash: str,
        mtp_state: Any,
    ) -> None:
        """Serialize and save MTP state for a context."""
        try:
            # Serialize state to bytes
            state_bytes = pickle.dumps(mtp_state)

            # Store in cache
            self._states[context_hash] = state_bytes
            self._stats["saves"] += 1

            # Evict oldest if over limit
            while len(self._states) > self._max_states:
                self._states.popitem(last=False)

        except Exception as e:
            logger.warning("Failed to save MTP state: %s", e)

    def load_state(
        self,
        context_hash: str,
    ) -> Any | None:
        """Load MTP state for a context if available."""
        if context_hash in self._states:
            self._stats["hits"] += 1
            self._stats["loads"] += 1
            try:
                return pickle.loads(self._states[context_hash])
            except Exception as e:
                logger.warning("Failed to load MTP state: %s", e)
                self._stats["misses"] += 1
        else:
            self._stats["misses"] += 1

        return None

    def get_state_key(
        self,
        messages: list[dict[str, Any]],
        overlap_tokens: int = 128,
    ) -> str:
        """Generate state key for context.

        Uses the overlap region to ensure state continuity.
        """
        # Get the last N tokens of context for state key
        content = "".join(m.get("content", "") for m in messages)
        overlap = content[-overlap_tokens:] if len(content) > overlap_tokens else content
        return hashlib.md5(overlap.encode()).hexdigest()[:16]

    def align_prediction_boundary(
        self,
        messages: list[dict[str, Any]],
        target_boundary: int = 128,
    ) -> list[dict[str, Any]]:
        """Align context to MTP prediction boundary.

        MTP predictions work best when context length is a multiple of
        the prediction boundary (typically 128 tokens for Qwen3.6-35B-A3B-MTP).

        This method pads or trims context to align to the boundary,
        preserving prediction quality.
        """
        if not messages:
            return messages

        # Calculate current context length in tokens
        total_chars = sum(len(m.get("content", "")) for m in messages)
        # Rough estimate: ~4 chars per token
        current_tokens = total_chars // 4

        # Find the boundary multiple
        remainder = current_tokens % target_boundary
        if remainder == 0:
            return messages  # Already aligned

        # Calculate padding needed
        padding_needed = target_boundary - remainder

        # Add padding to the last message (non-assistant to preserve chat template)
        # Find the last non-assistant message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "assistant":
                # Add padding as newlines (invisible to model but counts as tokens)
                result = [dict(m) for m in messages]
                result[i] = {
                    **result[i],
                    "content": result[i].get("content", "") + "\n" * padding_needed,
                }
                return result

        return messages

    def get_stats(self) -> dict[str, int]:
        """Get state management statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all saved states."""
        self._states.clear()
        self._stats = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
        }


# Global state manager instance
_state_manager: MTPStateManager | None = None


def get_mtp_state_manager() -> MTPStateManager:
    """Get or create the global MTP state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = MTPStateManager()
    return _state_manager