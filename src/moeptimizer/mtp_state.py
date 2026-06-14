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