"""MTP-Head State Checkpointing.

Persists per-head hidden states for recurring function signatures,
re-using them when the same signature appears again.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class MTPHeadStateCheckpoint:
    """
    Persists per-head hidden states for recurring function signatures.

    When the same function signature appears again, the cached MTP head
    states are reused instead of recomputing from scratch, improving
    MTP prediction accuracy and reducing computation.
    """

    def __init__(self, max_checkpoints: int = 256) -> None:
        self._checkpoints: OrderedDict[str, dict[int, Any]] = OrderedDict()
        self._max_checkpoints = max_checkpoints
        self._stats: dict[str, int] = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
        }

    def _signature_key(self, signature: str) -> str:
        """Generate a stable key for a function signature.

        Uses 32 hex chars (128 bits) to minimize collision risk.
        """
        normalized = signature.strip().lower()
        return hashlib.md5(normalized.encode("utf8")).hexdigest()[:32]

    def save_head_states(
        self,
        signature: str,
        head_states: dict[int, Any],
    ) -> str:
        """Save MTP head states for a function signature.

        Args:
            signature: The function signature (e.g., "def foo(x: int) -> str")
            head_states: Dict mapping head_idx -> hidden state

        Returns:
            The checkpoint key
        """
        key = self._signature_key(signature)

        # Serialize each head state
        serialized: dict[int, bytes] = {}
        for head_idx, state in head_states.items():
            try:
                serialized[head_idx] = pickle.dumps(state)
            except Exception as e:
                logger.debug("[MTPCheckpoint] Failed to serialize head %d: %s", head_idx, e)

        self._checkpoints[key] = serialized
        self._checkpoints.move_to_end(key)

        # Evict oldest if over limit
        while len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints.popitem(last=False)

        self._stats["saves"] += 1
        logger.debug("[MTPCheckpoint] Saved states for signature key=%s", key[:16])
        return key

    def load_head_states(
        self,
        signature: str,
    ) -> dict[int, Any] | None:
        """Load MTP head states for a function signature.

        Args:
            signature: The function signature to look up

        Returns:
            Dict mapping head_idx -> deserialized hidden state, or None
        """
        key = self._signature_key(signature)

        if key in self._checkpoints:
            self._stats["hits"] += 1
            self._stats["loads"] += 1
            serialized = self._checkpoints[key]
            self._checkpoints.move_to_end(key)

            try:
                result: dict[int, Any] = {}
                for head_idx, state_bytes in serialized.items():
                    result[head_idx] = pickle.loads(state_bytes)
                logger.debug("[MTPCheckpoint] Loaded states for signature key=%s", key[:16])
                return result
            except Exception as e:
                logger.warning("[MTPCheckpoint] Failed to deserialize states: %s", e)
                self._stats["misses"] += 1
                self._checkpoints.pop(key, None)
                return None

        self._stats["misses"] += 1
        return None

    def get_or_create(
        self,
        signature: str,
        create_fn,
    ) -> dict[int, Any]:
        """Load cached states or create and cache new ones.

        Args:
            signature: The function signature
            create_fn: Function to call if not cached (receives signature)

        Returns:
            Dict mapping head_idx -> hidden state
        """
        cached = self.load_head_states(signature)
        if cached is not None:
            return cached

        new_states = create_fn(signature)
        self.save_head_states(signature, new_states)
        return new_states

    def has_checkpoint(self, signature: str) -> bool:
        """Check if a checkpoint exists for a signature."""
        key = self._signature_key(signature)
        return key in self._checkpoints

    def invalidate(self, signature: str) -> None:
        """Invalidate a checkpoint for a signature."""
        key = self._signature_key(signature)
        self._checkpoints.pop(key, None)

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._stats = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
        }

    def get_stats(self) -> dict[str, int]:
        """Get checkpoint statistics."""
        return dict(self._stats)

    def get_signature_keys(self) -> list[str]:
        """Get list of all cached signature keys (truncated)."""
        return [k[:16] for k in self._checkpoints]


# Global instance
_mtp_checkpoint: MTPHeadStateCheckpoint | None = None


def get_mtp_head_checkpoint(max_checkpoints: int = 256) -> MTPHeadStateCheckpoint:
    """Get or create the global MTP head state checkpoint."""
    global _mtp_checkpoint
    if _mtp_checkpoint is None:
        _mtp_checkpoint = MTPHeadStateCheckpoint(max_checkpoints=max_checkpoints)
    return _mtp_checkpoint
