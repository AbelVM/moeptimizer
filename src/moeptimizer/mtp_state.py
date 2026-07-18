"""MTP state management — NON-FUNCTIONAL placeholder (see review03.md §2.1/§10).

MTP (multi-token prediction) is a model-internal decoder optimization. A
client-side OpenAI proxy CANNOT read or write MTP hidden states or draft tokens
— there is no OpenAI field for them. This module therefore cannot preserve MTP
state. It is retained only as inert scaffolding behind a disabled-by-default
config flag so existing imports keep working; `save_state` is never called from
the optimizer. Do not rely on this for any MTP optimization.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class MTPStateManager:
    """
    Inert MTP-state scaffolding (non-functional; see module docstring).

    MTP hidden state cannot be captured or restored by an OpenAI client proxy.
    This class exists only so imports resolve; it performs no MTP optimization.
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
        encode: Callable[[str], list[int]] | None = None,
    ) -> str:
        """Generate a state key from the trailing context (review §6 bug #6).

        The previous implementation sliced the last ``overlap_tokens`` *characters*
        despite the parameter name saying "tokens", which collided across very
        different contexts. This now tokenizes and hashes the last
        ``overlap_tokens`` *tokens* when a tokenizer is available, falling back to
        characters only if tokenization fails. (This key is inert — see module
        docstring — but the bug is fixed so the scaffolding is at least correct.)

        ``encode`` is an optional ``str -> list[int]`` callable (e.g. the
        optimizer's ``TokenCounter._encode``) to avoid reloading a tokenizer.
        """
        content = "".join(m.get("content", "") for m in messages)
        if not content:
            return hashlib.md5(b"").hexdigest()[:32]
        if encode is not None:
            try:
                ids = encode(content)
                tail = ids[-overlap_tokens:] if len(ids) > overlap_tokens else ids
                overlap = " ".join(str(t) for t in tail)
                return hashlib.md5(overlap.encode()).hexdigest()[:32]
            except Exception:
                pass
        try:  # pragma: no cover - best effort; depends on transformers availability
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-32B", local_files_only=True, trust_remote_code=False
            )
            ids = tok.encode(content, add_special_tokens=False)
            tail = ids[-overlap_tokens:] if len(ids) > overlap_tokens else ids
            overlap = " ".join(str(t) for t in tail)
        except Exception:
            overlap = content[-overlap_tokens:] if len(content) > overlap_tokens else content
        return hashlib.md5(overlap.encode()).hexdigest()[:32]

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
