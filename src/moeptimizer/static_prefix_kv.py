"""Static-prefix fast path (NOT a real KV cache).

HISTORICAL NOTE / honesty: this module was originally named and documented as a
"KV-cache reuse" feature. It is NOT. A client-side OpenAI proxy cannot read or
write the backend model's KV-cache tensors — there is no OpenAI field for them.
What this module actually does is store a *text memo* (the system + first-user
prefix bytes) and short-circuit the optimization pipeline when the incoming
prefix is byte-identical and already under budget. The real KV reuse is done by
the backend's own native prefix cache; this module only avoids redundant proxy
work. See review03.md §2.1 / §9.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistence path for the prefix text memo
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "kv_cache.pkl"


class StaticPrefixKVCache:
    """
    Fast path for an unchanged static prefix (text memo, NOT model KV).

    When the same static prefix (system prompt + first user message) is detected
    across requests, the pipeline can be short-circuited. This stores the prefix
    *text*, not KV tensors, and therefore does not reuse model KV — the backend's
    own prefix cache does that. Keep enabled for the latency win, but do not rely
    on it for KV reuse.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._last_context_changed = False

    def get_static_prefix(self, messages: list[dict[str, Any]]) -> str:
        """Extract the static prefix (system + first user) from messages."""
        static_parts: list[str] = []
        found_first_user = False

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                static_parts.append(f"system:{content}")
            elif role == "user" and not found_first_user:
                static_parts.append(f"user:{content}")
                found_first_user = True
                break

        return "\n".join(static_parts)

    def get_cache_key(self, static_prefix: str) -> str:
        """Generate a 32-hex-char cache key for a static prefix."""
        return hashlib.md5(static_prefix.encode("utf8")).hexdigest()[:32]

    def get(self, messages: list[dict[str, Any]]) -> bytes | None:
        """Get cached KV-cache for the static prefix, if available."""
        prefix = self.get_static_prefix(messages)
        if not prefix:
            return None

        key = self.get_cache_key(prefix)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            logger.debug("[StaticPrefixKV] Cache hit for prefix key=%s", key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(self, messages: list[dict[str, Any]], kv_data: bytes) -> str:
        """Store KV-cache data for the static prefix.

        Only marks the cache as changed (triggering a disk write) when the key is
        new or the stored data actually differs, so repeated puts of an identical
        stable prefix do not rewrite the pickle every turn.
        """
        prefix = self.get_static_prefix(messages)
        if not prefix:
            return ""

        key = self.get_cache_key(prefix)
        existing = self._cache.get(key)
        if existing is not None and existing == kv_data:
            # Identical prefix already cached: no change, skip the disk write.
            self._cache.move_to_end(key)
            return key

        self._cache[key] = kv_data
        self._cache.move_to_end(key)
        self._last_context_changed = True

        # Evict oldest if over limit
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

        logger.debug("[StaticPrefixKV] Cached KV for prefix key=%s", key)
        return key

    def invalidate(self, messages: list[dict[str, Any]]) -> None:
        """Invalidate cached KV-cache for a static prefix."""
        prefix = self.get_static_prefix(messages)
        if not prefix:
            return

        key = self.get_cache_key(prefix)
        if self._cache.pop(key, None) is not None:
            self._last_context_changed = True

    def clear(self) -> None:
        """Clear all cached KV-cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._last_context_changed = True

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 4),
        }

    def save_to_disk(self, force: bool = False) -> None:
        """Persist KV-cache to disk for cross-session reuse."""
        if not force and not self._last_context_changed:
            return
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PERSISTENCE_PATH.write_bytes(pickle.dumps(dict(self._cache)))
            self._last_context_changed = False
        except Exception as e:
            logger.warning("[StaticPrefixKV] Failed to save to disk: %s", e)

    def load_from_disk(self) -> None:
        """Load KV-cache from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = pickle.loads(_PERSISTENCE_PATH.read_bytes())
            self._cache = OrderedDict(data)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        except Exception as e:
            logger.warning("[StaticPrefixKV] Failed to load from disk: %s", e)


# Global instance
_kv_cache: StaticPrefixKVCache | None = None


def get_static_prefix_kv_cache() -> StaticPrefixKVCache:
    """Get or create the global static prefix KV-cache."""
    global _kv_cache
    if _kv_cache is None:
        _kv_cache = StaticPrefixKVCache()
        _kv_cache.load_from_disk()
    return _kv_cache
