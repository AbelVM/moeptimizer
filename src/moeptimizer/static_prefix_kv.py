"""Static Prefix KV-Cache Reuse for MoE models.

Pre-computes and re-uses KV-cache for unchanging system/static tokens,
reducing cache fill overhead on repeated requests with the same static prefix.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistence path for KV-cache entries
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "kv_cache.pkl"


class StaticPrefixKVCache:
    """
    Pre-computes and caches KV-cache for static prefix content.

    When the same static prefix (system prompt + first user message) is
    detected across requests, the pre-computed KV-cache is reused instead
    of recomputing, significantly reducing prefill time for MoE models.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

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
        """Store KV-cache data for the static prefix."""
        prefix = self.get_static_prefix(messages)
        if not prefix:
            return ""

        key = self.get_cache_key(prefix)
        self._cache[key] = kv_data
        self._cache.move_to_end(key)

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
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached KV-cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 4),
        }

    def save_to_disk(self) -> None:
        """Persist KV-cache to disk for cross-session reuse."""
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PERSISTENCE_PATH.write_bytes(pickle.dumps(dict(self._cache)))
        except Exception as e:
            logger.warning("[StaticPrefixKV] Failed to save to disk: %s", e)

    def load_from_disk(self) -> None:
        """Load KV-cache from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = pickle.loads(_PERSISTENCE_PATH.read_bytes())
            self._cache = OrderedDict(data)
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
