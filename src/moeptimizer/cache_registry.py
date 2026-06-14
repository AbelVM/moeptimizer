"""Cache key registry for tracking context cache hits.

Predicts cache hit rate before sending context to the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Cache block size for Qwen models
CACHE_BLOCK_SIZE = 128

# Persistence path
PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "cache_registry.json"


@dataclass
class CacheEntry:
    """Entry in the cache key registry."""

    key: str
    timestamp: float
    hit_count: int = 0
    miss_count: int = 0
    context_size: int = 0
    context_hash: str = ""


class CacheKeyRegistry:
    """
    Registry for tracking context cache keys.

    Predicts whether a context will hit the model's prefix cache
    based on previous interactions.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    def record_cache_hit(
        self,
        context: str | list[dict[str, Any]],
        hit_tokens: int,
    ) -> None:
        """Record actual cache hit from backend response.

        This should be called after each request to track real cache performance.
        """
        if isinstance(context, list):
            context = "".join(m.get("content", "") for m in context)
        key = self._hash_context(context)

        if key in self._entries:
            entry = self._entries[key]
            entry.hit_count += 1
            entry.context_size = len(context)
            self._entries.move_to_end(key)
        else:
            # Record the hit for future prediction
            now = time.time()
            entry = CacheEntry(
                key=key,
                timestamp=now,
                hit_count=1,
                miss_count=0,
                context_size=len(context),
                context_hash=self._hash_content(context),
            )
            self._entries[key] = entry

        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def register_context(
        self,
        context: str | list[dict[str, Any]],
        hit: bool = True,
    ) -> str:
        """Register a context and whether it hit the cache.

        Accepts either a string or a list of message dicts.
        """
        if isinstance(context, list):
            context = "".join(m.get("content", "") for m in context)
        key = self._hash_context(context)
        now = time.time()

        if key in self._entries:
            entry = self._entries[key]
            if hit:
                entry.hit_count += 1
            else:
                entry.miss_count += 1
            entry.timestamp = now
            self._entries.move_to_end(key)
        else:
            entry = CacheEntry(
                key=key,
                timestamp=now,
                hit_count=1 if hit else 0,
                miss_count=0 if hit else 1,
                context_size=len(context),
                context_hash=self._hash_content(context),
            )
            self._entries[key] = entry

        # Evict oldest if over limit
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

        return key

    def predict_hit_rate(
        self,
        context: str | list[dict[str, Any]],
    ) -> float:
        """Predict the cache hit rate for a context.

        Accepts either a string or a list of message dicts.
        """
        if isinstance(context, list):
            context = "".join(m.get("content", "") for m in context)
        key = self._hash_context(context)
        if key in self._entries:
            entry = self._entries[key]
            total = entry.hit_count + entry.miss_count
            if total > 0:
                return entry.hit_count / total
        return 0.0

    def get_cache_stats(
        self,
    ) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._entries.values())
        total_misses = sum(e.miss_count for e in self._entries.values())
        total = total_hits + total_misses

        return {
            "total_entries": len(self._entries),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(total_hits / max(total, 1), 4),
            "unique_contexts": len(self._entries),
        }

    def _hash_context(
        self,
        context: str,
    ) -> str:
        """Hash context for cache key lookup.

        Uses 32 hex chars (128 bits) to minimize collision risk.
        """
        return hashlib.md5(context.encode("utf8")).hexdigest()[:32]

    def _hash_content(
        self,
        content: str,
    ) -> str:
        """Hash content for full comparison."""
        return hashlib.md5(content.encode("utf8")).hexdigest()

    def clear(
        self,
    ) -> None:
        """Clear the registry."""
        self._entries.clear()

    def save_to_disk(self) -> None:
        """Persist cache registry to disk for cross-session reuse."""
        PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {
                k: asdict(v) for k, v in self._entries.items()
            },
        }
        PERSISTENCE_PATH.write_text(json.dumps(data))

    def load_from_disk(self) -> None:
        """Load cache registry from disk."""
        if not PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(PERSISTENCE_PATH.read_text())
            for k, v in data.get("entries", {}).items():
                self._entries[k] = CacheEntry(**v)
        except Exception:
            pass  # Ignore errors, start fresh


# Global registry instance
_registry: CacheKeyRegistry | None = None


def get_cache_registry() -> CacheKeyRegistry:
    """Get or create the global cache registry."""
    global _registry
    if _registry is None:
        _registry = CacheKeyRegistry()
    return _registry