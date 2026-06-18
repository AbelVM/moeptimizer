"""Cache key registry for tracking context cache hits.

Predicts cache hit rate before sending context to the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
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
        self._prefix_entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._last_context_changed = False

    def record_cache_hit(
        self,
        context: str | list[dict[str, Any]],
        hit_tokens: int,
    ) -> None:
        """Record actual cache hit from backend response.

        This should be called after each request to track real cache performance.
        """
        if isinstance(context, list):
            context = self._serialize_context(context)
        prefix = self._static_prefix(context)

        self._record_cache_hit_for_key(
            self._hash_context(context),
            context,
            hit_tokens,
            self._entries,
        )
        self._record_cache_hit_for_key(
            self._hash_context(prefix),
            prefix,
            hit_tokens,
            self._prefix_entries,
        )

        self._evict_old_entries(self._entries)
        self._evict_old_entries(self._prefix_entries)

    def _record_cache_hit_for_key(
        self,
        key: str,
        context: str,
        hit_tokens: int,
        entries: OrderedDict[str, CacheEntry],
    ) -> None:
        """Record a cache hit in the provided entry map."""
        if key in entries:
            entry = entries[key]
            entry.hit_count += 1
            entry.context_size = len(context)
            entries.move_to_end(key)
        else:
            now = time.time()
            entry = CacheEntry(
                key=key,
                timestamp=now,
                hit_count=1,
                miss_count=0,
                context_size=len(context),
                context_hash=self._hash_content(context),
            )
            entries[key] = entry
            self._last_context_changed = True

    def register_context(
        self,
        context: str | list[dict[str, Any]],
        hit: bool = True,
    ) -> str:
        """Register a context and whether it hit the cache.

        Accepts either a string or a list of message dicts.
        """
        if isinstance(context, list):
            context = self._serialize_context(context)
        prefix = self._static_prefix(context)
        now = time.time()

        self._register_context_for_key(
            self._hash_context(context),
            context,
            hit,
            now,
            self._entries,
        )
        self._register_context_for_key(
            self._hash_context(prefix),
            prefix,
            hit,
            now,
            self._prefix_entries,
        )

        self._evict_old_entries(self._entries)
        self._evict_old_entries(self._prefix_entries)

        return self._hash_context(context)

    def _register_context_for_key(
        self,
        key: str,
        context: str,
        hit: bool,
        timestamp: float,
        entries: OrderedDict[str, CacheEntry],
    ) -> None:
        """Register a context hit/miss in the provided entry map."""
        if key in entries:
            entry = entries[key]
            if hit:
                entry.hit_count += 1
            else:
                entry.miss_count += 1
            entry.timestamp = timestamp
            entries.move_to_end(key)
        else:
            entry = CacheEntry(
                key=key,
                timestamp=timestamp,
                hit_count=1 if hit else 0,
                miss_count=0 if hit else 1,
                context_size=len(context),
                context_hash=self._hash_content(context),
            )
            entries[key] = entry
            self._last_context_changed = True

    def predict_hit_rate(
        self,
        context: str | list[dict[str, Any]],
    ) -> float:
        """Predict the cache hit rate for a context.

        Accepts either a string or a list of message dicts. Uses the maximum of
        exact-context and static-prefix hit rates so growing conversations still
        benefit from stable system/first-user prefix cache entries.
        """
        if isinstance(context, list):
            context = self._serialize_context(context)
        prefix = self._static_prefix(context)
        exact_hit_rate = self._predict_hit_rate_for_key(
            self._hash_context(context),
            self._entries,
        )
        prefix_hit_rate = self._predict_hit_rate_for_key(
            self._hash_context(prefix),
            self._prefix_entries,
        )
        return max(exact_hit_rate, prefix_hit_rate)

    def predict_static_prefix_hit_rate(
        self,
        context: str | list[dict[str, Any]],
    ) -> float:
        """Predict cache hit rate for the stable static prefix only."""
        if isinstance(context, list):
            context = self._serialize_context(context)
        prefix = self._static_prefix(context)
        return self._predict_hit_rate_for_key(
            self._hash_context(prefix),
            self._prefix_entries,
        )

    def _predict_hit_rate_for_key(
        self,
        key: str,
        entries: OrderedDict[str, CacheEntry],
    ) -> float:
        """Return hit rate for a key in the provided entry map."""
        entry = entries.get(key)
        if entry is None:
            return 0.0
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

        prefix_hits = sum(e.hit_count for e in self._prefix_entries.values())
        prefix_misses = sum(e.miss_count for e in self._prefix_entries.values())
        prefix_total = prefix_hits + prefix_misses

        return {
            "total_entries": len(self._entries),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(total_hits / max(total, 1), 4),
            "unique_contexts": len(self._entries),
            "prefix_entries": len(self._prefix_entries),
            "prefix_hits": prefix_hits,
            "prefix_misses": prefix_misses,
            "prefix_hit_rate": round(prefix_hits / max(prefix_total, 1), 4),
        }

    def _serialize_context(
        self,
        context: list[dict[str, Any]],
    ) -> str:
        """Serialize messages with role/order for stable cache keys."""
        parts: list[str] = []
        for msg in context:
            content = msg.get("content", "")
            parts.append(f"{msg.get('role', '')}:{content}")
        return "\n".join(parts)

    def _static_prefix(self, context: str) -> str:
        """Extract the stable system + first-user prefix from serialized context."""
        prefix_parts: list[str] = []
        seen_first_user = False

        for line in context.splitlines():
            role, separator, _content = line.partition(":")
            if not separator:
                continue
            if role == "system":
                prefix_parts.append(line)
            elif role == "user" and not seen_first_user:
                prefix_parts.append(line)
                seen_first_user = True
                break

        return "\n".join(prefix_parts)

    def _evict_old_entries(
        self,
        entries: OrderedDict[str, CacheEntry],
    ) -> None:
        """Evict the oldest entries from a cache entry map."""
        while len(entries) > self._max_size:
            entries.popitem(last=False)

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
        self._prefix_entries.clear()
        self._last_context_changed = True

    def save_to_disk(self, force: bool = False) -> None:
        """Persist cache registry to disk for cross-session reuse."""
        if not force and not self._last_context_changed:
            return
        PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {
                k: asdict(v) for k, v in self._entries.items()
            },
            "prefix_entries": {
                k: asdict(v) for k, v in self._prefix_entries.items()
            },
        }
        PERSISTENCE_PATH.write_text(json.dumps(data))
        self._last_context_changed = False

    def load_from_disk(self) -> None:
        """Load cache registry from disk."""
        if not PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(PERSISTENCE_PATH.read_text())
            for k, v in data.get("entries", {}).items():
                self._entries[k] = CacheEntry(**v)
            for k, v in data.get("prefix_entries", {}).items():
                self._prefix_entries[k] = CacheEntry(**v)
            self._evict_old_entries(self._entries)
            self._evict_old_entries(self._prefix_entries)
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
