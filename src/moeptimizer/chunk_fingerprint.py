"""Chunk Fingerprinting & Reuse for compressed code chunks.

Generates SHA-256 fingerprints for compressed code chunks; identical chunks
are re-used across turns, eliminating redundant compression and embedding work.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ChunkFingerprintCache:
    """
    Caches compressed code chunks by SHA-256 fingerprint.

    When the same code chunk is encountered again (even after compression),
    the cached result is reused instead of recomputing. This eliminates
    redundant compression and embedding work across turns.
    """

    def __init__(self, max_entries: int = 2048) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def fingerprint(self, content: str) -> str:
        """Generate a SHA-256 fingerprint for content.

        Uses the full SHA-256 (64 hex chars) for maximum collision resistance.
        """
        return hashlib.sha256(content.encode("utf8")).hexdigest()

    def get(self, content: str) -> dict[str, Any] | None:
        """Get cached result for a content fingerprint.

        Args:
            content: The content to look up

        Returns:
            Cached result dict or None if not found
        """
        fp = self.fingerprint(content)
        if fp in self._cache:
            self._hits += 1
            self._cache.move_to_end(fp)
            logger.debug("[ChunkFingerprint] Cache hit for fp=%s", fp[:16])
            return self._cache[fp]

        self._misses += 1
        return None

    def put(self, content: str, result: dict[str, Any]) -> str:
        """Store a result keyed by content fingerprint.

        Args:
            content: The original content
            result: The computed result to cache

        Returns:
            The fingerprint key
        """
        fp = self.fingerprint(content)
        self._cache[fp] = result
        self._cache.move_to_end(fp)

        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

        logger.debug("[ChunkFingerprint] Cached result for fp=%s", fp[:16])
        return fp

    def get_or_compute(
        self,
        content: str,
        compute_fn,
    ) -> dict[str, Any]:
        """Get cached result or compute and cache it.

        Args:
            content: The content to process
            compute_fn: Function to call if not cached (receives content)

        Returns:
            The cached or newly computed result
        """
        cached = self.get(content)
        if cached is not None:
            return cached

        result = compute_fn(content)
        self.put(content, result)
        return result

    def invalidate(self, content: str) -> None:
        """Invalidate a cached entry."""
        fp = self.fingerprint(content)
        self._cache.pop(fp, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all entries whose original content starts with prefix.

        Returns:
            Number of entries invalidated
        """
        to_remove = [
            fp for fp, entry in self._cache.items()
            if entry.get("original_content", "").startswith(prefix)
        ]
        for fp in to_remove:
            self._cache.pop(fp, None)
        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
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

    def get_cached_fingerprints(self) -> list[str]:
        """Get list of all cached fingerprint keys (truncated)."""
        return [fp[:16] for fp in self._cache]


# Global instance
_fingerprint_cache: ChunkFingerprintCache | None = None


def get_chunk_fingerprint_cache(max_entries: int = 2048) -> ChunkFingerprintCache:
    """Get or create the global chunk fingerprint cache."""
    global _fingerprint_cache
    if _fingerprint_cache is None:
        _fingerprint_cache = ChunkFingerprintCache(max_entries=max_entries)
    return _fingerprint_cache
