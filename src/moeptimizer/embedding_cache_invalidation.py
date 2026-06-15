"""Embedding Cache Invalidation & Batching.

Tracks file mtime to invalidate only changed embeddings; groups embedding
queries into batches to hide I/O latency.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingCacheEntry:
    """Entry in the embedding cache with invalidation metadata."""

    def __init__(
        self,
        key: str,
        embedding: Any,
        source_path: str | None = None,
        mtime: float | None = None,
    ) -> None:
        self.key = key
        self.embedding = embedding
        self.source_path = source_path
        self.mtime = mtime
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def is_valid(self, current_mtime: float | None) -> bool:
        """Check if this cache entry is still valid."""
        if self.source_path is None or current_mtime is None:
            return True  # No source path, always valid
        if self.mtime is None:
            return True  # No stored mtime, assume valid
        return self.mtime == current_mtime

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class EmbeddingCacheWithInvalidation:
    """
    Embedding cache with file-based invalidation and batch support.

    Tracks file modification times to invalidate only changed embeddings.
    Groups embedding queries into batches to hide I/O latency.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._cache: OrderedDict[str, EmbeddingCacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._pending_batch: list[tuple[str, str, str | None]] = []
        self._batch_size = 32
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "batches_sent": 0,
            "total_batched": 0,
        }

    def _make_key(self, text: str, source_path: str | None = None) -> str:
        """Generate cache key from text and optional source path."""
        base = f"{source_path or ''}:{text}"
        return hashlib.md5(base.encode("utf8")).hexdigest()[:32]

    def _get_mtime(self, source_path: str | None) -> float | None:
        """Get modification time for a source file."""
        if source_path is None:
            return None
        try:
            return Path(source_path).stat().st_mtime
        except OSError:
            return None

    def get(
        self,
        text: str,
        source_path: str | None = None,
    ) -> Any | None:
        """Get embedding from cache, checking invalidation.

        Args:
            text: The text to look up
            source_path: Optional file path for mtime-based invalidation

        Returns:
            Cached embedding or None if not found/invalidated
        """
        key = self._make_key(text, source_path)
        current_mtime = self._get_mtime(source_path)

        if key in self._cache:
            entry = self._cache[key]
            if entry.is_valid(current_mtime):
                entry.touch()
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
                return entry.embedding
            else:
                # Invalidated — remove it
                self._cache.pop(key, None)
                self._stats["invalidations"] += 1

        self._stats["misses"] += 1
        return None

    def put(
        self,
        text: str,
        embedding: Any,
        source_path: str | None = None,
    ) -> str:
        """Store embedding in cache with invalidation metadata.

        Args:
            text: The text content
            embedding: The computed embedding
            source_path: Optional file path for mtime tracking

        Returns:
            Cache key
        """
        key = self._make_key(text, source_path)
        mtime = self._get_mtime(source_path)

        entry = EmbeddingCacheEntry(
            key=key,
            embedding=embedding,
            source_path=source_path,
            mtime=mtime,
        )

        self._cache[key] = entry
        self._cache.move_to_end(key)

        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

        return key

    def add_to_batch(
        self,
        text: str,
        source_path: str | None = None,
    ) -> str:
        """Add a text to the pending batch for batch embedding.

        Args:
            text: The text to embed
            source_path: Optional source file path

        Returns:
            Cache key for this entry
        """
        key = self._make_key(text, source_path)
        self._pending_batch.append((key, text, source_path))

        if len(self._pending_batch) >= self._batch_size:
            return key  # Signal that batch is ready

        return key

    def get_pending_batch(self) -> list[tuple[str, str, str | None]]:
        """Get and clear the pending batch.

        Returns:
            List of (key, text, source_path) tuples
        """
        batch = list(self._pending_batch)
        self._pending_batch.clear()
        if batch:
            self._stats["batches_sent"] += 1
            self._stats["total_batched"] += len(batch)
        return batch

    def batch_put(
        self,
        results: list[tuple[str, Any]],
    ) -> None:
        """Store a batch of computed embeddings.

        Args:
            results: List of (key, embedding) tuples
        """
        for key, embedding in results:
            self._cache[key] = EmbeddingCacheEntry(
                key=key,
                embedding=embedding,
            )
            self._cache.move_to_end(key)

        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def invalidate_source(self, source_path: str) -> int:
        """Invalidate all embeddings from a specific source file.

        Args:
            source_path: The file path to invalidate

        Returns:
            Number of entries invalidated
        """
        to_remove = [
            key for key, entry in self._cache.items()
            if entry.source_path == source_path
        ]
        for key in to_remove:
            self._cache.pop(key, None)
        self._stats["invalidations"] += len(to_remove)
        return len(to_remove)

    def invalidate_all(self) -> None:
        """Invalidate all cached embeddings."""
        self._cache.clear()
        self._pending_batch.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "entries": len(self._cache),
            "pending_batch": len(self._pending_batch),
            "hit_rate": round(self._stats["hits"] / max(total, 1), 4),
        }

    def clear(self) -> None:
        """Clear all cache entries and reset stats."""
        self._cache.clear()
        self._pending_batch.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "batches_sent": 0,
            "total_batched": 0,
        }


# Global instance
_embedding_cache: EmbeddingCacheWithInvalidation | None = None


def get_embedding_cache_with_invalidation() -> EmbeddingCacheWithInvalidation:
    """Get or create the global embedding cache with invalidation."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCacheWithInvalidation()
    return _embedding_cache
