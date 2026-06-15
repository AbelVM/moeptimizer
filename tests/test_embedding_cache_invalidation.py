"""Tests for embedding_cache_invalidation module."""

import time

import pytest

from moeptimizer.embedding_cache_invalidation import (
    EmbeddingCacheWithInvalidation,
    get_embedding_cache_with_invalidation,
)


class TestEmbeddingCacheWithInvalidation:
    def setup_method(self) -> None:
        self.cache = EmbeddingCacheWithInvalidation(max_entries=10)

    def test_put_and_get(self) -> None:
        embedding = [0.1, 0.2, 0.3]
        self.cache.put("hello world", embedding)
        result = self.cache.get("hello world")
        assert result is not None
        assert list(result) == embedding

    def test_get_miss(self) -> None:
        result = self.cache.get("nonexistent")
        assert result is None

    def test_cache_hit_count(self) -> None:
        self.cache.put("text", [0.1])
        self.cache.get("text")
        self.cache.get("text")
        stats = self.cache.get_stats()
        assert stats["hits"] == 2

    def test_add_to_batch(self) -> None:
        key = self.cache.add_to_batch("text1")
        assert key != ""
        batch = self.cache.get_pending_batch()
        assert len(batch) == 1
        assert batch[0][1] == "text1"

    def test_batch_put(self) -> None:
        self.cache.add_to_batch("text1")
        self.cache.add_to_batch("text2")
        batch = self.cache.get_pending_batch()
        assert len(batch) == 2

        results = [(batch[0][0], [0.1]), (batch[1][0], [0.2])]
        self.cache.batch_put(results)

        assert self.cache.get("text1") is not None
        assert self.cache.get("text2") is not None

    def test_invalidate_source(self) -> None:
        self.cache.put("text", [0.1], source_path="/tmp/test.py")
        assert self.cache.get("text", source_path="/tmp/test.py") is not None

        count = self.cache.invalidate_source("/tmp/test.py")
        assert count == 1
        assert self.cache.get("text", source_path="/tmp/test.py") is None

    def test_invalidate_all(self) -> None:
        self.cache.put("a", [0.1])
        self.cache.put("b", [0.2])
        self.cache.invalidate_all()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_clear(self) -> None:
        self.cache.put("a", [0.1])
        self.cache.clear()
        stats = self.cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0

    def test_global_instance(self) -> None:
        cache = get_embedding_cache_with_invalidation()
        assert isinstance(cache, EmbeddingCacheWithInvalidation)
