"""Tests for expert routing cache."""

import pytest

from moeptimizer.expert_cache import (
    ExpertRoutingCache,
    get_expert_cache,
    hash_for_expert_routing,
)


class TestExpertRoutingCache:
    def test_cache_get_miss(self) -> None:
        """Get from empty cache returns None."""
        cache = ExpertRoutingCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_put_and_get(self) -> None:
        """Put and get from cache works."""
        cache = ExpertRoutingCache(max_size=10)
        cache.put("pattern", (1, 2, 3, 4))
        result = cache.get("pattern")
        assert result == (1, 2, 3, 4)

    def test_cache_lru_eviction(self) -> None:
        """Cache evicts oldest entries when full."""
        cache = ExpertRoutingCache(max_size=3)
        cache.put("a", (1,))
        cache.put("b", (2,))
        cache.put("c", (3,))
        cache.put("d", (4,))  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_cache_stats(self) -> None:
        """Cache statistics are tracked."""
        cache = ExpertRoutingCache(max_size=10)
        cache.get("miss")
        cache.put("hit", (1,))
        cache.get("hit")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_cache_clear(self) -> None:
        """Cache clear works."""
        cache = ExpertRoutingCache(max_size=10)
        cache.put("a", (1,))
        cache.clear()
        assert cache.get("a") is None

    def test_hash_for_expert_routing(self) -> None:
        """Hash generation for expert routing."""
        hash1 = hash_for_expert_routing("def foo():", "function")
        hash2 = hash_for_expert_routing("def foo():", "function")
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_predict_expert_for_code(self) -> None:
        """Predict expert for code patterns."""
        cache = ExpertRoutingCache(max_size=10)
        result = cache.predict_expert_for_code("def foo(): pass", "function_definition")
        # Returns None if not cached
        assert result is None

    def test_warm_cache_for_static_layer(self) -> None:
        """Warm cache with static layer patterns."""
        cache = ExpertRoutingCache(max_size=100)
        cache.warm_cache_for_static_layer("import os\n\ndef foo(): pass")
        # Should have cached some patterns
        # The cache is pre-filled, not accessed, so check size
        assert len(cache._cache) > 0

    def test_get_expert_cache_singleton(self) -> None:
        """Get expert cache returns singleton."""
        cache1 = get_expert_cache()
        cache2 = get_expert_cache()
        assert cache1 is cache2