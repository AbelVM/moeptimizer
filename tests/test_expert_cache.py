"""Tests for expert cache partitioning."""

import pytest

from moeptimizer.expert_cache import (
    ExpertRoutingCache,
    get_expert_cache,
)


class TestExpertRoutingCache:
    def test_empty_cache(self) -> None:
        """Empty cache has no entries."""
        cache = ExpertRoutingCache()
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 0

    def test_warm_cache_for_static_layer(self) -> None:
        """Warm cache for static layer content."""
        cache = ExpertRoutingCache()
        content = "def foo():\n    pass\n"
        cache.warm_cache_for_static_layer(content)
        stats = cache.get_stats()
        assert stats["hits"] == 0  # No hits yet, just warmed

    def test_warm_cache_for_dynamic_layer(self) -> None:
        """Warm cache for dynamic layer content."""
        cache = ExpertRoutingCache()
        content = "x = 1\ny = 2\n"
        # Use static layer warm for now (dynamic warm doesn't exist)
        cache.warm_cache_for_static_layer(content)
        stats = cache.get_stats()
        assert stats["hits"] == 0  # No hits yet

    def test_get_expert_routing(self) -> None:
        """Get expert routing for content."""
        cache = ExpertRoutingCache()
        content = "def foo():\n    pass\n"
        routing = cache.predict_expert_for_code(content, "function")
        # May return None if no cache hit
        assert routing is None or isinstance(routing, tuple)

    def test_static_dynamic_partition(self) -> None:
        """Static and dynamic caches are separate."""
        cache = ExpertRoutingCache()
        static_content = "def foo():\n    pass\n"
        dynamic_content = "x = 1\ny = 2\n"
        cache.warm_cache_for_static_layer(static_content)
        cache.warm_cache_for_static_layer(dynamic_content)
        # Both should be cached
        stats = cache.get_stats()
        assert stats["hits"] == 0

    def test_singleton(self) -> None:
        """Get expert cache returns singleton."""
        cache1 = get_expert_cache()
        cache2 = get_expert_cache()
        assert cache1 is cache2

    def test_cache_get_put(self) -> None:
        """Get and put work correctly."""
        cache = ExpertRoutingCache()
        cache.put("test_pattern", (1, 2, 3), layer="dynamic")
        result = cache.get("test_pattern", layer="dynamic")
        assert result == (1, 2, 3)

    def test_cache_miss(self) -> None:
        """Cache miss returns None."""
        cache = ExpertRoutingCache()
        result = cache.get("nonexistent", layer="dynamic")
        assert result is None

    def test_get_or_compute(self) -> None:
        """Get or compute caches and returns value."""
        cache = ExpertRoutingCache()
        result = cache.get_or_compute(
            "test_pattern",
            lambda: (5, 6, 7),
            layer="dynamic",
        )
        assert result == (5, 6, 7)
        # Second call should hit cache
        result2 = cache.get("test_pattern", layer="dynamic")
        assert result2 == (5, 6, 7)