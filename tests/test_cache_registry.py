"""Tests for cache registry."""

import pytest

from moeptimizer.cache_registry import (
    CacheKeyRegistry,
    get_cache_registry,
)


class TestCacheKeyRegistry:
    def test_empty_registry(self) -> None:
        """Empty registry has no contexts."""
        registry = CacheKeyRegistry()
        stats = registry.get_cache_stats()
        assert stats["total_hits"] == 0

    def test_register_context(self) -> None:
        """Register context in registry."""
        registry = CacheKeyRegistry()
        messages = [{"role": "user", "content": "Test"}]
        key = registry.register_context(messages)
        assert key is not None
        assert len(key) == 32  # MD5 truncated to 32 chars (128 bits)

    def test_predict_hit_rate(self) -> None:
        """Predict hit rate for context."""
        registry = CacheKeyRegistry()
        messages = [{"role": "user", "content": "Test"}]
        # First call - no history, should be 0.0
        rate = registry.predict_hit_rate(messages)
        assert rate == 0.0
        # Register a hit
        registry.register_context(messages, hit=True)
        rate = registry.predict_hit_rate(messages)
        assert rate == 1.0

    def test_singleton(self) -> None:
        """Get cache registry returns singleton."""
        registry1 = get_cache_registry()
        registry2 = get_cache_registry()
        assert registry1 is registry2