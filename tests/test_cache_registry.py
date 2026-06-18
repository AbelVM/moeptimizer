"""Tests for cache registry."""


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

    def test_predict_static_prefix_hit_rate_survives_dynamic_growth(self) -> None:
        """Static prefix prediction remains useful as conversation context grows."""
        registry = CacheKeyRegistry()
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first user prompt"},
            {"role": "assistant", "content": "first response"},
        ]
        registry.register_context(messages, hit=True)

        growing_messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first user prompt"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "new user prompt"},
            {"role": "assistant", "content": "new response"},
        ]

        assert registry.predict_static_prefix_hit_rate(growing_messages) == 1.0
        assert registry.predict_hit_rate(growing_messages) == 1.0

    def test_context_hash_includes_role_and_order(self) -> None:
        """Same text with different roles does not share a cache key."""
        registry = CacheKeyRegistry()
        user_first = [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "same"},
        ]
        assistant_first = [
            {"role": "assistant", "content": "same"},
            {"role": "user", "content": "same"},
        ]
        assert registry.register_context(user_first) != registry.register_context(assistant_first)

    def test_save_to_disk_skips_unchanged_registry(self) -> None:
        """Disk persistence is skipped unless a new cache entry was created."""
        registry = CacheKeyRegistry()
        messages = [{"role": "user", "content": "Test"}]
        registry.register_context(messages)
        registry.save_to_disk()
        registry.save_to_disk()
        assert registry.get_cache_stats()["total_entries"] == 1

    def test_singleton(self) -> None:
        """Get cache registry returns singleton."""
        registry1 = get_cache_registry()
        registry2 = get_cache_registry()
        assert registry1 is registry2
