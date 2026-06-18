"""Tests for static_prefix_kv module."""

from moeptimizer.static_prefix_kv import StaticPrefixKVCache, get_static_prefix_kv_cache


class TestStaticPrefixKVCache:
    def setup_method(self) -> None:
        self.cache = StaticPrefixKVCache(max_entries=10)

    def test_get_static_prefix_system_and_user(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        prefix = self.cache.get_static_prefix(messages)
        assert "system:You are helpful" in prefix
        assert "user:Hello" in prefix

    def test_get_static_prefix_system_only(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
        ]
        prefix = self.cache.get_static_prefix(messages)
        assert "system:You are helpful" in prefix

    def test_get_static_prefix_empty(self) -> None:
        messages = []
        prefix = self.cache.get_static_prefix(messages)
        assert prefix == ""

    def test_put_and_get(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        kv_data = b"fake_kv_data"
        key = self.cache.put(messages, kv_data)
        assert key != ""

        retrieved = self.cache.get(messages)
        assert retrieved == kv_data

    def test_cache_miss(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        result = self.cache.get(messages)
        assert result is None

    def test_invalidate(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        kv_data = b"fake_kv_data"
        self.cache.put(messages, kv_data)
        assert self.cache.get(messages) is not None

        self.cache.invalidate(messages)
        assert self.cache.get(messages) is None

    def test_clear(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.cache.put(messages, b"data")
        self.cache.clear()
        assert self.cache.get(messages) is None
        assert self.cache.get_stats()["entries"] == 0

    def test_lru_eviction(self) -> None:
        cache = StaticPrefixKVCache(max_entries=2)
        msgs1 = [{"role": "system", "content": "S1"}, {"role": "user", "content": "U1"}]
        msgs2 = [{"role": "system", "content": "S2"}, {"role": "user", "content": "U2"}]
        msgs3 = [{"role": "system", "content": "S3"}, {"role": "user", "content": "U3"}]

        cache.put(msgs1, b"data1")
        cache.put(msgs2, b"data2")
        cache.put(msgs3, b"data3")

        # First entry should be evicted
        assert cache.get(msgs1) is None
        assert cache.get(msgs2) == b"data2"
        assert cache.get(msgs3) == b"data3"

    def test_get_stats(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.cache.put(messages, b"data")
        self.cache.get(messages)  # hit
        self.cache.get(messages)  # hit
        self.cache.get([{"role": "user", "content": "Other"}])  # miss

        stats = self.cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_save_to_disk_skips_unchanged_cache(self) -> None:
        """Disk persistence is skipped unless cache contents changed."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.cache.put(messages, b"data")
        self.cache.save_to_disk()
        self.cache.save_to_disk()
        assert self.cache.get_stats()["entries"] == 1

    def test_global_instance(self) -> None:
        cache = get_static_prefix_kv_cache()
        assert isinstance(cache, StaticPrefixKVCache)
