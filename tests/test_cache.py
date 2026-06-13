"""Tests for cache module."""

from collections import OrderedDict

from moeptimizer.cache import cache_get, cache_key, cache_put


class TestCacheKey:
    def test_deterministic(self) -> None:
        assert cache_key("hello") == cache_key("hello")

    def test_different_inputs(self) -> None:
        assert cache_key("hello") != cache_key("world")


class TestCacheGetPut:
    def test_put_and_get(self) -> None:
        cache: OrderedDict = OrderedDict()
        cache_put(cache, "key1", "value1", 10)
        result = cache_get(cache, "key1")
        assert result == "value1"

    def test_get_missing_key(self) -> None:
        cache: OrderedDict = OrderedDict()
        result = cache_get(cache, "missing")
        assert result is None

    def test_lru_eviction(self) -> None:
        cache: OrderedDict = OrderedDict()
        for i in range(5):
            cache_put(cache, f"key{i}", f"value{i}", 3)
        assert len(cache) == 3
        assert "key0" not in cache
        assert "key2" in cache

    def test_lru_update_on_hit(self) -> None:
        cache: OrderedDict = OrderedDict()
        cache_put(cache, "key1", "value1", 3)
        cache_put(cache, "key2", "value2", 3)
        cache_put(cache, "key3", "value3", 3)
        # Access key1 to move it to end
        cache_get(cache, "key1")
        # Add one more to trigger eviction
        cache_put(cache, "key4", "value4", 3)
        # key2 should be evicted (it was least recently used after key1 was accessed)
        assert "key2" not in cache
