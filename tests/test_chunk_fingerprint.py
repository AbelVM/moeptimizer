"""Tests for chunk_fingerprint module."""


from moeptimizer.chunk_fingerprint import ChunkFingerprintCache, get_chunk_fingerprint_cache


class TestChunkFingerprintCache:
    def setup_method(self) -> None:
        self.cache = ChunkFingerprintCache(max_entries=10)

    def test_fingerprint_deterministic(self) -> None:
        fp1 = self.cache.fingerprint("hello world")
        fp2 = self.cache.fingerprint("hello world")
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256

    def test_fingerprint_different(self) -> None:
        fp1 = self.cache.fingerprint("hello")
        fp2 = self.cache.fingerprint("world")
        assert fp1 != fp2

    def test_put_and_get(self) -> None:
        result = {"compressed": "data"}
        self.cache.put("code snippet", result)
        retrieved = self.cache.get("code snippet")
        assert retrieved == result

    def test_get_miss(self) -> None:
        result = self.cache.get("nonexistent")
        assert result is None

    def test_get_or_compute(self) -> None:
        calls = []

        def compute(content):
            calls.append(content)
            return {"result": content}

        result1 = self.cache.get_or_compute("input", compute)
        result2 = self.cache.get_or_compute("input", compute)
        assert result1 == result2
        assert len(calls) == 1  # Only called once

    def test_invalidate(self) -> None:
        self.cache.put("code", {"data": 1})
        self.cache.invalidate("code")
        assert self.cache.get("code") is None

    def test_clear(self) -> None:
        self.cache.put("a", {"data": 1})
        self.cache.put("b", {"data": 2})
        self.cache.clear()
        assert self.cache.get_stats()["entries"] == 0

    def test_lru_eviction(self) -> None:
        cache = ChunkFingerprintCache(max_entries=2)
        cache.put("a", {"data": 1})
        cache.put("b", {"data": 2})
        cache.put("c", {"data": 3})
        assert cache.get("a") is None  # Evicted
        assert cache.get("b") == {"data": 2}
        assert cache.get("c") == {"data": 3}

    def test_get_stats(self) -> None:
        self.cache.put("a", {"data": 1})
        self.cache.get("a")  # hit
        self.cache.get("b")  # miss
        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_global_instance(self) -> None:
        cache = get_chunk_fingerprint_cache()
        assert isinstance(cache, ChunkFingerprintCache)
