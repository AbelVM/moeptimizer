"""Tests for kv_cache_warmup module."""

import pytest

from moeptimizer.kv_cache_warmup import KVCacheWarmup, get_kv_cache_warmup


class TestKVCacheWarmup:
    def setup_method(self) -> None:
        self.warmup = KVCacheWarmup(max_warmups=10)

    def test_get_warmup_key(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        key = self.warmup.get_warmup_key(messages)
        assert len(key) == 32

    def test_has_warmup_miss(self) -> None:
        messages = [{"role": "system", "content": "System"}]
        assert self.warmup.has_warmup(messages) is False

    def test_store_and_get_warmup(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        warmup_data = {"static_tokens": 100, "kv_cache": b"data"}
        self.warmup.store_warmup(messages, warmup_data)
        assert self.warmup.has_warmup(messages) is True
        retrieved = self.warmup.get_warmup_data(messages)
        assert retrieved == warmup_data

    def test_should_warmup_no_cache(self) -> None:
        messages = [{"role": "system", "content": "System"}]
        assert self.warmup.should_warmup(messages) is True

    def test_should_warmup_has_cache(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.warmup.store_warmup(messages, {"static_tokens": 100})
        assert self.warmup.should_warmup(messages) is False

    def test_should_warmup_force(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.warmup.store_warmup(messages, {"static_tokens": 100})
        assert self.warmup.should_warmup(messages, force=True) is True

    def test_get_warmup_payload(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.warmup.store_warmup(messages, {"static_tokens": 100})
        payload = self.warmup.get_warmup_payload(messages)
        assert payload is not None
        assert payload["kv_cache_warmup"]["enabled"] is True

    def test_get_warmup_payload_none(self) -> None:
        messages = [{"role": "system", "content": "System"}]
        payload = self.warmup.get_warmup_payload(messages)
        assert payload is None

    def test_record_warmup_time(self) -> None:
        self.warmup.record_warmup_time(150.0)
        stats = self.warmup.get_stats()
        assert stats["warmup_time_ms"] == 150

    def test_clear(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.warmup.store_warmup(messages, {"static_tokens": 100})
        self.warmup.clear()
        assert self.warmup.has_warmup(messages) is False

    def test_get_stats(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        self.warmup.store_warmup(messages, {"static_tokens": 100})
        self.warmup.has_warmup(messages)
        stats = self.warmup.get_stats()
        assert stats["warmups_performed"] == 1
        assert stats["warmup_hits"] == 1

    def test_global_instance(self) -> None:
        warmup = get_kv_cache_warmup()
        assert isinstance(warmup, KVCacheWarmup)
