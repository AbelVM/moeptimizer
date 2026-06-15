"""Tests for delta_encoder module."""

import pytest

from moeptimizer.delta_encoder import CodeDeltaEncoder, get_delta_encoder


class TestCodeDeltaEncoder:
    def setup_method(self) -> None:
        self.encoder = CodeDeltaEncoder(max_snapshots=10)

    def test_store_snapshot_first(self) -> None:
        key = self.encoder.store_snapshot("test.py", "def foo():\n    pass\n")
        assert key != ""
        assert self.encoder.reconstruct(key) == "def foo():\n    pass\n"

    def test_store_snapshot_delta(self) -> None:
        code1 = "def foo():\n    x = 1\n    return x\n"
        code2 = "def foo():\n    x = 2\n    y = 3\n    return x + y\n"
        self.encoder.store_snapshot("test.py", code1)
        key2 = self.encoder.store_snapshot("test.py", code2)
        # Second snapshot should have a delta
        delta_size = self.encoder.get_delta_size(key2)
        full_size = self.encoder.get_full_size(key2)
        # Delta should be smaller than full for non-trivial changes
        assert delta_size <= full_size

    def test_reconstruct_missing(self) -> None:
        result = self.encoder.reconstruct("nonexistent")
        assert result is None

    def test_get_compression_ratio(self) -> None:
        self.encoder.store_snapshot("test.py", "def foo():\n    pass\n")
        key = self.encoder.store_snapshot("test.py", "def foo():\n    return 1\n")
        ratio = self.encoder.get_compression_ratio(key)
        assert 0.0 <= ratio <= 1.0

    def test_get_or_create_new(self) -> None:
        key, is_new = self.encoder.get_or_create("test.py", "code")
        assert is_new is True
        assert key != ""

    def test_get_or_create_existing(self) -> None:
        key1, is_new1 = self.encoder.get_or_create("test.py", "code")
        key2, is_new2 = self.encoder.get_or_create("test.py", "code")
        assert key1 == key2
        assert is_new1 is True
        assert is_new2 is False

    def test_clear(self) -> None:
        self.encoder.store_snapshot("test.py", "code")
        self.encoder.clear()
        stats = self.encoder.get_stats()
        assert stats["total_snapshots"] == 0

    def test_get_stats(self) -> None:
        self.encoder.store_snapshot("test.py", "def foo():\n    pass\n")
        self.encoder.store_snapshot("test.py", "def foo():\n    return 1\n")
        stats = self.encoder.get_stats()
        assert stats["total_snapshots"] >= 1
        assert "overall_compression" in stats

    def test_global_instance(self) -> None:
        encoder = get_delta_encoder()
        assert isinstance(encoder, CodeDeltaEncoder)
