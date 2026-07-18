"""Tests for delta_encoder module."""


from moeptimizer.delta_encoder import CodeDeltaEncoder, get_delta_encoder


class TestCodeDeltaEncoder:
    def setup_method(self) -> None:
        self.encoder = CodeDeltaEncoder(max_snapshots=10)

    def test_store_snapshot_first(self) -> None:
        key = self.encoder.store_snapshot("test.py", "def foo():\n    pass\n")
        assert key != ""
        assert self.encoder.reconstruct(key) == "def foo():\n    pass\n"

    def test_store_snapshot_delta(self) -> None:
        # A large file with a small edit: the delta must be far smaller than the
        # full content (the whole point of delta encoding).
        base = "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n"
        edited = base.replace("line_100 = 100", "line_100 = 999")
        self.encoder.store_snapshot("test.py", base)
        key2 = self.encoder.store_snapshot("test.py", edited)
        # A delta must have been produced (previously a no-op, so never stored).
        assert self.encoder._stats["deltas_stored"] >= 1
        delta_size = self.encoder.get_delta_size(key2)
        full_size = self.encoder.get_full_size(key2)
        # For a small edit to a large file the delta is a tiny fraction of full.
        assert delta_size < full_size / 2

    def test_reconstruct_missing(self) -> None:
        result = self.encoder.reconstruct("nonexistent")
        assert result is None

    def test_get_compression_ratio(self) -> None:
        # Large file, single-line edit -> ratio well under 1.0.
        base = "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n"
        edited = base.replace("line_50 = 50", "line_50 = 51")
        self.encoder.store_snapshot("test.py", base)
        key = self.encoder.store_snapshot("test.py", edited)
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
