"""Tests for TokenCounter."""


from unittest.mock import MagicMock

from moeptimizer.token_counter import TokenCounter


class TestTokenCounter:
    def test_count_empty(self) -> None:
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_whitespace_only(self) -> None:
        counter = TokenCounter()
        assert counter.count("   \n  ") == 0

    def test_count_generic(self) -> None:
        counter = TokenCounter()
        # Test with realistic text that has multiple tokens
        text = "The quick brown fox jumps over the lazy dog"
        tokens = counter.count(text, "generic")
        # Should be at least 9 tokens (9 words)
        assert tokens >= 9

    def test_count_python(self) -> None:
        counter = TokenCounter()
        # ~4.0 chars per token for python
        text = "def foo():\n    pass\n"
        tokens = counter.count(text, "python")
        assert tokens >= 1

    def test_count_messages(self) -> None:
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        total = counter.count_messages(messages)
        # 3 messages * 5 overhead + content tokens
        assert total > 15

    def test_count_messages_with_content_list(self) -> None:
        counter = TokenCounter()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello world"},
                ],
            },
        ]
        total = counter.count_messages(messages)
        assert total > 5

    def test_estimate_kv_cache_usage(self) -> None:
        counter = TokenCounter()
        assert "KV slots" in counter.estimate_kv_cache_usage(100)
        assert "near context limit" in counter.estimate_kv_cache_usage(50000)

    def test_count_messages_memoized_by_fingerprint(self) -> None:
        """Identical message lists return the same count and hit the fingerprint cache."""
        counter = TokenCounter(max_cache=16)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello " + "x" * 500},
            {"role": "assistant", "content": "Hi there " + "y" * 500},
        ]
        first = counter.count_messages(messages)
        # A fresh list with identical content must produce the same count and
        # be served from the fingerprint cache (no re-tokenization).
        again = counter.count_messages(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello " + "x" * 500},
                {"role": "assistant", "content": "Hi there " + "y" * 500},
            ]
        )
        assert again == first
        assert counter._cache  # fingerprint cache was populated

    def test_count_messages_cache_invalidated_on_content_change(self) -> None:
        """Changing content yields a different count (distinct fingerprint)."""
        counter = TokenCounter(max_cache=16)
        a = counter.count_messages([{"role": "user", "content": "alpha"}])
        b = counter.count_messages(
            [{"role": "user", "content": "alpha " + "word " * 20}]
        )
        assert b > a

    def test_remote_count_uses_backend_when_enabled(self) -> None:
        """When capability_probe reports remote_tokenize, _remote_count returns the backend count."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = 42
        counter = TokenCounter(capability_probe=probe)
        assert counter._remote_count("hello world") == 42
        probe.tokenize_count_sync.assert_called_once_with("hello world")

    def test_remote_count_cache_hit_skips_probe(self) -> None:
        """Repeated _remote_count calls for the same text hit the bounded remote cache."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = 42
        counter = TokenCounter(capability_probe=probe)
        assert counter._remote_count("hello world") == 42
        assert counter._remote_count("hello world") == 42
        probe.tokenize_count_sync.assert_called_once()

    def test_remote_count_disabled_when_probe_reports_no_remote(self) -> None:
        """If the probe reports remote_tokenize=False, remote counting is disabled."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=False)
        counter = TokenCounter(capability_probe=probe)
        assert counter._remote_count("hello") is None
        assert counter._use_remote is False

    def test_remote_count_disabled_after_failure(self) -> None:
        """If the backend /tokenize call fails, remote is disabled for the instance."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = None
        counter = TokenCounter(capability_probe=probe)
        assert counter._remote_count("hello") is None
        assert counter._use_remote is False

    def test_count_messages_uses_remote_path(self) -> None:
        """count_messages uses the remote path when enabled; overhead comes from the calibrated delta (B0.6)."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = 10
        counter = TokenCounter(capability_probe=probe)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        # No calibration yet -> delta defaults to 0, so count == remote joined count.
        assert counter.count_messages(messages) == 10

    def test_count_messages_applies_calibration_delta(self) -> None:
        """B0.6: the remote path applies the calibrated per-message overhead delta, not a fixed +5."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = 10
        counter = TokenCounter(capability_probe=probe)
        # Distinct message lists so the fingerprint cache does not mask the delta.
        before_msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        after_msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi"},
        ]
        # Before calibration: remote joined count only (delta=0).
        assert counter.count_messages(before_msgs) == 10
        # Calibrate: backend reported 19 prompt_tokens for this joined content.
        counter.set_token_calibration_delta(19 - 10)
        assert counter._calibration_delta == 9
        # After calibration: remote + delta (not 10 + 3*5 = 25).
        assert counter.count_messages(after_msgs) == 19

    def test_count_messages_remote_raw_excludes_delta(self) -> None:
        """count_messages_remote_raw returns the joined count without the overhead delta (B0.6)."""
        probe = MagicMock()
        probe.cached.return_value = MagicMock(remote_tokenize=True)
        probe.tokenize_count_sync.return_value = 10
        counter = TokenCounter(capability_probe=probe)
        messages = [{"role": "user", "content": "Hello"}]
        counter.set_token_calibration_delta(7)
        # Raw joined count is unaffected by the delta.
        assert counter.count_messages_remote_raw(messages) == 10

    def test_set_token_calibration_delta_clamps_negative(self) -> None:
        """A negative measured delta is clamped to 0 (never under-count below joined content)."""
        counter = TokenCounter()
        counter.set_token_calibration_delta(-5)
        assert counter._calibration_delta == 0
