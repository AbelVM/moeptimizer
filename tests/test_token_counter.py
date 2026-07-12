"""Tests for TokenCounter."""


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
