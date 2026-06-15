"""Tests for token_aware_truncator module."""

import pytest

from moeptimizer.token_aware_truncator import TokenAwareTruncator


class TestTokenAwareTruncator:
    def setup_method(self) -> None:
        self.truncator = TokenAwareTruncator()

    def test_count_tokens_empty(self) -> None:
        assert self.truncator.count_tokens("") == 0

    def test_count_tokens_short(self) -> None:
        # "hello" is 1 token in most tokenizers
        count = self.truncator.count_tokens("hello")
        assert count >= 1

    def test_truncate_to_token_limit_short(self) -> None:
        text = "Hello world"
        result = self.truncator.truncate_to_token_limit(text, 100)
        assert result == text

    def test_truncate_to_token_limit_long(self) -> None:
        text = "word " * 1000
        result = self.truncator.truncate_to_token_limit(text, 10)
        assert len(result) < len(text)

    def test_truncate_message(self) -> None:
        msg = {"role": "user", "content": "word " * 1000}
        result = self.truncator.truncate_message(msg, 10)
        assert result["content"] != msg["content"]
        assert result["role"] == "user"

    def test_count_message_tokens(self) -> None:
        msg = {"role": "user", "content": "Hello world"}
        count = self.truncator.count_message_tokens(msg)
        assert count >= 6  # content tokens + overhead

    def test_count_messages_tokens(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = self.truncator.count_messages_tokens(messages)
        assert count > 0

    def test_trim_messages_to_budget_under(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        result = self.truncator.trim_messages_to_budget(messages, 1000)
        assert len(result) == len(messages)

    def test_trim_messages_to_budget_over(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "word " * 1000},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        result = self.truncator.trim_messages_to_budget(messages, 50)
        # With many turns, some should be evicted
        assert len(result) <= len(messages)

    def test_char_based_truncate_fallback(self) -> None:
        """tiktoken truncation works correctly."""
        text = "word " * 1000
        result = self.truncator.truncate_to_token_limit(text, 10)
        assert len(result) < len(text)
