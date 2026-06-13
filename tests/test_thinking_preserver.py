"""Tests for ThinkingPreserver — pass-through for front-loading eviction.

Since the compactor now uses pure eviction (dropping entire turns), the
ThinkingPreserver is a pass-through that preserves all messages unchanged.
"""

from moeptimizer.thinking_preserver import ThinkingPreserver


class TestThinkingPreserver:
    def test_pass_through_all_messages(self) -> None:
        """All messages are preserved unchanged — no compression."""
        preserver = ThinkingPreserver()
        messages = [
            {"role": "assistant", "content": "Old thinking"},
            {"role": "assistant", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent 2"},
        ]
        result = preserver.process_messages(messages)
        # All messages preserved as-is (pass-through)
        assert len(result) == 3
        assert result[0]["content"] == "Old thinking"
        assert result[1]["content"] == "Recent 1"
        assert result[2]["content"] == "Recent 2"

    def test_preserves_reasoning_tags(self) -> None:
        """Reasoning tags are preserved unchanged."""
        preserver = ThinkingPreserver()
        content = "<thinking>I need to check the database...Actually let me search...OK I will call search_users</thinking>Here is the call: search_users()"
        messages = [
            {"role": "assistant", "content": content},
            {"role": "assistant", "content": "Recent"},
        ]
        result = preserver.process_messages(messages)
        # Content is preserved entirely unchanged
        assert result[0]["content"] == content
        assert "<thinking>" in result[0]["content"]
        assert "</thinking>" in result[0]["content"]

    def test_no_thinking_tags_unchanged(self) -> None:
        preserver = ThinkingPreserver()
        messages = [
            {"role": "assistant", "content": "No thinking here"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = preserver.process_messages(messages)
        assert result[0]["content"] == "No thinking here"

    def test_preserves_non_assistant_messages(self) -> None:
        preserver = ThinkingPreserver()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = preserver.process_messages(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_order_preserved(self) -> None:
        preserver = ThinkingPreserver()
        messages = [
            {"role": "assistant", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "assistant", "content": "Third"},
        ]
        result = preserver.process_messages(messages)
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"
        assert result[2]["content"] == "Third"
