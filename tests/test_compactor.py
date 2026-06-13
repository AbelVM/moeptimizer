"""Tests for ScratchpadCompactor — front-loading eviction strategy."""

from moeptimizer.compactor import ScratchpadCompactor


class TestScratchpadCompactor:
    def test_small_message_list_unchanged(self) -> None:
        compactor = ScratchpadCompactor(keep_full=3)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = compactor.compact_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "You are helpful"

    def test_evicts_historical_turns_from_front(self) -> None:
        """Historical user-assistant pairs are dropped from the front of the evictable body.

        System anchor: system + first turn (user1 + assistant1)
        Evictable body: turn2
        Protected tail: turn3 + turn4
        """
        compactor = ScratchpadCompactor(keep_full=2)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Third task"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Fourth task"},
            {"role": "assistant", "content": "Response 4"},
        ]
        result = compactor.compact_messages(messages)

        # System anchor (3 msgs) + protected tail (4 msgs) = 7
        # Evictable body (turn2: user2 + assistant2) is dropped
        assert len(result) == 7
        # System anchor preserved
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "First task"
        assert result[2]["content"] == "Response 1"
        # Protected tail starts at index 3
        assert result[3]["content"] == "Third task"
        assert result[6]["content"] == "Response 4"

    def test_system_always_preserved(self) -> None:
        compactor = ScratchpadCompactor(keep_full=1)
        messages = [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Old"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = compactor.compact_messages(messages)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "System instructions"

    def test_first_user_always_preserved(self) -> None:
        compactor = ScratchpadCompactor(keep_full=1)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Original prompt"},
            {"role": "assistant", "content": "Old"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = compactor.compact_messages(messages)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Original prompt"

    def test_tool_messages_attached_to_turn(self) -> None:
        """Tool messages belonging to a surviving turn are preserved."""
        compactor = ScratchpadCompactor(keep_full=1)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "tool", "content": "Tool output 1"},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "tool", "content": "Tool output 2"},
        ]
        result = compactor.compact_messages(messages)

        # System anchor: system + user1 + assistant1 + tool1 (4 msgs)
        # Protected tail: user2 + assistant2 + tool2 (3 msgs)
        # No evictable turns (only 1 turn after anchor, keep_full=1)
        assert len(result) == 7
        # First turn tools preserved in anchor
        assert result[3]["role"] == "tool"
        assert result[3]["content"] == "Tool output 1"
        # Second turn preserved
        assert result[4]["content"] == "Second task"
        assert result[6]["content"] == "Tool output 2"

    def test_no_content_modification(self) -> None:
        """Eviction drops whole turns — no summarization or truncation."""
        compactor = ScratchpadCompactor(keep_full=1)
        original_content = "This is a very detailed response with lots of information"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": original_content},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Recent response"},
        ]
        result = compactor.compact_messages(messages)

        # The preserved turn should have original content, untouched
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        # First assistant (part of system anchor) preserved
        assert assistant_msgs[0]["content"] == original_content
        # Second assistant preserved
        assert assistant_msgs[1]["content"] == "Recent response"

    def test_no_archived_markers_in_assistant_content(self) -> None:
        """Assistant messages must not contain [ARCHIVED] markers from compactor."""
        compactor = ScratchpadCompactor(keep_full=1)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Task 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        result = compactor.compact_messages(messages)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "[ARCHIVED" not in content, (
                    f"[ARCHIVED marker found in assistant: {content[:100]}"
                )

    def test_tool_summary(self) -> None:
        compactor = ScratchpadCompactor()
        msg = {"role": "tool", "content": "line1\nline2\nline3\nline4\nline5\nline6"}
        summary = compactor._summarize_message(msg)
        assert "ARCHIVED TOOL" in summary
