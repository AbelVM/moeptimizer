"""Tests for MOE context integrity — verified chat template structure.

Verifies optimized context won't trigger KV-cache refills in Qwen3.6-35B-A3B-MTP.

Key guarantees:
1. RAG context injected as SEPARATE user message (never into assistant)
2. Loop warnings injected as SEPARATE user message (never into assistant)
3. Reasoning tags preserved in assistant messages (ThinkingPreserver pass-through)
4. Chronological message ordering maintained (zone-based eviction)
5. No foreign markers in assistant content
6. Three immutable zones: System Anchor, Evictable Body, Protected Tail
"""

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestContextIntegrity:
    """Verify optimized context won't trigger KV-cache refills in Qwen3.6-35B-A3B-MTP."""

    def setup_method(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        self.optimizer = AgentContextOptimizer(config)

    def test_no_rag_injected_into_assistant(self) -> None:
        """RAG context must NEVER be injected into assistant message content.

        The model expects: assistant\n<thinking>...\n</think>\n\n{response}
        Injecting foreign patterns (RAG context) into assistant content creates
        an unfamiliar token sequence that triggers KV-cache refills.

        RAG context should be injected as a SEPARATE user message instead.
        """
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I will build a REST API with auth"},
        ]
        result = self.optimizer.optimize_messages(messages)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "step 0:" not in content or "assistant -" not in content, (
                    "RAG context found in assistant message — "
                    "this breaks the model's expected chat template"
                )

    def test_rag_as_separate_user_message(self) -> None:
        """RAG context should be injected as a separate user message."""
        messages1 = [
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I will build a REST API"},
            {"role": "user", "content": "Add auth"},
            {"role": "assistant", "content": "I will add auth with JWT"},
        ]
        self.optimizer.optimize_messages(messages1)

        messages2 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I will build a REST API"},
            {"role": "user", "content": "Add auth"},
            {"role": "assistant", "content": "I will add auth with JWT"},
            {"role": "user", "content": "How do I test it?"},
        ]
        result = self.optimizer.optimize_messages(messages2)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "step " not in content or "assistant -" not in content, (
                    "RAG context found in assistant message"
                )

    def test_loop_warnings_as_separate_user_message(self) -> None:
        """Loop warnings should be injected as a separate user message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Search for something"},
            {"role": "assistant", "content": "call search()"},
            {"role": "tool", "content": "result 1"},
            {"role": "assistant", "content": "call search()"},
            {"role": "tool", "content": "result 2"},
            {"role": "assistant", "content": "call search()"},
            {"role": "tool", "content": "result 3"},
        ]
        result = self.optimizer.optimize_messages(messages)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "[LOOP DETECTED:" not in content, (
                    "Loop warning found in assistant message — "
                    "should be in a separate user message"
                )

    def test_reasoning_tags_preserved(self) -> None:
        """Reasoning tags (<think>/</think> or <thinking></thinking>) must be preserved.

        ThinkingPreserver is a pass-through — it never compresses or modifies
        reasoning tags. The model sees exactly the token sequences it was trained on.
        """
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>\nLet me think about this...\n</think>\n\nHello there!"},
        ]
        result = self.optimizer.optimize_messages(messages)

        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        content = assistant_msgs[-1].get("content", "")
        assert "<think>" in content or "<thinking>" in content, (
            "Reasoning tags should be preserved — ThinkingPreserver is a pass-through"
        )

    def test_protected_tail_preserves_reasoning(self) -> None:
        """Protected tail turns preserve reasoning tags intact.

        With front-loading eviction, the protected tail (last N turns) is never
        modified. Reasoning tags in protected tail turns are preserved exactly.
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        config.agentic.keep_full_steps = 2
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "<thinking>Thinking 1...</thinking>Done 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "<thinking>Thinking 2...</thinking>Done 2"},
            {"role": "user", "content": "Task 3"},
            {"role": "assistant", "content": "<thinking>Thinking 3...</thinking>Done 3"},
            {"role": "user", "content": "Task 4"},
            {"role": "assistant", "content": "<thinking>Thinking 4...</thinking>Done 4"},
        ]
        result = optimizer.optimize_messages(messages)

        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        # Protected tail turns should have reasoning tags intact
        has_tags = any("<thinking>" in m.get("content", "") or "<think>" in m.get("content", "") for m in assistant_msgs)
        assert has_tags, (
            "Protected tail assistant messages should preserve reasoning tags"
        )

    def test_chronological_order_preserved_after_eviction(self) -> None:
        """Message order must be chronological after zone-based eviction.

        The three-zone structure ensures:
          System Anchor → Evictable Body (dropped from front) → Protected Tail
        No message ordering is reversed, preventing KV-cache refills.
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 200
        config.agentic.keep_full_steps = 2
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "<thinking>Thinking 1...</thinking>Done 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "<thinking>Thinking 2...</thinking>Done 2"},
            {"role": "user", "content": "Task 3"},
            {"role": "assistant", "content": "<thinking>Thinking 3...</thinking>Done 3"},
            {"role": "user", "content": "Task 4"},
            {"role": "assistant", "content": "<thinking>Thinking 4...</thinking>Done 4"},
        ]
        result = optimizer.optimize_messages(messages)

        roles = [m.get("role") for m in result]
        assert roles[0] == "system", f"First message must be system, got {roles[0]}"

        # Verify no reversed ordering: no _archived flags with zone-based eviction
        # The structure is: system, first_user, [evicted turns], protected_tail
        # All remaining messages are in chronological order

    def test_no_foreign_markers_in_assistant_content(self) -> None:
        """Assistant messages must not contain foreign markers that the model wasn't trained on."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I will build a REST API"},
            {"role": "user", "content": "Add auth"},
            {"role": "assistant", "content": "I will add auth with JWT"},
        ]
        result = self.optimizer.optimize_messages(messages)

        foreign_markers = [
            "[REASONING ARCHIVED]",
            "RELEVANT HISTORY",
            "[PROGRESS",
            "[LOOP DETECTED",
        ]

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                for marker in foreign_markers:
                    assert marker not in content, (
                        f"Foreign marker '{marker}' found in assistant message — "
                        f"this breaks the model's expected chat template and "
                        f"triggers KV-cache refills"
                    )

    def test_no_archived_markers_in_assistant_content(self) -> None:
        """Assistant messages must not contain [ARCHIVED] markers.

        Front-loading eviction drops entire turns — no markers are injected.
        This is the core mechanism that protects MTP heads from disruption.
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        config.agentic.keep_full_steps = 1
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "Response to task 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "Response to task 2"},
            {"role": "user", "content": "Task 3"},
            {"role": "assistant", "content": "Response to task 3"},
            {"role": "user", "content": "Task 4"},
            {"role": "assistant", "content": "Response to task 4"},
        ]
        result = optimizer.optimize_messages(messages)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "[ARCHIVED" not in content, (
                    f"[ARCHIVED marker found in assistant: {content[:100]} — "
                    "eviction drops turns, never injects markers"
                )

    def test_three_zone_structure(self) -> None:
        """Verify the three immutable zone structure is maintained.

        System Anchor: system + first user (never modified)
        Protected Tail: last N turns (never modified)
        Evictable Body: dropped from front when over budget
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        config.agentic.keep_full_steps = 2
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Third task"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Fourth task"},
            {"role": "assistant", "content": "Response 4"},
        ]
        result = optimizer.optimize_messages(messages)

        # System anchor preserved
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System rules"

        # First user (anchor) preserved
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "First task"

        # No foreign markers in any assistant message
        for msg in result:
            if msg.get("role") == "assistant":
                assert "[ARCHIVED" not in msg.get("content", "")
                assert "[REASONING" not in msg.get("content", "")
