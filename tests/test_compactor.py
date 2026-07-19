"""Tests for scratchpad compactor."""


from moeptimizer.compactor import (
    ScratchpadCompactor,
)


class TestScratchpadCompactor:
    def test_empty_compactor(self) -> None:
        """Empty compactor has no state."""
        compactor = ScratchpadCompactor()
        assert compactor is not None

    def test_compact_messages(self) -> None:
        """Compact messages."""
        compactor = ScratchpadCompactor()
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        result = compactor.compact_messages(messages)
        assert len(result) == len(messages)

    def test_compact_with_archived(self) -> None:
        """Compact messages with archived flag."""
        compactor = ScratchpadCompactor()
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response", "_archived": True},
        ]
        result = compactor.compact_messages(messages)
        # Archived messages should be handled
        assert len(result) >= 1

    def test_compactor_never_evicts_rolling_summary_block(self) -> None:
        """The rolling-summary block must survive front-eviction.

        The optimizer's Step 7 (pre-compaction) folds evicted turns into the
        append-only ``_summary_id`` block. If the compactor then drops that
        block (it is a trailing user message and lands outside the protected
        tail when there are > keep_full complete turns), all the folded task
        state is lost on every turn — the turn-10+ faithfulness/recall collapse.
        """
        compactor = ScratchpadCompactor(keep_full=2, cache_stable_mode=True, frozen_prefix_turns=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn1 user"},
            {"role": "assistant", "content": "turn1 asst"},
        ]
        # 6 full turns so there are > keep_full complete turns to evict from.
        for i in range(6):
            messages.append({"role": "user", "content": f"user {i} content here"})
            messages.append({"role": "assistant", "content": f"asst {i} content here"})
        summary = {
            "role": "user",
            "content": "Context summary (rolling):\nfolded task state from evicted turns",
            "_summary_id": "abc123",
            "_rolling_summary": True,
        }
        messages.append(summary)

        result = compactor.compact_messages(messages)

        surviving = [m for m in result if m.get("_summary_id") == "abc123"]
        assert surviving, "rolling-summary block was evicted by the compactor"
        assert surviving[0]["content"] == summary["content"]
        # The summary block must be present in the final output.
        assert any(m.get("_summary_id") == "abc123" for m in result)
