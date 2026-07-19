"""Tests for scratchpad compactor."""


from typing import Any

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
        # P0.5: the summary block sits IMMEDIATELY AFTER the frozen prefix (not at
        # the tail). The compactor must keep it there so [frozen][summary] stays
        # byte-stable for backend prefix-cache reuse. With frozen_prefix_turns=1 the
        # frozen prefix is system + first user + first assistant + 1 more complete
        # turn (user0/asst0), so the summary lands right after that boundary.
        from moeptimizer.context_aligner import get_context_aligner

        frozen_end = get_context_aligner().frozen_prefix_end(messages, 1)
        messages.insert(frozen_end, summary)

        result = compactor.compact_messages(messages)

        surviving = [m for m in result if m.get("_summary_id") == "abc123"]
        assert surviving, "rolling-summary block was evicted by the compactor"
        assert surviving[0]["content"] == summary["content"]
        # The summary block must be present in the final output.
        assert any(m.get("_summary_id") == "abc123" for m in result)
        # P0.5: it must be right after the frozen prefix, NOT at the tail.
        assert result[frozen_end].get("_summary_id") == "abc123"
        assert result[frozen_end] is not result[-1]

    def test_compactor_honors_shrink_floor(self) -> None:
        """P0.6: a per-turn shrink floor bounds one-shot front-eviction.

        The v0.7.21 turn-13 break was an 8.5K->2K token collapse that wiped the
        entire cached body in a single call. When ``min_keep_tokens`` is set and a
        token counter is present, the compactor must retain enough of the
        evictable body that the resulting context never drops below the floor in
        one call — even when the hard budget would otherwise drop everything.
        """
        # Deterministic fake token counter: 100 tokens per message.
        class _FakeCounter:
            def count_messages(self, msgs: list[dict[str, Any]]) -> int:
                return len(msgs) * 100

        compactor = ScratchpadCompactor(
            keep_full=2,
            cache_stable_mode=False,
            frozen_prefix_turns=0,
            token_counter=_FakeCounter(),
        )
        # system + 1 user + 1 asst + 10 evictable turns (20 msgs)
        # + 2 protected tail turns (4 msgs). Total = 27 msgs = 2700 tokens.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "frozen user"},
            {"role": "assistant", "content": "frozen asst"},
        ]
        for i in range(10):
            messages.append({"role": "user", "content": f"evictable user {i}"})
            messages.append({"role": "assistant", "content": f"evictable asst {i}"})
        for i in range(2):
            messages.append({"role": "user", "content": f"tail user {i}"})
            messages.append({"role": "assistant", "content": f"tail asst {i}"})

        # Floor of 2000 tokens: system(100)+frozen(200)+tail(400)=700 baseline,
        # so at least 13 evictable messages (1300 tok) must be retained to reach
        # the floor. Without the floor the compactor would drop all 20 evictable.
        result = compactor.compact_messages(messages, min_keep_tokens=2000)

        result_tokens = len(result) * 100
        assert result_tokens >= 2000, (
            f"shrink floor violated: {result_tokens} tok < 2000 tok floor"
        )
        # The protected tail must survive intact.
        assert any(m["content"] == "tail user 1" for m in result)
        assert any(m["content"] == "tail asst 1" for m in result)
        # The system anchor must survive intact.
        assert any(m["content"] == "frozen user" for m in result)

    def test_compactor_no_floor_drops_evictable_body(self) -> None:
        """Without a shrink floor the legacy drop-all-evictable behavior holds."""

        class _FakeCounter:
            def count_messages(self, msgs: list[dict[str, Any]]) -> int:
                return len(msgs) * 100

        compactor = ScratchpadCompactor(
            keep_full=2,
            cache_stable_mode=False,
            frozen_prefix_turns=0,
            token_counter=_FakeCounter(),
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "frozen user"},
            {"role": "assistant", "content": "frozen asst"},
        ]
        for i in range(10):
            messages.append({"role": "user", "content": f"evictable user {i}"})
            messages.append({"role": "assistant", "content": f"evictable asst {i}"})
        for i in range(2):
            messages.append({"role": "user", "content": f"tail user {i}"})
            messages.append({"role": "assistant", "content": f"tail asst {i}"})

        # No min_keep_tokens -> legacy behavior drops the whole evictable body.
        result = compactor.compact_messages(messages)
        assert not any("evictable" in m["content"] for m in result)
        assert any(m["content"] == "tail user 1" for m in result)
