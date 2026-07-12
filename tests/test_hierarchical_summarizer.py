"""Tests for hierarchical_summarizer module."""

from moeptimizer.hierarchical_summarizer import (
    HierarchicalSummarizer,
    get_hierarchical_summarizer,
)


class TestHierarchicalSummarizer:
    def setup_method(self) -> None:
        self.summarizer = HierarchicalSummarizer(max_full_turns=3, max_summary_turns=10)

    def test_summarize_short(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.summarizer.summarize_turns(messages)
        assert len(result) == len(messages)

    def test_summarize_long(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Turn 4"},
            {"role": "assistant", "content": "Response 4"},
        ]
        result = self.summarizer.summarize_turns(messages)
        # Should have fewer messages after summarization
        assert len(result) < len(messages)
        # Should have a summary message
        summary_msgs = [m for m in result if m.get("_summary_id")]
        assert len(summary_msgs) == 1

    def test_expand_summary(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        result = self.summarizer.summarize_turns(messages)
        summary_msg = next(m for m in result if m.get("_summary_id"))
        expanded = self.summarizer.expand_summary(summary_msg)
        assert expanded["_summary_level"] == 2

    def test_get_summary(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.summarizer.summarize_turns(messages)
        summary_msg = [m for m in result if m.get("_summary_id")]
        if summary_msg:
            sid = summary_msg[0]["_summary_id"]
            stored = self.summarizer.get_summary(sid)
            assert stored is not None

    def test_clear(self) -> None:
        self.summarizer.summarize_turns([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        self.summarizer.clear()
        stats = self.summarizer.get_stats()
        assert stats["turns_summarized"] == 0

    def test_get_stats(self) -> None:
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        self.summarizer.summarize_turns(messages)
        stats = self.summarizer.get_stats()
        assert "turns_summarized" in stats

    def test_global_instance(self) -> None:
        summarizer = get_hierarchical_summarizer()
        assert isinstance(summarizer, HierarchicalSummarizer)

    def test_summarize_cache_stable_short(self) -> None:
        """Short dynamic layer is returned unchanged (nothing to summarize)."""
        summarizer = HierarchicalSummarizer(max_full_turns=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Resp 2"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=2)
        assert result == messages

    def test_summarize_cache_stable_long_places_block_after_frozen(self) -> None:
        """Long dynamic layer gets a rolling summary block right after frozen prefix."""
        summarizer = HierarchicalSummarizer(max_full_turns=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Turn 4"},
            {"role": "assistant", "content": "Resp 4"},
            {"role": "user", "content": "Turn 5"},
            {"role": "assistant", "content": "Resp 5"},
        ]
        # Frozen prefix = system + first user + 2 turns (indices 0..4).
        # Dynamic layer = Turn2..Turn5 (4 turns, > max_full_turns=3) so the
        # oldest folds into the summary and the 3 most recent stay verbatim.
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=5)

        # Frozen prefix preserved verbatim at the front.
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]
        assert result[3] == messages[3]
        assert result[4] == messages[4]

        # The rolling summary block is now a TRAILING turn (review §1/§9, priority
        # fix #1): inserting it right after the frozen prefix shifts every later
        # turn's token offset and invalidates the backend's prefix cache. As a
        # trailing turn it never alters the stable leading prefix, so the backend
        # reuses its cached KV for the frozen + recent turns.
        summary_block = result[-1]
        assert summary_block.get("_summary_id")
        assert summary_block.get("_rolling_summary") is True
        assert summary_block["role"] == "user"
        assert "Context summary (rolling):" in summary_block["content"]

        # Most recent max_full_turns (3) turns retained in full, verbatim, between
        # the frozen prefix and the trailing summary block. Dynamic layer =
        # Turn2..Turn5 (4 turns); the oldest folds into the summary, so the 3 most
        # recent (Turn3..Turn5) stay verbatim at result[5:-1].
        assert result[5:-1] == messages[7:]

    def test_summarize_cache_stable_retains_constraints(self) -> None:
        """The rolling summary retains the task's 'don't' constraints."""
        summarizer = HierarchicalSummarizer(max_full_turns=2)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # Two OLD turns carrying the constraints that must survive in the summary.
            {"role": "user", "content": "Please refactor, but do not change the public API."},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Also avoid renaming the database tables."},
            {"role": "assistant", "content": "Resp 2"},
            # Two RECENT turns kept in full (no constraint needed there).
            {"role": "user", "content": "Now add tests."},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Then run the suite."},
            {"role": "assistant", "content": "Resp 4"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=3)
        summary_block = next(m for m in result if m.get("_rolling_summary"))
        content = summary_block["content"]
        assert "do not change the public API" in content
        assert "avoid renaming the database tables" in content

    def test_summarize_cache_stable_append_only(self) -> None:
        """Across growing turns the rolling text only appends (leading bytes stable)."""
        summarizer = HierarchicalSummarizer(max_full_turns=2)

        def conv(n: int) -> list[dict]:
            msgs = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First user"},
                {"role": "assistant", "content": "Resp 0"},
            ]
            for i in range(n):
                msgs.append({"role": "user", "content": f"Turn {i}"})
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        r1 = summarizer.summarize_turns_cache_stable(conv(5), frozen_prefix_end=3)
        block1 = next(m for m in r1 if m.get("_rolling_summary"))
        text1 = block1["content"]

        r2 = summarizer.summarize_turns_cache_stable(conv(6), frozen_prefix_end=3)
        block2 = next(m for m in r2 if m.get("_rolling_summary"))
        text2 = block2["content"]

        # The second summary extends the first; the first's text is a prefix.
        assert text2.startswith(text1)
        assert len(text2) >= len(text1)
        # The summary id is stable across turns.
        assert block1["_summary_id"] == block2["_summary_id"]
