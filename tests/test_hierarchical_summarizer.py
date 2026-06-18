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
