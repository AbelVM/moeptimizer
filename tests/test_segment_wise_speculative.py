"""Tests for segment_wise_speculative module."""

import pytest

from moeptimizer.segment_wise_speculative import (
    SegmentWiseSpeculativeDecoder,
    get_segment_wise_decoder,
)


class TestSegmentWiseSpeculativeDecoder:
    def setup_method(self) -> None:
        self.decoder = SegmentWiseSpeculativeDecoder(
            confidence_threshold=0.7,
            max_draft_tokens=4,
        )

    def test_split_into_segments_code_and_text(self) -> None:
        text = "Here is code:\n```python\ndef foo():\n    pass\n```\nAnd more text."
        segments = self.decoder.split_into_segments(text)
        assert len(segments) >= 2
        types = [s["type"] for s in segments]
        assert "text" in types
        assert "code" in types

    def test_split_into_segments_text_only(self) -> None:
        text = "Just plain text without any code blocks."
        segments = self.decoder.split_into_segments(text)
        assert all(s["type"] == "text" for s in segments)

    def test_split_into_segments_empty(self) -> None:
        segments = self.decoder.split_into_segments("")
        assert segments == []

    def test_should_draft_new_segment(self) -> None:
        segment = {"type": "code", "content": "def foo():\n    pass"}
        assert self.decoder.should_draft_segment(segment, None) is True

    def test_should_draft_unchanged(self) -> None:
        segment = {"type": "code", "content": "def foo():\n    pass"}
        prev = {"type": "code", "content": "def foo():\n    pass"}
        assert self.decoder.should_draft_segment(segment, prev) is False

    def test_should_draft_changed(self) -> None:
        segment = {"type": "code", "content": "def bar():\n    pass"}
        prev = {"type": "code", "content": "def foo():\n    pass"}
        assert self.decoder.should_draft_segment(segment, prev) is True

    def test_should_draft_short_text(self) -> None:
        segment = {"type": "text", "content": "hi"}
        prev = {"type": "text", "content": "hello"}
        assert self.decoder.should_draft_segment(segment, prev) is False

    def test_process_segments(self) -> None:
        text = "```python\ndef foo():\n    pass\n```"
        result = self.decoder.process_segments(text)
        assert "segments" in result
        assert "changed_segments" in result
        assert len(result["segments"]) >= 1

    def test_get_stats(self) -> None:
        self.decoder.process_segments("```python\nx=1\n```")
        stats = self.decoder.get_stats()
        assert "total_segments" in stats

    def test_reset_stats(self) -> None:
        self.decoder.process_segments("```python\nx=1\n```")
        self.decoder.reset_stats()
        stats = self.decoder.get_stats()
        assert stats["total_segments"] == 0

    def test_global_instance(self) -> None:
        decoder = get_segment_wise_decoder()
        assert isinstance(decoder, SegmentWiseSpeculativeDecoder)
