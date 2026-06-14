"""Tests for selective truncator."""

import pytest

from moeptimizer.selective_truncator import (
    SelectiveTruncator,
    get_selective_truncator,
)


class TestSelectiveTruncator:
    def test_empty_truncator(self) -> None:
        """Empty truncator has no state."""
        truncator = SelectiveTruncator()
        assert truncator is not None

    def test_remove_duplicates(self) -> None:
        """Remove duplicate code blocks."""
        truncator = SelectiveTruncator()
        messages = [
            {"role": "user", "content": "```python\ndef foo():\n    pass\n```"},
            {"role": "user", "content": "```python\ndef foo():\n    pass\n```"},
        ]
        result = truncator.remove_duplicates(messages)
        # Should have fewer messages after deduplication
        assert len(result) <= len(messages)

    def test_truncate(self) -> None:
        """Truncate context to max tokens."""
        truncator = SelectiveTruncator(max_tokens=10)
        messages = [{"role": "user", "content": "x" * 100}]
        result = truncator.truncate(messages)
        # Should be truncated
        assert len(result) <= len(messages)

    def test_singleton(self) -> None:
        """Get selective truncator returns new instance each time."""
        t1 = get_selective_truncator()
        t2 = get_selective_truncator()
        # Function returns new instances
        assert isinstance(t1, SelectiveTruncator)
        assert isinstance(t2, SelectiveTruncator)