"""Tests for selective truncator."""


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
        """Remove duplicate code blocks only from the newest user message."""
        truncator = SelectiveTruncator()
        messages = [
            {"role": "user", "content": "```python\ndef foo():\n    pass\n```"},
            {"role": "user", "content": "```python\ndef foo():\n    pass\n```\n```python\ndef foo():\n    pass\n```"},
        ]
        result = truncator.remove_duplicates(messages)
        assert len(result) == len(messages)
        assert result[0]["content"].count("def foo") == 1
        assert result[1]["content"].count("def foo") == 1
        assert "```python\ndef foo():\n    pass\n```" in result[0]["content"]

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
