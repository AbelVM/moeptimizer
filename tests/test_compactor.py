"""Tests for scratchpad compactor."""

import pytest

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