"""Tests for context aligner."""

import pytest

from moeptimizer.context_aligner import (
    ContextAligner,
    get_context_aligner,
)


class TestContextAligner:
    def test_empty_aligner(self) -> None:
        """Empty aligner has default block size."""
        aligner = ContextAligner()
        assert aligner._block_size == 128

    def test_singleton(self) -> None:
        """Get context aligner returns new instance each time."""
        aligner1 = get_context_aligner()
        aligner2 = get_context_aligner()
        # Function returns new instances
        assert isinstance(aligner1, ContextAligner)
        assert isinstance(aligner2, ContextAligner)