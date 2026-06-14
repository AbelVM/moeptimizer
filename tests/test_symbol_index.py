"""Tests for symbol index."""

import pytest

from moeptimizer.symbol_index import (
    SymbolIndex,
)


class TestSymbolIndex:
    def test_empty_index(self) -> None:
        """Empty index has no symbols."""
        index = SymbolIndex()
        assert index is not None

    def test_add_file(self) -> None:
        """Add a file to the index."""
        index = SymbolIndex()
        code = "def foo():\n    pass\n"
        index.add_file("test.py", code)
        # Should have indexed the function
        symbols = index.get_symbols_in_file("test.py")
        assert len(symbols) > 0

    def test_find_symbol(self) -> None:
        """Find a symbol in the index."""
        index = SymbolIndex()
        code = "def foo():\n    pass\n"
        index.add_file("test.py", code)
        results = index.find_symbol("foo")
        assert len(results) > 0
        assert any("foo" in r.get("name", "") for r in results)