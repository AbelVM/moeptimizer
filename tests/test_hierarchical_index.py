"""Tests for hierarchical repository indexing."""

import pytest

from moeptimizer.hierarchical_index import (
    HierarchicalIndex,
    get_hierarchical_index,
)


class TestHierarchicalIndex:
    def test_empty_index(self) -> None:
        """Empty index has no symbols."""
        index = HierarchicalIndex()
        assert index.find_symbols("foo") == []

    def test_add_file_python(self) -> None:
        """Add Python file to index."""
        index = HierarchicalIndex()
        content = "import os\n\nclass Foo:\n    def bar(self):\n        pass\n"
        index.add_file("src/foo.py", content)
        results = index.find_symbols("Foo")
        assert len(results) > 0
        assert any("Foo" in r["name"] for r in results)

    def test_find_symbols(self) -> None:
        """Find symbols by query."""
        index = HierarchicalIndex()
        content = "def hello():\n    pass\n"
        index.add_file("src/test.py", content)
        results = index.find_symbols("hello")
        assert len(results) > 0

    def test_get_context_files(self) -> None:
        """Get files in same context."""
        index = HierarchicalIndex()
        index.add_file("src/pkg/module.py", "def foo(): pass")
        index.add_file("src/pkg/other.py", "def bar(): pass")
        files = index.get_context_files("src/pkg/module.py")
        assert "src/pkg/module.py" in files

    def test_singleton(self) -> None:
        """Get hierarchical index returns singleton."""
        index1 = get_hierarchical_index()
        index2 = get_hierarchical_index()
        assert index1 is index2