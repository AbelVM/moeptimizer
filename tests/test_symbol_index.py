"""Tests for symbol index with fuzzy matching."""

import pytest

from moeptimizer.symbol_index import SymbolIndex


class TestSymbolIndex:
    def test_empty_index(self) -> None:
        """Empty index returns no symbols."""
        index = SymbolIndex()
        assert index.find_symbol("foo") == []

    def test_add_file_python(self) -> None:
        """Add Python file and extract symbols."""
        index = SymbolIndex()
        code = '''
import os
from typing import List

class MyClass:
    def __init__(self):
        pass

    def my_method(self):
        return 42

def my_function():
    pass
'''
        index.add_file("test.py", code, language="python")
        symbols = index.find_symbol("my")
        assert len(symbols) >= 2
        names = [s["name"] for s in symbols]
        assert "my_method" in names or "my_function" in names

    def test_fuzzy_matching(self) -> None:
        """Fuzzy matching finds similar symbols."""
        index = SymbolIndex()
        code = '''
def calculate_total():
    pass

def calculate_average():
    pass
'''
        index.add_file("test.py", code, language="python")
        # Fuzzy match for "calculate"
        symbols = index.find_symbol("calculat", fuzzy=True)
        assert len(symbols) >= 2

    def test_exact_search(self) -> None:
        """Exact search for symbol name."""
        index = SymbolIndex()
        code = '''
def my_function():
    pass
'''
        index.add_file("test.py", code, language="python")
        symbols = index.find_symbol("my_function", fuzzy=False)
        assert len(symbols) == 1
        assert symbols[0]["name"] == "my_function"

    def test_get_symbols_in_file(self) -> None:
        """Get all symbols in a file."""
        index = SymbolIndex()
        code = '''
class Foo:
    pass

def bar():
    pass
'''
        index.add_file("test.py", code, language="python")
        symbols = index.get_symbols_in_file("test.py")
        assert len(symbols) == 2

    def test_get_symbols_by_type(self) -> None:
        """Get symbols by type."""
        index = SymbolIndex()
        code = '''
class MyClass:
    pass

def my_function():
    pass
'''
        index.add_file("test.py", code, language="python")
        classes = index.get_symbols_by_type("class_definition")
        functions = index.get_symbols_by_type("function_definition")
        assert len(classes) == 1
        assert len(functions) == 1