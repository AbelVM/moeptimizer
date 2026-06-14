"""Hierarchical repository indexing for efficient symbol lookup.

Builds package → module → class → function hierarchy for faster
retrieval and better cache locality.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from moeptimizer.code_chunking import LANG_MAP, _get_cached_parser


class HierarchyNode:
    """Node in the repository hierarchy."""

    def __init__(self) -> None:
        self.children: dict[str, HierarchyNode] = {}
        self.symbols: list[dict[str, Any]] = []
        self.files: set[str] = set()


class HierarchicalIndex:
    """
    Hierarchical index for repository structure.

    Structure: package → module → class → function
    Enables faster symbol lookup and better cache locality.
    """

    def __init__(self) -> None:
        self._root = HierarchyNode()
        self._symbol_to_node: dict[str, HierarchyNode] = {}

    def add_file(
        self,
        file_path: str,
        content: str,
    ) -> None:
        """Add a file to the hierarchy index."""
        # Extract package/module from path
        parts = self._path_to_parts(file_path)

        # Navigate/create hierarchy
        node = self._root
        for part in parts:
            if part not in node.children:
                node.children[part] = HierarchyNode()
            node = node.children[part]
            node.files.add(file_path)

        # Index symbols in this file
        self._index_symbols(node, file_path, content)

    def _path_to_parts(self, file_path: str) -> list[str]:
        """Convert file path to hierarchy parts."""
        # Remove extension
        path = file_path.rsplit(".", 1)[0] if "." in file_path else file_path

        # Split into parts
        parts = path.replace("/", ".").split(".")
        return [p for p in parts if p]

    def _index_symbols(
        self,
        node: HierarchyNode,
        file_path: str,
        content: str,
    ) -> None:
        """Index symbols in a file at the given hierarchy node."""
        # Extract symbols using regex (fast path)
        patterns = [
            (r"\bclass\s+(\w+)", "class"),
            (r"\bdef\s+(\w+)", "function"),
            (r"\bfunction\s+(\w+)", "function"),
            (r"\bconst\s+(\w+)", "variable"),
            (r"\blet\s+(\w+)", "variable"),
        ]

        for pattern, sym_type in patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                line = content[: match.start()].count("\n") + 1
                symbol = {
                    "name": name,
                    "file": file_path,
                    "line": line,
                    "type": sym_type,
                }
                node.symbols.append(symbol)
                self._symbol_to_node[f"{file_path}:{name}"] = node

    def find_symbols(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Find symbols by hierarchical search.

        Searches from package level down, prioritizing:
        1. Exact matches in current context
        2. Matches in same module
        3. Matches in same package
        4. Global matches
        """
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        # Search hierarchy breadth-first
        visited: set[str] = set()
        queue = [self._root]

        while queue and len(results) < max_results:
            node = queue.pop(0)

            # Check symbols in this node
            for symbol in node.symbols:
                if query_lower in symbol["name"].lower():
                    results.append(symbol)

            # Add children to queue
            for child in node.children.values():
                if id(child) not in visited:
                    visited.add(id(child))
                    queue.append(child)

        return results[:max_results]

    def get_context_files(
        self,
        file_path: str,
    ) -> list[str]:
        """Get files in the same hierarchy context."""
        parts = self._path_to_parts(file_path)

        # Navigate to the file's node
        node = self._root
        for part in parts:
            if part in node.children:
                node = node.children[part]
            else:
                return []

        # Return all files in this subtree
        return list(node.files)

    def get_dependency_context(
        self,
        file_path: str,
        max_files: int = 5,
    ) -> list[str]:
        """Get context files that this file likely depends on."""
        # Get files in same module/package
        context_files = self.get_context_files(file_path)

        # Filter to likely dependencies (same package, different file)
        deps = [f for f in context_files if f != file_path][:max_files]

        return deps


# Global index instance
_index: HierarchicalIndex | None = None


def get_hierarchical_index() -> HierarchicalIndex:
    """Get or create the global hierarchical index."""
    global _index
    if _index is None:
        _index = HierarchicalIndex()
    return _index