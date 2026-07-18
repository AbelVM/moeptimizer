"""Symbol index with fuzzy matching for code retrieval.

Provides fast symbol lookup for coding agents with:
- Trie-based symbol storage
- Fuzzy matching for typos/partial names
- Symbol → file → chunk mappings
- Integration with AST skeletons
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from moeptimizer.code_chunking import LANG_MAP, _get_cached_parser


class SymbolNode:
    """Node in the symbol trie."""

    def __init__(self) -> None:
        self.children: dict[str, SymbolNode] = {}
        self.symbols: list[dict[str, Any]] = []  # (file, line, type)


class SymbolIndex:
    """
    Trie-based symbol index for fast code retrieval.

    Supports fuzzy matching for partial symbol names and typos.
    Integrates with AST skeletons for precise symbol location.
    """

    def __init__(self) -> None:
        self._root = SymbolNode()
        self._file_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._type_index: dict[str, list[str]] = defaultdict(list)

    def add_file(
        self,
        file_path: str,
        content: str,
        language: str | None = None,
    ) -> None:
        """Index all symbols in a file."""
        if language is None:
            language = self._detect_language(file_path, content)

        if language == "generic":
            # Fall back to regex-based extraction
            self._extract_symbols_regex(file_path, content)
        else:
            self._extract_symbols_treesitter(file_path, content, language)

    def _detect_language(self, file_path: str, content: str) -> str:
        """Detect language from file path and content."""
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in LANG_MAP:
            return LANG_MAP[ext]
        return "generic"

    def _extract_symbols_treesitter(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> None:
        """Extract symbols using tree-sitter AST."""
        parser = _get_cached_parser(language)
        if parser is None:
            self._extract_symbols_regex(file_path, content)
            return

        try:
            tree = parser.parse(content)
            root = tree.root_node()
            self._walk_node(root, content, file_path, language)
        except Exception:
            self._extract_symbols_regex(file_path, content)

    def _walk_node(
        self,
        node: Any,
        content: str,
        file_path: str,
        language: str,
    ) -> None:
        """Walk AST node and extract symbols."""
        symbol_types = {
            "function_definition",
            "function_declaration",
            "method_definition",
            "class_definition",
            "class_declaration",
            "struct_item",
            "impl_item",
            "trait_item",
            "interface_declaration",
            "enum_declaration",
            "type_alias",
            "variable_declaration",
        }

        node_type = node.kind()
        if node_type in symbol_types:
            # Extract symbol name
            name = self._get_node_name(node, content)
            if name:
                br = node.byte_range()
                line_start = content[: br.start].count("\n") + 1
                self._add_symbol(name, file_path, line_start, node_type)

        # Recurse into children
        for i in range(node.child_count()):
            self._walk_node(node.child(i), content, file_path, language)

    def _get_node_name(self, node: Any, content: str) -> str | None:
        """Extract symbol name from AST node."""
        # Try to find identifier child
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() in ("identifier", "name"):
                br = child.byte_range()
                return content[br.start : br.end]
        return None

    def _extract_symbols_regex(
        self,
        file_path: str,
        content: str,
    ) -> None:
        """Extract symbols using regex patterns."""
        patterns = [
            (r"\bdef\s+(\w+)", "function_definition"),
            (r"\bclass\s+(\w+)", "class_definition"),
            (r"\bfunction\s+(\w+)", "function_declaration"),
            (r"\bconst\s+(\w+)", "variable_declaration"),
            (r"\blet\s+(\w+)", "variable_declaration"),
        ]

        for pattern, sym_type in patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                line = content[: match.start()].count("\n") + 1
                self._add_symbol(name, file_path, line, sym_type)

    def _add_symbol(
        self,
        name: str,
        file_path: str,
        line: int,
        sym_type: str,
    ) -> None:
        """Add symbol to trie and indexes."""
        # Add to trie
        node = self._root
        for char in name.lower():
            if char not in node.children:
                node.children[char] = SymbolNode()
            node = node.children[char]
        node.symbols.append({
            "name": name,
            "file": file_path,
            "line": line,
            "type": sym_type,
        })

        # Add to file index
        self._file_index[file_path].append({
            "name": name,
            "line": line,
            "type": sym_type,
        })

        # Add to type index
        self._type_index[sym_type].append(name)

    def find_symbol(
        self,
        query: str,
        fuzzy: bool = True,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Find symbols matching query.

        Uses fuzzy matching if enabled, otherwise exact match.
        Returns list of symbol info with file, line, and type.
        """
        if fuzzy:
            return self._fuzzy_search(query, max_results)
        return self._exact_search(query, max_results)

    def _exact_search(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Exact symbol search."""
        node = self._root
        for char in query.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        return node.symbols[:max_results]

    def _fuzzy_search(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Fuzzy symbol search using Levenshtein distance."""
        results: list[tuple[dict[str, Any], int]] = []
        query_lower = query.lower()

        # Collect all symbols (limit for performance)
        all_symbols = self._collect_all_symbols()

        for symbol in all_symbols:
            name_lower = symbol["name"].lower()
            # Simple fuzzy: check if query is substring or close match
            if query_lower in name_lower:
                results.append((symbol, 0))
            elif self._levenshtein_distance(query_lower, name_lower) <= 2:
                results.append((symbol, 1))

        # Sort by distance, then by type priority
        type_priority = {
            "class_definition": 0,
            "function_definition": 1,
            "method_definition": 2,
            "function_declaration": 3,
        }
        results.sort(
            key=lambda x: (
                x[1],
                type_priority.get(x[0]["type"], 99),
            ),
        )

        return [s[0] for s in results[:max_results]]

    def _collect_all_symbols(self) -> list[dict[str, Any]]:
        """Collect all symbols from trie (for fuzzy search)."""
        results: list[dict[str, Any]] = []
        self._collect_from_node(self._root, results)
        return results

    def _collect_from_node(
        self,
        node: SymbolNode,
        results: list[dict[str, Any]],
    ) -> None:
        """Recursively collect symbols from trie node."""
        results.extend(node.symbols)
        for child in node.children.values():
            self._collect_from_node(child, results)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_symbols_in_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all symbols defined in a file."""
        return self._file_index.get(file_path, [])

    def get_symbols_by_type(self, sym_type: str) -> list[str]:
        """Get all symbols of a given type."""
        return self._type_index.get(sym_type, [])

    def get_context_for_symbol(
        self,
        symbol: dict[str, Any],
        content_getter: Any,
    ) -> str:
        """Get context for a symbol (for RAG injection)."""
        file_path = symbol["file"]
        line = symbol["line"]

        # Get file content
        content = content_getter(file_path)
        if not content:
            return ""

        # Extract surrounding context
        lines = content.split("\n")
        start = max(0, line - 10)
        end = min(len(lines), line + 10)

        return "\n".join(lines[start:end])
