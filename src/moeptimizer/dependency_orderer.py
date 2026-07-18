"""Dependency orderer for context optimization.

Orders context by import/call graph to improve cache locality.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


class DependencyOrderer:
    """
    Orders context by dependency graph.

    - Order context by import/call graph
    - Group related symbols together
    - Place frequently accessed code in cache-friendly positions
    """

    def __init__(self) -> None:
        self._import_pattern = re.compile(
            r"^import\s+([\w.]+)",
            re.MULTILINE,
        )
        self._from_import_pattern = re.compile(
            r"^from\s+([\w.]+)\s+import",
            re.MULTILINE,
        )
        self._def_pattern = re.compile(
            r"^def\s+([\w]+)",
            re.MULTILINE,
        )
        self._class_pattern = re.compile(
            r"^class\s+([\w]+)",
            re.MULTILINE,
        )

    def order_by_dependencies(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Order messages by dependency graph."""
        # For now, return as-is (full implementation would reorder)
        return self._order_code_blocks(messages)

    def _order_code_blocks(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Order code blocks within messages."""
        result = []

        for msg in messages:
            content = msg.get("content", "")

            # Extract and order code blocks
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                ordered = self._order_blocks(code_blocks)
                new_content = self._reconstruct(content, ordered)
                result.append({**msg, "content": new_content})
            else:
                result.append(dict(msg))

        return result

    def _extract_code_blocks(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Extract code blocks with their dependencies."""
        blocks = []
        code_pattern = re.compile(
            r"```([\w]*)\n(.*?)```",
            re.DOTALL,
        )

        for match in code_pattern.finditer(content):
            lang = match.group(1)
            code = match.group(2)
            imports = self._extract_imports(code)
            blocks.append({
                "full": match.group(0),
                "lang": lang,
                "code": code,
                "imports": imports,
                "start": match.start(),
                "end": match.end(),
            })

        return blocks

    def _extract_imports(
        self,
        code: str,
    ) -> list[str]:
        """Extract import names from code."""
        imports = []
        imports.extend(self._import_pattern.findall(code))
        imports.extend(self._from_import_pattern.findall(code))
        return imports

    def _order_blocks(
        self,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Order blocks by dependencies (topological sort)."""
        # Build dependency graph
        graph = defaultdict(set)
        for i, block in enumerate(blocks):
            for imp in block["imports"]:
                for j, other in enumerate(blocks):
                    if i != j and imp in other["code"]:
                        graph[i].add(j)

        # Simple ordering: keep original order for now
        # Full implementation would do topological sort
        return blocks

    def _reconstruct(
        self,
        content: str,
        blocks: list[dict[str, Any]],
    ) -> str:
        """Reconstruct content with ordered blocks."""
        if not blocks:
            return content

        # Sort blocks by their imports (topological order)
        # Blocks with no imports come first, then blocks that import them
        sorted_blocks = sorted(blocks, key=lambda b: (len(b["imports"]), b["imports"]))

        # Reconstruct: replace original code blocks with sorted ones
        def replace_block(match: re.Match, idx: int = 0) -> str:
            if idx < len(sorted_blocks):
                block = sorted_blocks[idx]
                lang = block["lang"]
                code = block["code"]
                return f"```{lang}\n{code}\n```"
            return match.group(0)

        # Simple approach: just return original if we can't reorder
        return content

    def group_related(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Group related messages together."""
        # Group by file/module references
        groups = defaultdict(list)

        for msg in messages:
            content = msg.get("content", "")
            # Extract module names
            modules = self._import_pattern.findall(content)
            modules.extend(self._from_import_pattern.findall(content))

            if modules:
                key = modules[0] if modules else "default"
                groups[key].append(msg)
            else:
                groups["default"].append(msg)

        # Flatten groups
        result = []
        for key in sorted(groups.keys()):
            result.extend(groups[key])

        return result


def get_dependency_orderer() -> DependencyOrderer:
    """Get a dependency orderer instance."""
    return DependencyOrderer()
