"""Context compressor for cache optimization.

Compresses code to skeletons while preserving cache-friendly structure.
"""

from __future__ import annotations

import re
from typing import Any

from moeptimizer.code_chunking import LANG_MAP, _get_cached_parser


class ContextCompressor:
    """
    Compresses context to skeletons.

    - Compress code to skeletons
    - Keep cache-friendly structure
    - Use consistent compression patterns
    """

    def __init__(
        self,
        preserve_signatures: bool = True,
        preserve_short_code_blocks: bool = True,
        short_code_block_max_chars: int = 128,
    ) -> None:
        self._preserve_signatures = preserve_signatures
        self._preserve_short_code_blocks = preserve_short_code_blocks
        self._short_code_block_max_chars = short_code_block_max_chars

    def compress(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress context to skeletons.

        Compresses code in all messages to reduce context size.
        Preserves the model's expected chat template structure.
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only compress code in user/system messages, not assistant responses
            # Assistant responses should be preserved as-is for quality
            if role in ("system", "user"):
                content = self._compress_content(content)

            result.append({**msg, "content": content})

        return result

    def _compress_content(
        self,
        content: str,
    ) -> str:
        """Compress a single content string."""
        # Extract code blocks
        code_pattern = re.compile(
            r"```([\w]*)\n(.*?)```",
            re.DOTALL,
        )

        def replace_code(match: re.Match) -> str:
            lang = match.group(1)
            code = match.group(2)
            skeleton = self._compress_code(code, lang)
            return f"```{lang}\n{skeleton}\n```"

        return code_pattern.sub(replace_code, content)

    def _compress_code(
        self,
        code: str,
        lang: str = "",
    ) -> str:
        """Compress code to skeleton using tree-sitter for proper parsing."""
        if not self._preserve_signatures:
            return code

        # Small code snippets are often the original problem statement or
        # minimal reproducible examples. Compressing them removes the exact
        # semantics the model needs, so preserve them verbatim while still
        # skeletonizing larger code bodies.
        if self._preserve_short_code_blocks and len(code.strip()) <= self._short_code_block_max_chars:
            return code

        # Try tree-sitter first for proper AST-based skeleton extraction
        lang_id = LANG_MAP.get(lang, lang) if lang else "generic"
        parser = _get_cached_parser(lang_id)

        if parser is not None:
            try:
                tree = parser.parse(code)
                root = tree.root_node()
                return self._extract_skeleton_with_treesitter(code, root)
            except Exception:
                pass

        # Fallback to line-based skeleton extraction
        return self._extract_skeleton_fallback(code)

    def _extract_skeleton_with_treesitter(
        self,
        code: str,
        root: Any,
    ) -> str:
        """Extract an AST skeleton that preserves signatures and structure."""
        skeleton_lines: list[str] = []

        def _get_text(node: Any) -> str:
            br = node.byte_range()
            return code[br.start : br.end]

        def _get_kind(node: Any) -> str:
            return node.kind()

        def _indent(level: int) -> str:
            return "    " * level

        def _normalize_signature(text: str) -> str:
            return " ".join(text.strip().split())

        def _function_signature(node: Any) -> str:
            text = _get_text(node)
            colon = text.find(":")
            if colon != -1:
                return _normalize_signature(text[: colon + 1])

            first_line = text.splitlines()[0].strip() if text.splitlines() else ""
            if first_line.endswith(":"):
                return first_line
            return _normalize_signature(first_line)

        def _class_signature(node: Any) -> str:
            first_line = _get_text(node).splitlines()[0].strip()
            return _normalize_signature(first_line)

        def _append_function(node: Any, level: int) -> None:
            for child in node.children:
                if _get_kind(child) == "decorator":
                    skeleton_lines.append(f"{_indent(level)}{_get_text(child).strip()}")
            signature = _function_signature(node)
            if signature:
                skeleton_lines.append(f"{_indent(level)}{signature}")
                skeleton_lines.append(f"{_indent(level)}    ...")

        def _append_class(node: Any, level: int) -> None:
            skeleton_lines.append(f"{_indent(level)}{_class_signature(node)}")
            child_lines_start = len(skeleton_lines)
            for child in node.children:
                kind = _get_kind(child)
                if kind in def_types:
                    _append_function(child, level + 1)
                elif kind in class_types:
                    _append_class(child, level + 1)
                elif kind == "comment":
                    skeleton_lines.append(f"{_indent(level + 1)}{_get_text(child).strip()}")

            if len(skeleton_lines) == child_lines_start:
                skeleton_lines.append(f"{_indent(level + 1)}...")

        def _append_node(node: Any, level: int) -> None:
            kind = _get_kind(node)
            if kind in def_types:
                _append_function(node, level)
            elif kind in class_types:
                _append_class(node, level)
            elif kind in import_types or kind == "comment":
                skeleton_lines.append(f"{_indent(level)}{_get_text(node).strip()}")
            else:
                for child in node.children:
                    _append_node(child, level)

        # Node types that represent function/class definitions
        def_types = {
            "function_definition",
            "function_declaration",
            "method_definition",
            "def",
            "function_item",
            "function",
        }
        class_types = {
            "class_definition",
            "class_declaration",
            "class",
            "class_specifier",
        }
        import_types = {
            "import_statement",
            "import_declaration",
            "import_from_statement",
            "use_declaration",
            "package_clause",
        }

        _append_node(root, 0)
        return "\n".join(skeleton_lines)

    def _extract_skeleton_fallback(
        self,
        code: str,
    ) -> str:
        """Fallback skeleton extraction using regex patterns."""
        lines = code.split("\n")
        skeleton_lines = []

        for line in lines:
            stripped = line.strip()

            # Keep signatures and structure
            if any(
                stripped.startswith(kw)
                for kw in ("def ", "class ", "import ", "from ", "function ", "pub fn ", "fn ")
            ):
                skeleton_lines.append(line)
            elif stripped and not stripped.startswith("#"):
                # Replace body with ellipsis to indicate continuation
                indent = len(line) - len(line.lstrip())
                skeleton_lines.append(" " * indent + "...")
            else:
                skeleton_lines.append(line)

        return "\n".join(skeleton_lines)

    def compress_to_outline(
        self,
        code: str,
    ) -> str:
        """Compress code to outline form."""
        lines = code.split("\n")
        outline = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith(("def ", "class ")) or stripped.startswith(("import ", "from ")):
                outline.append(stripped)

        return "\n".join(outline)

    def get_compression_ratio(
        self,
        original: str,
        compressed: str,
    ) -> float:
        """Get compression ratio."""
        if not original:
            return 0.0
        return round(len(compressed) / len(original), 2)


def get_context_compressor(
    preserve_signatures: bool = True,
) -> ContextCompressor:
    """Get a context compressor instance."""
    return ContextCompressor(preserve_signatures=preserve_signatures)
