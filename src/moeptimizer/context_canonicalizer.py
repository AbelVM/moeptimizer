"""Context canonicalizer for cache-friendly formatting.

Normalizes code formatting to maximize cache hits.
"""

from __future__ import annotations

import re
from typing import Any


class ContextCanonicalizer:
    """
    Canonicalizes context for cache-friendly formatting.

    - Sorts imports alphabetically
    - Normalizes code formatting
    - Removes redundant whitespace
    - Standardizes indentation
    """

    def __init__(self) -> None:
        self._import_pattern = re.compile(
            r"^import\s+(.+)$",
            re.MULTILINE,
        )
        self._from_import_pattern = re.compile(
            r"^from\s+(\S+)\s+import\s+(.+)$",
            re.MULTILINE,
        )

    def canonicalize(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Canonicalize all message content.

        Only applies to system and user messages to preserve assistant chat template.
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only canonicalize system and user messages
            # Assistant messages must preserve the model's expected template
            if role in ("system", "user"):
                content = self._canonicalize_content(content)

            result.append({**msg, "content": content})

        return result

    def _canonicalize_content(
        self,
        content: str,
    ) -> str:
        """Canonicalize a single content string."""
        # Normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize indentation (4 spaces)
        content = self._normalize_indentation(content)

        # Remove trailing whitespace
        content = "\n".join(
            line.rstrip() for line in content.split("\n")
        )

        # Normalize multiple blank lines to single
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content

    def _normalize_indentation(
        self,
        content: str,
    ) -> str:
        """Normalize indentation to 4 spaces."""
        lines = content.split("\n")
        result = []
        for line in lines:
            if not line.strip():
                result.append("")
                continue
            # Convert tabs to spaces
            line = line.replace("\t", "    ")
            # Find minimum indentation
            stripped = line.lstrip(" ")
            if stripped:
                indent = len(line) - len(stripped)
                # Normalize to multiples of 4
                normalized_indent = (indent // 4) * 4
                result.append(" " * normalized_indent + stripped)
            else:
                result.append(line)
        return "\n".join(result)

    def sort_imports(
        self,
        code: str,
    ) -> str:
        """Sort imports alphabetically in code."""
        lines = code.split("\n")
        import_lines = []
        from_import_lines = []
        other_lines = []
        in_import_section = True

        for line in lines:
            stripped = line.strip()
            if in_import_section and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
            ):
                if stripped.startswith("import "):
                    import_lines.append(stripped)
                else:
                    from_import_lines.append(stripped)
            else:
                in_import_section = False
                other_lines.append(line)

        # Sort imports
        import_lines.sort()
        from_import_lines.sort()

        return "\n".join(import_lines + from_import_lines + other_lines)


def get_context_canonicalizer() -> ContextCanonicalizer:
    """Get a context canonicalizer instance."""
    return ContextCanonicalizer()