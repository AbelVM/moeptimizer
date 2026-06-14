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
        """Canonicalize a single content string.

        CRITICAL: Does NOT modify content inside code blocks to preserve
        MTP prediction patterns. Only normalizes non-code content.
        """
        # Extract code blocks to preserve them
        code_pattern = re.compile(
            r"(```[\w]*\n.*?```)",
            re.DOTALL,
        )

        # Split into code and non-code sections
        parts = []
        last_end = 0

        for match in code_pattern.finditer(content):
            # Add text before (canonicalize this part)
            if match.start() > last_end:
                before = content[last_end : match.start()]
                parts.append(self._canonicalize_text(before))

            # Add code block unchanged (preserve MTP patterns)
            parts.append(match.group(1))
            last_end = match.end()

        # Add remaining text
        if last_end < len(content):
            parts.append(self._canonicalize_text(content[last_end:]))

        return "".join(parts)

    def _canonicalize_text(self, text: str) -> str:
        """Canonicalize non-code text only.

        Does NOT modify indentation or structure - only normalizes
        line endings and removes trailing whitespace.
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace only
        text = "\n".join(
            line.rstrip() for line in text.split("\n")
        )

        # Normalize multiple blank lines to single
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _normalize_indentation(
        self,
        content: str,
    ) -> str:
        """Normalize indentation to 4 spaces.

        DEPRECATED: This method is kept for API compatibility but
        should NOT be used as it breaks MTP prediction patterns.
        """
        # Return content unchanged to preserve MTP patterns
        return content

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