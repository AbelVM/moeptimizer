"""Pattern injector for cache-stable context.

Adds consistent section markers and delimiters.
"""

from __future__ import annotations

import re
from typing import Any


class PatternInjector:
    """
    Injects patterns for cache-stable context.

    - Adds consistent section markers
    - Uses predictable delimiters
    - Maintains code structure patterns
    """

    SECTION_MARKERS = {
        "static": "<!-- STATIC_LAYER -->",
        "context": "<!-- CONTEXT_LAYER -->",
        "dynamic": "<!-- DYNAMIC_LAYER -->",
    }

    def __init__(self) -> None:
        self._marker_pattern = re.compile(
            r"<!-- (STATIC|CONTEXT|DYNAMIC)_LAYER -->",
        )

    def inject_markers(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject section markers into context.

        Only adds markers to system and user messages, NOT assistant messages,
        to preserve the model's expected chat template.
        """
        if not messages:
            return messages

        result = []
        static_done = False
        context_done = False

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only add markers to system and user messages
            # Assistant messages must preserve the model's expected template
            if role == "system" and not static_done:
                content = f"{self.SECTION_MARKERS['static']}\n{content}"
                static_done = True
            elif role == "user" and not context_done:
                content = f"{self.SECTION_MARKERS['context']}\n{content}"
                context_done = True
            # Skip markers for assistant messages to preserve chat template

            result.append({**msg, "content": content})

        return result

    def remove_markers(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove section markers from context."""
        return [
            {
                **msg,
                "content": self._marker_pattern.sub("", msg.get("content", "")),
            }
            for msg in messages
        ]

    def add_delimiters(
        self,
        content: str,
        delimiter: str = "\n\n",
    ) -> str:
        """Add consistent delimiters between sections."""
        # Normalize multiple newlines
        content = re.sub(r"\n{3,}", delimiter, content)
        return content

    def ensure_pattern(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ensure consistent patterns in context."""
        return self.inject_markers(messages)


def get_pattern_injector() -> PatternInjector:
    """Get a pattern injector instance."""
    return PatternInjector()