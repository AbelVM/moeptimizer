"""Selective truncator for context optimization.

Truncates verbose content while preserving cache-friendly structure.
"""

from __future__ import annotations

import re
from typing import Any


class SelectiveTruncator:
    """
    Selectively truncates context to reduce token usage.

    - Truncates verbose explanations (keeps code)
    - Removes duplicate code blocks
    - Summarizes old turns into structured outlines
    """

    def __init__(
        self,
        max_tokens: int = 4000,
    ) -> None:
        self._max_tokens = max_tokens

    def truncate(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Truncate context to max tokens while preserving structure."""
        # Estimate current token count
        total_chars = sum(
            len(m.get("content", "")) for m in messages
        )
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self._max_tokens:
            return messages

        # Truncate from oldest to newest
        result = []
        remaining = self._max_tokens * 4

        for msg in messages:
            content = msg.get("content", "")
            if len(content) <= remaining:
                result.append(dict(msg))
                remaining -= len(content)
            else:
                # Truncate this message
                truncated = self._truncate_message(content, remaining)
                result.append({**msg, "content": truncated})
                break

        return result

    def _truncate_message(
        self,
        content: str,
        max_chars: int,
    ) -> str:
        """Truncate a message, preferring to keep code."""
        # Find code blocks
        code_blocks = list(
            re.finditer(r"```[\w]*\n(.*?)```", content, re.DOTALL)
        )

        if not code_blocks:
            return content[:max_chars]

        # Keep code blocks, truncate explanations
        result = []
        last_end = 0

        for match in code_blocks:
            # Add text before code block
            before = content[last_end : match.start()]
            if before:
                # Truncate before text
                before = before[: max_chars // 4]
                result.append(before)
                max_chars -= len(before)

            # Add code block
            code = match.group(0)
            if len(code) <= max_chars:
                result.append(code)
                max_chars -= len(code)
            else:
                # Truncate code block
                result.append(code[:max_chars])
                max_chars = 0
                break

            last_end = match.end()

        return "".join(result)

    def remove_duplicates(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate code blocks from context."""
        seen_code = set()
        result = []

        for msg in messages:
            content = msg.get("content", "")
            # Extract code blocks (including language identifier)
            code_blocks = re.findall(
                r"(```[\w]*\n.*?)```", content, re.DOTALL
            )

            if not code_blocks:
                result.append(dict(msg))
                continue

            # Check for duplicates
            new_content = content
            for code in code_blocks:
                if code in seen_code:
                    # Remove this code block (including the opening fence)
                    new_content = new_content.replace(
                        f"{code}```", "", 1
                    )
                else:
                    seen_code.add(code)

            result.append({**msg, "content": new_content})

        return result

    def summarize_old_turns(
        self,
        messages: list[dict[str, Any]],
        keep_last: int = 2,
    ) -> list[dict[str, Any]]:
        """Summarize old turns into structured outlines."""
        if len(messages) <= keep_last:
            return messages

        result = []
        for i, msg in enumerate(messages[:-keep_last]):
# Summarize old messages
            content = msg.get("content", "")
            if len(content) > 200:
                # Create outline
                lines = content.split("\n")
                outline_lines = [
                    line for line in lines if line.strip().startswith(("#", "def ", "class "))
                ]
                if outline_lines:
                    result.append({
                        **msg,
                        "content": "# Summary of previous turn:\n" + "\n".join(outline_lines[:10]),
                    })
            else:
                result.append(dict(msg))

        # Keep last messages intact
        result.extend(messages[-keep_last:])

        return result


def get_selective_truncator(
    max_tokens: int = 4000,
) -> SelectiveTruncator:
    """Get a selective truncator instance."""
    return SelectiveTruncator(max_tokens=max_tokens)