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
        """Drop whole old messages from the front without content truncation."""
        if not messages:
            return messages

        result = []
        remaining = self._max_tokens * 4

        for msg in messages:
            msg_size = len(msg.get("content") or "")
            if msg_size <= remaining:
                result.append(dict(msg))
                remaining -= msg_size
            else:
                break

        return result

    def _truncate_message(
        self,
        content: str,
        max_chars: int,
    ) -> str:
        """No-op for API compatibility; content truncation is disabled."""
        del max_chars
        return content

    def remove_duplicates(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate code blocks only from the newest user message."""
        result = [dict(msg) for msg in messages]
        last_user_idx = -1
        for idx in range(len(result) - 1, -1, -1):
            if result[idx].get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx < 0:
            return result

        msg = result[last_user_idx]
        content = msg.get("content", "")
        if not isinstance(content, str):
            return result

        seen_code = set()
        code_blocks = re.findall(
            r"(```[\w]*\n.*?)```", content, re.DOTALL
        )
        if not code_blocks:
            return result

        new_content = content
        for code in code_blocks:
            if code in seen_code:
                new_content = new_content.replace(
                    f"{code}```", "", 1
                )
            else:
                seen_code.add(code)

        result[last_user_idx] = {**msg, "content": new_content}
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
        for msg in messages[:-keep_last]:
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
