"""Summarize old turns for long conversation context reduction.

Reduces context size by summarizing old conversation turns while preserving
key information for the model.
"""

from __future__ import annotations

import re
from typing import Any


class SummarizeOldTurns:
    """
    Summarizes old conversation turns to reduce context size.

    - Identifies old turns to summarize
    - Creates compact summaries preserving key information
    - Maintains conversation flow for the model
    """

    def __init__(self, max_turns_to_keep: int = 10) -> None:
        self._max_turns_to_keep = max_turns_to_keep

    def summarize(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Summarize old turns if context is too long.

        Keeps the most recent turns intact and summarizes older ones.
        """
        if len(messages) <= self._max_turns_to_keep:
            return messages

        # Find the system anchor (system + first user)
        system_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_end = i + 1
            elif (msg.get("role") == "user" and system_end > 0) or (msg.get("role") == "user" and system_end == 0):
                system_end = i + 1
                break

        # Get messages to potentially summarize
        system_anchor = messages[:system_end]
        rest = messages[system_end:]

        # If we have more than max_turns_to_keep, summarize the oldest
        if len(rest) > self._max_turns_to_keep:
            # Keep the most recent turns intact
            keep_count = self._max_turns_to_keep
            to_summarize = rest[:-keep_count]
            keep_recent = rest[-keep_count:]

            # Create summary
            summary = self._create_summary(to_summarize)

            return [*system_anchor, summary, *keep_recent]

        return messages

    def _create_summary(
        self,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a compact summary of old turns."""
        # Extract key information from turns
        code_blocks = []
        key_facts = []

        for msg in turns:
            content = msg.get("content", "")
            if not content:
                continue

            # Extract code blocks
            code_pattern = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
            for match in code_pattern.finditer(content):
                code = match.group(1).strip()
                if code:
                    # Get function signatures only
                    sigs = self._extract_signatures(code)
                    if sigs:
                        code_blocks.append(sigs)

            # Extract key facts (first sentence of user messages)
            if msg.get("role") == "user":
                first_sentence = content.split(".")[0].strip()
                if len(first_sentence) > 10 and len(first_sentence) < 200:
                    key_facts.append(first_sentence)

        # Build summary content
        summary_parts = ["[Previous conversation summary]"]
        if key_facts:
            summary_parts.append("Key topics: " + "; ".join(key_facts[:3]))
        if code_blocks:
            summary_parts.append("Code discussed: " + "; ".join(code_blocks[:3]))

        return {
            "role": "user",
            "content": "\n".join(summary_parts),
        }

    def _extract_signatures(self, code: str) -> str:
        """Extract function signatures from code."""
        lines = code.split("\n")
        sigs = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "function ")):
                # Get just the signature
                if ":" in stripped:
                    sigs.append(stripped.split(":")[0] + ":")
                else:
                    sigs.append(stripped)
        return " ".join(sigs) if sigs else ""


def get_summarize_old_turns(
    max_turns_to_keep: int = 10,
) -> SummarizeOldTurns:
    """Get a summarize old turns instance."""
    return SummarizeOldTurns(max_turns_to_keep=max_turns_to_keep)
