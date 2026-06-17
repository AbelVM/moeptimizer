"""Hierarchical Summarization for long conversation context.

Summarizes older turns into a single "recall" token that can be expanded
on demand, keeping context lean while preserving key information.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistence path
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "hierarchical_summaries.json"


class HierarchicalSummarizer:
    """
    Summarizes older conversation turns hierarchically.

    Creates multi-level summaries:
    - Level 0: Full turn (original content)
    - Level 1: Compact summary (key points, ~10% size)
    - Level 2: Recall token (single token representation, ~1% size)

    Older turns are progressively summarized to keep context lean.
    """

    def __init__(
        self,
        max_full_turns: int = 5,
        max_summary_turns: int = 15,
    ) -> None:
        self._max_full_turns = max_full_turns
        self._max_summary_turns = max_summary_turns
        self._summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stats: dict[str, int] = {
            "turns_summarized": 0,
            "turns_compressed": 0,
            "recall_tokens_created": 0,
        }

    def summarize_turns(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Summarize older turns to reduce context size.

        Keeps the most recent turns intact and summarizes older ones
        into compact summaries.

        Args:
            messages: The message list to summarize

        Returns:
            Message list with older turns summarized
        """
        if len(messages) <= self._max_full_turns:
            return messages

        # Find the system anchor (system + first user)
        system_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_end = i + 1
            elif (msg.get("role") == "user" and system_end > 0) or (msg.get("role") == "user" and system_end == 0):
                system_end = i + 1
                break

        system_anchor = messages[:system_end]
        system_only = system_anchor[:1] if system_anchor else []
        first_user = system_anchor[1:]
        rest = messages[system_end:]

        if len(rest) <= self._max_full_turns:
            return messages

        # Keep recent turns full, summarize older ones
        keep_count = self._max_full_turns
        to_summarize = rest[:-keep_count] if len(rest) > keep_count else []
        keep_recent = rest[-keep_count:]

        if not to_summarize:
            return messages

        # Create hierarchical summary
        summary = self._create_hierarchical_summary(to_summarize)

        result = [*system_only, summary, *first_user, *keep_recent]
        self._stats["turns_summarized"] += len(to_summarize)
        self._stats["turns_compressed"] += 1

        return result

    def _create_hierarchical_summary(
        self,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a hierarchical summary of old turns.

        Creates a compact summary that preserves key information:
        - Topics discussed
        - Code patterns seen
        - Decisions made
        - Current state

        Args:
            turns: List of turns to summarize

        Returns:
            Summary message dict
        """
        # Generate a stable ID for this summary
        turn_ids = [m.get("step_id", str(i)) for i, m in enumerate(turns)]
        summary_id = hashlib.md5(
            json.dumps(turn_ids).encode()
        ).hexdigest()[:16]

        # Extract key information
        topics: list[str] = []
        code_patterns: list[str] = []
        tool_uses: list[str] = []

        for msg in turns:
            content = msg.get("content", "")
            if not content:
                continue

            role = msg.get("role", "")

            # Extract topics from user messages
            if role == "user":
                # Get first meaningful sentence
                sentences = content.replace("?", ".").replace("!", ".").split(".")
                for sent in sentences[:2]:
                    sent = sent.strip()
                    if 10 < len(sent) < 150:
                        topics.append(sent)

            # Extract code patterns from assistant messages
            elif role == "assistant":
                import re
                for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
                    lang = match.group(1)
                    code = match.group(2)
                    # Extract function/class signatures
                    for line in code.split("\n")[:5]:
                        stripped = line.strip()
                        if stripped.startswith(("def ", "class ", "function ", "fn ", "pub fn ")):
                            code_patterns.append(f"{lang}:{stripped[:60]}")
                            break

            # Track tool usage
            elif role == "tool":
                tool_name = msg.get("tool_name", msg.get("metadata", {}).get("name", ""))
                if tool_name:
                    tool_uses.append(tool_name)

        # Build compact summary
        summary_parts = [f"[Recall:{summary_id}]"]

        if topics:
            summary_parts.append(f"Topics: {'; '.join(topics[:3])}")

        if code_patterns:
            summary_parts.append(f"Code: {'; '.join(code_patterns[:3])}")

        if tool_uses:
            unique_tools = list(dict.fromkeys(tool_uses))[:5]
            summary_parts.append(f"Tools: {', '.join(unique_tools)}")

        if not topics and not code_patterns and not tool_uses:
            summary_parts.append(f"History: {len(turns)} summarized turns")

        # Store full summary for potential expansion
        full_summary = {
            "summary_id": summary_id,
            "topics": topics[:5],
            "code_patterns": code_patterns[:5],
            "tool_uses": list(dict.fromkeys(tool_uses))[:10],
            "turn_count": len(turns),
            "created_at": time.time(),
            "level": 1,  # Level 1 = compact summary
        }

        self._summaries[summary_id] = full_summary
        while len(self._summaries) > 100:
            self._summaries.popitem(last=False)

        return {
            "role": "user",
            "content": " | ".join(summary_parts),
            "_summary_id": summary_id,
            "_summary_level": 1,
        }

    def expand_summary(
        self,
        summary_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Expand a recall token back to a fuller summary.

        Args:
            summary_message: The summary message to expand

        Returns:
            Expanded message with more detail
        """
        summary_id = summary_message.get("_summary_id", "")
        if not summary_id or summary_id not in self._summaries:
            return summary_message

        stored = self._summaries[summary_id]
        level = summary_message.get("_summary_level", 1)

        if level >= 2:
            # Already at max expansion
            return summary_message

        # Expand to level 2
        expanded_parts = [f"[Expanded:{summary_id}]"]

        if stored.get("topics"):
            expanded_parts.append(f"Topics: {'; '.join(stored['topics'])}")

        if stored.get("code_patterns"):
            expanded_parts.append(f"Code: {'; '.join(stored['code_patterns'])}")

        if stored.get("tool_uses"):
            expanded_parts.append(f"Tools: {', '.join(stored['tool_uses'])}")

        expanded_parts.append(f"({stored['turn_count']} turns summarized)")

        result = {**summary_message}
        result["content"] = " | ".join(expanded_parts)
        result["_summary_level"] = 2

        self._stats["recall_tokens_created"] += 1
        return result

    def get_summary(self, summary_id: str) -> dict[str, Any] | None:
        """Get a stored summary by ID."""
        return self._summaries.get(summary_id)

    def get_stats(self) -> dict[str, int]:
        """Get summarization statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all stored summaries."""
        self._summaries.clear()
        self._stats = {
            "turns_summarized": 0,
            "turns_compressed": 0,
            "recall_tokens_created": 0,
        }

    def save_to_disk(self) -> None:
        """Persist summaries to disk."""
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: v for k, v in self._summaries.items()
            }
            _PERSISTENCE_PATH.write_text(json.dumps(data))
        except Exception as e:
            logger.warning("[HierarchicalSummary] Failed to save: %s", e)

    def load_from_disk(self) -> None:
        """Load summaries from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(_PERSISTENCE_PATH.read_text())
            self._summaries = OrderedDict(data)
        except Exception as e:
            logger.warning("[HierarchicalSummary] Failed to load: %s", e)


# Global instance
_hierarchical_summarizer: HierarchicalSummarizer | None = None


def get_hierarchical_summarizer(max_full_turns: int = 5) -> HierarchicalSummarizer:
    """Get or create the global hierarchical summarizer."""
    global _hierarchical_summarizer
    if _hierarchical_summarizer is None:
        _hierarchical_summarizer = HierarchicalSummarizer(max_full_turns=max_full_turns)
        _hierarchical_summarizer.load_from_disk()
    return _hierarchical_summarizer
