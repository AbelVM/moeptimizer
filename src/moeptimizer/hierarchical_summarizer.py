"""Hierarchical Summarization for long conversation context.

Summarizes older turns into a single "recall" token that can be expanded
on demand, keeping context lean while preserving key information.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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

    # Keywords that mark a constraint / "don't" the model must keep in context.
    # Retaining these in the rolling summary is what stops the 2.17x verbosity
    # regression: when the proxy drops them, the model re-derives them verbosely.
    _CONSTRAINT_HINTS = (
        "don't", "do not", "doesn't", "does not", "dont", "dont",
        "must not", "mustn't", "should not", "shouldn't",
        "cannot", "can't", "can not", "won't", "will not",
        "avoid", "never", "no longer", "not allowed", "prohibited",
        "forbidden", "refrain", "instead of", "without", "only",
        "make sure", "ensure", "keep", "preserve", "don't change",
        "do not change", "don't modify", "do not modify", "unchanged",
    )

    def __init__(
        self,
        max_full_turns: int = 5,
        max_summary_turns: int = 15,
        max_rolling_summary_chars: int = 4000,
    ) -> None:
        self._max_full_turns = max_full_turns
        self._max_summary_turns = max_summary_turns
        # Cap the append-only rolling summary so a very long session does not
        # grow it without bound (review §8.5). When exceeded, the OLDEST lines
        # are dropped — the most-recent task state is what the model needs.
        self._max_rolling_summary_chars = max(256, int(max_rolling_summary_chars))
        self._summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stats: dict[str, int] = {
            "turns_summarized": 0,
            "turns_compressed": 0,
            "recall_tokens_created": 0,
        }
        self._last_context_changed = False
        # Cache-stable rolling-summary state (review §1/§3/§5, #7). The rolling
        # summary block only ever grows by appending, so its leading bytes stay
        # byte-identical across turns and the backend reuses the prefix cache.
        self._rolling_summary_text: str = ""
        self._rolling_summary_id: str = ""
        self._summarized_turn_count: int = 0

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

    def summarize_turns_cache_stable(
        self,
        messages: list[dict[str, Any]],
        frozen_prefix_end: int,
    ) -> list[dict[str, Any]]:
        """Cache-stable tiered rolling-summary compaction (review §1/§3/§5, #7).

        Folds older dynamic turns into a single append-only rolling summary
        block placed immediately after the frozen prefix. The block retains
        constraints (the task's "don'ts") and key decisions so the model does
        not re-derive them verbosely (the 2.17x verbosity regression). Because
        the block only ever grows by appending, its leading bytes stay
        byte-identical across turns, so the backend's prefix cache reuses the
        frozen prefix + summary head instead of re-prefilling.

        Args:
            messages: Full optimized message list.
            frozen_prefix_end: Index just past the stable prefix block
                (system + first user + frozen early turns).

        Returns:
            Message list with older dynamic turns replaced by the rolling
            summary block, or ``messages`` unchanged when there is nothing to
            summarize.
        """
        if frozen_prefix_end < 0 or frozen_prefix_end > len(messages):
            return messages

        frozen = messages[:frozen_prefix_end]
        rest = messages[frozen_prefix_end:]
        if len(rest) <= self._max_full_turns:
            # Nothing old enough to summarize; reset the rolling counter so a
            # later long context starts fresh.
            self._summarized_turn_count = 0
            return messages

        # Group the dynamic layer into user-led turns.
        turns = self._group_turns(rest)
        total_turns = len(turns)
        keep = self._max_full_turns
        if total_turns <= keep:
            self._summarized_turn_count = 0
            return messages

        # Turns already folded into the rolling summary (append-only, stable).
        end = total_turns - keep
        start = min(self._summarized_turn_count, end)
        new_turns = turns[start:end]

        if new_turns:
            new_text = self._extract_constraints(new_turns)
            if new_text:
                self._rolling_summary_text = (
                    f"{self._rolling_summary_text}\n{new_text}"
                    if self._rolling_summary_text
                    else new_text
                )
                # Cap growth: drop oldest lines when over budget (review §8.5).
                if len(self._rolling_summary_text) > self._max_rolling_summary_chars:
                    lines = self._rolling_summary_text.split("\n")
                    while (
                        lines
                        and len("\n".join(lines)) > self._max_rolling_summary_chars
                    ):
                        lines.pop(0)
                    self._rolling_summary_text = "\n".join(lines)
                self._stats["turns_summarized"] += sum(len(t) for t in new_turns)
                self._stats["turns_compressed"] += 1
            self._summarized_turn_count = end

        keep_recent = [m for t in turns[end:] for m in t]
        # Append the rolling summary as a TRAILING turn (review §1/§9, priority
        # fix #1): inserting it right after the frozen prefix shifts every later
        # turn's token offset and invalidates the backend's prefix cache. As a
        # trailing turn it never alters the stable leading prefix, so the backend
        # reuses its cached KV for the frozen + recent turns.
        return [*frozen, *keep_recent, self._build_rolling_summary_block()]

    def _build_rolling_summary_block(self) -> dict[str, Any]:
        """Return the single rolling-summary message (append-only content)."""
        if not self._rolling_summary_id:
            self._rolling_summary_id = hashlib.md5(
                b"rolling-summary"
            ).hexdigest()[:16]
        text = self._rolling_summary_text or "Earlier context summarized."
        return {
            "role": "user",
            "content": f"Context summary (rolling):\n{text}",
            "_summary_id": self._rolling_summary_id,
            "_summary_level": 1,
            "_rolling_summary": True,
        }

    @staticmethod
    def _group_turns(
        messages: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group a message list into user-led turns (user + following asst/tool)."""
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                if current:
                    turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns

    # Patterns that identify a file path the agent is working on. Capturing
    # these is what lets the rolling summary retain *which file* was edited when
    # the full turn is evicted (review P0.3).
    _FILE_PATH_RE = re.compile(
        r"""(?:\b(?:file|path|module|read|write|edit|open|import|from)\b[^\n]{0,80}?|
            (?:["'`])([./]?[\w./\-/]+\.\w{1,6})\1)""",
        re.VERBOSE,
    )
    # A line that looks like a runtime/test error the model must keep in mind.
    _ERROR_RE = re.compile(
        r"""(?ix)
        (?:error|exception|traceback|failed|failure|assert|raise[d]?||
           \b\d{3}\b\s*(?:error|forbidden|not\s*found)||
           module\s+not\s+found|no\s+such\s+file|syntax\s+error|type\s+error)
        """,
    )

    def _extract_constraints(
        self,
        turns: list[list[dict[str, Any]]],
    ) -> str:
        """Extract a task-STATE summary from evicted turns (review P0.3).

        The old version only kept lines containing "don't"/"must not"/"avoid"
        keywords, which dropped the actual bug, code, and decisions — causing the
        quality collapse (proxy emitted no code, semantic_similarity ~0.25). The
        new version retains the *state* the model needs to keep working:

        - file paths touched (so the model knows what it was editing),
        - the last error / failure message (so it knows what to fix),
        - the current plan / goal (user requests + assistant "I will" statements),
        - explicit constraints ("don't"/"must not"/"avoid") as before.

        Falls back to a short topic line so the block is never empty.
        """
        constraints: list[str] = []
        plans: list[str] = []
        errors: list[str] = []
        files: list[str] = []
        topics: list[str] = []
        seen_files: set[str] = set()

        for turn in turns:
            for msg in turn:
                content = msg.get("content", "")
                if not isinstance(content, str) or not content:
                    continue
                role = msg.get("role", "")
                # File paths (from any role).
                for m in self._FILE_PATH_RE.finditer(content):
                    fp = m.group(1)
                    if fp and fp not in seen_files and len(fp) < 120:
                        seen_files.add(fp)
                        files.append(fp)
                # Errors (from tool/assistant output primarily).
                if role in ("tool", "assistant"):
                    for raw_line in content.splitlines():
                        line = raw_line.strip()
                        if 8 < len(line) < 240 and self._ERROR_RE.search(line):
                            errors.append(line)
                for raw_line in content.splitlines():
                    line = raw_line.strip()
                    low_line = line.lower()
                    if 12 < len(line) < 200 and any(
                        hint in low_line for hint in self._CONSTRAINT_HINTS
                    ):
                        constraints.append(line)
                if role == "user":
                    sentences = re.split(r"[.?!]", content)
                    for sent in sentences:
                        sent = sent.strip()
                        if 12 < len(sent) < 160:
                            topics.append(sent)
                            break
                if role == "assistant":
                    # Capture plan-like statements ("I will", "Next,", "Let's").
                    for raw_line in content.splitlines():
                        line = raw_line.strip()
                        if 12 < len(line) < 200 and re.match(
                            r"(?i)(i\s+will|next,|let's|now\s+we|the\s+plan|step\s*\d)", line
                        ):
                            plans.append(line)

        parts: list[str] = []
        if files:
            uniq_files = list(dict.fromkeys(files))[:10]
            parts.append("Files touched: " + ", ".join(uniq_files))
        if errors:
            seen_e: set[str] = set()
            uniq_e: list[str] = []
            for e in errors:
                if e not in seen_e:
                    seen_e.add(e)
                    uniq_e.append(e)
            parts.append("Last errors: " + " | ".join(uniq_e[:3]))
        if plans:
            seen_p: set[str] = set()
            uniq_p: list[str] = []
            for p in plans:
                if p not in seen_p:
                    seen_p.add(p)
                    uniq_p.append(p)
            parts.append("Plan: " + " ".join(uniq_p[:3]))
        if constraints:
            seen_c: set[str] = set()
            uniq_c: list[str] = []
            for c in constraints:
                if c not in seen_c:
                    seen_c.add(c)
                    uniq_c.append(c)
            parts.append("Constraints: " + "; ".join(uniq_c[:6]))
        if topics:
            parts.append("Topic: " + "; ".join(topics[:2]))
        return "\n".join(parts)

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
        self._last_context_changed = True
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
        self._last_context_changed = True
        self._rolling_summary_text = ""
        self._rolling_summary_id = ""
        self._summarized_turn_count = 0

    def save_to_disk(self, force: bool = False) -> None:
        """Persist summaries to disk."""
        if not force and not self._last_context_changed:
            return
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: v for k, v in self._summaries.items()
            }
            _PERSISTENCE_PATH.write_text(json.dumps(data))
            self._last_context_changed = False
        except Exception as e:
            logger.warning("[HierarchicalSummary] Failed to save: %s", e)

    def load_from_disk(self) -> None:
        """Load summaries from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(_PERSISTENCE_PATH.read_text())
            self._summaries = OrderedDict(data)
            while len(self._summaries) > 100:
                self._summaries.popitem(last=False)
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
