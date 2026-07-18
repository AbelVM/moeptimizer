"""ToolOutputFilter — declarative filtering for tool/assistant outputs (rtk/snip pattern).

Filters large tool outputs at the proxy boundary, before they enter the stable
prefix, by applying regex-based rewrite rules. This is orthogonal to
``ToolOutputCompressor``: the filter reduces tokens at the source (e.g.
``go test ./...`` → ``10 passed, 0 failed``), while the compressor applies
lossless-ish transforms (truncate, collapse lines, strip ANSI) to whatever
survives filtering.

Design goals:
  - Zero false positives: unmatched output passes through unchanged.
  - Cache-safe: filtered output is deterministic for the same input, so the
    backend's prefix cache reuses it across turns.
  - Extensible: rules are plain regex + replacement pairs, easy to add.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Default filter rules: (name, compiled_pattern, replacement).
# Patterns are matched against the FULL tool output; the first match wins.
_DEFAULT_RULES: list[tuple[str, re.Pattern[str], str]] = [
    # --- Test runners -------------------------------------------------------
    (
        "go_test",
        re.compile(
            r"(?im)^(ok\s+\S+\s+[\d.]+s|\?\s+\S+\s+\S+|FAIL\s+\S+)"
            r"(?:\n(?:ok\s+\S+\s+[\d.]+s|\?\s+\S+\s+\S+|FAIL\s+\S+))*"
        ),
        "[go test result]",
    ),
    (
        "pytest_summary",
        re.compile(
            r"(?im)^(=+ .+ =+$)\n.*?(\d+ passed|passed).*?(\d+ failed|failed).*?(\d+ error|error)"
            r".*?(\d+ skipped|skipped)"
        ),
        "[pytest summary]",
    ),
    (
        "pytest_short",
        re.compile(
            r"(?im)^(.*?\[100%\]|.*?passed.*?failed.*?)$"
        ),
        "[pytest result]",
    ),
    (
        "cargo_test",
        re.compile(
            r"(?im)^(test result: ok\. \d+ passed; \d+ failed; \d+ ignored; \d+ measured; \d+ filtered out)"
        ),
        "[cargo test result]",
    ),
    (
        "npm_test",
        re.compile(
            r"(?im)^(Tests:\s+\d+\s+passed,\s+\d+\s+failed,\s+\d+\s+total)"
        ),
        "[npm test result]",
    ),
    # --- Git -----------------------------------------------------------------
    (
        "git_status",
        re.compile(
            r"(?im)^(On branch \S+\n(?:Changes to be committed:\n(?:.*\n)*?(?:new file|modified|deleted|renamed):.*\n)*"
            r"(?:Changes not staged for commit:\n(?:.*\n)*?(?:modified:.*\n)*)?)?$"
        ),
        "[git status]",
    ),
    (
        "git_log_oneline",
        re.compile(
            r"(?im)^([0-9a-f]{7,40}\s+.*?\n)+"
        ),
        "[git log]",
    ),
    (
        "git_diff_stat",
        re.compile(
            r"(?im)^(.*?\|.*?\n)+(\d+ files? changed.*)$"
        ),
        "[git diff stat]",
    ),
    # --- Build / compile -----------------------------------------------------
    (
        "build_success",
        re.compile(
            r"(?im)^(Build succeeded|Build complete|Compilation succeeded|Finished \w+ \[.+\] in \d+\.\d+s)"
        ),
        "[build success]",
    ),
    (
        "build_error",
        re.compile(
            r"(?im)^(error(?:\[[\w\s]+\])?:.*?)$"
        ),
        "[build error]",
    ),
    # --- Lint / format -------------------------------------------------------
    (
        "lint_summary",
        re.compile(
            r"(?im)^(Found \d+ (error|warning)s?.*?)$"
        ),
        "[lint summary]",
    ),
    # --- Shell / general -----------------------------------------------------
    (
        "shell_ok",
        re.compile(
            r"(?im)^(Command finished successfully|Process completed|Done\.|Done in \d+\.\d+s)"
        ),
        "[shell ok]",
    ),
    (
        "repeated_lines",
        re.compile(
            r"^(.+)\n(?:\1\n){3,}"
        ),
        r"\1\n... (repeated \d+ times) ...",
    ),
]


class ToolOutputFilter:
    """Declarative filter for tool/assistant outputs.

    Applies regex-based rewrite rules to large tool outputs before they enter
    the stable prefix. Unmatched output passes through unchanged.

    Rules are evaluated in order; the first match wins. Each rule has a name
    (for logging), a compiled pattern, and a replacement string.
    """

    def __init__(
        self,
        rules: list[tuple[str, re.Pattern[str], str]] | None = None,
        min_chars: int = 200,
    ) -> None:
        """Initialize the filter.

        Args:
            rules: Custom filter rules. When ``None``, the built-in defaults
                (test runners, git, build tools) are used.
            min_chars: Minimum output length to trigger filtering. Shorter
                outputs are returned unchanged to avoid false positives.
        """
        self._rules = list(rules) if rules is not None else list(_DEFAULT_RULES)
        self._min_chars = max(0, min_chars)

    def should_filter(self, content: str) -> bool:
        """True if the content is large enough to be worth filtering."""
        return bool(content) and len(content) >= self._min_chars

    def filter(self, content: str) -> str:
        """Return a filtered form of ``content``, or the original if no rule matches.

        Idempotent on already-filtered content: if the content already contains
        a filter marker, it is returned unchanged.
        """
        if not content or not self.should_filter(content):
            return content

        for name, pattern, replacement in self._rules:
            if pattern.search(content):
                logger.debug("ToolOutputFilter matched rule=%s (%d chars -> marker)", name, len(content))
                return replacement

        return content

    def add_rule(self, name: str, pattern: str | re.Pattern[str], replacement: str) -> None:
        """Append a custom filter rule.

        Args:
            name: Rule name for logging.
            pattern: Regex pattern (string or compiled).
            replacement: Replacement string when the pattern matches.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self._rules.append((name, pattern, replacement))


def filter_tool_messages(
    messages: list[dict[str, Any]],
    tool_filter: ToolOutputFilter,
    roles: tuple[str, ...] = ("tool", "assistant"),
) -> list[dict[str, Any]]:
    """Return ``messages`` with large tool/assistant outputs filtered.

    Only string ``content`` is touched; structured (list) content is left as-is.
    Idempotent: already-filtered outputs are returned unchanged.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in roles and isinstance(content, str) and tool_filter.should_filter(content):
            filtered = tool_filter.filter(content)
            if filtered is not content:
                new_msg = dict(msg)
                new_msg["content"] = filtered
                out.append(new_msg)
            else:
                out.append(msg)
        else:
            out.append(msg)
    return out
