"""LoopDetector — Detect repeated tool calls / actions."""

from __future__ import annotations

import re
from collections import defaultdict, deque

from moeptimizer.models import AgentStep, LoopWarning


class LoopDetector:
    """
    Detects when an agent is stuck in repetitive patterns.

    Monitors:
      - Same tool called N times consecutively (tool_repeat)
      - Same action taken N times (action_repeat)
      - Repeated thinking without progress (thinking_loop)
    """

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._recent_tools: deque[str] = deque(maxlen=threshold + 1)
        self._recent_actions: deque[str] = deque(maxlen=threshold + 1)
        self._thinking_count = 0
        self._recent_warnings: list[LoopWarning] = []

    def analyze_step(self, step: AgentStep) -> LoopWarning | None:
        """Analyze a single step for loop patterns."""
        warning: LoopWarning | None = None

        if step.tool_name:
            self._recent_tools.append(step.tool_name)
            if len(self._recent_tools) >= self.threshold:
                tool_counts: dict[str, int] = defaultdict(int)
                for t in self._recent_tools:
                    tool_counts[t] += 1
                for tool, count in tool_counts.items():
                    if count >= self.threshold:
                        warning = LoopWarning(
                            loop_type="tool_repeat",
                            tool_name=tool,
                            repeat_count=count,
                            message=(
                                f"Agent called tool '{tool}' {count} times in a row. "
                                "Consider changing strategy."
                            ),
                        )
                        break

        if step.role == "assistant":
            action = self._extract_action_signature(step.content)
            if action:
                self._recent_actions.append(action)
                if len(self._recent_actions) >= self.threshold:
                    action_counts: dict[str, int] = defaultdict(int)
                    for a in self._recent_actions:
                        action_counts[a] += 1
                    for action_sig, count in action_counts.items():
                        if count >= self.threshold:
                            warning = LoopWarning(
                                loop_type="action_repeat",
                                repeat_count=count,
                                message=(
                                    f"Agent repeated action '{action_sig}' {count} times. "
                                    "Context drift detected."
                                ),
                            )
                            break

        if step.role == "thinking":
            self._thinking_count += 1
            if (
                self._thinking_count >= self.threshold
                and (
                    len(self._recent_actions) == 0
                    or self._recent_actions[-1] == self._recent_actions[0]
                )
            ):
                warning = LoopWarning(
                    loop_type="thinking_loop",
                    repeat_count=self._thinking_count,
                    message=(
                        f"Agent has {self._thinking_count} thinking steps without "
                        "new actions. Breaking loop."
                    ),
                )
        elif step.role == "assistant":
            self._thinking_count = 0

        if warning:
            self._recent_warnings.append(warning)

        return warning

    def _extract_action_signature(self, content: str) -> str | None:
        """Extract a normalized action signature from assistant content."""
        patterns = [
            (r"(?:tool_calls|function_call|call)\s*[:=]\s*(\w+)", "call:"),
            (r"call\s+(\w+)\s*\(", "call:"),
            (r"invoke\s+(\w+)\s*\(", "call:"),
            (r"execute\s+(\w+)\s*\(", "call:"),
            (r"run\s+(\w+)\s*\(", "call:"),
        ]
        for pattern, prefix in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return f"{prefix}{match.group(1)}"

        code_changes = re.findall(
            r"(?:create|modify|update|delete|add|remove)\s+(\w+)",
            content,
            re.IGNORECASE,
        )
        if code_changes:
            return f"code:{code_changes[0]}"

        search_patterns = [
            r"(?:search|find|query|lookup)\s+(\w+)",
        ]
        for pattern in search_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return f"search:{match.group(1)}"

        return None

    def get_warning_message(self, warning: LoopWarning) -> str:
        """Get a user-facing warning message."""
        return f"[LOOP DETECTED: {warning.loop_type}] {warning.message}"

    def get_recent_warnings(self, n: int = 5) -> list[LoopWarning]:
        """Get recent loop warnings."""
        return self._recent_warnings[-n:]

    def reset_warnings(self) -> None:
        """Clear recent warnings."""
        self._recent_warnings = []
