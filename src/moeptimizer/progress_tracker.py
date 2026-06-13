"""ProgressTracker — Track goal completion."""

from __future__ import annotations

import re
import time
from typing import ClassVar

from moeptimizer.models import AgentStep, ProgressSnapshot


class ProgressTracker:
    """
    Tracks the agent's progress toward its goal.

    Uses heuristics based on:
      - Subtask completion signals in messages
      - Tool usage patterns
      - Goal decomposition
    """

    COMPLETION_SIGNALS: ClassVar[list[str]] = [
        "completed", "done", "finished", "implemented", "resolved",
        "fixed", "deployed", "merged", "submitted", "pushed",
    ]

    START_SIGNALS: ClassVar[list[str]] = [
        "starting", "beginning", "now implementing", "now fixing",
        "now updating", "now creating", "starting to",
    ]

    def __init__(
        self,
        goal: str = "",
        subtasks: list[str] | None = None,
    ) -> None:
        self.goal = goal
        self.subtasks: list[str] = subtasks or []
        self._tracked_subtasks: dict[str, str] = {}
        self._tools_used: set[str] = set()
        self._step_count = 0

    def record_step(self, step: AgentStep) -> None:
        """Record a step and update progress."""
        self._step_count += 1
        if step.tool_name:
            self._tools_used.add(step.tool_name)

        content = step.content.lower()

        for signal in self.COMPLETION_SIGNALS:
            if signal in content:
                for st, status in self._tracked_subtasks.items():
                    if status in ("active", "pending"):
                        self._tracked_subtasks[st] = "completed"
                        break
                break

        for signal in self.START_SIGNALS:
            if signal in content:
                match = re.search(
                    r"(?:starting|beginning|now)\s+(?:to\s+)?(\w+)",
                    content,
                    re.IGNORECASE,
                )
                if match:
                    subtask = match.group(1)
                    if subtask not in self._tracked_subtasks:
                        self._tracked_subtasks[subtask] = "active"
                break

    def get_progress(self) -> ProgressSnapshot:
        """Get current progress snapshot."""
        total = max(len(self._tracked_subtasks), len(self.subtasks), 1)
        completed = sum(
            1 for s in self._tracked_subtasks.values() if s == "completed"
        )

        completion = completed / total if total > 0 else 0.0

        if not self._tracked_subtasks:
            completion = min(1.0, self._step_count / 20.0)

        is_complete = completion >= 0.9 or (
            len(self._tracked_subtasks) > 0
            and all(s == "completed" for s in self._tracked_subtasks.values())
        )

        active = [
            st for st, status in self._tracked_subtasks.items()
            if status == "active"
        ]

        return ProgressSnapshot(
            total_steps=self._step_count,
            completed_subtasks=[
                st for st, s in self._tracked_subtasks.items() if s == "completed"
            ],
            active_subtasks=active,
            tools_used=self._tools_used,
            estimated_completion=completion,
            is_complete=is_complete,
            last_update=time.time(),
        )

    def set_subtasks(self, subtasks: list[str]) -> None:
        """Manually set subtasks for tracking."""
        self.subtasks = subtasks
        for st in subtasks:
            if st not in self._tracked_subtasks:
                self._tracked_subtasks[st] = "pending"
