"""GoalDecomposer — Break goals into subtasks."""

from __future__ import annotations

import re


class GoalDecomposer:
    """
    Decomposes a high-level goal into actionable subtasks.

    Uses pattern matching on the goal text to identify common task types
    and generate appropriate subtasks.
    """

    def decompose(self, goal: str) -> list[str]:
        """Decompose a goal into subtasks."""
        goal_lower = goal.lower()

        if self._is_web_task(goal_lower):
            return self._decompose_web_task()
        elif self._is_api_task(goal_lower):
            return self._decompose_api_task()
        elif self._is_bugfix_task(goal_lower):
            return self._decompose_bugfix_task()
        elif self._is_refactor_task(goal_lower):
            return self._decompose_refactor_task()
        else:
            return self._generic_decompose(goal)

    def _is_web_task(self, goal: str) -> bool:
        return any(kw in goal for kw in (
            "website", "web app", "frontend", "ui", "html", "css",
            "react", "vue", "angular",
        ))

    def _is_api_task(self, goal: str) -> bool:
        return any(kw in goal for kw in (
            "api", "endpoint", "route", "rest", "graphql", "server", "backend",
        ))

    def _is_bugfix_task(self, goal: str) -> bool:
        return any(kw in goal for kw in (
            "fix", "bug", "error", "crash", "issue", "problem", "broken",
        ))

    def _is_refactor_task(self, goal: str) -> bool:
        return any(kw in goal for kw in (
            "refactor", "restructure", "reorganize", "cleanup", "optimize",
        ))

    def _decompose_web_task(self) -> list[str]:
        return [
            "Design page structure and layout",
            "Implement HTML/CSS markup",
            "Add JavaScript interactivity",
            "Test responsiveness and cross-browser compatibility",
        ]

    def _decompose_api_task(self) -> list[str]:
        return [
            "Define API schema and data models",
            "Implement route handlers",
            "Add input validation and error handling",
            "Write integration tests",
        ]

    def _decompose_bugfix_task(self) -> list[str]:
        return [
            "Reproduce the issue and identify root cause",
            "Implement fix in relevant module",
            "Add test case for the bug",
            "Verify fix does not break existing functionality",
        ]

    def _decompose_refactor_task(self) -> list[str]:
        return [
            "Map current code structure and dependencies",
            "Design new module organization",
            "Implement refactoring changes",
            "Run tests to verify behavior is preserved",
        ]

    def _generic_decompose(self, goal: str) -> list[str]:
        """Generic decomposition for unrecognized task types."""
        sentences = re.split(r"[.\n]", goal)
        if len(sentences) > 1:
            return [s.strip() + "." for s in sentences if s.strip()]

        return [
            "Understand the current codebase and requirements",
            "Plan the implementation approach",
            "Implement the changes",
            "Verify the changes work correctly",
        ]
