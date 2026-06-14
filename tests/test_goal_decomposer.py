"""Tests for goal decomposer."""

import pytest

from moeptimizer.goal_decomposer import (
    GoalDecomposer,
)


class TestGoalDecomposer:
    def test_empty_decomposer(self) -> None:
        """Empty decomposer has no state."""
        decomposer = GoalDecomposer()
        assert decomposer is not None

    def test_decompose(self) -> None:
        """Decompose a goal into subtasks."""
        decomposer = GoalDecomposer()
        subtasks = decomposer.decompose("Write a web application")
        assert isinstance(subtasks, list)

    def test_decompose_empty(self) -> None:
        """Decompose empty goal."""
        decomposer = GoalDecomposer()
        subtasks = decomposer.decompose("")
        assert isinstance(subtasks, list)