"""Tests for progress tracker."""

import pytest

from moeptimizer.progress_tracker import (
    ProgressTracker,
)
from moeptimizer.models import AgentStep


class TestProgressTracker:
    def test_empty_tracker(self) -> None:
        """Empty tracker has no steps."""
        tracker = ProgressTracker()
        assert tracker is not None

    def test_record_step(self) -> None:
        """Record a step."""
        tracker = ProgressTracker()
        step = AgentStep(role="user", content="Test")
        tracker.record_step(step)
        progress = tracker.get_progress()
        assert progress.total_steps >= 1

    def test_set_subtasks(self) -> None:
        """Set subtasks."""
        tracker = ProgressTracker()
        subtasks = ["Task 1", "Task 2"]
        tracker.set_subtasks(subtasks)
        # Check that subtasks were set
        assert len(tracker.subtasks) == 2
        assert "Task 1" in tracker.subtasks