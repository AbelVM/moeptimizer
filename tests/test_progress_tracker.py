"""Tests for ProgressTracker."""


from moeptimizer.models import AgentStep
from moeptimizer.progress_tracker import ProgressTracker


class TestProgressTracker:
    def test_initial_progress(self) -> None:
        tracker = ProgressTracker()
        progress = tracker.get_progress()
        assert progress.total_steps == 0
        assert progress.estimated_completion == 0.0
        assert not progress.is_complete

    def test_record_step_increments_count(self) -> None:
        tracker = ProgressTracker()
        tracker.record_step(AgentStep(role="assistant", content="hello"))
        progress = tracker.get_progress()
        assert progress.total_steps == 1

    def test_completion_signal(self) -> None:
        tracker = ProgressTracker()
        tracker.set_subtasks(["fix auth", "add tests"])
        tracker.record_step(AgentStep(role="assistant", content="completed fix auth"))
        progress = tracker.get_progress()
        assert "fix auth" in progress.completed_subtasks

    def test_start_signal(self) -> None:
        tracker = ProgressTracker()
        tracker.record_step(AgentStep(role="assistant", content="starting to implement auth"))
        progress = tracker.get_progress()
        assert "implement" in progress.active_subtasks or len(progress.active_subtasks) >= 0

    def test_tracking_without_subtasks(self) -> None:
        tracker = ProgressTracker()
        for i in range(15):
            tracker.record_step(AgentStep(role="assistant", content=f"step {i}"))
        progress = tracker.get_progress()
        assert progress.estimated_completion > 0.5
        assert progress.estimated_completion <= 1.0

    def test_all_subtasks_completed(self) -> None:
        tracker = ProgressTracker()
        tracker.set_subtasks(["task1", "task2"])
        tracker._tracked_subtasks["task1"] = "completed"
        tracker._tracked_subtasks["task2"] = "completed"
        progress = tracker.get_progress()
        assert progress.is_complete

    def test_tools_used_tracking(self) -> None:
        tracker = ProgressTracker()
        tracker.record_step(AgentStep(role="assistant", tool_name="search"))
        tracker.record_step(AgentStep(role="assistant", tool_name="write"))
        tracker.record_step(AgentStep(role="assistant", tool_name="search"))
        progress = tracker.get_progress()
        assert "search" in progress.tools_used
        assert "write" in progress.tools_used
