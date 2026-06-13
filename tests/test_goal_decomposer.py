"""Tests for GoalDecomposer."""


from moeptimizer.goal_decomposer import GoalDecomposer


class TestGoalDecomposer:
    def setup_method(self) -> None:
        self.decomposer = GoalDecomposer()

    def test_web_task_decomposition(self) -> None:
        subtasks = self.decomposer.decompose("Build a React frontend with CSS")
        assert len(subtasks) >= 2
        assert any("HTML" in s or "CSS" in s for s in subtasks)

    def test_api_task_decomposition(self) -> None:
        subtasks = self.decomposer.decompose("Create a REST API with auth")
        assert len(subtasks) >= 2
        assert any("route" in s.lower() or "schema" in s.lower() for s in subtasks)

    def test_bugfix_task_decomposition(self) -> None:
        subtasks = self.decomposer.decompose("Fix the login bug")
        assert len(subtasks) >= 2
        assert any("root cause" in s.lower() or "reproduce" in s.lower() for s in subtasks)

    def test_refactor_task_decomposition(self) -> None:
        subtasks = self.decomposer.decompose("Refactor the auth module")
        assert len(subtasks) >= 2
        assert any("structure" in s.lower() or "dependencies" in s.lower() for s in subtasks)

    def test_generic_decomposition(self) -> None:
        subtasks = self.decomposer.decompose("Do something complex")
        assert len(subtasks) >= 2
        assert any("Understand" in s or "Plan" in s for s in subtasks)

    def test_multi_sentence_generic(self) -> None:
        subtasks = self.decomposer.decompose("Build the frontend. Add the backend. Write tests.")
        assert len(subtasks) >= 2
