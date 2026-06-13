"""Tests for AgentStateStore and StateBasedRAG."""



from moeptimizer.models import AgentStep
from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore


class TestAgentStateStore:
    def test_add_step_indexes_by_tool(self) -> None:
        store = AgentStateStore()
        step = AgentStep(role="assistant", tool_name="search")
        store.add_step(step)
        assert step.step_id in store.tool_index["search"]

    def test_add_step_indexes_by_subtask(self) -> None:
        store = AgentStateStore()
        step = AgentStep(
            role="assistant",
            content="# Subtask: auth implementation",
        )
        store.add_step(step)
        assert "auth implementation" in store.subtask_index

    def test_add_step_inferred_subtask_from_tool(self) -> None:
        store = AgentStateStore()
        step = AgentStep(role="assistant", tool_name="file_read")
        store.add_step(step)
        assert "file_read" in store.subtask_index

    def test_set_goal(self) -> None:
        store = AgentStateStore()
        goal_id = store.set_goal("Build a REST API")
        goal = store.get_goal()
        assert goal is not None
        assert goal.goal_id == goal_id
        assert goal.original_prompt == "Build a REST API"

    def test_get_recent_steps(self) -> None:
        store = AgentStateStore()
        for i in range(5):
            store.add_step(AgentStep(role="assistant", content=f"step {i}"))
        recent = store.get_recent_steps(3)
        assert len(recent) == 3
        assert recent[0].content == "step 2"
        assert recent[2].content == "step 4"

    def test_get_archived_steps(self) -> None:
        store = AgentStateStore()
        for i in range(6):
            store.add_step(AgentStep(role="assistant", content=f"step {i}"))
        archived = store.get_archived_steps()
        assert len(archived) == 3  # 6 - archive_threshold(3) = 3
        assert archived[0].content == "step 0"

    def test_get_related_context_same_subtask(self) -> None:
        store = AgentStateStore()
        store.add_step(AgentStep(role="assistant", content="# Subtask: auth", metadata={"subtask": "auth"}))
        store.add_step(AgentStep(role="assistant", content="# Subtask: auth", metadata={"subtask": "auth"}))
        store.add_step(AgentStep(role="assistant", content="# Subtask: other", metadata={"subtask": "other"}))

        current = AgentStep(role="assistant", metadata={"subtask": "auth"})
        related = store.get_related_context(current)
        assert len(related) == 2

    def test_get_related_context_same_tool(self) -> None:
        store = AgentStateStore()
        store.add_step(AgentStep(role="assistant", tool_name="search"))
        store.add_step(AgentStep(role="assistant", tool_name="search"))
        store.add_step(AgentStep(role="assistant", tool_name="write"))

        current = AgentStep(role="assistant", tool_name="search")
        related = store.get_related_context(current)
        assert len(related) == 2

    def test_compacted_history(self) -> None:
        store = AgentStateStore()
        for i in range(5):
            store.add_step(AgentStep(role="assistant", content=f"step {i}"))
        history = store.get_compacted_history()
        # First 2 archived, last 3 recent
        archived = [h for h in history if h.get("archived")]
        recent = [h for h in history if not h.get("archived")]
        assert len(archived) == 2
        assert len(recent) == 3

    def test_serialize_deserialize(self) -> None:
        store = AgentStateStore()
        store.set_goal("Test goal")
        store.add_step(AgentStep(role="assistant", content="hello", tool_name="test"))
        serialized = store.serialize()
        restored = AgentStateStore.deserialize(serialized)
        assert restored.get_goal().original_prompt == "Test goal"
        assert len(restored.steps) == 1
        assert restored.steps[0].tool_name == "test"

    def test_generate_summary_tool(self) -> None:
        store = AgentStateStore()
        step = AgentStep(role="tool", tool_name="file_write", content="line1\nline2\nline3\nline4\nline5\nline6")
        summary = store._generate_summary(step)
        assert "returned 6 lines" in summary

    def test_generate_summary_assistant_with_tool(self) -> None:
        store = AgentStateStore()
        step = AgentStep(role="assistant", tool_name="search", content="some content")
        summary = store._generate_summary(step)
        assert "called tool 'search'" in summary


class TestStateBasedRAG:
    def test_get_context_for_step(self) -> None:
        store = AgentStateStore()
        store.add_step(AgentStep(role="assistant", content="step 1", metadata={"subtask": "auth"}))
        store.add_step(AgentStep(role="assistant", content="step 2", metadata={"subtask": "auth"}))
        store.add_step(AgentStep(role="assistant", content="step 3", metadata={"subtask": "other"}))

        rag = StateBasedRAG(store)
        current = AgentStep(role="assistant", metadata={"subtask": "auth"})
        context = rag.get_context_for_step(current)
        # Model-friendly format: "step N: role - content"
        assert "step 0: assistant - step 1" in context or "step 1: assistant - step 2" in context
        # Should not include "other" subtask step
        assert "step 3" not in context

    def test_get_context_empty(self) -> None:
        store = AgentStateStore()
        rag = StateBasedRAG(store)
        current = AgentStep(role="assistant")
        context = rag.get_context_for_step(current)
        assert context == ""
