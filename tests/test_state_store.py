"""Tests for state store."""


from moeptimizer.models import AgentStep
from moeptimizer.state_store import (
    AgentStateStore,
)


class TestAgentStateStore:
    def test_empty_store(self) -> None:
        """Empty store has no steps."""
        store = AgentStateStore()
        assert len(store.steps) == 0

    def test_add_step(self) -> None:
        """Add a step to the store."""
        store = AgentStateStore()
        step = AgentStep(role="user", content="Test")
        store.add_step(step)
        assert len(store.steps) == 1

    def test_get_goal(self) -> None:
        """Get goal from store."""
        store = AgentStateStore()
        goal = store.get_goal()
        assert goal is None

    def test_set_goal(self) -> None:
        """Set goal in store."""
        store = AgentStateStore()
        store.set_goal("Test goal")
        goal = store.get_goal()
        assert goal is not None
        assert goal.original_prompt == "Test goal"

    def test_serialize_deserialize(self) -> None:
        """Serialize and deserialize store."""
        store = AgentStateStore()
        store.add_step(AgentStep(role="user", content="Test"))
        store.set_goal("Goal")
        data = store.serialize()
        store2 = AgentStateStore.deserialize(data)
        assert len(store2.steps) == 1
        assert store2.get_goal() is not None
