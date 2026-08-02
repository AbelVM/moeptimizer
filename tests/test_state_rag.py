"""Tests for state RAG module."""


from moeptimizer.state_rag import (
    StateBasedRAG,
)
from moeptimizer.state_store import AgentStateStore


class TestStateBasedRAG:
    def test_empty_rag(self) -> None:
        """Empty RAG has no context."""
        store = AgentStateStore()
        rag = StateBasedRAG(store)
        assert rag is not None

    def test_get_context_for_step(self) -> None:
        """Get context for a step."""
        store = AgentStateStore()
        rag = StateBasedRAG(store)
        from moeptimizer.models import AgentStep
        step = AgentStep(role="user", content="Test")
        context = rag.get_context_for_step(step)
        # May return None or context string
        assert context is None or isinstance(context, str)

    def test_get_context_for_query_is_bounded_and_excludes_current_turn(self) -> None:
        from moeptimizer.models import AgentStep

        store = AgentStateStore()
        store.add_step(AgentStep(role="user", content="Fix the cache registry lookup"))
        current = AgentStep(role="assistant", content="I fixed the cache registry lookup")
        store.add_step(current)
        rag = StateBasedRAG(store)

        context = rag.get_context_for_query(
            current.content,
            exclude_step_id=current.step_id,
            exclude_content=current.content,
            max_chars=140,
        )

        assert len(context) <= 140
        assert "step 1:" not in context
        assert context.startswith("step 0:")

    def test_get_context_for_query_ranks_overlap_before_fallback(self) -> None:
        from moeptimizer.models import AgentStep

        store = AgentStateStore()
        store.add_step(AgentStep(role="user", content="unrelated deployment notes"))
        store.add_step(AgentStep(role="assistant", content="cache registry lookup fixed"))
        rag = StateBasedRAG(store)

        context = rag.get_context_for_query(
            "cache registry lookup",
            max_results=1,
            max_chars=80,
        )

        assert context.startswith("step 1: assistant - cache registry lookup fixed")
        assert len(context) <= 80

    def test_get_dependency_context(self) -> None:
        """Get dependency context for a file."""
        store = AgentStateStore()
        rag = StateBasedRAG(store)
        context = rag.get_dependency_context("test.py")
        assert context is None or isinstance(context, str)
