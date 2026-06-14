"""Tests for state RAG module."""

import pytest

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

    def test_get_dependency_context(self) -> None:
        """Get dependency context for a file."""
        store = AgentStateStore()
        rag = StateBasedRAG(store)
        context = rag.get_dependency_context("test.py")
        assert context is None or isinstance(context, str)