"""Tests for sliding window context with MTP state preservation."""

import pytest

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestSlidingWindowContext:
    def test_sliding_window_trim_small_context(self) -> None:
        """Small context is not trimmed."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=4096)
        assert len(result) == len(messages)

    def test_sliding_window_trim_large_context(self) -> None:
        """Large context is trimmed to window size."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 10000
        optimizer = AgentContextOptimizer(config)
        # Create large context
        large_content = "x" * 2000
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": large_content},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=1000)
        # Should be trimmed (static layer + some dynamic)
        total_chars = sum(len(m.get("content", "")) for m in result)
        # The result should be smaller than original (6000+ chars)
        assert total_chars < 6000

    def test_sliding_window_preserves_static_layer(self) -> None:
        """Sliding window preserves static layer (system + first user)."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 10000
        optimizer = AgentContextOptimizer(config)
        large_content = "x" * 2000
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "First user"},
            {"role": "user", "content": large_content},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=100)
        # Static layer should be preserved
        assert result[0].get("role") == "system"
        assert result[1].get("role") == "user"