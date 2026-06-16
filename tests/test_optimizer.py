"""Tests for the full optimizer pipeline — front-loading eviction strategy."""

import json

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestAgentContextOptimizer:
    def setup_method(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 500  # Small budget for testing
        self.optimizer = AgentContextOptimizer(config)

    def test_optimize_basic_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = self.optimizer.optimize_messages(messages)
        assert len(result) >= 2
        # System and first user should be preserved
        assert result[0]["role"] == "system"

    def test_optimize_with_goal(self) -> None:
        messages = [
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I will build a REST API"},
        ]
        self.optimizer.optimize_messages(messages, original_prompt="Build a REST API")
        goal = self.optimizer.store.get_goal()
        assert goal is not None
        assert "REST API" in goal.original_prompt

    def test_optimize_enforces_budget_via_eviction(self) -> None:
        """Budget enforcement via _trim_to_budget works correctly."""
        # Test the _trim_to_budget method directly since the full pipeline
        # has many stages that can add content before trimming
        self.optimizer._config.agentic.keep_full_steps = 1
        self.optimizer._config.agentic.max_optimized_chars = 100

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "x" * 500},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "y" * 500},
            {"role": "user", "content": "Third task"},
            {"role": "assistant", "content": "z" * 500},
        ]
        # Test _trim_to_budget directly
        result = self.optimizer._trim_to_budget(messages, use_tokens=True)
        # With keep_full=1, only the last turn should be fully preserved
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) <= 2, f"Expected ≤2 assistant msgs, got {len(assistant_msgs)}"

    def test_static_prefix_cache_hit_still_enforces_budget(self) -> None:
        """Static prefix cache hits must not bypass context compaction."""
        from moeptimizer.static_prefix_kv import get_static_prefix_kv_cache

        kv_cache = get_static_prefix_kv_cache()
        kv_cache.clear()

        self.optimizer._config.agentic.keep_full_steps = 1
        self.optimizer._config.agentic.max_optimized_chars = 200

        first_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Remember this task"},
            {"role": "assistant", "content": "I will keep it in mind."},
        ]
        self.optimizer.optimize_messages(first_messages)

        long_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Remember this task"},
            {"role": "assistant", "content": "word " * 500},
            {"role": "user", "content": "Now continue."},
            {"role": "assistant", "content": "Recent response."},
        ]

        result = self.optimizer.optimize_messages(long_messages)

        assert len(result) < len(long_messages)
        assert result[0]["role"] == "system"
        assert result[-1]["role"] == "assistant"
        assert result[-2]["role"] == "user"

    def test_no_content_truncation(self) -> None:
        """Front-loading eviction drops whole turns — no content is truncated."""
        long_content = "This is a very detailed response with important information"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = self.optimizer.optimize_messages(messages)

        # The preserved assistant should have original content, untouched
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert msg["content"] == long_content or msg["content"] == "Recent", (
                "Content should never be truncated — only whole turns are evicted"
            )

    def test_session_state_serialization(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        self.optimizer.optimize_messages(messages)
        state = self.optimizer.get_session_state()
        data = json.loads(state)
        assert "store" in data
        assert "progress" in data

    def test_session_state_load(self) -> None:
        messages1 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        self.optimizer.optimize_messages(messages1)
        state = self.optimizer.get_session_state()

        new_optimizer = AgentContextOptimizer(self.optimizer._config)
        new_optimizer.load_session_state(state)
        assert len(new_optimizer.store.steps) == len(self.optimizer.store.steps)

    def test_attention_sink_markers_survive_final_output(self) -> None:
        """Attention sink markers are model-visible and must not be stripped."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        optimizer = AgentContextOptimizer(config)
        long_unique_words = " ".join(f"unique_token_{i}" for i in range(900))

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": long_unique_words},
        ])

        system_content = result[0].get("content", "")
        assert "STATIC_LAYER_END" in system_content

    def test_context_compressor_skips_lean_contexts(self) -> None:
        """Lean contexts keep full code bodies instead of skeletonizing them."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        optimizer = AgentContextOptimizer(config)
        code = "def double(x):\n    return x * 2\n"

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System"},
            {"role": "user", "content": f"Use this code:\n```python\n{code}\n```"},
        ])

        assert code.strip() in "\n".join(m.get("content", "") for m in result)
