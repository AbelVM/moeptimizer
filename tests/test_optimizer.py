"""Tests for the full optimizer pipeline — front-loading eviction strategy."""

import json
from unittest.mock import patch

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

    def test_proactive_trim_preserves_recent_turn_when_newest_message_is_large(self) -> None:
        """Proactive trimming must not collapse to only the static prefix."""
        self.optimizer._config.agentic.keep_full_steps = 1
        self.optimizer._config.agentic.max_optimized_chars = 1000

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Middle task"},
            {"role": "assistant", "content": "Middle response"},
            {"role": "user", "content": "Latest task"},
            {"role": "assistant", "content": "x" * 2000},
        ]

        result = self.optimizer._proactive_trim(messages, target=100, use_tokens=True)

        assert result[0]["role"] == "system"
        assert result[-2]["role"] == "user"
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == "x" * 2000

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
        """Attention sink markers are model-visible when explicitly enabled."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        config.agentic.attention_sinks_enabled = True
        optimizer = AgentContextOptimizer(config)
        long_unique_words = " ".join(f"unique_token_{i}" for i in range(900))

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": long_unique_words},
        ])

        system_content = result[0].get("content", "")
        assert "STATIC_LAYER_END" in system_content

    def test_fast_path_preserves_lean_context_and_strips_proxy_fields(self) -> None:
        """Lean contexts should bypass transformations while removing proxy-only fields."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        optimizer = AgentContextOptimizer(config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Explain this function", "chunk_index": 0},
            {"role": "assistant", "content": "I can help."},
        ]

        result = optimizer.optimize_messages(messages)

        assert len(result) == len(messages)
        assert "chunk_index" not in result[1]
        assert result[1]["content"] == "Explain this function"

    def test_padding_and_code_optimization_are_disabled_by_default(self) -> None:
        """Quality target prefers exact prompts and exact code unless explicitly enabled."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        config.agentic.static_layer_alignment_enabled = False
        config.agentic.reasoning_preseed_enabled = False
        config.agentic.optimize_code_blocks = False
        optimizer = AgentContextOptimizer(config)
        code = "def double(x):\n    return x * 2\n"

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System"},
            {"role": "user", "content": f"Use this code:\n```python\n{code}\n```"},
        ])

        joined = "\n".join(m.get("content", "") for m in result)
        assert "Use this code:" in joined
        assert code.strip() in joined
        assert "Let's reason step by step" not in joined

    def test_large_code_blocks_are_skeletonized_after_proactive_threshold(self) -> None:
        """Huge codebases should become lean skeletons once proactive pressure starts."""
        config = AppConfig()
        config.agentic.max_optimized_tokens = 1000
        config.agentic.proactive_trim_ratio = 0.001
        config.agentic.compaction_trigger_ratio = 0.9
        config.agentic.fast_path_enabled = False
        config.agentic.semantic_dedup_enabled = False
        config.agentic.code_skeleton_enabled = True
        optimizer = AgentContextOptimizer(config)
        code = (
            "import os\n\n"
            "def process(values):\n"
            "    total = 0\n"
            "    for value in values:\n"
            "        total += value * 2\n"
            "        if value > 10:\n"
            "            total += value * 3\n"
            "    return total\n"
        )

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System"},
            {"role": "user", "content": f"Use this code:\n```python\n{code}\n```"},
        ])

        joined = "\n".join(m.get("content", "") for m in result)
        assert "def process(values):" in joined
        assert "import os" in joined
        assert "total += value * 2" not in joined
        assert "..." in joined

    def test_small_code_blocks_remain_exact_during_skeletonization(self) -> None:
        """Small snippets should remain exact because they often define the task."""
        config = AppConfig()
        config.agentic.max_optimized_tokens = 1000
        config.agentic.proactive_trim_ratio = 0.001
        config.agentic.fast_path_enabled = False
        config.agentic.semantic_dedup_enabled = False
        config.agentic.code_skeleton_enabled = True
        optimizer = AgentContextOptimizer(config)
        code = "def double(x):\n    return x * 2\n"

        result = optimizer.optimize_messages([
            {"role": "system", "content": "System"},
            {"role": "user", "content": f"Use this code:\n```python\n{code}\n```"},
        ])

        joined = "\n".join(m.get("content", "") for m in result)
        assert code.strip() in joined

    def test_proactive_trim_uses_configured_threshold(self) -> None:
        """Proactive trimming should follow the configured ratio, not a hardcoded value."""
        config = AppConfig()
        config.agentic.max_optimized_tokens = 1000
        config.agentic.proactive_trim_ratio = 0.4
        config.agentic.compaction_trigger_ratio = 0.9
        config.agentic.fast_path_enabled = False
        config.agentic.semantic_dedup_enabled = False
        optimizer = AgentContextOptimizer(config)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "word " * 700},
            {"role": "assistant", "content": "ack"},
        ]

        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def capture_trim(msgs: object, target: int, **kwargs: object) -> object:
            calls.append(((msgs, target), kwargs))
            return msgs

        with patch.object(
            optimizer,
            "_proactive_trim",
            side_effect=capture_trim,
        ):
            optimizer.optimize_messages(messages)

        assert len(calls) == 1
        assert calls[0][0][1] == 400
        assert calls[0][1]["use_tokens"] is True

    def test_semantic_dedup_runs_after_proactive_threshold(self) -> None:
        """Semantic dedup supports leanness once proactive threshold is exceeded."""
        config = AppConfig()
        config.agentic.max_optimized_tokens = 1000
        config.agentic.proactive_trim_ratio = 0.4
        config.agentic.compaction_trigger_ratio = 0.9
        config.agentic.fast_path_enabled = False
        config.agentic.semantic_dedup_enabled = True
        optimizer = AgentContextOptimizer(config)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "word " * 700},
            {"role": "assistant", "content": "ack"},
        ]

        with patch.object(
            optimizer.semantic_deduplicator,
            "deduplicate",
            side_effect=lambda msgs, embedding_service: msgs,
        ) as dedup:
            optimizer.optimize_messages(messages)

        dedup.assert_called_once()
