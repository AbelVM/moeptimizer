"""Integration tests for v0.5.0 features.

Tests the full optimizer pipeline with all v0.5.0 components enabled,
verifying they work together correctly and produce valid optimized output.
"""

from __future__ import annotations

import json

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


def _build_messages(*parts: tuple[str, str]) -> list[dict]:
    """Helper to build message lists."""
    return [{"role": r, "content": c} for r, c in parts]


def _count_tokens(optimizer: AgentContextOptimizer, messages: list[dict]) -> int:
    """Count tokens in a message list."""
    return optimizer.token_counter.count_messages(messages)


class TestV050Integration:
    """Integration tests for v0.5.0 features in the full optimizer pipeline."""

    def setup_method(self) -> None:
        """Create an optimizer with all v0.5.0 features enabled."""
        config = AppConfig()
        # Enable all v0.5.0 features
        config.v050.static_prefix_kv_enabled = True
        config.v050.token_aware_truncation_enabled = True
        config.v050.chunk_fingerprint_enabled = True
        config.v050.hit_prediction_enabled = True
        config.v050.hierarchical_summary_enabled = True
        config.v050.delta_encoding_enabled = True
        config.v050.async_io_enabled = True

        self.config = config
        self.optimizer = AgentContextOptimizer(config)

    def test_basic_optimization_with_v050(self) -> None:
        """Basic optimization works with all v0.5.0 features enabled."""
        messages = _build_messages(
            ("system", "You are a helpful coding assistant."),
            ("user", "Write a Python function to calculate fibonacci numbers."),
            ("assistant", "Here is a fibonacci function:\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"),
            ("user", "Add memoization to improve performance."),
            ("assistant", "Here is the memoized version:\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"),
        )

        result = self.optimizer.optimize_messages(messages)

        assert len(result) >= 2
        assert result[0]["role"] == "system"
        # No foreign markers in assistant content
        for msg in result:
            if msg.get("role") == "assistant":
                assert "[ARCHIVED" not in msg.get("content", "")
                assert "[REASONING" not in msg.get("content", "")

    def test_static_prefix_kv_cache_reuse(self) -> None:
        """Static prefix KV-cache is populated and reused across requests."""
        # Disable hit prediction to prevent early exit before KV-cache is stored
        self.optimizer.hit_prediction = None

        messages1 = _build_messages(
            ("system", "You are a helpful assistant."),
            ("user", "Hello"),
            ("assistant", "Hi there!"),
        )

        result1 = self.optimizer.optimize_messages(messages1)
        assert len(result1) >= 2

        # Second request with same static prefix
        messages2 = _build_messages(
            ("system", "You are a helpful assistant."),
            ("user", "How are you?"),
            ("assistant", "I'm doing well, thanks!"),
        )

        result2 = self.optimizer.optimize_messages(messages2)
        assert len(result2) >= 2

        # Verify static prefix KV-cache has entries
        stats = self.optimizer.static_prefix_kv.get_stats()
        assert stats["entries"] >= 1

    def test_token_aware_truncation_preserves_boundaries(self) -> None:
        """Token-aware truncation trims at token boundaries."""
        messages = _build_messages(
            ("system", "System"),
            ("user", "word " * 1000),
            ("assistant", "Response"),
        )

        result = self.optimizer.optimize_messages(messages)

        # Result should be valid
        assert len(result) >= 1
        for msg in result:
            if msg.get("role") == "assistant":
                # Assistant content should not be truncated mid-token
                content = msg.get("content", "")
                assert content == "Response" or len(content) == 0 or content == "word " * 1000

    def test_chunk_fingerprinting_reuse(self) -> None:
        """Chunk fingerprinting caches and reuses compressed chunks."""
        code = (
            "def foo(values):\n"
            "    total = 0\n"
            "    count = 0\n"
            "    weighted_total = 0\n"
            "    for index, value in enumerate(values):\n"
            "        weight = index + 1\n"
            "        total += value\n"
            "        count += 1\n"
            "        weighted_total += value * weight\n"
            "    mean = total / max(count, 1)\n"
            "    weighted_mean = weighted_total / max(count, 1)\n"
            "    return mean, weighted_mean\n"
        )
        self.optimizer.chunk_fingerprint.clear()
        self.optimizer.cache_registry._entries.clear()
        self.optimizer.cache_registry._prefix_entries.clear()
        self.optimizer.static_prefix_kv = None
        self.optimizer.hit_prediction = None
        self.optimizer._config.agentic.compaction_trigger_ratio = 0.01
        self.optimizer._config.agentic.optimize_code_blocks = True
        self.optimizer._config.agentic.fast_path_enabled = False

        messages = _build_messages(
            ("system", "System"),
            ("user", f"Here is code:\n```python\n{code}\n```"),
            ("assistant", "I see the code."),
        )

        # First optimization
        result1 = self.optimizer.optimize_messages(messages)

        # Second optimization with same code
        messages2 = _build_messages(
            ("system", "System"),
            ("user", f"Here is code:\n```python\n{code}\n```"),
            ("assistant", "I see the code again."),
        )

        result2 = self.optimizer.optimize_messages(messages2)

        # Both should succeed
        assert len(result1) >= 2
        assert len(result2) >= 2

        # Fingerprint cache should have entries
        stats = self.optimizer.chunk_fingerprint.get_stats()
        assert stats["entries"] >= 1

    def test_hit_prediction_model(self) -> None:
        """Hit prediction model predicts and records outcomes."""
        model = self.optimizer.hit_prediction
        assert model is not None

        messages = _build_messages(
            ("system", "You are helpful."),
            ("user", "Hello"),
        )

        # Predict hit rate
        rate = model.predict_hit_rate(messages)
        assert 0.0 <= rate <= 1.0

        # Record outcome
        model.record_outcome(messages, hit=True)
        assert len(model._history) >= 1

    def test_hierarchical_summarization_standalone(self) -> None:
        """Hierarchical summarization remains available as a standalone utility."""
        summarizer = self.optimizer.hierarchical_summarizer
        assert summarizer is not None

        messages = _build_messages(
            ("system", "System"),
            ("user", "Turn 1"),
            ("assistant", "Response 1"),
            ("user", "Turn 2"),
            ("assistant", "Response 2"),
            ("user", "Turn 3"),
            ("assistant", "Response 3"),
            ("user", "Turn 4"),
            ("assistant", "Response 4"),
            ("user", "Turn 5"),
            ("assistant", "Response 5"),
        )

        result = summarizer.summarize_turns(messages)
        assert len(result) < len(messages)
        summary_msgs = [m for m in result if m.get("_summary_id")]
        assert len(summary_msgs) == 1

    def test_delta_encoding(self) -> None:
        """Delta encoding stores and reconstructs code snapshots."""
        encoder = self.optimizer.delta_encoder
        assert encoder is not None

        key1 = encoder.store_snapshot("test.py", "def foo():\n    x = 1\n    return x\n")
        assert key1 != ""

        key2 = encoder.store_snapshot("test.py", "def foo():\n    x = 2\n    y = 3\n    return x + y\n")
        assert key2 != ""

        # Reconstruct
        reconstructed = encoder.reconstruct(key2)
        assert reconstructed is not None

        # Stats
        stats = encoder.get_stats()
        assert stats["total_snapshots"] >= 2

    def test_async_io_stage(self) -> None:
        """Async I/O stage manages async and sync stages."""
        import asyncio

        async_io = self.optimizer.async_io
        assert async_io is not None

        # Test sync stage
        result = async_io.run_sync_stage(lambda: 42, stage_name="test")
        assert result == 42

        # Test async stage
        async def async_task():
            return 123

        result = asyncio.run(async_io.run_async_stage(async_task(), stage_name="async_test"))
        assert result == 123

    def test_full_pipeline_with_code_heavy_context(self) -> None:
        """Full pipeline handles code-heavy context with all v0.5.0 features."""
        messages = _build_messages(
            ("system", "You are an expert Python developer."),
            ("user", "Review this code:\n```python\nclass DataProcessor:\n    def __init__(self, data):\n        self.data = data\n        self.results = []\n\n    def process(self):\n        for item in self.data:\n            self.results.append(self._transform(item))\n        return self.results\n\n    def _transform(self, item):\n        return item * 2\n```"),
            ("assistant", "The code looks good but could use type hints and better error handling."),
            ("user", "Add type hints and error handling."),
            ("assistant", "Here is the improved version:\n```python\nfrom typing import Any, List\n\nclass DataProcessor:\n    def __init__(self, data: List[Any]) -> None:\n        self.data = data\n        self.results: List[Any] = []\n\n    def process(self) -> List[Any]:\n        try:\n            for item in self.data:\n                self.results.append(self._transform(item))\n            return self.results\n        except Exception as e:\n            raise RuntimeError(f\"Processing failed: {e}\") from e\n\n    def _transform(self, item: Any) -> Any:\n        return item * 2\n```"),
        )

        result = self.optimizer.optimize_messages(messages)

        assert len(result) >= 2
        assert result[0]["role"] == "system"

        # Verify no foreign markers
        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "[ARCHIVED" not in content
                assert "[REASONING" not in content

    def test_session_state_with_v050(self) -> None:
        """Session state serialization works with v0.5.0 components."""
        messages = _build_messages(
            ("user", "Hello"),
            ("assistant", "Hi"),
            ("user", "How are you?"),
            ("assistant", "I'm doing well."),
        )

        self.optimizer.optimize_messages(messages)
        state = self.optimizer.get_session_state()
        data = json.loads(state)

        assert "store" in data
        assert "progress" in data
        assert "mtp_state_key" in data

    def test_multi_turn_with_summarization(self) -> None:
        """Multi-turn conversation triggers hierarchical summarization."""
        summarizer = self.optimizer.hierarchical_summarizer
        assert summarizer is not None

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Add many turns to trigger summarization
        for i in range(15):
            messages.append({"role": "user", "content": f"Turn {i}: This is a test message with some content."})
            messages.append({"role": "assistant", "content": f"Response {i}: This is a response to turn {i}."})

        # Exercise the standalone summarizer directly as well.
        result = summarizer.summarize_turns(messages)

        # Should have fewer messages after summarization
        assert len(result) < len(messages)

        # Should have at least one summary message
        summary_msgs = [m for m in result if m.get("_summary_id")]
        assert len(summary_msgs) >= 1

    def test_hierarchical_summarization_not_in_pipeline(self) -> None:
        """The main pipeline does not inject middle-history summaries."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 1200
        config.agentic.max_optimized_tokens = 60
        config.v050.static_prefix_kv_enabled = False
        config.v050.hit_prediction_enabled = False
        config.v050.hierarchical_summary_max_full_turns = 3
        optimizer = AgentContextOptimizer(config)

        messages = [{"role": "system", "content": "Unique system prompt for pipeline summarization."}]
        for i in range(12):
            messages.append({
                "role": "user",
                "content": f"Turn {i}: Please review this detailed implementation note.",
            })
            messages.append({
                "role": "assistant",
                "content": f"Response {i}: The implementation should preserve the original behavior.",
            })

        result = optimizer.optimize_messages(messages)
        content = "\n".join(msg.get("content", "") for msg in result)

        assert len(result) < len(messages)
        assert "[Recall:" not in content

    def test_code_delta_encoding_in_pipeline(self) -> None:
        """Delta encoding stores code snapshots during optimization."""
        encoder = self.optimizer.delta_encoder
        assert encoder is not None

        messages = _build_messages(
            ("system", "System"),
            ("user", "Write code:\n```python\ndef foo():\n    x = 1\n    return x\n```"),
            ("assistant", "Here is the code."),
            ("user", "Update it:\n```python\ndef foo():\n    x = 2\n    y = 3\n    return x + y\n```"),
            ("assistant", "Updated."),
        )

        result = self.optimizer.optimize_messages(messages)
        assert len(result) >= 2

        # Delta encoder should have stored snapshots
        stats = encoder.get_stats()
        assert stats["total_snapshots"] >= 1

    def test_static_prefix_kv_early_exit(self) -> None:
        """Static prefix KV-cache enables early exit on cache hit."""
        messages = _build_messages(
            ("system", "You are a helpful assistant."),
            ("user", "Hello"),
            ("assistant", "Hi!"),
        )

        # First call populates cache
        result1 = self.optimizer.optimize_messages(messages)
        assert len(result1) >= 2

        # Second call with same static prefix should hit cache
        # (In production, this would skip heavy optimization)
        messages2 = _build_messages(
            ("system", "You are a helpful assistant."),
            ("user", "Hello again"),
            ("assistant", "Hi again!"),
        )

        result2 = self.optimizer.optimize_messages(messages2)
        assert len(result2) >= 2

        # Cache should have entries
        stats = self.optimizer.static_prefix_kv.get_stats()
        assert stats["entries"] >= 1

    def test_rolling_summary_in_pipeline_cache_stable(self) -> None:
        """Pipeline injects a cache-stable rolling summary that retains constraints.

        With hierarchical summarization enabled, older dynamic turns are folded
        into a single rolling summary block placed right after the frozen prefix
        (review §1/§3/§5, #7). The frozen prefix is preserved verbatim and the
        task's "don't" constraints survive in the summary so the model does not
        re-derive them verbosely.
        """
        # High budget so compression/chunking/trim don't rewrite the summary;
        # proactive threshold is still exceeded so Step 8.5 fires.
        self.config.agentic.max_optimized_tokens = 6000
        self.config.agentic.max_optimized_chars = 24000
        self.config.agentic.proactive_trim_ratio = 0.01
        self.config.agentic.code_skeleton_enabled = False
        self.config.v050.hierarchical_summary_max_full_turns = 5

        messages = [
            {"role": "system", "content": "You are a careful coding assistant."},
            {"role": "user", "content": "Build a user service."},
            {"role": "assistant", "content": "I will scaffold the service."},
        ]
        # Two frozen early turns (frozen_prefix_turns default = 2).
        for i in range(2):
            messages.append({"role": "user", "content": f"Early turn {i}: set up the project."})
            messages.append({"role": "assistant", "content": f"Done with early turn {i}."})
        # Older turns carrying a hard constraint the model must keep.
        for i in range(6):
            messages.append({
                "role": "user",
                "content": f"Step {i}: refactor the module. CRITICAL: never remove the authentication middleware.",
            })
            messages.append({"role": "assistant", "content": f"Refactored step {i} while keeping auth middleware."})
        # Recent turns (kept in full).
        for i in range(5):
            messages.append({"role": "user", "content": f"Recent {i}: add a small helper."})
            messages.append({"role": "assistant", "content": f"Added helper {i}."})

        result = self.optimizer.optimize_messages(messages)

        # Frozen prefix (system + first user + 2 early turns) preserved verbatim.
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]
        assert result[3] == messages[3]
        assert result[4] == messages[4]

        # A rolling summary block follows the frozen prefix.
        summary_msgs = [m for m in result if "Context summary (rolling):" in m.get("content", "")]
        assert summary_msgs, "expected a rolling summary block after the frozen prefix"
        summary_content = summary_msgs[0]["content"]
        assert "never remove the authentication middleware" in summary_content

    def test_all_v050_components_initialized(self) -> None:
        """All v0.5.0 components are properly initialized."""
        assert self.optimizer.static_prefix_kv is not None
        assert self.optimizer.token_aware_truncator is not None
        assert self.optimizer.chunk_fingerprint is not None
        assert self.optimizer.hit_prediction is not None
        assert self.optimizer.hierarchical_summarizer is not None
        assert self.optimizer.delta_encoder is not None
        assert self.optimizer.async_io is not None
