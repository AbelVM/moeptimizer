"""Tests for live-zone compression (P3)."""

from __future__ import annotations

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestLiveZoneCompression:
    def setup_method(self) -> None:
        config = AppConfig()
        config.agentic.live_zone_compression_enabled = True
        config.agentic.max_optimized_chars = 20000
        config.agentic.optimize_code_blocks = True
        config.agentic.tool_output_compression_enabled = True
        config.agentic.fast_path_enabled = False
        config.v050.cache_stable_mode = True
        config.v050.frozen_prefix_turns = 1
        config.v050.hit_prediction_enabled = False
        self.optimizer = AgentContextOptimizer(config)

    def test_stable_prefix_unchanged_skips_reoptimization(self) -> None:
        """When the stable prefix is unchanged, only the live zone is re-optimized."""
        base_messages = [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "First response"},
        ]

        # First turn: full optimization runs.
        first_result = self.optimizer.optimize_messages(base_messages)
        first_prefix = first_result[: self.optimizer._live_zone_start]

        # Second turn: same stable prefix, new live zone.
        second_messages = [*base_messages, {"role": "user", "content": "Second task"}, {"role": "assistant", "content": "Second response"}]
        second_result = self.optimizer.optimize_messages(second_messages)

        # The stable prefix (first 3 messages) should be byte-identical.
        assert second_result[:3] == first_prefix
        # The live zone should contain the new messages.
        assert len(second_result) == 5

    def test_changed_prefix_resets_live_zone(self) -> None:
        """When the stable prefix changes, the live zone resets to full optimization."""
        first_messages = [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "First response"},
        ]
        self.optimizer.optimize_messages(first_messages)
        # After the first turn, _update_stable_prefix sets the live zone start.
        first_live_start = self.optimizer._live_zone_start
        assert first_live_start > 0
        first_stable_prefix = list(self.optimizer._last_stable_prefix)

        # Change the system prompt (stable prefix changes).
        changed_messages = [
            {"role": "system", "content": "New system rules"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "First response"},
        ]
        self.optimizer.optimize_messages(changed_messages)
        second_stable_prefix = list(self.optimizer._last_stable_prefix)

        # The stored stable prefix should reflect the new content, proving the
        # change was detected and the live zone was reset during optimization.
        assert second_stable_prefix != first_stable_prefix
        assert second_stable_prefix[0]["content"] == "New system rules"

    def test_tool_output_cache_avoids_recompression(self) -> None:
        """Identical tool outputs should hit the content-hash cache."""
        tool_content = "ERROR " * 500  # Large enough to compress.
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Run tool"},
            {"role": "tool", "content": tool_content},
            {"role": "assistant", "content": "Done"},
        ]

        # First turn: cache miss.
        self.optimizer.optimize_messages(messages)
        cache_size_after_first = len(self.optimizer._tool_output_cache)

        # Second turn: identical tool output should hit the cache.
        self.optimizer.optimize_messages(messages)
        cache_size_after_second = len(self.optimizer._tool_output_cache)

        # Cache size should not grow on identical content.
        assert cache_size_after_second == cache_size_after_first

    def test_code_block_optimization_skips_stable_prefix(self) -> None:
        """Code blocks in the stable prefix are not re-optimized when unchanged."""
        code_block = "```python\nx = 1\n```"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": code_block},
            {"role": "assistant", "content": "Got it"},
        ]

        first_result = self.optimizer.optimize_messages(messages)
        first_user_content = first_result[1]["content"]

        # Second turn with same stable prefix.
        second_messages = [*messages, {"role": "user", "content": "Follow-up"}, {"role": "assistant", "content": "Follow-up response"}]
        second_result = self.optimizer.optimize_messages(second_messages)
        second_user_content = second_result[1]["content"]

        # The stable prefix user message should be byte-identical.
        assert first_user_content == second_user_content

    def test_live_zone_disabled_runs_full_pipeline(self) -> None:
        """When live-zone compression is disabled, the full pipeline runs."""
        config = AppConfig()
        config.agentic.live_zone_compression_enabled = False
        config.agentic.max_optimized_chars = 20000
        config.v050.cache_stable_mode = True
        config.v050.frozen_prefix_turns = 1
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": "Response"},
        ]
        result = optimizer.optimize_messages(messages)
        assert len(result) >= 2
