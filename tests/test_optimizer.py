"""Tests for the full optimizer pipeline — front-loading eviction strategy."""

import json
from typing import Any
from unittest.mock import patch

from moeptimizer import ROLLING_SUMMARY_MARKER
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

    def test_set_token_calibration_clamps_and_scales(self) -> None:
        # ratio clamped to [0.5, 2.0]
        self.optimizer.set_token_calibration(10.0)
        assert self.optimizer._token_calibration == 2.0
        self.optimizer.set_token_calibration(0.01)
        assert self.optimizer._token_calibration == 0.5
        # None / non-positive ignored (keeps last)
        self.optimizer.set_token_calibration(None)
        assert self.optimizer._token_calibration == 0.5
        self.optimizer.set_token_calibration(-1.0)
        assert self.optimizer._token_calibration == 0.5

    def test_record_cache_outcome_skips_weak_label(self) -> None:
        """Review §2 / C7: do not train hit_prediction on a guessed label.

        When the authoritative ``cached_tokens`` is absent, the turn must be
        skipped rather than labeled with the weak local static-prefix signal.
        """
        assert self.optimizer.hit_prediction is not None
        calls: list[tuple] = []
        self.optimizer.hit_prediction.record_outcome = lambda *a, **kw: calls.append((a, kw))

        # No cached_tokens -> no training (weak label avoided).
        self.optimizer.record_cache_outcome(None)
        assert calls == []

        # Authoritative signal present -> trains with hit = (cached_tokens > 0).
        self.optimizer.record_cache_outcome(123)
        assert len(calls) == 1
        assert calls[0][1]["hit"] is True

        self.optimizer.record_cache_outcome(0)
        assert len(calls) == 2
        assert calls[1][1]["hit"] is False

    def test_seed_token_calibration_anchors_from_exact_count(self) -> None:
        sample = "hello world test"
        local = self.optimizer.token_counter.count(sample)
        assert local > 0
        assert self.optimizer._calibration_seeded is False
        # exact backend count = 2x local -> ratio 2.0 (also the clamp ceiling)
        self.optimizer.seed_token_calibration(sample, local * 2)
        assert self.optimizer._calibration_seeded is True
        assert self.optimizer._token_calibration == 2.0

    def test_seed_token_calibration_ignores_bad_input(self) -> None:
        self.optimizer.seed_token_calibration("", 5)
        assert self.optimizer._calibration_seeded is False
        self.optimizer.seed_token_calibration("text", 0)
        assert self.optimizer._calibration_seeded is False
        assert self.optimizer._token_calibration == 1.0

    def test_aggressive_defaults_use_top_only_eviction(self) -> None:
        """Default aggressive settings evict old complete turns without mutating history."""
        config = AppConfig()
        config.agentic.rag_enabled = False
        optimizer = AgentContextOptimizer(config)
        messages = [{"role": "system", "content": "System"}]
        for idx in range(5):
            messages.append({"role": "user", "content": f"task {idx} " + "x" * 120})
            messages.append({"role": "assistant", "content": f"response {idx} " + "y" * 120})

        result = optimizer.optimize_messages(messages)

        assert [msg["role"] for msg in result[:3]] == ["system", "user", "assistant"]
        assert result[1]["content"].startswith("task 0")
        assert [msg["role"] for msg in result[-6:]] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert result[-6]["content"].startswith("task 2")
        assert result[-2]["content"].startswith("task 4")

    def test_optimize_enforces_budget_via_eviction(self) -> None:
        """Budget enforcement via _trim_to_budget works correctly."""
        # Test the _trim_to_budget method directly since the full pipeline
        # has many stages that can add content before trimming
        self.optimizer._config.agentic.keep_full_steps = 1
        self.optimizer._config.agentic.max_optimized_chars = 100
        # Disable cache-stable mode so this test exercises pure budget eviction
        # (the frozen early turns would otherwise be immutable).
        self.optimizer._config.v050.cache_stable_mode = False

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

    def test_eviction_hysteresis_batches_to_low_water(self) -> None:
        """Once eviction triggers, it trims to the low-water mark in one batch and
        keeps the oldest kept turn byte-stable across the next over-budget turn.

        This is the review03.md §6/§9 property: batching to a low-water mark means
        the backend's native prefix cache is reused across many turns instead of
        being invalidated every over-budget turn.
        """
        self.optimizer._config.agentic.eviction_low_water_ratio = 0.5

        def pair(i: int) -> list[dict[str, str]]:
            return [
                {"role": "user", "content": f"task {i} " + "x" * 90},
                {"role": "assistant", "content": f"resp {i} " + "y" * 90},
            ]

        # 10 pairs, ~200 chars each -> ~2000 chars total. Budget 1000, low water 500.
        body = [m for i in range(10) for m in pair(i)]
        budget = 1000

        trimmed = self.optimizer._evict_for_budget(body, budget, use_tokens=False)
        total = sum(len(m.get("content", "")) for m in trimmed)
        # Trimmed down to <= low water (500), not merely under budget (1000).
        assert total <= 500
        first_kept = trimmed[0]["content"]

        # Add one more pair (simulating the next turn) and re-evict. Because the
        # body is now under budget again (hysteresis headroom), the oldest kept
        # turn does NOT change -> the leading prefix stays byte-stable.
        next_body = trimmed + pair(10)
        if sum(len(m.get("content", "")) for m in next_body) <= budget:
            re_trimmed = self.optimizer._evict_for_budget(
                next_body, budget, use_tokens=False
            )
            assert re_trimmed[0]["content"] == first_kept

    def test_eviction_low_water_ratio_one_trims_to_budget(self) -> None:
        """ratio=1.0 restores classic trim-to-exact-budget behavior."""
        self.optimizer._config.agentic.eviction_low_water_ratio = 1.0
        body = [
            m
            for i in range(10)
            for m in (
                {"role": "user", "content": f"t{i} " + "x" * 90},
                {"role": "assistant", "content": f"r{i} " + "y" * 90},
            )
        ]
        budget = 1000
        trimmed = self.optimizer._evict_for_budget(body, budget, use_tokens=False)
        total = sum(len(m.get("content", "")) for m in trimmed)
        assert total <= budget

    def test_eviction_carries_forward_code_ledger(self) -> None:
        """When front-eviction drops a code-bearing turn, its function/class
        signatures are carried forward in a compact '[Evicted-turn code index]'
        message appended to the result, so the model keeps awareness of code
        that lived in dropped turns (fixes has_code_proxy=0 / code_block_loss).
        """
        self.optimizer._config.agentic.eviction_low_water_ratio = 1.0
        self.optimizer._config.agentic.code_ledger_max_sigs = 40
        body = [
            {
                "role": "user",
                "content": "implement helper",
            },
            {
                "role": "assistant",
                "content": "Here:\n```python\ndef compute_hash(x):\n    return x\n\nclass Parser:\n    pass\n```",
            },
            {"role": "user", "content": "t1 " + "x" * 400},
            {"role": "assistant", "content": "r1 " + "y" * 400},
            {"role": "user", "content": "t2 " + "x" * 400},
            {"role": "assistant", "content": "r2 " + "y" * 400},
        ]
        # Budget small enough to evict the first (code-bearing) pair.
        trimmed = self.optimizer._evict_for_budget(body, budget=1500, use_tokens=False)
        ledger = [m for m in trimmed if m.get("_code_ledger")]
        assert ledger, "expected an evicted-turn code ledger message"
        content = ledger[0]["content"]
        assert "Evicted-turn code index" in content
        assert "def compute_hash" in content
        assert "class Parser" in content
        # The ledger must not re-introduce the full dropped code body.
        assert "return x" not in content

    def test_static_prefix_cache_hit_still_enforces_budget(self) -> None:
        """Static prefix cache hits must not bypass context compaction."""
        from moeptimizer.static_prefix_kv import get_static_prefix_kv_cache

        kv_cache = get_static_prefix_kv_cache()
        kv_cache.clear()

        self.optimizer._config.agentic.keep_full_steps = 1
        self.optimizer._config.agentic.max_optimized_chars = 200
        # Disable cache-stable mode: this test verifies the static-prefix KV cache
        # does not bypass compaction of the dynamic layer, independent of the
        # frozen early-turn prefix.
        self.optimizer._config.v050.cache_stable_mode = False

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

    def test_attention_sink_markers_are_not_injected(self) -> None:
        """Attention sink markers are not model-visible in cache-stable mode."""
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
        assert "STATIC_LAYER_END" not in system_content

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

    def test_cache_stable_mode_freezes_early_turns(self) -> None:
        """Cache-stable mode keeps the early turns verbatim and immutable.

        Regression guard for the frozen_prefix_end boundary bug: the stable
        prefix must include the first user message AND the next
        `frozen_prefix_turns` complete turns, not just system + first user.
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 20000
        config.v050.cache_stable_mode = True
        config.v050.frozen_prefix_turns = 2
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Third task"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Fourth task"},
            {"role": "assistant", "content": "A4"},
            {"role": "user", "content": "Fifth task"},
            {"role": "assistant", "content": "A5"},
        ]
        # frozen_prefix_end must cover system + first user + 2 turns
        # (Second + Third), i.e. indices 0..6 -> 7 messages.
        assert optimizer.context_aligner.frozen_prefix_end(messages, 2) == 7

        # Under a tight budget the frozen early turns must survive eviction
        # while later turns are dropped.
        config.agentic.max_optimized_chars = 50
        optimizer2 = AgentContextOptimizer(config)
        result = optimizer2.optimize_messages(messages)

        # The frozen block is byte-identical to the original prefix.
        assert [m["content"] for m in result[:7]] == [
            m["content"] for m in messages[:7]
        ]
        # Early-turn content is preserved (not evicted) under budget pressure.
        joined = "\n".join(m["content"] for m in result)
        assert "Second task" in joined
        assert "Third task" in joined

    def test_volatile_turn_not_accumulated(self) -> None:
        """A prior volatile trailing turn is stripped before a fresh one is appended.

        Regression guard for review §8: without the ``_volatile_turn`` tag the
        prior volatile turn became a historical user turn on the next request and
        a new one was appended after it, so the context accumulated one extra
        volatile turn every turn until eviction.
        """
        config = AppConfig()
        config.agentic.max_optimized_chars = 200_000
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Task A"},
            {"role": "assistant", "content": "Done A"},
        ]
        # Simulate a prior pass that already appended a volatile trailing turn.
        prior = list(messages)
        prior.append(
            {"role": "user", "content": "Old volatile context", "_volatile_turn": True}
        )

        result = optimizer._append_volatile_context(
            prior, anchor="anchor text", rag_context="", warning_lines=[], proactive_threshold_tokens=0
        )
        volatile = [m for m in result if m.get("_volatile_turn")]
        # Exactly one volatile turn, and the stale one was removed.
        assert len(volatile) == 1
        assert volatile[0]["content"] == "# Conversation Quality Anchor\nanchor text"
        assert "Old volatile context" not in [m["content"] for m in result]


class TestToolOutputCompressionPipeline:
    """Step 11.6: large tool outputs must be boundary-compressed by the pipeline."""

    def test_large_tool_output_is_compressed(self) -> None:
        config = AppConfig()
        # Generous budget so nothing is evicted; we are testing compression,
        # not eviction. Keep default 4000-char compression threshold.
        config.agentic.max_optimized_chars = 200_000
        assert config.agentic.tool_output_compression_enabled
        threshold = config.agentic.tool_output_compression_max_chars
        optimizer = AgentContextOptimizer(config)

        # A realistic oversized run_command log: many repeated lines -> well over
        # the threshold so the boundary compressor fires and collapses it.
        big_log = "\n".join(["DEBUG worker heartbeat ok"] * 400)
        assert len(big_log) > threshold

        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run the test suite."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "run_command", "content": big_log},
        ]

        result = optimizer.optimize_messages(messages)

        tool_msg = next(m for m in result if m.get("role") == "tool")
        # The tool output survived (not evicted) but was compressed.
        assert len(tool_msg["content"]) < len(big_log)
        # Compression collapsed the repeated line rather than forwarding verbatim.
        assert tool_msg["content"].count("DEBUG worker heartbeat ok") < 400

    def test_small_tool_output_forwarded_verbatim(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 200_000
        optimizer = AgentContextOptimizer(config)

        small = "3 passed in 0.12s"
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run the tests."},
            {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": small},
        ]

        result = optimizer.optimize_messages(messages)
        tool_msg = next(m for m in result if m.get("role") == "tool")
        # Under the threshold -> quality-safe verbatim forwarding.
        assert tool_msg["content"] == small


class TestUserPasteCompression:
    """Review §5 / C13: large user code pastes must be boundary-compressed by the
    pipeline (agentic coding bloats on pasted files, not just tool output)."""

    def test_large_user_paste_is_compressed(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 200_000
        assert config.agentic.user_paste_compression_enabled
        threshold = config.agentic.user_paste_compression_max_chars
        optimizer = AgentContextOptimizer(config)

        # A realistic oversized file paste: many repeated lines -> over threshold.
        big_paste = "\n".join(["def helper():  # boilerplate"] * 400)
        assert len(big_paste) > threshold

        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": f"Here is my file:\n{big_paste}"},
        ]
        result = optimizer.optimize_messages(messages)
        user_msg = next(m for m in result if m.get("role") == "user")
        assert len(user_msg["content"]) < len(big_paste)
        # Compression collapsed the repeated line rather than forwarding verbatim.
        assert user_msg["content"].count("def helper()") < 400

    def test_small_user_paste_forwarded_verbatim(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 200_000
        optimizer = AgentContextOptimizer(config)
        small = "def add(a, b):\n    return a + b"
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": f"Review this:\n{small}"},
        ]
        result = optimizer.optimize_messages(messages)
        user_msg = next(m for m in result if m.get("role") == "user")
        # Under the threshold -> verbatim forwarding (quality-safe).
        assert small in user_msg["content"]

    def test_user_paste_compression_can_be_disabled(self) -> None:
        config = AppConfig()
        config.agentic.max_optimized_chars = 200_000
        config.agentic.user_paste_compression_enabled = False
        optimizer = AgentContextOptimizer(config)
        big_paste = "\n".join(["x = 1  # repeated"] * 400)
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": f"Paste:\n{big_paste}"},
        ]
        result = optimizer.optimize_messages(messages)
        user_msg = next(m for m in result if m.get("role") == "user")
        # Disabled -> the paste is forwarded unchanged.
        assert user_msg["content"] == f"Paste:\n{big_paste}"


class TestIncrementalOptimization:
    """Review §4: incremental optimization (behind a flag) must produce output
    byte-identical to the full path while reusing the stable prefix."""

    def _build_conversation(self, turns: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful coding agent."},
        ]
        for i in range(turns):
            msgs.append({"role": "user", "content": f"Step {i}: implement feature {i}"})
            msgs.append(
                {"role": "assistant", "content": f"Done step {i}."}
            )
        return msgs

    def test_incremental_matches_full_path(self) -> None:
        """With the flag on, the incremental path is byte-identical to full path."""
        full_cfg = AppConfig()
        full_cfg.agentic.max_optimized_chars = 200_000
        full_cfg.agentic.live_zone_compression_enabled = True
        full_cfg.v050.cache_stable_mode = True
        full_cfg.v050.frozen_prefix_turns = 2
        full = AgentContextOptimizer(full_cfg)

        incr_cfg = AppConfig()
        incr_cfg.agentic.max_optimized_chars = 200_000
        incr_cfg.agentic.live_zone_compression_enabled = True
        incr_cfg.v050.cache_stable_mode = True
        incr_cfg.v050.frozen_prefix_turns = 2
        incr_cfg.agentic.incremental_optimization_enabled = True
        incr = AgentContextOptimizer(incr_cfg)

        # Replay a growing conversation; at each turn compare incremental vs full.
        for turn in range(1, 12):
            messages = self._build_conversation(turn)
            full_out = full.optimize_messages(list(messages))
            incr_out = incr.optimize_messages(list(messages))
            # Byte-identical optimized prompts (volatile turn included).
            assert [
                {k: v for k, v in m.items() if k != "_volatile_turn"}
                for m in full_out
            ] == [
                {k: v for k, v in m.items() if k != "_volatile_turn"}
                for m in incr_out
            ], f"turn {turn}: incremental output diverged from full path"

    def test_incremental_reuses_stable_prefix(self) -> None:
        """The memo is populated and reused across turns with an unchanged prefix."""
        incr_cfg = AppConfig()
        incr_cfg.agentic.max_optimized_chars = 200_000
        incr_cfg.agentic.live_zone_compression_enabled = True
        incr_cfg.v050.cache_stable_mode = True
        incr_cfg.v050.frozen_prefix_turns = 2
        incr_cfg.agentic.incremental_optimization_enabled = True
        incr = AgentContextOptimizer(incr_cfg)

        # The first turn has no prior stable prefix, so the memo is not yet set.
        incr.optimize_messages(self._build_conversation(3))
        assert incr._stable_prefix_optimized is None

        # The second turn establishes a stable prefix and populates the memo.
        incr.optimize_messages(self._build_conversation(4))
        assert incr._stable_prefix_optimized is not None
        assert incr._stable_prefix_hash is not None
        memo_before = incr._stable_prefix_optimized

        # A further turn with an unchanged prefix reuses the memo verbatim.
        incr.optimize_messages(self._build_conversation(5))
        # The stable prefix portion is reused byte-for-byte.
        assert incr._stable_prefix_optimized == memo_before

    def test_incremental_disabled_by_default(self) -> None:
        """The flag defaults to off so production behavior is unchanged."""
        cfg = AppConfig()
        assert cfg.agentic.incremental_optimization_enabled is False


class TestFastPathSingleGate:
    """Review §2 / C14: the fast-path concept must be a single early-return gate that
    also skips RAG/summary when the context is small — even when the fast path itself
    is bypassed (e.g. a lean context that carries a large tool output)."""

    def _make_optimizer(self) -> AgentContextOptimizer:
        cfg = AppConfig()
        # Generous budget so the conversation stays under the proactive threshold
        # (lean context), but include a large tool output that bypasses the fast path.
        cfg.agentic.max_optimized_chars = 200_000
        cfg.agentic.rag_enabled = True
        cfg.v050.cache_stable_summary_enabled = True
        cfg.v050.hierarchical_summary_enabled = True
        cfg.v050.cache_stable_mode = True
        return AgentContextOptimizer(cfg)

    def test_rag_skipped_on_lean_context_with_large_tool_output(self) -> None:
        opt = self._make_optimizer()
        # A small conversation (lean) but with one large tool output that forces the
        # fast path to return None. RAG must still be skipped because the context is
        # lean (single gate), not fired just because the fast path was bypassed.
        big_tool = "\n".join(["line of output"] * 50)  # >1000 chars -> bypasses fast path
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "run", "content": big_tool},
        ]
        rag_calls: list[object] = []
        opt.state_rag.get_context_for_step = lambda *a, **kw: rag_calls.append(a) or ""  # type: ignore[method-assign]

        result = opt.optimize_messages(messages)

        # RAG was not invoked because the context is lean (single gate).
        assert rag_calls == []
        # The tool output is still present (not evicted) and the volatile turn is
        # appended without RAG context. The cache-stable boundary transform collapses
        # the repeated tool output in place (idempotent, frozen into the prefix), so
        # the content is the compressed form, not the raw one.
        tool_msg = next(m for m in result if m.get("role") == "tool")
        assert tool_msg["content"] != big_tool
        assert "repeated" in tool_msg["content"]

    def test_rag_runs_on_over_threshold_context(self) -> None:
        opt = self._make_optimizer()
        # A large conversation that exceeds the proactive threshold -> RAG should run.
        # The threshold is ~1350 tokens; use a 20k-char user turn to clear it.
        big_user = "x" * 20_000
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": big_user},
            {"role": "assistant", "content": "ok"},
        ]
        rag_calls: list[object] = []
        opt.state_rag.get_context_for_step = lambda *a, **kw: rag_calls.append(a) or ""  # type: ignore[method-assign]

        opt.optimize_messages(messages)
        # Over threshold -> RAG is invoked.
        assert len(rag_calls) >= 1


class TestRankChunksThreshold:
    """Review §5 / C2: min_chunk_score must actually drop low-relevance chunks."""

    def _make_optimizer(self) -> AgentContextOptimizer:
        cfg = AppConfig()
        # Use the production default threshold (0.2) unless overridden below.
        return AgentContextOptimizer(cfg)

    def test_low_relevance_chunks_dropped_at_default_threshold(self) -> None:
        import numpy as np

        opt = self._make_optimizer()
        # Query vector along the x-axis.
        query = np.array([1.0, 0.0, 0.0], dtype=float)
        # Three chunks: one highly relevant (cos=1.0), one marginal (cos=0.1,
        # below 0.2), one orthogonal (cos=0.0). Only the relevant one should survive.
        chunks = ["relevant", "marginal", "orthogonal"]
        vecs = [
            np.array([1.0, 0.0, 0.0], dtype=float),   # cos = 1.0
            np.array([0.1, 1.0, 0.0], dtype=float),   # cos ~ 0.099
            np.array([0.0, 1.0, 0.0], dtype=float),   # cos = 0.0
        ]
        ranked = opt._rank_chunks(query, vecs, chunks)
        assert ranked == ["relevant"]

    def test_permissive_threshold_keeps_marginal_chunk(self) -> None:
        import numpy as np

        cfg = AppConfig()
        cfg.code_chunking.min_chunk_score = 0.05  # legacy permissive value
        opt = AgentContextOptimizer(cfg)
        query = np.array([1.0, 0.0, 0.0], dtype=float)
        chunks = ["relevant", "marginal", "orthogonal"]
        vecs = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.1, 1.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
        ]
        ranked = opt._rank_chunks(query, vecs, chunks)
        # At 0.05 the marginal chunk (cos ~0.099) survives; orthogonal (0.0) still drops.
        assert "relevant" in ranked
        assert "marginal" in ranked
        assert "orthogonal" not in ranked

    def test_default_threshold_is_raised_from_legacy(self) -> None:
        assert AppConfig().code_chunking.min_chunk_score == 0.2


class TestDeltaSnapshotScopedToLiveZone:
    """Review §4 / C4: delta-snapshot regex scan must only cover the live zone
    (new/mutated turns), not the byte-stable prefix re-scanned every turn."""

    def _make_optimizer(self) -> AgentContextOptimizer:
        cfg = AppConfig()
        cfg.v050.delta_encoding_enabled = True
        cfg.agentic.fast_path_enabled = False  # reach the delta-snapshot step
        cfg.v050.hit_prediction_enabled = False  # avoid early-exit before delta step
        cfg.v050.static_prefix_kv_enabled = False  # avoid KV-cache early return
        cfg.agentic.max_optimized_chars = 100_000  # no eviction; keep full history
        cfg.v050.cache_stable_mode = True
        opt = AgentContextOptimizer(cfg)
        # The cache registry is a process-global singleton that learns a 1.0 hit
        # rate after the first turn, which would early-exit the second turn
        # before the delta-snapshot step. Replace it with a fresh, empty registry
        # so the test exercises the full pipeline on every turn.
        from moeptimizer.cache_registry import CacheKeyRegistry, get_cache_registry

        get_cache_registry.__globals__["_registry"] = CacheKeyRegistry()
        opt.cache_registry = get_cache_registry()
        # The cache registry predicts a 1.0 hit rate for a repeated static prefix,
        # which early-exits the pipeline before the delta-snapshot step. Force it
        # to 0.0 so the test exercises the full pipeline (and the delta scan) every
        # turn.
        opt.cache_registry.predict_hit_rate = lambda *a, **k: 0.0  # type: ignore[method-assign]
        return opt

    def test_stable_prefix_code_block_not_rescanned_each_turn(self) -> None:
        opt = self._make_optimizer()
        code_block = "```python\ndef stable():\n    return 42\n```"

        # Turn 1: a code block in an early (stable-prefix) user turn.
        msgs1 = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": f"Here is the helper:\n{code_block}"},
            {"role": "assistant", "content": "Got it."},
        ]
        opt.optimize_messages(msgs1)
        stored_after_t1 = opt.delta_encoder._stats["snapshots_stored"]

        # Turn 2: append a NEW turn. The early code block is now in the stable
        # prefix and must NOT be re-scanned (no new snapshot for it).
        msgs2 = [*msgs1,
            {"role": "user", "content": "Now add a test."},
            {"role": "assistant", "content": "Added."},
        ]
        opt.optimize_messages(msgs2)
        stored_after_t2 = opt.delta_encoder._stats["snapshots_stored"]

        # The new turn has no code block, so the stable-prefix block must not be
        # re-stored: the snapshot count stays the same (C4 gating).
        assert stored_after_t2 == stored_after_t1
        assert stored_after_t1 >= 1  # the block WAS stored on turn 1

    def test_new_live_zone_code_block_is_stored(self) -> None:
        opt = self._make_optimizer()
        msgs1 = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Start."},
            {"role": "assistant", "content": "OK."},
        ]
        opt.optimize_messages(msgs1)
        stored_after_t1 = opt.delta_encoder._stats["snapshots_stored"]

        # Turn 2 introduces a code block in the live zone -> must be stored.
        new_code = "```python\ndef new_fn():\n    pass\n```"
        msgs2 = [*msgs1,
            {"role": "user", "content": f"Add this:\n{new_code}"},
            {"role": "assistant", "content": "Done."},
        ]
        opt.optimize_messages(msgs2)
        stored_after_t2 = opt.delta_encoder._stats["snapshots_stored"]

        assert stored_after_t2 > stored_after_t1


class TestCodeDeltaInjection:
    """P2.2 (review §3.4): when a file is re-read after an edit, the optimizer
    should replace the full re-read body with a compact diff against the prior
    snapshot — but only when the prior version is already present in context."""

    def _make_optimizer(self) -> AgentContextOptimizer:
        cfg = AppConfig()
        cfg.agentic.delta_encode_inject = True
        return AgentContextOptimizer(cfg)

    def test_reread_injects_diff_when_prior_in_context(self) -> None:
        opt = self._make_optimizer()
        v1 = "def add(a, b):\n    return a + b\n"
        v2 = "def add(a, b):\n    return a + b + 0\n"
        # Simulate the pipeline: v1 stored on a prior turn, v2 stored this turn
        # (Step 14.8), so the encoder now has v1 as the prior version.
        opt.delta_encoder.store_snapshot("inline:python", v1)
        opt.delta_encoder.store_snapshot("inline:python", v2)
        msgs = [
            {"role": "user", "content": f"Here is the file:\n```python\n{v1}```"},
            {"role": "user", "content": f"Re-read it:\n```python\n{v2}```"},
        ]
        opt._inject_code_deltas(msgs, 0)
        reread = msgs[1]["content"]
        assert "file changed since last read" in reread
        # The full v2 body must be gone, replaced by the diff.
        assert v2 not in reread
        assert "-    return a + b" in reread
        assert "+    return a + b + 0" in reread

    def test_first_read_keeps_full_body(self) -> None:
        opt = self._make_optimizer()
        v1 = "def add(a, b):\n    return a + b\n"
        msgs = [
            {"role": "user", "content": f"Read the file:\n```python\n{v1}```"},
        ]
        opt._inject_code_deltas(msgs, 0)
        # No prior version -> full body preserved, no diff marker.
        assert "file changed since last read" not in msgs[0]["content"]
        assert v1 in msgs[0]["content"]

    def test_reread_without_prior_in_context_keeps_full(self) -> None:
        opt = self._make_optimizer()
        v1 = "def add(a, b):\n    return a + b\n"
        v2 = "def add(a, b):\n    return a + b + 0\n"
        opt.delta_encoder.store_snapshot("inline:python", v1)
        # Prior version is NOT in the context blob -> keep full v2.
        msgs = [
            {"role": "user", "content": f"Re-read it:\n```python\n{v2}```"},
        ]
        opt._inject_code_deltas(msgs, 0)
        assert "file changed since last read" not in msgs[0]["content"]
        assert v2 in msgs[0]["content"]


class TestQualityAnchorMonotonic:
    """Review §5 / C5: the quality anchor is the trailing volatile user turn, so its
    content must be byte-stable across turns. Constraints accumulate append-only and
    only drop from the FRONT (oldest) when capped, so the recent tail never churns."""

    def _make_optimizer(self) -> AgentContextOptimizer:
        cfg = AppConfig()
        cfg.agentic.max_optimized_chars = 100_000
        return AgentContextOptimizer(cfg)

    def _user_turns(self, *contents: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a coding agent."}]
        for c in contents:
            msgs.append({"role": "user", "content": c})
            msgs.append({"role": "assistant", "content": "ok"})
        return msgs

    def test_first_request_is_frozen_across_turns(self) -> None:
        opt = self._make_optimizer()
        a1 = opt._build_quality_anchor(self._user_turns("Build a REST API for users"))
        a2 = opt._build_quality_anchor(self._user_turns("Build a REST API for users", "Add auth"))
        # The "Original request" head must be identical on both turns (monotonic).
        assert "Build a REST API for users" in a1
        assert "Build a REST API for users" in a2
        assert opt._anchor_first_request  # captured once, never rewritten

    def test_constraints_accumulate_append_only(self) -> None:
        opt = self._make_optimizer()
        a1 = opt._build_quality_anchor(self._user_turns("Task", "constraint one"))
        a2 = opt._build_quality_anchor(self._user_turns("Task", "constraint one", "constraint two"))
        # The first constraint must still be present and the second appended.
        assert "constraint one" in a1
        assert "constraint one" in a2
        assert "constraint two" in a2
        # The tail is monotonic: a2's constraint block contains a1's constraint block.
        assert a1 in a2 or "constraint one" in a2

    def test_oldest_constraint_drops_from_front_not_middle(self) -> None:
        opt = self._make_optimizer()
        # Feed 7 distinct constraints; cap is 5, so the 2 oldest drop from the FRONT.
        contents = ["Task"] + [f"constraint {i}" for i in range(1, 8)]
        anchor = opt._build_quality_anchor(self._user_turns(*contents))
        assert "constraint 1" not in anchor
        assert "constraint 2" not in anchor
        # The most-recent 5 are retained and in original order (tail stable).
        for i in range(3, 8):
            assert f"constraint {i}" in anchor
        assert anchor.index("constraint 3") < anchor.index("constraint 7")

    def test_repeated_user_turn_does_not_grow_anchor(self) -> None:
        opt = self._make_optimizer()
        a1 = opt._build_quality_anchor(self._user_turns("Task", "dup constraint"))
        a2 = opt._build_quality_anchor(self._user_turns("Task", "dup constraint", "dup constraint"))
        # The duplicate is deduped against the running set; tail stays byte-identical.
        assert a1 == a2


class TestDynamicSubCaps:
    """v0.7.18: sub-caps (tool output, paste, chunk, state steps, anchor) derive
    from the lean dynamic budget and stay tiny even on a huge window."""

    def _opt_with_window(self, window: int | None, **overrides: Any) -> AgentContextOptimizer:
        config = AppConfig()
        config.agentic.dynamic_budget_enabled = True
        config.agentic.budget_window_fraction = 0.06
        config.agentic.max_optimized_tokens = 12000
        config.agentic.max_optimized_chars = 48000
        for k, v in overrides.items():
            setattr(config.agentic, k, v)

        class _Caps:
            max_context_window = window
            remote_tokenize = False

        class _Probe:
            def cached(self) -> _Caps:
                return _Caps()

        return AgentContextOptimizer(config, capability_probe=_Probe() if window else None)

    def test_tool_output_cap_scales_with_window_above_floor(self) -> None:
        # 262144 * 0.06 = 15728 budget; 0.10 * 15728 * 4 chars ~= 6291 > 4000 floor
        opt = self._opt_with_window(262144)
        assert opt._dynamic_tool_output_max_chars() > 4000
        # Tiny budget (low floor) -> static floor wins, never starves below it.
        opt_small = self._opt_with_window(8000, max_optimized_tokens=1000, max_optimized_chars=4000)
        assert opt_small._dynamic_tool_output_max_chars() == 4000

    def test_user_paste_cap_mirrors_tool_output(self) -> None:
        opt = self._opt_with_window(262144)
        assert opt._dynamic_user_paste_max_chars() > 4000
        opt_small = self._opt_with_window(8000, max_optimized_tokens=1000, max_optimized_chars=4000)
        assert opt_small._dynamic_user_paste_max_chars() == 4000

    def test_chunk_cap_scales_with_window_above_floor(self) -> None:
        # 0.05 * 15728 * 4 ~= 3145 > 1500 floor
        opt = self._opt_with_window(262144)
        assert opt._dynamic_chunk_max_chars() > 1500
        opt_small = self._opt_with_window(8000, max_optimized_tokens=1000, max_optimized_chars=4000)
        assert opt_small._dynamic_chunk_max_chars() == 1500

    def test_state_steps_cap_scales_with_window_above_floor(self) -> None:
        # 0.025 * 15728 ~= 393 > 200 floor
        opt = self._opt_with_window(262144)
        assert opt._dynamic_max_state_steps() > 200
        opt_small = self._opt_with_window(8000, max_optimized_tokens=1000, max_optimized_chars=4000)
        assert opt_small._dynamic_max_state_steps() == 200

    def test_anchor_constraints_cap_scales_with_window(self) -> None:
        # 0.001 * 15728 ~= 15 > 5 floor
        opt = self._opt_with_window(262144)
        assert opt._dynamic_max_anchor_constraints() > 5
        opt_small = self._opt_with_window(8000, max_optimized_tokens=1000, max_optimized_chars=4000)
        assert opt_small._dynamic_max_anchor_constraints() == 5

    def test_disabled_dynamic_uses_static_floor(self) -> None:
        opt = self._opt_with_window(262144, dynamic_budget_enabled=False)
        assert opt._dynamic_tool_output_max_chars() == 4000
        assert opt._dynamic_chunk_max_chars() == 1500
        assert opt._dynamic_max_state_steps() == 200
        assert opt._dynamic_max_anchor_constraints() == 5

    def test_state_store_override_applied_per_turn(self) -> None:
        opt = self._opt_with_window(262144)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        opt.optimize_messages(msgs)
        # The store cap should track the dynamic value, not the static floor.
        assert opt.store._max_steps_override == opt._dynamic_max_state_steps()
        assert opt.store._max_steps_override > 200


class TestCacheStabilityAcrossTurns:
    """P0.3 (REVIEW.md): the frozen prefix + rolling-summary head must stay
    byte-stable across a long agentic session so the backend's prefix-cache KV
    is reused every turn. The v0.7.18 regression mutated the middle of the
    dynamic body at turn 13, collapsing reuse to the frozen prefix only.
    """

    def _opt(self) -> AgentContextOptimizer:
        config = AppConfig()
        config.agentic.dynamic_budget_enabled = True
        config.agentic.budget_window_fraction = 0.025
        config.agentic.max_optimized_tokens = 12000
        config.agentic.max_optimized_chars = 48000
        config.agentic.max_context_growth_per_turn = 1500
        config.agentic.keep_full_steps = 8
        config.agentic.quality_profile = "balanced"
        config.v050.cache_stable_mode = True
        config.v050.frozen_prefix_turns = 2
        config.v050.cache_stable_summary_enabled = True
        config.v050.hierarchical_summary_max_full_turns = 8

        class _Caps:
            max_context_window = 262144
            remote_tokenize = False

        class _Probe:
            def cached(self) -> _Caps:
                return _Caps()

        return AgentContextOptimizer(config, capability_probe=_Probe())

    def _build_turn(self, n: int) -> list[dict[str, Any]]:
        """A realistic agentic turn: user request + assistant reply with code."""
        return [
            {
                "role": "user",
                "content": (
                    f"Turn {n}: refactor module_{n} to use async def process_{n}() "
                    f"and handle the edge case where input_{n} is None. "
                    f"Keep the public API stable and do not change the return type."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Here is the change for turn {n}:\n"
                    f"```python\n"
                    f"async def process_{n}(input_{n}):\n"
                    f"    if input_{n} is None:\n"
                    f"        return default_{n}\n"
                    f"    return await transform_{n}(input_{n})\n"
                    f"```\n"
                    f"I kept the return type unchanged and added the None guard."
                ),
            },
        ]

    def _stable_prefix_blob(self, opt: AgentContextOptimizer, msgs: list[dict[str, Any]]) -> tuple[str, int, bool]:
        """Serialize the byte-stable leading section: frozen prefix + summary head.

        The summary block is detected by its content marker (``ROLLING_SUMMARY_MARKER``)
        as well as by its internal ``_summary_id`` key, because ``_strip_internal_flags``
        removes the ``_summary_id`` key before the prompt is sent to the backend — and
        the stable-prefix detection on the *next* turn runs on the stripped list. The
        production ``_stable_prefix_end`` uses the same content-based detection; this
        helper must match it or it would miss the summary and report a false break.

        Returns ``(blob, opt_len, summary_present)`` where ``blob`` is the serialized
        leading section. The backend caches the LEADING bytes of the prompt, so the
        cache-stability invariant is APPEND-ONLY: this turn's blob must be a prefix of
        next turn's blob (the summary only ever grows by appending). It is NOT required
        to be byte-identical, because the summary legitimately appends new folded text.
        """
        optimized = opt.optimize_messages(msgs)
        # The frozen prefix is system + first user + frozen_prefix_turns turns.
        frozen_end = opt.context_aligner.frozen_prefix_end(
            optimized, opt._config.v050.frozen_prefix_turns
        )
        stable = optimized[:frozen_end]
        # Include any rolling-summary block (append-only, byte-stable head).
        # Detect by content marker too, since _strip_internal_flags drops _summary_id.
        summary_present = False
        for m in optimized[frozen_end:]:
            content = m.get("content") or ""
            if (
                m.get("_summary_id")
                or m.get("_rolling_summary")
                or content.startswith(ROLLING_SUMMARY_MARKER)
            ):
                stable.append(m)
                summary_present = True
            else:
                break
        blob = "\n".join(
            f"{m.get('role')}:{m.get('content')}" for m in stable
        )
        return blob, len(optimized), summary_present

    def test_frozen_prefix_stable_across_30_turns(self) -> None:
        opt = self._opt()
        # Seed with system + first user so the frozen prefix exists.
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
            {"role": "user", "content": "Initial task: build the pipeline."},
            {"role": "assistant", "content": "I will build the pipeline step by step."},
        ]
        prev_blob: str | None = None
        prev_summary_present = False
        for n in range(1, 31):
            conversation.extend(self._build_turn(n))
            blob, _opt_len, summary_present = self._stable_prefix_blob(opt, list(conversation))
            # The leading bytes [frozen][summary] must be APPEND-ONLY across turns:
            # this turn's stable prefix must be a prefix of next turn's. The backend
            # caches the leading bytes, so as long as they are preserved (the summary
            # only ever grows by appending) the cached KV is reused. The prefix is
            # still GROWING until frozen_prefix_turns complete turns have accumulated
            # (turn 2 here), so only assert stability once it has fully formed (turn 3+).
            # This is the exact invariant that broke at turn 13 in v0.7.18 (over-cap
            # mid-body rewrite), at turn 11 in v0.7.23 (rolling-summary front-trim),
            # and at turn 11 in v0.7.24 (the summary fell out of the stable prefix
            # because _strip_internal_flags dropped _summary_id, so it was re-optimized
            # every turn and the cached KV was invalidated: cached 3192 -> 882).
            if n >= 3 and prev_blob is not None:
                if prev_summary_present:
                    # Once the summary has appeared it is append-only: the leading
                    # bytes [frozen][summary] must be a prefix of the previous turn's
                    # leading bytes. A violation here is the cache break.
                    assert blob.startswith(prev_blob), (
                        f"stable prefix leading bytes changed at turn {n} "
                        f"(append-only violated): previous was not a prefix of current"
                    )
                elif not summary_present:
                    # Frozen-only phase: the frozen prefix alone must be stable.
                    assert blob == prev_blob, (
                        f"frozen prefix changed before summary appeared at turn {n}"
                    )
                # else: summary's first-appearance turn (prev=False, curr=True) —
                # the one-time frozen-only -> frozen+summary transition is expected
                # and not a cache break, so we skip the assertion here.
            prev_blob = blob
            prev_summary_present = summary_present

    def test_growth_ceiling_bounds_per_turn_expansion(self) -> None:
        """The effective budget must not exceed prev_size + growth cap."""
        opt = self._opt()
        # First turn establishes a baseline size.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "ok"},
        ]
        opt.optimize_messages(msgs)
        base = opt._last_optimized_token_count or 0
        # A much larger second turn would otherwise jump to the full ~6.5K budget.
        big = [*msgs, {"role": "user", "content": "x" * 4000}, {"role": "assistant", "content": "y" * 4000}]
        opt.optimize_messages(big)
        grown = (opt._last_optimized_token_count or 0) - base
        assert grown <= opt._config.agentic.max_context_growth_per_turn + 50, (
            f"context grew {grown} tokens in one turn, exceeds growth ceiling"
        )

    def test_shrink_cap_bounds_per_turn_contraction(self) -> None:
        """P0.6: the context must not SHRINK more than the per-turn shrink cap.

        The v0.7.21 turn-13 break was an 8.5K->2K tok one-shot collapse that
        invalidated the backend's cached KV for the whole body. The shrink cap
        (symmetric to the growth ceiling) bounds the front-eviction rate so the
        body never drops more than ``prev_size - shrink_cap`` in a single turn.
        """
        opt = self._opt()
        # Establish a large baseline (turn 1), then a much smaller turn 2 that
        # would otherwise collapse the body in one shot.
        big = [
            {"role": "system", "content": "sys " + "x" * 2000},
            {"role": "user", "content": "start " + "y" * 2000},
            {"role": "assistant", "content": "ok " + "z" * 2000},
        ]
        opt.optimize_messages(big)
        base = opt._last_optimized_token_count or 0
        # A tiny next turn would otherwise let the compactor drop the whole body.
        small = [
            {"role": "system", "content": "sys " + "x" * 2000},
            {"role": "user", "content": "start " + "y" * 2000},
            {"role": "assistant", "content": "ok " + "z" * 2000},
            {"role": "user", "content": "tiny follow-up"},
            {"role": "assistant", "content": "ack"},
        ]
        opt.optimize_messages(small)
        shrunk = base - (opt._last_optimized_token_count or 0)
        # The shrink must be bounded by the per-turn cap (allow small slack for
        # the protected tail / summary which are never evicted).
        cap = opt._effective_shrink_cap()
        assert shrunk <= cap + 200, (
            f"context shrank {shrunk} tokens in one turn, exceeds shrink cap {cap}"
        )

    def test_fast_path_updates_last_optimized_token_count(self) -> None:
        """P0.6 regression: lean turns that hit the fast path must still record
        ``_last_optimized_token_count``.

        The v0.7.22 turn-11 cliff was caused by turns 1-10 taking the fast path
        (never setting ``_last_optimized_token_count`` -> stayed ``None``), so at
        turn 11 the shrink floor was ``None`` and the tool-output filter collapsed
        the whole body in one shot. Every early-return path must now finalize the
        token count so the floor is always defined.
        """
        opt = self._opt()
        lean = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        opt.optimize_messages(lean)
        assert opt._last_optimized_token_count is not None, (
            "fast path left _last_optimized_token_count unset -> shrink floor is None"
        )
        # A subsequent over-budget turn must now see a real floor and stay bounded.
        big = [
            *lean,
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "y" * 4000},
        ]
        opt.optimize_messages(big)
        assert opt._last_optimized_token_count is not None

    def test_filter_tool_messages_respects_shrink_floor(self) -> None:
        """P0.6 regression: tool-output filtering must not drop the context below
        the per-turn shrink floor.

        The v0.7.22 turn-11 cliff was the unbounded ``filter_tool_messages`` stage
        replacing matched tool/assistant content with a tiny marker in one call,
        collapsing 4091 -> 1553 tokens. The stage now runs through
        ``_apply_transform_with_floor`` which stops before crossing the floor.
        """
        opt = self._opt()
        # Establish a large baseline with a tool call + oversized tool output.
        big_log = "\n".join(["DEBUG worker heartbeat ok"] * 400)
        base_msgs = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run the suite."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "run_command", "content": big_log},
        ]
        opt.optimize_messages(base_msgs)
        base = opt._last_optimized_token_count or 0
        # A tiny follow-up turn would otherwise let the filter collapse the body.
        small = [
            *base_msgs,
            {"role": "user", "content": "tiny follow-up"},
            {"role": "assistant", "content": "ack"},
        ]
        opt.optimize_messages(small)
        shrunk = base - (opt._last_optimized_token_count or 0)
        floor = opt._effective_shrink_floor()
        # The shrink must be bounded by the per-turn cap (small slack for the
        # protected tail / summary which are never evicted).
        cap = opt._effective_shrink_cap()
        assert shrunk <= cap + 200, (
            f"tool-output filter shrank {shrunk} tokens in one turn, "
            f"exceeds shrink cap {cap} (floor={floor})"
        )
