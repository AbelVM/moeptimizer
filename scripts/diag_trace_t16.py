"""Detailed trace of summary block through pipeline at turn 16."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moeptimizer import ROLLING_SUMMARY_MARKER
from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer

import benchmark as bench  # noqa: E402


class _FakeCaps:
    max_context_window = 262144
    remote_tokenize = False


class _FakeProbe:
    def cached(self):
        return _FakeCaps()

    def get(self):
        return _FakeCaps()


def make_optimizer() -> AgentContextOptimizer:
    config = AppConfig()
    config.agentic.quality_profile = "balanced"
    config.v050.static_prefix_kv_enabled = True
    config.v050.token_aware_truncation_enabled = True
    config.v050.hierarchical_summary_enabled = True
    config.v050.delta_encoding_enabled = True
    config.agentic.dynamic_budget_enabled = True
    opt = AgentContextOptimizer(config)
    opt._capability_probe = _FakeProbe()
    return opt


def main() -> None:
    num_turns = 17  # Stop at turn 16 (0-indexed: 15)
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges: list[list[dict]] = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    opt = make_optimizer()
    messages: list[dict] = [{"role": "system", "content": proxy_system_prompt}]

    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)

        # Trace summary block presence before optimization
        has_summary_before = any(
            (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER) for m in messages
        )
        summary_count_before = sum(
            1 for m in messages if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER)
        )

        result = opt.optimize_messages(messages)

        has_summary_after = any(
            (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER) for m in result
        )
        summary_count_after = sum(
            1 for m in result if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER)
        )

        lz = opt._live_zone_start
        tok = opt.token_counter.count_messages(result)

        print(
            f"turn {local_turn + 1:2d}: n_raw={len(messages):3d} n_opt={len(result):4d} "
            f"lz={lz:3d} tok={tok:5d} "
            f"summary_before={has_summary_before}({summary_count_before}) "
            f"summary_after={has_summary_after}({summary_count_after})"
        )

        if local_turn == 15:  # Turn 16 - the cliff
            print("  === DETAILED TRACE AT TURN 16 ===")
            # Trace through pipeline stages manually
            from moeptimizer.optimizer import AgentContextOptimizer

            # Re-run with tracing
            opt2 = make_optimizer()
            opt2._capability_probe = _FakeProbe()

            # Monkey-patch to trace
            orig_summarize = opt2.hierarchical_summarizer.summarize_turns_cache_stable
            orig_compact = opt2.compactor.compact_messages
            orig_proactive = opt2._proactive_trim
            orig_sliding = opt2._sliding_window_trim
            orig_trim = opt2._trim_to_budget
            orig_strip = opt2._strip_internal_flags

            def traced_summarize(msgs, frozen_end):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_summarize(msgs, frozen_end)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    summarize_turns: {before} -> {after} summary blocks")
                return res

            def traced_compact(msgs, min_keep_tokens=None):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_compact(msgs, min_keep_tokens)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    compact: {before} -> {after} summary blocks, n={len(msgs)}->{len(res)}")
                return res

            def traced_proactive(msgs, target, use_tokens=False, shrink_floor=None):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_proactive(msgs, target, use_tokens, shrink_floor)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    proactive_trim: {before} -> {after} summary blocks, n={len(msgs)}->{len(res)}")
                return res

            def traced_sliding(msgs, window_size=None, overlap_size=256, use_tokens=False, shrink_floor=None):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_sliding(msgs, window_size, overlap_size, use_tokens, shrink_floor)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    sliding_window: {before} -> {after} summary blocks, n={len(msgs)}->{len(res)}")
                return res

            def traced_trim(msgs, use_tokens=False):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_trim(msgs, use_tokens)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    trim_to_budget: {before} -> {after} summary blocks, n={len(msgs)}->{len(res)}")
                return res

            def traced_strip(msgs):
                before = sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                res = orig_strip(msgs)
                after = sum(1 for m in res if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))
                print(f"    strip_flags: {before} -> {after} summary blocks")
                return res

            opt2.hierarchical_summarizer.summarize_turns_cache_stable = traced_summarize
            opt2.compactor.compact_messages = traced_compact
            opt2._proactive_trim = traced_proactive
            opt2._sliding_window_trim = traced_sliding
            opt2._trim_to_budget = traced_trim
            opt2._strip_internal_flags = traced_strip

            # Build messages up to turn 16
            msgs16: list[dict] = [{"role": "system", "content": proxy_system_prompt}]
            for t in range(16):
                msgs16.extend(turn_exchanges[t % len(turn_exchanges)])

            result16 = opt2.optimize_messages(msgs16)
            print(f"  Final: n_opt={len(result16)}, summary={any((m.get('content') or '').startswith(ROLLING_SUMMARY_MARKER) for m in result16)}")


if __name__ == "__main__":
    main()
