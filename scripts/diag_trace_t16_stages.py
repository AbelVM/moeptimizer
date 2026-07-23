"""Trace summary block through pipeline stages at turn 16 with stateful optimizer."""
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


def count_summaries(msgs):
    return sum(1 for m in msgs if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER))


def main() -> None:
    num_turns = 17
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

        if local_turn == 15:  # Turn 16 - trace it
            print("=== TRACING TURN 16 ===")

            # Monkey-patch key methods
            orig_summarize = opt.hierarchical_summarizer.summarize_turns_cache_stable
            orig_compact = opt.compactor.compact_messages
            orig_proactive = opt._proactive_trim
            orig_sliding = opt._sliding_window_trim
            orig_trim = opt._trim_to_budget
            orig_strip = opt._strip_internal_flags
            orig_boundary = opt._apply_boundary_transforms
            orig_volatile = opt._append_volatile_context

            def traced_summarize(msgs, frozen_end):
                before = count_summaries(msgs)
                res = orig_summarize(msgs, frozen_end)
                after = count_summaries(res)
                print(f"  summarize_turns: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_compact(msgs, min_keep_tokens=None):
                before = count_summaries(msgs)
                res = orig_compact(msgs, min_keep_tokens)
                after = count_summaries(res)
                print(f"  compact: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_proactive(msgs, target, use_tokens=False, shrink_floor=None):
                before = count_summaries(msgs)
                res = orig_proactive(msgs, target, use_tokens, shrink_floor)
                after = count_summaries(res)
                print(f"  proactive_trim: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_sliding(msgs, window_size=None, overlap_size=256, use_tokens=False, shrink_floor=None):
                before = count_summaries(msgs)
                res = orig_sliding(msgs, window_size, overlap_size, use_tokens, shrink_floor)
                after = count_summaries(res)
                print(f"  sliding_window: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_trim(msgs, use_tokens=False):
                before = count_summaries(msgs)
                res = orig_trim(msgs, use_tokens)
                after = count_summaries(res)
                print(f"  trim_to_budget: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_strip(msgs):
                before = count_summaries(msgs)
                res = orig_strip(msgs)
                after = count_summaries(res)
                print(f"  strip_flags: {before}->{after} summaries")
                return res

            def traced_boundary(msgs, live_zone_start):
                before = count_summaries(msgs)
                res = orig_boundary(msgs, live_zone_start)
                after = count_summaries(res)
                print(f"  boundary_transforms: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            def traced_volatile(msgs, anchor, rag, warnings, threshold):
                before = count_summaries(msgs)
                res = orig_volatile(msgs, anchor, rag, warnings, threshold)
                after = count_summaries(res)
                print(f"  volatile_context: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
                return res

            opt.hierarchical_summarizer.summarize_turns_cache_stable = traced_summarize
            opt.compactor.compact_messages = traced_compact
            opt._proactive_trim = traced_proactive
            opt._sliding_window_trim = traced_sliding
            opt._trim_to_budget = traced_trim
            opt._strip_internal_flags = traced_strip
            opt._apply_boundary_transforms = traced_boundary
            opt._append_volatile_context = traced_volatile

            result = opt.optimize_messages(messages)
            print(f"  FINAL: n_opt={len(result)}, summaries={count_summaries(result)}, lz={opt._live_zone_start}")
        else:
            result = opt.optimize_messages(messages)

        lz = opt._live_zone_start
        tok = opt.token_counter.count_messages(result)
        has_summary = any(
            (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER) for m in result
        )
        summary_count = count_summaries(result)
        print(
            f"turn {local_turn + 1:2d}: n_opt={len(result):4d} lz={lz:3d} "
            f"tok={tok:5d} summary={has_summary}({summary_count})"
        )


if __name__ == "__main__":
    main()
