"""Trace where summary blocks disappear in the full optimizer pipeline."""
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


def count_summaries(messages):
    return sum(
        1 for m in messages
        if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER)
    )


def make_optimizer():
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


def print_boundary(msgs, label):
    n = count_summaries(msgs)
    # Find first summary index
    first_summary = -1
    for i, m in enumerate(msgs):
        if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER):
            first_summary = i
            break
    print(f"   {label}: len={len(msgs)} summaries={n} first_summary_idx={first_summary}")


def main():
    num_turns = 17
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    opt = make_optimizer()
    messages = [{"role": "system", "content": proxy_system_prompt}]

    # Trace summary step
    orig_summary = opt.hierarchical_summarizer.summarize_turns_cache_stable
    def traced_summary(msgs, frozen_end):
        result = orig_summary(msgs, frozen_end)
        print_boundary(result, "after-summary")
        return result
    opt.hierarchical_summarizer.summarize_turns_cache_stable = traced_summary

    # Trace compactor
    orig_compact = opt.compactor.compact_messages
    def traced_compact(msgs, min_keep_tokens=None):
        result = orig_compact(msgs, min_keep_tokens)
        print_boundary(result, "after-compactor")
        return result
    opt.compactor.compact_messages = traced_compact

    # Trace proactive trim
    orig_proactive = opt._proactive_trim
    def traced_proactive(msgs, target, use_tokens=False, shrink_floor=None):
        result = orig_proactive(msgs, target, use_tokens, shrink_floor)
        print_boundary(result, "after-proactive")
        return result
    opt._proactive_trim = traced_proactive

    # Trace sliding window
    orig_sliding = opt._sliding_window_trim
    def traced_sliding(msgs, window_size=None, overlap_size=256, use_tokens=False, shrink_floor=None):
        result = orig_sliding(msgs, window_size, overlap_size, use_tokens, shrink_floor)
        print_boundary(result, "after-sliding")
        return result
    opt._sliding_window_trim = traced_sliding

    # Trace trim_to_budget
    orig_trim = opt._trim_to_budget
    def traced_trim(msgs, use_tokens=False):
        result = orig_trim(msgs, use_tokens)
        print_boundary(result, "after-trim")
        return result
    opt._trim_to_budget = traced_trim

    # Trace token_aware_truncator
    orig_truncator = opt.token_aware_truncator.trim_messages_to_budget
    def traced_truncator(msgs, max_tokens):
        result = orig_truncator(msgs, max_tokens)
        print_boundary(result, "after-truncator")
        return result
    opt.token_aware_truncator.trim_messages_to_budget = traced_truncator

    # Trace strip_internal_flags
    orig_strip = opt._strip_internal_flags
    def traced_strip(msgs):
        result = orig_strip(msgs)
        print_boundary(result, "after-strip")
        return result
    opt._strip_internal_flags = traced_strip

    # Trace _append_volatile_context
    orig_append = opt._append_volatile_context
    def traced_append(msgs, anchor, rag, warnings, threshold):
        result = orig_append(msgs, anchor, rag, warnings, threshold)
        print_boundary(result, "after-append")
        return result
    opt._append_volatile_context = traced_append

    # Trace freeze_static_prefix
    orig_freeze = opt.context_aligner.freeze_static_prefix
    def traced_freeze(msgs, original_msgs, frozen_prefix_turns=0):
        result = orig_freeze(msgs, original_msgs, frozen_prefix_turns)
        print_boundary(result, "after-freeze")
        return result
    opt.context_aligner.freeze_static_prefix = traced_freeze

    # Trace _update_stable_prefix
    orig_update = opt._update_stable_prefix
    def traced_update(optimized, live_zone_start=0, raw_messages=None):
        print_boundary(optimized, "before-update")
        orig_update(optimized, live_zone_start, raw_messages)
        print(f"   -> lz={opt._live_zone_start}")
    opt._update_stable_prefix = traced_update

    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)
        result = opt.optimize_messages(messages)
        lz = opt._live_zone_start
        head = result[:lz] if lz > 0 else result
        has_summary = any(
            (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER) for m in head
        )
        print(
            f"turn {local_turn + 1:2d}: n_opt={len(result):4d} lz={lz:3d} "
            f"summary={has_summary}"
        )


if __name__ == "__main__":
    main()
