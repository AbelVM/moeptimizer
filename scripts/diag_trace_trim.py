"""Trace token_aware_truncator at turn 16."""
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
    num_turns = 16
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
        opt.optimize_messages(messages)

    # Now trace turn 16
    messages.extend(turn_exchanges[16 % len(turn_exchanges)])

    orig_trim_budget = opt._trim_to_budget
    orig_token_trim = opt.token_aware_truncator.trim_messages_to_budget if opt.token_aware_truncator else None
    orig_strip = opt._strip_internal_flags

    def traced_trim_budget(msgs, use_tokens=False):
        before = count_summaries(msgs)
        res = orig_trim_budget(msgs, use_tokens)
        after = count_summaries(res)
        print(f"  trim_to_budget: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
        return res

    def traced_token_trim(msgs, max_tokens):
        before = count_summaries(msgs)
        res = orig_token_trim(msgs, max_tokens)
        after = count_summaries(res)
        print(f"  token_aware_truncator: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
        return res

    def traced_strip(msgs):
        before = count_summaries(msgs)
        res = orig_strip(msgs)
        after = count_summaries(res)
        print(f"  strip_flags: {before}->{after} summaries, n={len(msgs)}->{len(res)}")
        return res

    opt._trim_to_budget = traced_trim_budget
    if opt.token_aware_truncator:
        opt.token_aware_truncator.trim_messages_to_budget = traced_token_trim
    opt._strip_internal_flags = traced_strip

    result = opt.optimize_messages(messages)
    print(f"  FINAL: n_opt={len(result)}, summaries={count_summaries(result)}, lz={opt._live_zone_start}")


if __name__ == "__main__":
    main()
