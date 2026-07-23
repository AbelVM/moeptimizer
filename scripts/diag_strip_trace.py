"""Trace _strip_internal_flags behavior on summary blocks at turn 16."""
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

    # Now at turn 16, run one more optimization with detailed strip tracing
    messages.extend(turn_exchanges[16 % len(turn_exchanges)])

    # Run up to just before strip to get the pre-strip state
    # We'll manually run the pipeline steps
    from moeptimizer.optimizer import AgentContextOptimizer

    # Get the pre-strip state by running with a monkey-patched strip that captures state
    captured = {}
    _original_strip = AgentContextOptimizer._strip_internal_flags

    def traced_strip(self, msgs):
        captured["pre_strip"] = [dict(m) for m in msgs]
        captured["pre_strip_summaries"] = [
            (i, m.get("_summary_id", ""), (m.get("content") or "")[:60])
            for i, m in enumerate(msgs)
            if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER)
        ]
        res = _original_strip(self, msgs)
        captured["post_strip"] = [dict(m) for m in res]
        captured["post_strip_summaries"] = [
            (i, m.get("_summary_id", ""), (m.get("content") or "")[:60])
            for i, m in enumerate(res)
            if (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER)
        ]
        return res

    AgentContextOptimizer._strip_internal_flags = traced_strip

    try:
        result = opt.optimize_messages(messages)
    finally:
        AgentContextOptimizer._strip_internal_flags = _original_strip

    print("Pre-strip summary blocks:")
    for i, sid, content in captured["pre_strip_summaries"]:
        print(f"  [{i}] _summary_id={sid[:8] if sid else '':8s} content={content!r}")

    print(f"\nPost-strip summary blocks: {len(captured['post_strip_summaries'])}")
    for i, sid, content in captured["post_strip_summaries"]:
        print(f"  [{i}] _summary_id={sid[:8] if sid else '':8s} content={content!r}")

    # Show all messages around the summary blocks
    print("\nPre-strip messages around summary blocks:")
    for i, m in enumerate(captured["pre_strip"]):
        content = m.get("content") or ""
        is_summary = content.startswith(ROLLING_SUMMARY_MARKER)
        if is_summary or 20 <= i <= 30:
            role = m.get("role", "?")
            sid = m.get("_summary_id", "")
            print(f"  [{i:2d}] {role:10s} _summary_id={sid[:8] if sid else '':8s} content={content[:50]!r}")

    print("\nPost-strip messages around summary blocks:")
    for i, m in enumerate(captured["post_strip"]):
        content = m.get("content") or ""
        is_summary = content.startswith(ROLLING_SUMMARY_MARKER)
        if is_summary or 20 <= i <= 30:
            role = m.get("role", "?")
            keys = [k for k in m.keys() if k.startswith("_")]
            print(f"  [{i:2d}] {role:10s} internal_keys={keys} content={content[:50]!r}")


if __name__ == "__main__":
    main()
