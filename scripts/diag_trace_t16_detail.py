"""Trace message list at turn 16 to find summary block position."""
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
        result = opt.optimize_messages(messages)

        if local_turn == 15:  # Turn 16
            print("=== TURN 16 OPTIMIZED MESSAGES ===")
            lz = opt._live_zone_start
            frozen_end = opt.context_aligner.frozen_prefix_end(result, opt._config.v050.frozen_prefix_turns)
            stable_end = opt._stable_prefix_end(result)

            print(f"frozen_end={frozen_end}, stable_end={stable_end}, lz={lz}")
            print(f"Total messages: {len(result)}")
            for i, m in enumerate(result):
                role = m.get("role", "?")
                content = m.get("content") or ""
                is_summary = opt._is_summary_block(m)
                summary_id = m.get("_summary_id", "")
                prefix = "S" if is_summary else " "
                zone = "STABLE" if i < lz else "LIVE "
                print(f"  {prefix} [{i:2d}] {zone} {role:10s} _summary_id={summary_id[:8] if summary_id else '':8s} content_start={content[:40]!r}")

            print()
            print("=== RAW MESSAGES AT TURN 16 ===")
            print(f"Total raw messages: {len(messages)}")
            for i, m in enumerate(messages):
                role = m.get("role", "?")
                content = m.get("content") or ""
                print(f"  [{i:2d}] {role:10s} content_start={content[:40]!r}")

            print()
            print(f"_last_raw_prefix length: {len(opt._last_raw_prefix)}")
            print(f"_last_stable_prefix length: {len(opt._last_stable_prefix)}")
            print(f"_stable_prefix_optimized length: {len(opt._stable_prefix_optimized) if opt._stable_prefix_optimized else 0}")


if __name__ == "__main__":
    main()
