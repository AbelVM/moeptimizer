"""Diagnostic: compare actual optimized prompt text between turns via the live proxy."""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as bench
from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


def make_optimizer() -> AgentContextOptimizer:
    config = AppConfig()
    config.agentic.quality_profile = "balanced"
    config.v050.static_prefix_kv_enabled = True
    config.v050.token_aware_truncation_enabled = True
    config.v050.hierarchical_summary_enabled = True
    config.v050.delta_encoding_enabled = True
    config.agentic.dynamic_budget_enabled = True
    return AgentContextOptimizer(config)


def serialize_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def main() -> None:
    num_turns = 15
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    opt = make_optimizer()
    messages = [{"role": "system", "content": proxy_system_prompt}]
    prev_opt_blob = None

    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)
        result = opt.optimize_messages(messages)
        blob = serialize_prompt(result)

        if prev_opt_blob is None:
            status = "(first)"
        elif blob == prev_opt_blob:
            status = "IDENTICAL"
        elif blob.startswith(prev_opt_blob):
            status = "APPEND-ONLY"
        else:
            status = "*** BREAK ***"
            # Find first diff
            for i in range(min(len(blob), len(prev_opt_blob))):
                if blob[i] != prev_opt_blob[i]:
                    print(f"   first diff at char {i}:")
                    print(f"   prev: ...{prev_opt_blob[max(0,i-80):i+80]!r}")
                    print(f"   cur : ...{blob[max(0,i-80):i+80]!r}")
                    break
            else:
                print(f"   (prev len={len(prev_opt_blob)} cur len={len(blob)})")

        print(f"turn {local_turn + 1:2d}: opt_len={len(blob):6d} {status}")
        prev_opt_blob = blob


if __name__ == "__main__":
    main()
