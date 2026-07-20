"""Faithful replay of the opencode benchmark through the optimizer.

Mirrors scripts/benchmark.py's _collect_proxy_conversation message-building
exactly: proxy_system_prompt (with cache-bust marker), drift probe injected
into the first user turn, and cumulative turn_exchanges accumulation. Dumps the
serialized optimized prompt bytes per turn and flags where the stable prefix
diverges from the previous turn (the condition that breaks backend prefix-cache
reuse, which the live log shows as cached=724 at turns 4 and 10).
"""
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
    max_context_window = 262144  # live Lemonade/Qwen3.6-35B window
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
    # Reproduce the LIVE dynamic budget: the running proxy sees a 262K window,
    # so the effective cap is window*budget_window_fraction (~6.5K) grown by the
    # per-turn growth ceiling — NOT the static 12000 my offline replay would use
    # with no backend. This is the divergence that breaks the cache live.
    config.agentic.dynamic_budget_enabled = True
    opt = AgentContextOptimizer(config)
    opt._capability_probe = _FakeProbe()
    return opt


def serialize_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def main() -> None:
    num_turns = 30
    # Mirror benchmark.py: build base_tasks, inject drift probe, split into
    # turn_exchanges (agentic exchanges).
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges: list[list[dict]] = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    opt = make_optimizer()
    messages: list[dict] = [{"role": "system", "content": proxy_system_prompt}]
    prev_blob = None
    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)
        result = opt.optimize_messages(messages)
        lz = opt._live_zone_start
        head = result[:lz] if lz > 0 else result
        blob = serialize_prompt(head)
        tok = opt.token_counter.count_messages(head)
        has_summary = any(
            (m.get("content") or "").startswith(ROLLING_SUMMARY_MARKER) for m in head
        )
        if prev_blob is None:
            status = "(first)"
        elif blob == prev_blob:
            status = "STABLE"
        elif blob.startswith(prev_blob):
            status = "APPEND-ONLY"
        else:
            status = "*** BREAK ***"
        print(
            f"turn {local_turn + 1:2d}: n_opt={len(result):4d} lz={lz:3d} "
            f"head_tok={tok:5d} summary={has_summary} {status}"
        )
        if status == "*** BREAK ***":
            for i in range(min(len(blob), len(prev_blob))):
                if blob[i] != prev_blob[i]:
                    print(f"   first diff at char {i}:")
                    print(f"   prev: ...{prev_blob[max(0,i-80):i+80]!r}")
                    print(f"   cur : ...{blob[max(0,i-80):i+80]!r}")
                    break
            else:
                print(
                    f"   (prev len={len(prev_blob)} cur len={len(blob)}; "
                    f"prev is not a prefix of cur)"
                )
        prev_blob = blob


if __name__ == "__main__":
    main()
