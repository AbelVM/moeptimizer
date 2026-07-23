"""Detailed diagnostic for the turn-4 prefix break.
Traces each stage of the pipeline to find where the stable prefix changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moeptimizer import ROLLING_SUMMARY_MARKER
from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


def _big_tool(t: int) -> str:
    lines = [f"# scan report for batch {t}"]
    for i in range(30):
        lines.append(
            f"module_{t}_{i}.py: parsed {1000 + i*7} nodes, "
            f"imports=[os, sys, json, pathlib, asyncio, logging, dataclasses], "
            f"defs=[handler_{i}, route_{i}, parse_{i}], "
            f"sha256={hex((t * 2654435761 + i) & 0xFFFFFFFFFFFFFFFF)[2:].zfill(16)}"
        )
    lines.append(f"batch {t} complete: 30 modules indexed, graph built.")
    return "\n".join(lines)


_ASSISTANT = (
    "I'll scan the modules and report. Here is the result of the batch scan. "
    "The refactor touches src/handler.py and src/router.py; I'll keep the public "
    "API stable. Next I'll add a regression test that replays the original payload."
)


def build_conversation(n_turns: int) -> list[dict]:
    msgs: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
    ]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"Run scan batch {t} and refactor."})
        msgs.append(
            {"role": "assistant", "content": _ASSISTANT, "tool_calls": [{"id": f"c{t}"}]}
        )
        msgs.append({"role": "tool", "content": _big_tool(t), "tool_call_id": f"c{t}"})
        msgs.append({"role": "assistant", "content": _ASSISTANT})
    return msgs


def make_optimizer() -> AgentContextOptimizer:
    config = AppConfig()
    config.agentic.quality_profile = "balanced"
    config.v050.static_prefix_kv_enabled = True
    config.v050.token_aware_truncation_enabled = True
    config.v050.hierarchical_summary_enabled = True
    config.v050.cache_stable_mode = True
    config.v050.frozen_prefix_turns = 2
    return AgentContextOptimizer(config)


def run_trace() -> None:
    opt = make_optimizer()
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
    ]
    
    prev_head = None
    for n in range(1, 5):
        print(f"\n=== TURN {n} ===")
        turn_msgs = build_conversation(1)
        messages.extend(turn_msgs[1:])
        
        # Check frozen prefix end before optimization
        frozen_end = opt.context_aligner.frozen_prefix_end(
            messages, opt._config.v050.frozen_prefix_turns
        )
        print(f"  frozen_end (raw) = {frozen_end}")
        
        # Run optimization
        result = opt.optimize_messages(messages)
        
        # Check live zone start after optimization
        lz = opt._live_zone_start
        print(f"  _live_zone_start (after) = {lz}")
        
        # Show the head content
        head = result[:lz]
        print(f"  head messages ({len(head)}):")
        for i, m in enumerate(head):
            role = m.get("role")
            content = m.get("content") or ""
            print(f"    [{i}] {role}: {len(content)} chars, _summary_id={m.get('_summary_id')}")
        
        # Check if head changed
        head_blob = "\n".join(f"{m.get('role')}:{m.get('content')}" for m in head)
        if prev_head is not None:
            if head_blob == prev_head:
                print("  head: STABLE")
            elif head_blob.startswith(prev_head):
                print("  head: APPEND-ONLY")
            else:
                print("  head: *** BREAK ***")
                # Find where the break is
                for i in range(min(len(head), len(prev_head.split('\n')))):
                    if head_blob.split('\n')[i] != prev_head.split('\n')[i]:
                        print(f"    break at line {i}")
                        break
        prev_head = head_blob


if __name__ == "__main__":
    run_trace()