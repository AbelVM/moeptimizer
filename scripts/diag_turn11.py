"""Faithful reproduction of the live turn-11 prefix-cache break.

Builds a conversation with LARGE tool outputs (so the context crosses the
proactive threshold and the rolling summary first appears around turn 11,
matching the live benchmark). Traces the serialized leading bytes
[frozen prefix + summary block] through the FULL optimizer pipeline each turn
and reports when they change (a cache break).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moeptimizer import ROLLING_SUMMARY_MARKER
from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


# Large, realistic tool output that does NOT match any ToolOutputFilter rule
# (so it survives compression and accumulates, pushing the context over the
# proactive threshold and triggering the rolling summary around turn 11, like
# the live benchmark). Varied, non-repeating lines avoid the repeated_lines rule.
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


def _tool_for(t: int) -> str:
    return _big_tool(t)


def build_conversation(n_turns: int) -> list[dict]:
    msgs: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
    ]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"Run scan batch {t} and refactor."})
        msgs.append(
            {"role": "assistant", "content": _ASSISTANT, "tool_calls": [{"id": f"c{t}"}]}
        )
        msgs.append({"role": "tool", "content": _tool_for(t), "tool_call_id": f"c{t}"})
        msgs.append({"role": "assistant", "content": _ASSISTANT})
    return msgs


def make_optimizer() -> AgentContextOptimizer:
    """Build a fresh optimizer. The optimizer is STATEFUL (it caches the stable
    prefix / rolling summary across turns), so each independent conversation must
    get its own instance — reusing one across two conversations produces
    meaningless deltas and was the source of a previous false 'BREAK' reading.
    """
    config = AppConfig()
    config.agentic.quality_profile = "balanced"
    config.v050.static_prefix_kv_enabled = True
    config.v050.token_aware_truncation_enabled = True
    config.v050.hierarchical_summary_enabled = True
    config.v050.delta_encoding_enabled = True
    return AgentContextOptimizer(config)


def leading_bytes(opt: AgentContextOptimizer, result: list[dict]) -> tuple[str, int, bool]:
    # The stable prefix is everything before the live zone (opt._live_zone_start),
    # which the optimizer computes as the byte-stable boundary (frozen prefix +
    # append-only rolling-summary block). The backend caches the LEADING bytes of
    # the prompt, so the cache-stability invariant is APPEND-ONLY: this turn's
    # stable-prefix blob must be a prefix of next turn's (the summary only ever
    # grows by appending). We serialize the leading bytes into a single string and
    # compare with str.startswith, exactly like the unit test
    # test_frozen_prefix_stable_across_30_turns.
    lz = opt._live_zone_start
    head = result[:lz] if lz > 0 else result
    summary_present = False
    for m in head:
        c = m.get("content") or ""
        if m.get("_summary_id") or m.get("_rolling_summary") or c.startswith(
            ROLLING_SUMMARY_MARKER
        ):
            summary_present = True
    blob = "\n".join(f"{m.get('role')}:{m.get('content')}" for m in head)
    tok = opt.token_counter.count_messages(head)
    return blob, tok, summary_present


def run_trace(n_turns: int = 20) -> None:
    # Fresh optimizer per trace: the optimizer is stateful across turns, so a
    # brand-new instance is required for a clean, reproducible conversation.
    opt = make_optimizer()
    prev = None
    prev_tok = None
    # Build conversation CUMULATIVELY (like a real client does): each turn adds
    # to the same message list, rather than rebuilding from scratch. The optimizer
    # is stateful and expects cumulative message history; rebuilding from scratch
    # each turn causes the optimizer's internal state (_last_raw_prefix,
    # _live_zone_start) to mismatch the incoming messages, producing false BREAK
    # readings.
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
    ]
    for n in range(1, n_turns + 1):
        # Add one turn's worth of messages (user, assistant, tool, assistant)
        turn_msgs = build_conversation(1)
        # Skip the system message (already added) and append the rest
        messages.extend(turn_msgs[1:])
        result = opt.optimize_messages(messages)
        blob, tok, has_sum = leading_bytes(opt, result)
        if prev is None:
            status = "(first)"
        elif blob == prev:
            status = "STABLE"
        elif prev is not None and blob.startswith(prev):
            # Append-only growth: the leading bytes are preserved and new folded
            # text was appended (the summary legitimately grows each turn). This is
            # the correct cache-stable invariant — the backend reuses the cached KV
            # for the leading bytes. NOT a break.
            status = "APPEND-ONLY"
        else:
            status = "*** BREAK ***"
        # Real cache-break signal (matches the live log: cached 3192 -> 882): the
        # stable-prefix token size collapses versus the previous turn. Append-only
        # growth raises tok; a collapse means the leading bytes were rewritten and
        # the backend's cached KV was invalidated.
        if prev_tok is not None and tok < prev_tok - 50:
            status = "*** PREFIX BREAK ***"
        print(
            f"turn {n:2d}: n_opt={len(result):4d} head_tok={tok:5d} "
            f"summary={has_sum} {status}"
        )
        prev = blob
        prev_tok = tok


if __name__ == "__main__":
    run_trace()
