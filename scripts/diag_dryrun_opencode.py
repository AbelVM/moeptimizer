"""Replay the opencode scenario through the LIVE proxy's dry-run endpoint and
dump the optimized prompt bytes per turn, to find the byte-level divergence
that breaks the backend prefix cache (cached=724 at turns 4 and 10 in the live
log). Dry-run uses the real running optimizer (with real backend calibration)
but does NOT call the backend, so it's fast and safe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os  # noqa: E402
import uuid  # noqa: E402

import requests  # noqa: E402

import benchmark as bench  # noqa: E402
from moeptimizer.output_shaper import OutputShaper  # noqa: E402

SHAPER = OutputShaper(enabled=True)

PROXY = "http://127.0.0.1:8080/v1/chat/completions"
# Mirror the live benchmark: unique session per run, streaming on.
USE_UNIQUE_SESSION = os.environ.get("DIAG_UNIQUE_SESSION", "1") == "1"
STREAM = os.environ.get("DIAG_STREAM", "1") == "1"


def serialize(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def main() -> None:
    num_turns = 30
    base_tasks = bench._inject_drift_probe(bench._OPENCODE_SCENARIO_TASKS, num_turns)
    turn_exchanges = [t for t in base_tasks if isinstance(t, list)]
    proxy_marker = "P{diag dryrun proxy session}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT
    tools = bench.OPENCODE_TOOLS

    messages: list[dict] = [{"role": "system", "content": proxy_system_prompt}]
    prev_blob = None
    for local_turn in range(num_turns):
        messages.extend(turn_exchanges[local_turn % len(turn_exchanges)])
        session = uuid.uuid4().hex if USE_UNIQUE_SESSION else "diag-persistent-session"
        resp = requests.post(
            PROXY,
            json={
                "model": "Qwen3.6-35B-A3B-MTP-GGUF",
                "messages": messages,
                "tools": tools,
                "max_tokens": int(os.environ.get("DIAG_MAX_TOKENS", "16")),
                "stream": STREAM,
                "session_id": session,
            },
            headers={"X-MOEPT-Dry-Run": "true"},
            timeout=60,
        )
        data = resp.json()
        opt_msgs = data.get("optimized_messages", [])
        # Mirror app.py: apply output_shaper.shape_request (appends terse
        # instruction to system prompt + clamps max_tokens/reasoning_effort).
        shaped = SHAPER.shape_request({"messages": opt_msgs, "max_tokens": int(__import__("os").environ.get("DIAG_MAX_TOKENS", "16"))})
        opt_msgs = shaped["messages"]
        if local_turn + 1 in (11, 12):
            import json as _json

            with open(f"/tmp/diag_opt_t{local_turn + 1}.json", "w") as _f:
                _json.dump(opt_msgs, _f)
            seq = " ".join(
                f"{m.get('role')}"
                + ("[S]" if m.get("_summary_id") or m.get("_rolling_summary") else "")
                + f":{str(m.get('content'))[:24]!r}"
                for m in opt_msgs
            )
            print(f"   [turn {local_turn + 1}] ROLES {seq}")
            # Capture per-stage diagnostic dumps for this turn
            import glob
            for src in glob.glob("/tmp/diag_after-*.json"):
                import shutil
                dst = f"/tmp/diag_{local_turn + 1}_{src.split('/')[-1]}"
                shutil.copy2(src, dst)
        blob = serialize(opt_msgs)
        if local_turn + 1 in (3, 4, 10):
            sys_msg = next((m for m in opt_msgs if m.get("role") == "system"), None)
            c = sys_msg["content"]
            has_terse = "Be concise" in c
            print(f"   [turn {local_turn + 1}] system len={len(c)} has_terse={has_terse}")
        if prev_blob is None:
            status = "(first)"
        elif blob == prev_blob:
            status = "STABLE"
        elif blob.startswith(prev_blob) or prev_blob.startswith(blob):
            status = "APPEND-ONLY"
        else:
            status = "*** BREAK ***"
        print(f"turn {local_turn + 1:2d}: n_opt={len(opt_msgs):3d} {status}")
        if status == "*** BREAK ***":
            for i in range(min(len(blob), len(prev_blob))):
                if blob[i] != prev_blob[i]:
                    print(f"   first diff at char {i}:")
                    print(f"   prev: ...{prev_blob[max(0,i-100):i+100]!r}")
                    print(f"   cur : ...{blob[max(0,i-100):i+100]!r}")
                    break
            else:
                print(f"   (prev len={len(prev_blob)} cur len={len(blob)})")
        prev_blob = blob


if __name__ == "__main__":
    main()
