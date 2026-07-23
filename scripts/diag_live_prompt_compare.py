"""Diagnostic: compare actual optimized prompt text between turns via the live proxy."""
from __future__ import annotations

import sys
import json
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as bench


def main() -> None:
    num_turns = 5
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    messages = [{"role": "system", "content": proxy_system_prompt}]
    prev_opt_text = None

    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)

        body = {
            "model": bench.MODEL_ID,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
            "max_tokens": 256,
            "_session_id": f"diag-prompt-compare-{local_turn}",
        }

        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            print(f"turn {local_turn + 1:2d}: HTTP ERROR {e.code}")
            print(f"   body: {e.read().decode()[:500]}")
            continue

        opt_text = resp_headers.get("X-MOEPT-Optimized-Prompt-Text", "")
        if opt_text:
            opt_text = opt_text.replace("\\n", "\n")

        prefix_hit = resp_headers.get("X-Prefix-Cache-Hit-Tokens", "0")

        if prev_opt_text is None:
            status = "(first)"
        elif opt_text == prev_opt_text:
            status = "IDENTICAL"
        elif opt_text.startswith(prev_opt_text):
            status = "APPEND-ONLY"
        else:
            status = "*** BREAK ***"
            for i in range(min(len(opt_text), len(prev_opt_text))):
                if opt_text[i] != prev_opt_text[i]:
                    print(f"   first diff at char {i}:")
                    print(f"   prev: ...{prev_opt_text[max(0,i-80):i+80]!r}")
                    print(f"   cur : ...{opt_text[max(0,i-80):i+80]!r}")
                    break
            else:
                print(f"   (prev len={len(prev_opt_text)} cur len={len(opt_text)})")

        print(f"turn {local_turn + 1:2d}: opt_len={len(opt_text):6d} prefix_hit={prefix_hit:>6} {status}")
        prev_opt_text = opt_text


if __name__ == "__main__":
    main()
