"""Replay the opencode scenario through an owned, offline proxy dry-run endpoint.

The child runs the real optimizer and HTTP proxy path, but its backend URL is
deliberately unreachable and capability probing is disabled. This keeps the
check focused on local prompt optimization, prefix stability, and savings.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests

# Running this file directly puts ``benchmark/`` first on sys.path, where
# benchmark.py shadows the package. Keep the documented direct invocation
# pointed at the repository package.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark import benchmark as bench  # noqa: E402
from moeptimizer.output_shaper import OutputShaper  # noqa: E402

SHAPER = OutputShaper(enabled=True)

PROXY = ""
_PROXY_PROCESS: subprocess.Popen | None = None


def _proxy_is_running(port: int, timeout: float = 3.0) -> bool:
    """Check if the proxy is already listening on *port*."""
    try:
        import urllib.request

        url = f"http://127.0.0.1:{port}/v1/health"
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


def _start_proxy(port: int = 8080, wait: float = 60.0) -> subprocess.Popen | None:
    """Start the moeptimizer proxy as a background process and wait for it to be ready.

    Returns the Popen object on success, or *None* if the proxy was already
    running or failed to start.
    """
    global _PROXY_PROCESS

    if _proxy_is_running(port):
        print(f"  Proxy already running on port {port}", file=sys.stderr)
        return None

    print(f"  Starting moeptimizer proxy on port {port} ...")
    env = os.environ.copy()
    env["MOEPT_PORT"] = str(port)
    # Isolation contract: the child must be able to complete with Lemonade
    # unreachable. The endpoint still runs the real proxy optimizer and dry-run
    # response path; it simply cannot probe or call a backend.
    env["MOEPT_SERVER__URL"] = "http://127.0.0.1:1/api/v1"
    env["MOEPT_V050__CAPABILITY_AUTODETECT"] = "false"
    env["MOEPT_V050__REMOTE_TOKENIZE_ENABLED"] = "false"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "moeptimizer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        _PROXY_PROCESS = proc
    except OSError as e:
        print(f"  ERROR: could not start proxy: {e}", file=sys.stderr)
        return None

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _proxy_is_running(port):
            print(f"  Proxy ready on port {port}")
            return proc
        time.sleep(0.5)

    try:
        stdout, _ = proc.communicate(timeout=2)
    except Exception:
        proc.kill()
        stdout, _ = proc.communicate()
    print(f"  ERROR: proxy failed to start within {wait}s (exit={proc.returncode})", file=sys.stderr)
    if stdout:
        for line in stdout.decode("utf-8", errors="replace").strip().splitlines()[-10:]:
            print(f"    | {line}", file=sys.stderr)
    _PROXY_PROCESS = None
    return None


def _stop_proxy() -> None:
    """Stop the proxy if we started it."""
    global _PROXY_PROCESS
    proc = _PROXY_PROCESS
    _PROXY_PROCESS = None
    if proc is not None and proc.poll() is None:
        print("  Stopping proxy ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _clear_pycache(root: str = ".") -> None:
    """Remove __pycache__ directories under *root*."""
    cleared = 0
    for dirpath, dirnames, _filenames in os.walk(root, topdown=False):
        if "__pycache__" in dirnames:
            shutil.rmtree(os.path.join(dirpath, "__pycache__"))
            cleared += 1
    if cleared:
        print(f"  Cleared {cleared} __pycache__ directory(ies)")


def serialize(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _role_sequence(messages: list[dict[str, Any]]) -> list[str]:
    """Return a compact role sequence with summary markers."""
    roles: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        if m.get("_summary_id") or m.get("_rolling_summary"):
            role += "[S]"
        roles.append(role)
    return roles


def _content_lengths(messages: list[dict[str, Any]]) -> list[int]:
    """Return content lengths for each message."""
    return [len(m.get("content") or "") for m in messages]


def _diff_messages(prev: list[dict[str, Any]], cur: list[dict[str, Any]]) -> list[str]:
    """Return human-readable lines describing message-level changes."""
    lines: list[str] = []
    prev_roles = [m.get("role", "?") for m in prev]
    cur_roles = [m.get("role", "?") for m in cur]
    if prev_roles != cur_roles:
        lines.append(f"   roles: {prev_roles} -> {cur_roles}")
    if len(prev) != len(cur):
        lines.append(f"   count: {len(prev)} -> {len(cur)}")
    for i, (p, c) in enumerate(zip(prev, cur, strict=False)):
        if p.get("content") != c.get("content"):
            pc = (p.get("content") or "")[:80]
            cc = (c.get("content") or "")[:80]
            plen = len(p.get("content") or "")
            clen = len(c.get("content") or "")
            lines.append(f"   msg[{i}] {p.get('role')} content changed ({plen}->{clen} chars): {pc!r} -> {cc!r}")
    return lines


def _byte_diff(prev_blob: str, cur_blob: str) -> dict[str, Any]:
    """Return detailed byte-level diff information."""
    limit = min(len(prev_blob), len(cur_blob))
    first_diff = -1
    for i in range(limit):
        if prev_blob[i] != cur_blob[i]:
            first_diff = i
            break

    result: dict[str, Any] = {
        "prev_len": len(prev_blob),
        "cur_len": len(cur_blob),
        "first_diff_char": first_diff,
    }

    if first_diff >= 0:
        result["prev_context"] = prev_blob[max(0, first_diff - 80):first_diff + 80]
        result["cur_context"] = cur_blob[max(0, first_diff - 80):first_diff + 80]
        result["prev_char"] = repr(prev_blob[first_diff])
        result["cur_char"] = repr(cur_blob[first_diff])

        # Find the common prefix and suffix lengths
        prefix_len = first_diff
        suffix_len = 0
        for j in range(1, min(len(prev_blob), len(cur_blob)) - first_diff + 1):
            if prev_blob[-j] != cur_blob[-j]:
                break
            suffix_len = j
        result["common_prefix_len"] = prefix_len
        result["common_suffix_len"] = suffix_len

        # Show the divergent region
        div_start = max(0, first_diff - 40)
        div_end_prev = min(len(prev_blob), first_diff + 120)
        div_end_cur = min(len(cur_blob), first_diff + 120)
        result["prev_divergent"] = prev_blob[div_start:div_end_prev]
        result["cur_divergent"] = cur_blob[div_start:div_end_cur]
    else:
        result["common_prefix_len"] = limit
        result["note"] = "one is prefix of the other"

    return result


def _quality_issues(
    raw_tokens: int | None,
    optimized_tokens: int | None,
    previous_raw_tokens: int | None,
    previous_optimized_tokens: int | None,
) -> list[str]:
    """Flag prompt inflation and abrupt savings loss in one turn."""
    if raw_tokens is None or optimized_tokens is None:
        return []
    issues: list[str] = []
    saved = raw_tokens - optimized_tokens
    if saved < 0:
        issues.append("INFLATION")
    elif raw_tokens >= 4000 and saved == 0:
        issues.append("NO SAVINGS")
    if previous_raw_tokens and previous_optimized_tokens:
        previous_saved = previous_raw_tokens - previous_optimized_tokens
        raw_growth = raw_tokens / previous_raw_tokens
        optimized_growth = optimized_tokens / previous_optimized_tokens
        if previous_saved >= 0 and saved < 0:
            issues.append("SAVINGS CLIFF")
        elif raw_growth <= 1.2 and optimized_growth >= 1.35:
            issues.append("OPTIMIZED CLIFF")
    return issues


def _dump_messages(msgs: list[dict[str, Any]], label: str = "messages") -> str:
    """Return a formatted string showing role, content length, and summary markers."""
    lines = [f"  {label}:"]
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content") or ""
        markers = []
        if m.get("_summary_id"):
            markers.append(f"_summary_id={m['_summary_id']}")
        if m.get("_rolling_summary"):
            markers.append("_rolling_summary")
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        lines.append(f"    [{i}] {role}{marker_str}: {len(content)} chars")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay opencode scenario through proxy dry-run to detect local prefix breaks."
    )
    parser.add_argument("--turns", type=int, default=30, help="Number of turns to replay (default: 30)")
    parser.add_argument("--profile", type=str, default="balanced",
                        choices=["quality", "balanced", "aggressive"],
                        help="Benchmark context profile (default: balanced)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override the benchmark character budget")
    parser.add_argument("--port", type=int, default=18080,
                        help="Dedicated local proxy port (default: 18080)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens per request")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming in dry-run requests")
    parser.add_argument("--unique-session", action="store_true", default=True,
                        help="Use a unique session per turn (default: True)")
    parser.add_argument("--persistent-session", action="store_true",
                        help="Use a single persistent session across all turns")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-turn details even when stable")
    parser.add_argument("--dump-turn", type=int, nargs="+", default=[],
                        help="Dump full message content for specified turn numbers")
    parser.add_argument("--dump-all", action="store_true",
                        help="Dump full message content for every turn")
    parser.add_argument("--dump-dir", type=str, default="/tmp/diag_dryrun",
                        help="Directory for dump files (default: /tmp/diag_dryrun)")
    parser.add_argument("--cliff-only", action="store_true",
                        help="Only show turns around known cliff regions (11-15, 28-30)")
    parser.add_argument("--reuse-threshold", type=float, default=0.8,
                            help="Common-prefix reuse ratio at/above which a non-append-only turn "
                                "counts as locally prefix-stable (REUSED) instead of a BREAK. The volatile "
                             "trailing anchor differs every turn by design but does not break "
                             "backend prefix reuse, so reuse ratio — not strict append-only — "
                             "is the cache metric (default: 0.8)")
    parser.add_argument("--max-breaks", type=int, default=None,
                        help="CI gate: exit 0 when the number of BREAK turns is <= this value, "
                             "instead of requiring zero breaks. The rolling-summary fold breaks "
                             "the prefix cache by design on fold turns, so a zero-break gate is "
                             "unreachable; set this to the expected fold-break count (plus a little "
                             "headroom) to catch regressions like the every-turn eviction cliff "
                             "(which broke ~18 turns). Default: strict (any break exits 2).")
    parser.add_argument("--allow-quality-regressions", action="store_true",
                        help="Report but do not fail on prompt inflation or savings cliffs.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output machine-readable JSON summary (to stdout)")
    args = parser.parse_args()

    use_unique_session = not args.persistent_session
    if not args.persistent_session:
        env_unique = os.environ.get("DIAG_UNIQUE_SESSION", "1")
        if env_unique == "0":
            use_unique_session = False
    stream = not args.no_stream
    if args.no_stream:
        stream = False
    else:
        env_stream = os.environ.get("DIAG_STREAM", "1")
        if env_stream == "0":
            stream = False
    max_tokens = args.max_tokens or int(os.environ.get("DIAG_MAX_TOKENS", "8192"))
    global PROXY
    PROXY = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    # Clear stale bytecode and start the proxy so every run is fresh.
    _clear_pycache()
    if args.budget is not None:
        os.environ["MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS"] = str(args.budget)
    # Reuse the benchmark's exact profile defaults so dry-run and live runs
    # exercise the same optimizer configuration.
    bench._apply_profile_overrides(argparse.Namespace(profile=args.profile, budget=args.budget))
    _start_proxy(args.port)
    if not _proxy_is_running(args.port):
        print(f"  ERROR: proxy is not running on port {args.port}", file=sys.stderr)
        return 1

    try:
        base_tasks = list(bench._OPENCODE_SCENARIO_TASKS)
        base_tasks = bench._inject_drift_probe(base_tasks, args.turns)
        turn_exchanges = [t for t in base_tasks if isinstance(t, list)]

        proxy_marker = "P{diag dryrun proxy session}\n"
        proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT
        tools = bench.OPENCODE_TOOLS
        model = bench.MODEL_ID

        messages: list[dict[str, Any]] = [{"role": "system", "content": proxy_system_prompt}]
        prev_blob: str | None = None
        prev_opt: list[dict[str, Any]] = []
        breaks: list[int] = []
        report: list[dict[str, Any]] = []
        turn_times: list[float] = []
        quality_regressions: list[int] = []
        previous_raw_tokens: int | None = None
        previous_optimized_tokens: int | None = None
        dump_turns = set(args.dump_turn)

        for local_turn in range(args.turns):
            turn_start = time.monotonic()
            messages.extend(turn_exchanges[local_turn % len(turn_exchanges)])
            session = uuid.uuid4().hex if use_unique_session else "diag-persistent-session"

            try:
                resp = requests.post(
                    PROXY,
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "max_tokens": max_tokens,
                        "stream": stream,
                        "session_id": session,
                    },
                    headers={"X-MOEPT-Dry-Run": "true"},
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"   [ERROR] turn {local_turn + 1}: {exc}", file=sys.stderr)
                return 1

            turn_elapsed = time.monotonic() - turn_start
            turn_times.append(turn_elapsed)

            data = resp.json()
            opt_msgs = data.get("optimized_messages", [])
            tokens = data.get("tokens", {})
            raw_tokens = tokens.get("original")
            optimized_tokens = tokens.get("optimized")
            quality_issues = _quality_issues(
                raw_tokens,
                optimized_tokens,
                previous_raw_tokens,
                previous_optimized_tokens,
            )
            if quality_issues:
                quality_regressions.append(local_turn + 1)
            cache_key_prefix = data.get("cache_key_prefix", "")
            est_cache_hit = data.get("est_cache_hit", False)

            # Mirror app.py: apply output_shaper.shape_request
            shaped = SHAPER.shape_request({
                "messages": opt_msgs,
                "max_tokens": max_tokens,
            })
            opt_msgs = shaped["messages"]

            # Capture detailed dumps for requested turns and the default cliff pair.
            if local_turn + 1 in dump_turns or local_turn + 1 in (11, 12):
                with open(f"/tmp/diag_opt_t{local_turn + 1}.json", "w") as _f:
                    json.dump(opt_msgs, _f, indent=2)
                seq = " ".join(
                    f"{m.get('role')}"
                    + ("[S]" if m.get("_summary_id") or m.get("_rolling_summary") else "")
                    + f":{str(m.get('content'))[:24]!r}"
                    for m in opt_msgs
                )
                print(f"   [turn {local_turn + 1}] ROLES {seq}")
                for src in Path("/tmp").glob("diag_after-*.json"):
                    dst = f"/tmp/diag_{local_turn + 1}_{src.name}"
                    shutil.copy2(src, dst)

            blob = serialize(opt_msgs)

            # System-prompt sanity checks for turns 3, 4, 10.
            if local_turn + 1 in (3, 4, 10):
                sys_msg = next((m for m in opt_msgs if m.get("role") == "system"), None)
                c = sys_msg["content"] if sys_msg else ""
                has_terse = "Be concise" in c
                print(f"   [turn {local_turn + 1}] system len={len(c)} has_terse={has_terse}")

            reuse_ratio = 1.0
            if prev_blob is None:
                status = "(first)"
            elif blob == prev_blob:
                status = "STABLE"
            elif blob.startswith(prev_blob) or prev_blob.startswith(blob):
                status = "APPEND-ONLY"
            else:
                # Not byte-identical nor a strict prefix extension. The backend
                # still reuses the longest common PREFIX, and the volatile
                # trailing anchor differs every turn by design without breaking
                # that reuse — so classify by common-prefix reuse ratio rather
                # than failing the strict append-only check (a high ratio is
                # cache-stable; a low ratio is a real prefix-cache break).
                limit = min(len(prev_blob), len(blob))
                common = 0
                while common < limit and prev_blob[common] == blob[common]:
                    common += 1
                reuse_ratio = common / len(prev_blob) if prev_blob else 0.0
                if reuse_ratio >= args.reuse_threshold:
                    status = "REUSED"
                else:
                    status = "*** BREAK ***"
                    breaks.append(local_turn + 1)

            msg_diffs = _diff_messages(prev_opt, opt_msgs) if prev_opt else []
            byte_diff = _byte_diff(prev_blob, blob) if prev_blob is not None and status == "*** BREAK ***" else None
            turn_report: dict[str, Any] = {
                "turn": local_turn + 1,
                "n_opt": len(opt_msgs),
                "status": status,
                "reuse_ratio": round(reuse_ratio, 4),
                "tokens": tokens,
                "saved_tokens": (
                    raw_tokens - optimized_tokens
                    if raw_tokens is not None and optimized_tokens is not None
                    else None
                ),
                "quality_issues": quality_issues,
                "cache_key_prefix": cache_key_prefix,
                "est_cache_hit": est_cache_hit,
                "msg_diffs": msg_diffs,
                "byte_diff": byte_diff,
                "role_sequence": _role_sequence(opt_msgs),
                "content_lengths": _content_lengths(opt_msgs),
                "elapsed_ms": round(turn_elapsed * 1000, 1),
            }
            report.append(turn_report)

            # Determine whether to print this turn's details
            should_print = (
                status != "STABLE"
                or args.verbose
                or (local_turn + 1) in dump_turns
                or args.dump_all
                or (args.cliff_only and local_turn + 1 in range(11, 16))
                or (args.cliff_only and local_turn + 1 in range(28, min(args.turns + 1, 31)))
            )

            if should_print:
                prefix = "  " if args.cliff_only else ""
                print(f"{prefix}turn {local_turn + 1:2d}: n_opt={len(opt_msgs):3d} "
                      f"tok={tokens.get('optimized', '?'):>6} "
                      f"cache={cache_key_prefix[:8] if cache_key_prefix else 'none':>8} "
                      f"hit={est_cache_hit} "
                      f"{status} "
                      f"({turn_elapsed * 1000:.0f}ms)")
                if quality_issues:
                    print(f"{prefix}  QUALITY: {', '.join(quality_issues)}")
                if args.verbose or status != "STABLE" or (local_turn + 1) in dump_turns:
                    print(f"{prefix}  roles: {' -> '.join(_role_sequence(opt_msgs))}")
                    print(f"{prefix}  lengths: {_content_lengths(opt_msgs)}")
                for d in msg_diffs:
                    print(f"{prefix}  {d}")
                if byte_diff:
                    print(f"{prefix}  byte_diff: first_diff_at_char={byte_diff['first_diff_char']} "
                          f"prev_len={byte_diff['prev_len']} cur_len={byte_diff['cur_len']}")
                    print(f"{prefix}    prev_char={byte_diff.get('prev_char', '?')} "
                          f"cur_char={byte_diff.get('cur_char', '?')}")
                    print(f"{prefix}    common_prefix={byte_diff.get('common_prefix_len', '?')} "
                          f"common_suffix={byte_diff.get('common_suffix_len', '?')}")
                    if args.verbose:
                        print(f"{prefix}    prev_context={byte_diff.get('prev_context', '?')!r}")
                        print(f"{prefix}    cur_context={byte_diff.get('cur_context', '?')!r}")

            # Dump full message content for specified turns
            if (local_turn + 1) in dump_turns or args.dump_all:
                dump_dir = Path(args.dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"turn_{local_turn + 1:02d}.json"
                with open(dump_path, "w") as f:
                    json.dump({
                        "turn": local_turn + 1,
                        "status": status,
                        "messages": opt_msgs,
                        "serialized": blob,
                        "serialized_len": len(blob),
                    }, f, indent=2, default=str)
                print(f"  [dump] wrote {dump_path}")

            prev_blob = blob
            prev_opt = opt_msgs
            previous_raw_tokens = raw_tokens
            previous_optimized_tokens = optimized_tokens

        # ── Summary ──────────────────────────────────────────────
        if args.json_output:
            summary = {
                "config": {
                    "turns": args.turns,
                    "model": model,
                    "session": "unique" if use_unique_session else "persistent",
                    "stream": stream,
                    "max_tokens": max_tokens,
                },
                "turns": report,
                "breaks": breaks,
                "quality_regressions": quality_regressions,
                "stats": {
                    "total_turns": args.turns,
                    "n_breaks": len(breaks),
                    "n_stable": sum(1 for r in report if r["status"] == "STABLE"),
                    "n_append_only": sum(1 for r in report if r["status"] == "APPEND-ONLY"),
                    "n_reused": sum(1 for r in report if r["status"] == "REUSED"),
                    "n_first": sum(1 for r in report if r["status"] == "(first)"),
                    "min_reuse_ratio": min((r["reuse_ratio"] for r in report if r["turn"] > 1), default=1.0),
                    "avg_turn_ms": round(sum(turn_times) / len(turn_times) * 1000, 1) if turn_times else 0,
                    "break_turns": breaks,
                },
            }
            print(json.dumps(summary, indent=2, default=str))
        else:
            print("\n=== local prefix dry-run summary ===")
            print(f"  turns: {args.turns}")
            print(f"  breaks: {breaks if breaks else 'none'}")
            print(f"  quality regressions: {quality_regressions if quality_regressions else 'none'}")
            print(f"  model: {model}")
            print(f"  session: {'unique per turn' if use_unique_session else 'persistent'}")
            print(f"  stream: {stream}")
            print(f"  max_tokens: {max_tokens}")
            if turn_times:
                print(f"  avg turn time: {sum(turn_times) / len(turn_times) * 1000:.0f}ms")
                print(f"  total time: {sum(turn_times) * 1000:.0f}ms")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "config": {
                        "turns": args.turns,
                        "model": model,
                        "session": "unique" if use_unique_session else "persistent",
                        "stream": stream,
                        "max_tokens": max_tokens,
                    },
                    "turns": report,
                    "breaks": breaks,
                    "quality_regressions": quality_regressions,
                }, f, indent=2, default=str)
            print(f"  report written to {out_path}")

        # CI gate (review §4.12.2): with --max-breaks, pass when the break count is
        # within the expected fold-break budget; otherwise require zero breaks.
        allowed = args.max_breaks if args.max_breaks is not None else 0
        if len(breaks) > allowed:
            print(f"  GATE FAIL: {len(breaks)} breaks > allowed {allowed}")
            return 2
        if quality_regressions and not args.allow_quality_regressions:
            print(
                f"  GATE FAIL: quality regressions on turns {quality_regressions} "
                "(use --allow-quality-regressions to report only)"
            )
            return 2
        if len(breaks) <= allowed:
            return 0
        return 2
    finally:
        _stop_proxy()


if __name__ == "__main__":
    raise SystemExit(main())
