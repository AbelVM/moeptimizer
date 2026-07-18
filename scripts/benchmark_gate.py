#!/usr/bin/env python3
"""CI benchmark regression gate for MOE-ptimizer.

Compares a freshly produced benchmark JSON report (from
``python scripts/benchmark.py --json``) against a committed baseline report and
fails (exit code 2) when proxy quality or token savings regress beyond a
tolerance. This is the automated quality guardrail described in REVIEW.md §11.6
("Benchmark regression gate in CI").

The gate handles both report shapes the benchmark script can emit:

* **Single-scenario** report (``--scenario opencode --json``): metrics live at
  ``quality.<metric>.mean`` and ``tokens.token_savings_pct``.
* **Aggregated** report (``--scenario all --json``): metrics live at
  ``aggregated.quality.<metric>.mean`` and
  ``aggregated.token_savings_pct.mean``.

Both are normalized into a single ``{metric: value}`` dict so the same
tolerance logic applies regardless of which report the CI job produced.

Usage:
    # Compare a new report against the committed baseline.
    python scripts/benchmark_gate.py \
        --baseline scripts/benchmark_opencode_10_5_0.7.4_fixcross.json \
        --current  report.json \
        --tolerance 0.05

    # Or run the benchmark inline (auto-starts the proxy if needed) and gate it.
    python scripts/benchmark_gate.py \
        --baseline scripts/benchmark_opencode_10_5_0.7.4_fixcross.json \
        --scenario opencode --turns 10 --rounds 3

Exit codes:
    0  gate passed (within tolerance, or no baseline to compare).
    2  regression detected beyond tolerance.
    3  usage / IO error (missing files, malformed JSON).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Headline quality metrics, in priority order. prompt_faithfulness and
# evicted_content_recall are the PRIMARY optimizer signals (context retention);
# the lexical battery is secondary for this use case. Lower is a regression.
QUALITY_METRICS: tuple[str, ...] = (
    "prompt_faithfulness",
    "evicted_content_recall",
    "code_block_ratio",
    "rouge_l_f1",
    "token_jaccard",
    "edit_similarity",
)


def _drill(data: object, *keys: str) -> object | None:
    """Walk nested dicts; return ``None`` on any missing/non-dict step."""
    cur: object = data
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _mean_of(value: object) -> float | None:
    """Extract a numeric mean from either a scalar or a ``{"mean": x}`` dict."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        m = value.get("mean")
        if isinstance(m, (int, float)):
            return float(m)
    return None


def _normalize(report: dict) -> dict[str, float]:
    """Flatten a benchmark report into ``{metric: value}`` for the gate.

    Handles both single-scenario and aggregated shapes. Returns a dict with
    ``token_savings_pct`` plus every headline quality metric that is present.
    """
    out: dict[str, float] = {}

    # Token savings: single -> tokens.token_savings_pct; aggregated ->
    # aggregated.token_savings_pct.mean.
    ts = _mean_of(_drill(report, "tokens", "token_savings_pct"))
    if ts is None:
        ts = _mean_of(_drill(report, "aggregated", "token_savings_pct"))
    if ts is not None:
        out["token_savings_pct"] = ts

    # Quality metrics: try single first, then aggregated.
    for qm in QUALITY_METRICS:
        val = _mean_of(_drill(report, "quality", qm))
        if val is None:
            val = _mean_of(_drill(report, "aggregated", "quality", qm))
        if val is not None:
            out[qm] = val

    return out


def _load_report(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ⚠️  Could not read report {path!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"  ⚠️  Report {path!r} is not a JSON object", file=sys.stderr)
        return None
    return data


def _run_benchmark(args: argparse.Namespace) -> dict | None:
    """Run the benchmark inline and return its parsed --json report."""
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark.py")),
        "--json",
        "--scenario",
        args.scenario,
        "--turns",
        str(args.turns),
        "--rounds",
        str(args.rounds),
    ]
    if args.max_tokens:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if args.measure_ttft:
        cmd.append("--measure-ttft")
    print(f"  ▶ Running benchmark: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(f"  ⚠️  Benchmark exited {proc.returncode}", file=sys.stderr)
        return None
    # The --json payload is the last JSON object printed to stdout.
    text = proc.stdout.strip()
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        print("  ⚠️  Benchmark produced no JSON output", file=sys.stderr)
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  Could not parse benchmark JSON: {exc}", file=sys.stderr)
        return None


def _print_diff(current: dict[str, float], baseline: dict[str, float], tol: float) -> None:
    rows = [("token_savings_pct", "pp", 1.0)]
    rows += [(qm, "abs", 1.0) for qm in QUALITY_METRICS]
    print("\n  Regression gate diff (current vs baseline):")
    print(f"    {'metric':<24}{'baseline':>14}{'current':>14}{'delta':>14}")
    print("    " + "-" * 64)
    for label, _kind, _scale in rows:
        base = baseline.get(label)
        cur = current.get(label)
        if base is None and cur is None:
            continue
        b_str = f"{base:.2f}" if base is not None else "n/a"
        c_str = f"{cur:.2f}" if cur is not None else "n/a"
        if base is not None and cur is not None:
            delta = cur - base
            d_str = ("+" if delta >= 0 else "") + f"{delta:.2f}"
        else:
            d_str = "n/a"
        print(f"    {label:<24}{b_str:>14}{c_str:>14}{d_str:>14}")
    print(f"    tolerance: {tol:.4f} (quality) / {tol*100:.2f}pp (savings)")


def _check_gate(current: dict[str, float], baseline: dict[str, float], tol: float) -> list[str]:
    """Return a list of failure strings (empty == pass)."""
    failures: list[str] = []

    # Token savings: higher is better; a drop is a regression (pp scale).
    cur = current.get("token_savings_pct")
    base = baseline.get("token_savings_pct")
    if cur is not None and base is not None and base > 0:
        drop = base - cur
        if drop > tol * 100:
            failures.append(
                f"token_savings_pct regressed {drop:.2f}pp "
                f"(baseline {base:.2f} -> {cur:.2f}, tol {tol*100:.2f}pp)"
            )

    # Headline quality: lower is a regression (0..1 scale).
    for qm in QUALITY_METRICS:
        cur = current.get(qm)
        base = baseline.get(qm)
        if cur is None or base is None:
            continue
        drop = base - cur
        if drop > tol:
            failures.append(
                f"{qm} regressed {drop:.4f} (baseline {base:.4f} -> {cur:.4f}, tol {tol:.4f})"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="MOE-ptimizer CI benchmark regression gate")
    parser.add_argument("--baseline", required=True, help="Path to committed baseline JSON report")
    parser.add_argument("--current", default=None, help="Path to a new JSON report to gate (skips running the benchmark)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Absolute tolerance (0.05 = quality mean may drop 0.05; savings 5pp)")
    parser.add_argument("--scenario", default="opencode", help="Scenario to run when --current is omitted")
    parser.add_argument("--turns", type=int, default=10, help="Turns when running the benchmark inline")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds when running the benchmark inline")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per response when running inline")
    parser.add_argument("--measure-ttft", action="store_true", help="Measure TTFT when running inline")
    args = parser.parse_args()

    baseline_raw = _load_report(args.baseline)
    if baseline_raw is None:
        return 3
    baseline = _normalize(baseline_raw)
    if not baseline:
        print(f"  ⚠️  Baseline {args.baseline!r} has no comparable metrics", file=sys.stderr)
        return 3

    if args.current:
        current_raw = _load_report(args.current)
        if current_raw is None:
            return 3
        current = _normalize(current_raw)
    else:
        current_raw = _run_benchmark(args)
        if current_raw is None:
            return 3
        current = _normalize(current_raw)

    if not current:
        print("  ⚠️  Current report has no comparable metrics", file=sys.stderr)
        return 3

    _print_diff(current, baseline, args.tolerance)
    failures = _check_gate(current, baseline, args.tolerance)

    if failures:
        print("\n  ❌ BENCHMARK REGRESSION GATE FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 2

    print(f"\n  ✅ Benchmark regression gate passed (tolerance {args.tolerance:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
