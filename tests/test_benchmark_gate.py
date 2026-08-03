"""Tests for benchmark.py regression-similarity gate (review03.md §10)."""

from __future__ import annotations

import os
import socket
import sys
from types import SimpleNamespace

import pytest
import requests

pytest.importorskip("benchmark.benchmark")


def _args(min_similarity: float | None) -> SimpleNamespace:
    return SimpleNamespace(min_similarity=min_similarity)


def test_gate_passes_when_above_threshold() -> None:
    from benchmark.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(0.7), 0.85) == 0


def test_gate_fails_when_below_threshold() -> None:
    from benchmark.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(0.8), 0.70) == 2


def test_gate_disabled_when_none() -> None:
    from benchmark.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(None), 0.0) == 0


def test_regression_gate_includes_headline_retention_metrics() -> None:
    from benchmark.gate import QUALITY_METRICS, _normalize

    report = {
        "quality": {
            "prompt_source_token_recall": {"mean": 0.9},
            "evicted_content_recall": {"mean": 0.8},
            "code_syntax_validity": {"mean": 1.0},
        }
    }
    normalized = _normalize(report)
    assert "prompt_source_token_recall" in QUALITY_METRICS
    assert "code_syntax_validity" in QUALITY_METRICS
    assert normalized["prompt_source_token_recall"] == 0.9
    assert normalized["code_syntax_validity"] == 1.0


def test_regression_gate_preserves_zero_and_missing_task_anchor_recall() -> None:
    from benchmark.gate import _normalize

    zero = _normalize({"quality": {"task_anchor_recall": {"mean": 0.0}}})
    missing = _normalize({"quality": {"code_block_ratio": {"mean": 1.0}}})

    assert zero["task_anchor_recall"] == 0.0
    assert "task_anchor_recall" not in missing


def test_regression_gate_rejects_hard_quality_failures() -> None:
    from benchmark.gate import _hard_failures

    failures = _hard_failures({
        "quality": {
            "prompt_source_token_recall": {"mean": 0.7},
            "evicted_content_recall": {"mean": 0.8},
            "code_syntax_validity": {"mean": 1.0},
            "quality_skipped_turns": 1,
        },
        "turns": [{"quality": {"code_syntax_validity": 0.0}}],
    })
    assert any("prompt_source_token_recall below hard floor" in failure for failure in failures)
    assert any("quality_skipped_turns exceeded" in failure for failure in failures)
    assert any("turn 1 code_syntax_validity" in failure for failure in failures)


def test_backend_log_health_url_and_bounded_events() -> None:
    from benchmark.benchmark import BackendLogCollector, _health_url_for_lemonade

    assert _health_url_for_lemonade("http://localhost:13305/api/v1") == "http://localhost:13305/v1/health"
    collector = BackendLogCollector("http://localhost:13305/v1/health", cap=1)
    collector.set_round(2)
    collector._append({"seq": 1, "severity": "INFO", "tag": "a", "line": "one"})
    collector._append({"seq": 1, "line": "duplicate"})
    collector._append({"seq": 2, "line": "two"})

    summary = collector.summary()
    assert summary["event_count"] == 1
    assert summary["dropped_events"] == 1
    assert summary["last_seq"] == 2
    assert summary["events"][0]["round"] == 2


def test_backend_log_discovery_reports_missing_optional_port(monkeypatch) -> None:
    from benchmark import benchmark as bm

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"status":"ok"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    endpoint, status = bm._discover_backend_log_endpoint("http://localhost:13305/v1/health")
    assert endpoint is None
    assert status.startswith("unavailable:")


def test_backend_log_snapshot_uses_untyped_entries_envelope() -> None:
    from benchmark.benchmark import BackendLogCollector

    collector = BackendLogCollector("http://localhost:13305/v1/health")
    snapshot = {
        "entries": [
            {"seq": 1, "severity": "Info", "tag": "main", "line": "started"},
            {"seq": 2, "severity": "Warning", "tag": "llama.cpp", "line": "busy"},
        ]
    }
    collector._handle_message(snapshot)
    collector._handle_message({"type": "logs.snapshot"})

    assert collector.summary()["event_count"] == 0
    assert collector.summary()["first_seq"] is None
    assert collector.summary()["baseline_seq"] == 2


def test_fixtures_scenario_builds_and_grows() -> None:
    """The real-use-case fixture scenario is agentic and accumulates context."""
    from benchmark.benchmark import SCENARIOS

    tasks = SCENARIOS["fixtures"]["tasks"]
    assert len(tasks) == 30
    # The fixtures scenario is now an OpenCode-harness replay: each turn is a
    # full agentic exchange (list of role-tagged message dicts), not a plain
    # user string.
    assert all(isinstance(turn, list) for turn in tasks)
    for turn in tasks:
        roles = [m["role"] for m in turn]
        assert roles[0] == "user"
        assert "assistant" in roles
        assert "tool" in roles

    # Cumulative conversation size must grow monotonically as real files and
    # tool logs are appended turn-over-turn (genuine context accumulation).
    running = 0
    lens: list[int] = []
    for turn in tasks:
        running += sum(len(m.get("content") or "") for m in turn)
        lens.append(running)
    assert lens[0] < lens[-1]
    assert all(lens[i] < lens[i + 1] for i in range(len(lens) - 1))

    # At least one turn must ship a large run_command log (>4000 chars) so the
    # proxy's ToolOutputCompressor boundary compression actually fires on
    # benchmark traffic; file-read outputs stay small and are forwarded verbatim.
    big_logs = [
        len(m["content"])
        for turn in tasks
        for m in turn
        if m["role"] == "tool" and m.get("name") == "run_command" and len(m["content"]) > 4000
    ]
    assert big_logs, "expected at least one >4000-char run_command tool output"


def test_opencode_scenario_builds() -> None:
    """The OpenCode-harness scenario must ship full agentic tool exchanges."""
    import json

    from benchmark.benchmark import SCENARIOS

    tasks = SCENARIOS["opencode"]["tasks"]
    assert len(tasks) == 30
    # Each turn is a full exchange: a list of role-tagged message dicts.
    assert all(isinstance(turn, list) for turn in tasks)

    for turn in tasks:
        roles = [m["role"] for m in turn]
        # A realistic agent payload: user request, assistant tool_calls, tool results.
        assert roles[0] == "user"
        assert "assistant" in roles
        assert "tool" in roles

    # The assistant must emit real tool_calls and the tool role must carry
    # a matching tool_call_id so the payload is OpenAI-API-compliant.
    first = tasks[0]
    assistant = next(m for m in first if m["role"] == "assistant")
    tool_msg = next(m for m in first if m["role"] == "tool")
    assert assistant["tool_calls"]
    call = assistant["tool_calls"][0]
    assert call["function"]["name"]
    json.loads(call["function"]["arguments"])  # arguments must be valid JSON
    assert tool_msg["tool_call_id"] == call["id"]

    # Tool outputs must be real fixture content, not empty placeholders.
    assert tool_msg["content"].strip()


def test_synthetic_agentic_exchange_fires_compression() -> None:
    """Synthetic scenarios (via _agentic_exchange) also emit realistic tool I/O.

    The default read_file/run_command pair must ship a >4000-char run_command
    log so the proxy's ToolOutputCompressor fires on every scenario, not just
    the fixtures/opencode replay; the file read stays smaller (forwarded
    verbatim to protect quality).
    """
    import json

    from benchmark.benchmark import _agentic_exchange

    msgs = _agentic_exchange("Refactor calculate_stats for performance.", 0)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user"
    assert "assistant" in roles
    assert "tool" in roles

    # Assistant tool_calls must be OpenAI-API-compliant and match the tool role.
    for m in msgs:
        if m["role"] == "assistant":
            call = m["tool_calls"][0]
            assert call["function"]["name"]
            json.loads(call["function"]["arguments"])

    run_logs = [
        len(m["content"])
        for m in msgs
        if m["role"] == "tool" and m.get("name") == "run_command"
    ]
    assert run_logs and max(run_logs) > 4000

    read_outputs = [
        len(m["content"])
        for m in msgs
        if m["role"] == "tool" and m.get("name") == "read_file"
    ]
    assert read_outputs and all(n > 0 for n in read_outputs)


def test_fixture_replay_builds_direct_proxy_quality_comparison() -> None:
    from benchmark.benchmark import TurnMetrics, _build_turn_comparisons

    direct = [TurnMetrics(turn_index=1)]
    proxy = [
        TurnMetrics(
            turn_index=1,
            full_prompt_text="system task old context",
            optimized_prompt_text="system task",
        )
    ]
    content = "```python\ndef summarize():\n    return 1\n```"
    comparisons = _build_turn_comparisons(direct, proxy, [content], [content])

    assert len(comparisons) == 1
    assert comparisons[0].quality_computed is True
    assert comparisons[0].quality["code_block_ratio"] == 1.0
    assert comparisons[0].quality["prompt_faithfulness"] is not None


def test_report_aggregates_backend_mtp_metrics() -> None:
    from benchmark.benchmark import BenchmarkReport, TurnComparison

    report = BenchmarkReport(
        turns=[TurnComparison(turn_index=1)],
        cache_reuse=[
            {"round": 0, "mtp_samples": 2, "total_mtp_accepted_tokens": 12, "total_mtp_draft_tokens": 20},
            {"round": 1, "mtp_samples": 1, "total_mtp_accepted_tokens": 9, "total_mtp_draft_tokens": 10},
        ],
    )

    mtp = report.summary()["mtp"]
    assert mtp["samples"] == 3
    assert mtp["total_accepted_tokens"] == 21
    assert mtp["total_draft_tokens"] == 30
    assert mtp["acceptance_rate"] == 0.7
    assert len(mtp["per_round"]) == 2


def test_report_aggregates_mtp_from_backend_logs_when_metrics_are_empty() -> None:
    from benchmark.benchmark import BenchmarkReport, TurnComparison

    report = BenchmarkReport(
        turns=[TurnComparison(turn_index=1)],
        backend_logs={
            "mtp_per_turn": [
                {"round": 0, "phase": "proxy", "turn": 1, "samples": 2,
                 "accepted_tokens": 483, "draft_tokens": 489, "acceptance_rate": 0.9877},
            ]
        },
    )

    mtp = report.summary()["mtp"]
    assert mtp["source"] == "backend logs"
    assert mtp["samples"] == 2
    assert mtp["total_accepted_tokens"] == 483
    assert mtp["total_draft_tokens"] == 489
    assert mtp["acceptance_rate"] == 0.9877
    assert mtp["per_round"][0]["acceptance_rate"] == 0.9877


def test_report_aggregates_proxy_observability_metrics() -> None:
    from benchmark.benchmark import BenchmarkReport, TurnComparison

    report = BenchmarkReport(
        turns=[TurnComparison(turn_index=1)],
        cache_reuse=[
            {
                "round": 0,
                "requests": 2,
                "backend_errors": 1,
                "degradation_counts": {"compress": 2},
                "avg_optimizer_ms": 4.0,
                "optimizer_samples": 2,
                "avg_token_count_ms": 1.0,
                "token_count_samples": 2,
                "avg_fresh_prefill_tokens": 10.0,
                "fresh_prefill_samples": 2,
            },
        ],
    )

    observability = report.summary()["proxy_observability"]
    assert observability["requests"] == 2
    assert observability["backend_errors"] == 1
    assert observability["degradation_events"] == 2
    assert observability["avg_optimizer_ms"] == 4.0
    assert observability["avg_token_count_ms"] == 1.0
    assert observability["avg_fresh_prefill_tokens"] == 10.0


def test_report_derives_backend_independent_performance_metrics() -> None:
    from benchmark.benchmark import BenchmarkReport, TurnComparison, TurnMetrics

    report = BenchmarkReport(
        turns=[TurnComparison(
            turn_index=1,
            direct=TurnMetrics(prompt_tokens=100, completion_tokens=20, cached_tokens=25,
                               latency_ms=2000, ttft_ms=1000, full_prompt_text="direct prompt"),
            proxy=TurnMetrics(prompt_tokens=60, completion_tokens=18, cached_tokens=30,
                              latency_ms=1900, ttft_ms=700, optimized_prompt_text="proxy prompt"),
        )],
    )

    performance = report.summary()["performance"]
    assert performance["fresh_prefill_tokens"]["direct"]["mean"] == 75
    assert performance["fresh_prefill_tokens"]["proxy"]["mean"] == 30
    assert performance["cache_reuse_pct"]["direct"]["mean"] == 25
    assert performance["cache_reuse_pct"]["proxy"]["mean"] == 50
    assert performance["approximate_decode_tps"]["direct"]["mean"] == 20
    assert performance["approximate_decode_tps"]["proxy"]["mean"] == 15
    assert len(performance["per_turn"][0]["direct_prompt_sha256"]) == 16
    assert len(performance["per_turn"][0]["proxy_prompt_sha256"]) == 16


def test_report_excludes_only_turn_zero_from_cache_aggregates() -> None:
    from benchmark.benchmark import BenchmarkReport, TurnComparison, TurnMetrics

    report = BenchmarkReport(turns=[
        TurnComparison(
            turn_index=0,
            direct=TurnMetrics(prompt_tokens=100, cached_tokens=100),
            proxy=TurnMetrics(prompt_tokens=100, cached_tokens=100, prefix_cache_hit_tokens=999),
        ),
        TurnComparison(
            turn_index=1,
            direct=TurnMetrics(prompt_tokens=100, cached_tokens=0),
            proxy=TurnMetrics(prompt_tokens=100, cached_tokens=0, prefix_cache_hit_tokens=0),
        ),
        TurnComparison(
            turn_index=2,
            direct=TurnMetrics(prompt_tokens=100, cached_tokens=50),
            proxy=TurnMetrics(prompt_tokens=100, cached_tokens=50, prefix_cache_hit_tokens=10),
        ),
    ])

    summary = report.summary()
    assert summary["cache_reuse"]["total_prefix_cache_hit_tokens"] == 10
    assert summary["cache_reuse"]["per_turn_prefix_cache_hit_tokens"]["mean"] == 5
    assert summary["performance"]["cache_reuse_pct"]["proxy"]["mean"] == 25


def test_dry_run_prefix_classifier_respects_threshold_boundaries() -> None:
    from benchmark.diag_dryrun_opencode import _classify_prefix

    assert _classify_prefix(None, "abc", 0.9) == ("(first)", 1.0)
    assert _classify_prefix("abc", "abc", 0.9) == ("STABLE", 1.0)
    assert _classify_prefix("abc", "abcdef", 0.9) == ("APPEND-ONLY", 1.0)
    assert _classify_prefix("abcdef", "abcXYZ", 0.5) == ("REUSED", 0.5)
    assert _classify_prefix("abcdef", "abcXYZ", 0.51) == ("*** BREAK ***", 0.5)


def test_dry_run_parser_rejects_malformed_payloads() -> None:
    from benchmark.diag_dryrun_opencode import _parse_dry_run_response

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    with pytest.raises(ValueError, match="valid JSON"):
        _parse_dry_run_response(Response(ValueError("bad json")))
    with pytest.raises(ValueError, match="optimized_messages"):
        _parse_dry_run_response(Response({"tokens": {}}))
    with pytest.raises(ValueError, match="tokens"):
        _parse_dry_run_response(Response({"optimized_messages": [], "tokens": []}))


def test_dry_run_memory_probe_is_opt_in() -> None:
    from benchmark import benchmark as bm
    from benchmark.diag_dryrun_opencode import _dry_run_tasks

    plain = _dry_run_tasks(3, include_memory_probe=False)
    probed = _dry_run_tasks(3, include_memory_probe=True)
    assert plain == bm._OPENCODE_SCENARIO_TASKS
    assert probed != plain


def test_dry_run_gate_code_covers_break_and_quality_boundaries() -> None:
    from benchmark.diag_dryrun_opencode import _dry_run_gate_code

    assert _dry_run_gate_code(0, None, [], False) == 0
    assert _dry_run_gate_code(1, 1, [], False) == 0
    assert _dry_run_gate_code(2, 1, [], False) == 2
    assert _dry_run_gate_code(0, 1, [3], False) == 2
    assert _dry_run_gate_code(0, 1, [3], True) == 0


def test_dry_run_main_returns_success_for_valid_proxy_response(monkeypatch) -> None:
    from benchmark import diag_dryrun_opencode as diagnostic

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "optimized_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                ],
                "tokens": {"original": 10, "optimized": 8},
                "cache_key_prefix": "prefix",
                "est_cache_hit": False,
            }

    monkeypatch.setattr(sys, "argv", ["diag_dryrun_opencode", "--turns", "1", "--no-stream"])
    monkeypatch.setattr(diagnostic, "_clear_pycache", lambda: None)
    monkeypatch.setattr(diagnostic, "_start_proxy", lambda port: None)
    monkeypatch.setattr(diagnostic, "_proxy_is_running", lambda port: True)
    monkeypatch.setattr(diagnostic, "_stop_proxy", lambda: None)
    monkeypatch.setattr(diagnostic.requests, "post", lambda *args, **kwargs: Response())

    assert diagnostic.main() == 0


def test_dry_run_main_returns_failure_for_malformed_proxy_response(monkeypatch) -> None:
    from benchmark import diag_dryrun_opencode as diagnostic

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            raise ValueError("bad json")

    monkeypatch.setattr(sys, "argv", ["diag_dryrun_opencode", "--turns", "1", "--no-stream"])
    monkeypatch.setattr(diagnostic, "_clear_pycache", lambda: None)
    monkeypatch.setattr(diagnostic, "_start_proxy", lambda port: None)
    monkeypatch.setattr(diagnostic, "_proxy_is_running", lambda port: True)
    monkeypatch.setattr(diagnostic, "_stop_proxy", lambda: None)
    monkeypatch.setattr(diagnostic.requests, "post", lambda *args, **kwargs: Response())

    assert diagnostic.main() == 1


def test_dry_run_live_proxy_subprocess() -> None:
    if os.environ.get("MOEPT_RUN_LIVE_DRYRUN") != "1":
        pytest.skip("set MOEPT_RUN_LIVE_DRYRUN=1 to run the live proxy check")

    from benchmark import diag_dryrun_opencode as diagnostic

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    process = diagnostic._start_proxy(port, wait=15)
    assert process is not None
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "dry-run", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-MOEPT-Dry-Run": "true"},
            timeout=15,
        )
        response.raise_for_status()
        payload = diagnostic._parse_dry_run_response(response)
        assert payload["tokens"]["optimized"] <= payload["tokens"]["original"]
    finally:
        diagnostic._stop_proxy()


def test_aggregate_reports_excludes_compression_stress_from_default_metrics() -> None:
    from benchmark.benchmark import _aggregate_reports

    class Report:
        def __init__(self, scenario_kind: str, similarity: float) -> None:
            self.config = {"scenario_kind": scenario_kind}
            self.similarity = similarity

        def summary(self) -> dict[str, object]:
            return {
                "num_turns": 1,
                "latency_ms": {"proxy": {"mean": 1.0}},
                "quality": {
                    "semantic_similarity": {"mean": self.similarity},
                    "rouge_l_f1": {"mean": self.similarity},
                    "token_jaccard": {"mean": self.similarity},
                    "code_block_ratio": {"mean": self.similarity},
                    "edit_similarity": {"mean": self.similarity},
                    "quality_skipped_turns": 0,
                },
                "tokens": {"token_savings_pct": 1.0},
            }

    result = _aggregate_reports({
        "real": Report("realistic", 0.9),
        "compression_stress": Report("compression_stress", 0.1),
    })

    assert result["aggregated"]["semantic_similarity"]["mean"] == 0.9
    assert result["per_scenario"]["compression_stress"]["scenario_kind"] == "compression_stress"
    assert result["stress"]["aggregated"]["semantic_similarity"]["mean"] == 0.1
    assert result["quality"]["code_block_ratio"]["mean"] == 0.9
    assert result["secondary_quality"]["rouge_l_f1"]["mean"] == 0.9


def test_gate_normalizes_flattened_aggregate_quality_metrics() -> None:
    from benchmark.gate import _normalize

    normalized = _normalize({
        "aggregated": {
            "prompt_source_token_recall": {"mean": 0.9},
            "code_syntax_validity": {"mean": 1.0},
        }
    })

    assert normalized["prompt_source_token_recall"] == 0.9
    assert normalized["code_syntax_validity"] == 1.0


def test_baseline_quality_report_removes_stress_scenarios() -> None:
    from benchmark.benchmark import _baseline_quality_report

    report = _baseline_quality_report({
        "scenarios": ["real", "compression_stress"],
        "per_scenario": {
            "real": {"scenario_kind": "realistic"},
            "compression_stress": {"scenario_kind": "compression_stress"},
        },
        "aggregated": {"semantic_similarity": {"mean": 0.9}},
        "stress": {"aggregated": {"semantic_similarity": {"mean": 0.1}}},
    })

    assert report["report_scope"] == "realistic_baseline"
    assert report["scenarios"] == ["real"]
    assert list(report["per_scenario"]) == ["real"]
    assert "stress" not in report


def test_aggregate_reports_preserves_zero_quality_metrics() -> None:
    from benchmark.benchmark import _aggregate_reports

    class ZeroReport:
        def summary(self) -> dict:
            return {
                "num_turns": 1,
                "quality": {
                    "semantic_similarity": {"mean": 0.0},
                    "rouge_l_f1": {"mean": 0.0},
                    "token_jaccard": {"mean": 0.0},
                    "code_block_ratio": {"mean": 0.0},
                    "edit_similarity": {"mean": 0.0},
                },
                "tokens": {"token_savings_pct": 0.0},
                "latency_ms": {"proxy": {"mean": 0.0}},
                "ttft_ms": {"proxy": {"mean": 0.0}},
                "proxy_overhead_ms": {"mean": 0.0},
                "cost_usd": {"savings_pct": 0.0},
            }

    aggregated = _aggregate_reports({"zero": ZeroReport()})["aggregated"]
    assert aggregated["semantic_similarity"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
    assert aggregated["token_savings_pct"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
