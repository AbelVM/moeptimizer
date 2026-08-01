"""Tests for benchmark.py regression-similarity gate (review03.md §10)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("scripts.benchmark")


def _args(min_similarity: float | None) -> SimpleNamespace:
    return SimpleNamespace(min_similarity=min_similarity)


def test_gate_passes_when_above_threshold() -> None:
    from scripts.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(0.7), 0.85) == 0


def test_gate_fails_when_below_threshold() -> None:
    from scripts.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(0.8), 0.70) == 2


def test_gate_disabled_when_none() -> None:
    from scripts.benchmark import _check_similarity_gate

    assert _check_similarity_gate(_args(None), 0.0) == 0


def test_backend_log_health_url_and_bounded_events() -> None:
    from scripts.benchmark import BackendLogCollector, _health_url_for_lemonade

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
    from scripts import benchmark as bm

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
    from scripts.benchmark import BackendLogCollector

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
    from scripts.benchmark import SCENARIOS

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

    from scripts.benchmark import SCENARIOS

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

    from scripts.benchmark import _agentic_exchange

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
    from scripts.benchmark import TurnMetrics, _build_turn_comparisons

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
    from scripts.benchmark import BenchmarkReport, TurnComparison

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
    from scripts.benchmark import BenchmarkReport, TurnComparison

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
    from scripts.benchmark import BenchmarkReport, TurnComparison

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
    from scripts.benchmark import BenchmarkReport, TurnComparison, TurnMetrics

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
