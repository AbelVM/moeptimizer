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
