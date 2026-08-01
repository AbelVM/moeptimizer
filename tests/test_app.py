
"""Smoke tests for moeptimizer app endpoints, streaming, and metrics.

Covers:
- Health, models, and metrics endpoints
- Chat completions (streaming + non-streaming)
- OutputShaper integration
- Metrics recording and reset
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from moeptimizer.app import _ProxyMetrics, create_app
from moeptimizer.config import AppConfig

MODEL_ID = "Qwen3.6-35B-A3B-MTP-GGUF"
LEMONADE_URL = "http://localhost:13305/api/v1"


def _reset_metrics() -> None:
    """Reset the global proxy metrics singleton between tests."""
    import moeptimizer.app as app_module
    app_module.PROXY_METRICS.reset()


def _backend_response(model: str = MODEL_ID, cached_tokens: int = 10) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "Thinking...",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def _mock_backend(cached_tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.chat_completions_create = AsyncMock(return_value=MagicMock(
        model_dump=lambda: _backend_response(cached_tokens=cached_tokens)
    ))
    return mock


def test_metrics_snapshot_records_optional_mtp_usage() -> None:
    metrics = _ProxyMetrics()
    metrics.record_mtp_usage({
        "completion_tokens_details": {
            "accepted_prediction_tokens": 12,
            "draft_tokens": 20,
            "fallback_count": 1,
            "decode_ms": 42.5,
        }
    })
    snap = metrics.snapshot()
    assert snap["mtp_samples"] == 1
    assert snap["avg_mtp_accepted_tokens"] == 12.0
    assert snap["avg_mtp_draft_tokens"] == 20.0
    assert snap["total_mtp_accepted_tokens"] == 12
    assert snap["total_mtp_draft_tokens"] == 20
    assert snap["mtp_acceptance_rate"] == 0.6
    assert snap["total_mtp_fallbacks"] == 1
    assert snap["mtp_fallback_rate"] == 1.0
    assert snap["avg_mtp_decode_ms"] == 42.5


def test_metrics_request_trace_is_bounded_and_keeps_mtp_context() -> None:
    metrics = _ProxyMetrics()
    metrics.record_mtp_usage(
        {"completion_tokens_details": {"accepted_prediction_tokens": 2, "draft_tokens": 4}},
        request_id="req-1",
        session_id="session-1",
    )
    metrics.record_turn(
        request_id="req-1",
        session_id="session-1",
        prompt_hash="abc123",
        slot=7,
        prompt_tokens=20,
        cached_tokens=12,
    )
    traces = metrics.request_traces()
    assert traces[0]["mtp"]["accepted_tokens"] == 2
    assert traces[1]["prompt_hash"] == "abc123"
    assert traces[1]["slot"] == 7

    for index in range(600):
        metrics.record_turn(request_id=f"req-{index}")
    assert len(metrics.request_traces()) == 512


class TestHealthAndModels:
    """Smoke tests for non-chat endpoints."""

    def test_health_endpoint_exists(self) -> None:
        app = create_app(AppConfig())
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_models_endpoint_returns_model_list(self) -> None:
        app = create_app(AppConfig())
        client = TestClient(app)
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_metrics_endpoint_returns_aggregate(self) -> None:
        app = create_app(AppConfig())
        client = TestClient(app)
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "cache_hit_rate" in data
        assert "backend_errors" in data

    def test_request_debug_endpoint_returns_bounded_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        fake_backend = _mock_backend()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)
        client = TestClient(create_app(AppConfig()))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "trace me"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        assert response.headers["X-MOEPT-Request-Id"]
        debug = client.get("/v1/debug/requests").json()
        assert debug["object"] == "proxy.request_debug"
        assert debug["requests"][-1]["prompt_hash"]
        assert debug["requests"][-1]["cached_tokens"] == 10

    def test_metrics_reset_clears_counters(self) -> None:
        app = create_app(AppConfig())
        client = TestClient(app)
        # First, hit metrics to ensure it works
        client.get("/v1/metrics")
        # Reset
        response = client.post("/v1/metrics/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        # Verify counters are zeroed
        snap = client.get("/v1/metrics").json()
        assert snap["requests"] == 0
        assert snap["backend_errors"] == 0


class TestChatCompletionsNonStreaming:
    """Smoke tests for non-streaming chat completions."""

    def test_non_streaming_fulfils_expand_content_before_returning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module
        from moeptimizer.optimizer import AgentContextOptimizer
        from moeptimizer.session_manager import SessionManager

        optimizer = AgentContextOptimizer(AppConfig())
        handle = optimizer.content_store.put("full retained tool output")
        calls: list[dict[str, Any]] = []
        first = _backend_response()
        first["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_expand",
                "type": "function",
                "function": {
                    "name": "expand_content",
                    "arguments": f'{{"handle":"{handle}"}}',
                },
            }],
        }
        second = _backend_response()

        async def fake_create(*args: Any, **kwargs: Any):
            calls.append(kwargs)
            result = first if len(calls) == 1 else second
            return MagicMock(model_dump=lambda: result)

        fake_backend = MagicMock()
        fake_backend.chat_completions_create = fake_create
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)
        monkeypatch.setattr(
            SessionManager,
            "get_or_create",
            classmethod(lambda cls, session_id=None: optimizer),
        )

        response = TestClient(create_app(AppConfig())).post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "use the retained output"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello!"
        assert len(calls) == 2
        assert calls[1]["messages"][-1]["content"] == "full retained tool output"

    def test_non_streaming_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        fake_backend = _mock_backend()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)
        monkeypatch.setattr(app_module, "_serialize_messages_text", lambda _: "x" * 8001)

        app = create_app(AppConfig())
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_non_streaming_records_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        fake_backend = _mock_backend(cached_tokens=15)
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        snap = client.get("/v1/metrics").json()
        assert snap["requests"] == 1
        assert snap["total_cached_tokens"] == 15

    def test_non_streaming_sets_optimized_prompt_header_when_diagnostics_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        fake_backend = _mock_backend()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
            headers={"X-MOEPT-Diagnostics": "true"},
        )
        assert "X-MOEPT-Optimized-Prompt-Text" in response.headers

    def test_non_streaming_omits_prompt_header_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        fake_backend = _mock_backend()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert "X-MOEPT-Optimized-Prompt-Text" not in response.headers

    def test_diagnostics_reports_prompt_header_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        fake_backend = _mock_backend()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)
        monkeypatch.setattr(app_module, "_serialize_messages_text", lambda _: "x" * 8001)

        response = TestClient(create_app(AppConfig())).post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "x" * 7000}],
                "stream": False,
            },
            headers={"X-MOEPT-Diagnostics": "true"},
        )
        assert "X-MOEPT-Optimized-Prompt-Text" not in response.headers
        assert response.headers["X-MOEPT-Diagnostics-Limit"]


class TestChatCompletionsStreaming:
    """Smoke tests for streaming chat completions."""

    def test_streaming_fulfils_expand_content_before_client_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module
        from moeptimizer.optimizer import AgentContextOptimizer
        from moeptimizer.session_manager import SessionManager

        optimizer = AgentContextOptimizer(AppConfig())
        handle = optimizer.content_store.put("full retained tool output")
        calls: list[dict[str, Any]] = []

        async def fake_stream(*args: Any, **kwargs: Any):
            calls.append(kwargs)
            if len(calls) == 1:
                yield MagicMock(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant",
                            content=None,
                            reasoning_content=None,
                            tool_calls=[SimpleNamespace(
                                index=0,
                                id="call_expand",
                                function=SimpleNamespace(
                                    name="expand_content",
                                    arguments=f'{{"handle":"{handle}"}}',
                                ),
                            )],
                        ),
                        finish_reason=None,
                    )],
                    usage=None,
                )
                yield MagicMock(
                    choices=[MagicMock(
                        delta=MagicMock(
                            role=None,
                            content=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="tool_calls",
                    )],
                    usage=None,
                )
                return

            yield MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(
                        role="assistant",
                        content="done",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )],
                usage=None,
            )

        fake_backend = MagicMock()
        fake_backend.chat_completions_stream = fake_stream
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)
        monkeypatch.setattr(
            SessionManager,
            "get_or_create",
            classmethod(lambda cls, session_id=None: optimizer),
        )

        response = TestClient(create_app(AppConfig())).post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "use the retained output"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "done" in response.text
        assert "expand_content" not in response.text
        assert len(calls) == 2
        second_messages = calls[1]["messages"]
        assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "expand_content"
        assert second_messages[-1] == {
            "role": "tool",
            "tool_call_id": "call_expand",
            "content": "full retained tool output",
        }

    def test_streaming_returns_sse_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        async def fake_stream(*args: Any, **kwargs: Any):
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(role="assistant", content="H", reasoning_content=None), finish_reason=None)],
                usage=None,
            )
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(role=None, content="i", reasoning_content=None), finish_reason="stop")],
                usage=MagicMock(
                    prompt_tokens=20,
                    cache_hit_tokens=10,
                    prompt_tokens_details=MagicMock(cached_tokens=10),
                ),
            )

        fake_backend = MagicMock()
        fake_backend.chat_completions_stream = fake_stream
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        body = response.text
        assert "data: " in body
        assert "data: [DONE]" in body

    def test_streaming_records_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        async def fake_stream(*args: Any, **kwargs: Any):
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(role="assistant", content="H", reasoning_content=None), finish_reason=None)],
                usage=MagicMock(
                    prompt_tokens=20,
                    cache_hit_tokens=10,
                    prompt_tokens_details=MagicMock(cached_tokens=10),
                ),
            )

        fake_backend = MagicMock()
        fake_backend.chat_completions_stream = fake_stream
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        snap = client.get("/v1/metrics").json()
        assert snap["requests"] == 1
        assert snap["total_cached_tokens"] == 10

    def test_streaming_emits_context_budget_comment_when_evicted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C8 (review §11.4): when front-eviction dropped turns this turn, the
        streaming path must surface it as an SSE comment so the client knows
        history was compacted."""
        import moeptimizer.app as app_module
        from moeptimizer.optimizer import AgentContextOptimizer
        from moeptimizer.session_manager import SessionManager

        async def fake_stream(*args: Any, **kwargs: Any):
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(role="assistant", content="H", reasoning_content=None), finish_reason=None)],
                usage=None,
            )
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(role=None, content="i", reasoning_content=None), finish_reason="stop")],
                usage=MagicMock(
                    prompt_tokens=20,
                    cache_hit_tokens=10,
                    prompt_tokens_details=MagicMock(cached_tokens=10),
                ),
            )

        fake_backend = MagicMock()
        fake_backend.chat_completions_stream = fake_stream
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        # Force the optimizer to report that 3 turns were evicted this turn.
        # optimize_messages resets _last_evicted_turns to 0 at the start, so wrap
        # it to re-set the count after the real optimization runs.
        evicting_optimizer = AgentContextOptimizer(AppConfig())
        _real_optimize = evicting_optimizer.optimize_messages

        def _optimize_with_eviction(msgs, original=None):
            result = _real_optimize(msgs, original)
            evicting_optimizer._last_evicted_turns = 3
            return result

        monkeypatch.setattr(evicting_optimizer, "optimize_messages", _optimize_with_eviction)
        monkeypatch.setattr(
            SessionManager, "get_or_create",
            classmethod(lambda cls, session_id=None: evicting_optimizer),
        )

        app = create_app(AppConfig())
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "X-MOEPT-Context-Budget: evicted 3 turn(s)" in response.text

    def test_non_streaming_sets_context_budget_header_when_evicted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C8 (review §11.4): the non-streaming path surfaces eviction via a
        response header (headers are available before the body is sent)."""
        import moeptimizer.app as app_module
        from moeptimizer.optimizer import AgentContextOptimizer
        from moeptimizer.session_manager import SessionManager

        fake_backend = MagicMock()
        fake_backend.chat_completions_create = AsyncMock(return_value=MagicMock(
            model_dump=lambda: _backend_response(cached_tokens=10)
        ))
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        evicting_optimizer = AgentContextOptimizer(AppConfig())
        _real_optimize = evicting_optimizer.optimize_messages

        def _optimize_with_eviction(msgs, original=None):
            result = _real_optimize(msgs, original)
            evicting_optimizer._last_evicted_turns = 2
            return result

        monkeypatch.setattr(evicting_optimizer, "optimize_messages", _optimize_with_eviction)
        monkeypatch.setattr(
            SessionManager, "get_or_create",
            classmethod(lambda cls, session_id=None: evicting_optimizer),
        )

        app = create_app(AppConfig())
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.headers.get("X-MOEPT-Context-Budget") == "evicted 2 turn(s)"


class TestOutputShaperIntegration:
    """Hard-constraint guard (review §4.2.5): the proxy compacts ONLY the input
    context and must NOT tune response verbosity. OutputShaper is disabled by
    default, so the proxy must forward the client's generation params unchanged and
    inject no "be terse" instruction. These tests fail if response shaping is
    re-enabled."""

    def test_proxy_does_not_inject_terse_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        captured_bodies: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> MagicMock:
            captured_bodies.append(kwargs)
            return MagicMock(model_dump=lambda: _backend_response())

        fake_backend = MagicMock()
        fake_backend.chat_completions_create = fake_create
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Explain quantum computing"}],
                "stream": False,
            },
        )

        assert len(captured_bodies) == 1
        body = captured_bodies[0]
        messages = body.get("messages", [])
        # No proxy-injected verbosity instruction may reach the backend.
        for msg in messages:
            assert "Be concise" not in (msg.get("content") or "")

    def test_proxy_does_not_clamp_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        captured_bodies: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> MagicMock:
            captured_bodies.append(kwargs)
            return MagicMock(model_dump=lambda: _backend_response())

        fake_backend = MagicMock()
        fake_backend.chat_completions_create = fake_create
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 8192,
                "stream": False,
            },
        )

        assert len(captured_bodies) == 1
        body = captured_bodies[0]
        # The client's max_tokens passes through untouched: the backend/model
        # controls response size, not the proxy.
        assert body.get("max_tokens") == 8192


class TestMetricsSmoke:
    """Metrics endpoint smoke tests."""

    def test_metrics_snapshot_structure(self) -> None:
        metrics = _ProxyMetrics()
        snap = metrics.snapshot()
        expected_keys = {
            "requests",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate",
            "total_cached_tokens",
            "total_prompt_tokens",
            "prefix_cache_reuse_ratio",
            "total_saved_tokens",
            "total_latency_ms",
            "avg_latency_ms",
            "backend_errors",
            "avg_optimizer_ms",
            "avg_token_count_ms",
            "token_count_samples",
            "sessions",
        }
        assert expected_keys.issubset(snap.keys())

    def test_metrics_records_optimizer_duration(self) -> None:
        metrics = _ProxyMetrics()
        metrics.record_optimizer_duration(12.5)
        metrics.record_optimizer_duration(7.5)
        assert metrics.snapshot()["avg_optimizer_ms"] == 10.0

    def test_metrics_records_token_count_duration(self) -> None:
        metrics = _ProxyMetrics()
        metrics.record_token_count_duration(3.0, 2)
        metrics.record_token_count_duration(1.0, 1)
        snapshot = metrics.snapshot()
        assert snapshot["avg_token_count_ms"] == 1.333
        assert snapshot["token_count_samples"] == 3

    def test_metrics_reset_clears_everything(self) -> None:
        metrics = _ProxyMetrics()
        metrics.record_turn(
            session_id="s1",
            cached_tokens=10,
            prompt_tokens=20,
            saved_tokens=5,
            latency_ms=100.0,
        )
        metrics.record_backend_error(session_id="s1")
        snap_before = metrics.snapshot()
        assert snap_before["requests"] == 1
        assert snap_before["backend_errors"] == 1

        metrics.reset()
        snap_after = metrics.snapshot()
        assert snap_after["requests"] == 0
        assert snap_after["backend_errors"] == 0
        assert snap_after["sessions"] == {}

    def test_per_session_lru_eviction(self) -> None:
        metrics = _ProxyMetrics()
        # Record more sessions than the LRU cap
        for i in range(600):
            metrics.record_turn(session_id=f"sess-{i}")
        snap = metrics.snapshot()
        assert len(snap["sessions"]) <= metrics._max_sessions_tracked


class TestSessionDebugEndpoint:
    """Per-session debug dashboard endpoint (review §10, P4)."""

    def test_debug_endpoint_returns_live_zone_and_breaker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        fake_backend = _mock_backend(cached_tokens=15)
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        app = create_app(AppConfig())
        client = TestClient(app)
        sid = "debug-session-1"

        # Drive one turn so the optimizer builds live-zone + cache state.
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Implement a quicksort in Rust"}],
                "max_tokens": 8192,
                "stream": False,
                "_session_id": sid,
            },
        )

        resp = client.get(f"/v1/agent/sessions/{sid}/debug")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "agent.session.debug"
        assert data["session_id"] == sid
        # Live-zone boundary is always present and well-formed.
        assert "live_zone" in data
        assert isinstance(data["live_zone"]["live_zone_start"], int)
        assert isinstance(data["live_zone"]["stable_prefix_len"], int)
        # Embedding circuit breaker state is surfaced for dependency health.
        assert "embedding_breaker" in data
        assert "state" in data["embedding_breaker"]
        # Cache outcome + token savings are present.
        assert "cache" in data
        assert "last_saved_token_count" in data["cache"]
        # Per-session metrics are attached when available.
        assert "metrics" in data

    def test_debug_endpoint_unknown_session_still_200(self) -> None:
        app = create_app(AppConfig())
        client = TestClient(app)
        resp = client.get("/v1/agent/sessions/never-seen/debug")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "never-seen"


class TestOptimizerExecutor:
    """Review §9: the optimizer must run on a dedicated bounded executor."""

    def test_optimizer_executor_created_on_startup(self) -> None:
        import moeptimizer.app as app_module

        app = create_app(AppConfig())
        # Before the lifespan runs, the executor is None.
        assert app_module._OPTIMIZER_EXECUTOR is None
        with TestClient(app):
            # Inside the lifespan context the dedicated executor exists.
            assert app_module._OPTIMIZER_EXECUTOR is not None
            assert app_module._OPTIMIZER_EXECUTOR._max_workers >= 1
        # After shutdown the executor is torn down.
        assert app_module._OPTIMIZER_EXECUTOR is None

    def test_optimizer_runs_through_dedicated_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module
        from moeptimizer.optimizer import AgentContextOptimizer

        _reset_metrics()
        fake_backend = _mock_backend(cached_tokens=15)
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: fake_backend)

        # Record the thread name the optimizer actually runs on. The dedicated
        # executor uses thread_name_prefix="moept-optim", so a match proves the
        # optimizer was dispatched on the dedicated pool, not the default.
        seen_threads: list[str] = []
        real_optimize = AgentContextOptimizer.optimize_messages

        def _spy(self, messages, original_prompt=None):
            seen_threads.append(threading.current_thread().name)
            return real_optimize(self, messages, original_prompt)

        monkeypatch.setattr(AgentContextOptimizer, "optimize_messages", _spy)

        app = create_app(AppConfig())
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 8192,
                    "stream": False,
                },
            )
        # The optimizer ran on the dedicated "moept-optim" executor thread.
        assert seen_threads, "optimizer was never invoked"
        assert any(t.startswith("moept-optim") for t in seen_threads)


class TestDryRunEndpoint:
    """Review §11 / P4a: X-MOEPT-Dry-Run returns the optimized prompt without
    calling the backend."""

    def test_dry_run_returns_optimized_prompt_without_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        # The backend must NOT be called in dry-run mode. Track calls to the
        # backend's completion method (the client itself is built at startup).
        backend = _mock_backend(cached_tokens=15)
        backend.chat_completions_create = AsyncMock(side_effect=AssertionError(
            "backend must not be called in dry-run mode"
        ))
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: backend)

        app = create_app(AppConfig())
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"X-MOEPT-Dry-Run": "true"},
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "user", "content": "Write a function to sort a list"},
                    ],
                    "max_tokens": 8192,
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "moept.dry_run"
        assert "optimized_messages" in data
        assert "tokens" in data
        assert "original" in data["tokens"]
        assert "optimized" in data["tokens"]
        assert "saved" in data["tokens"]
        # The backend was never contacted in dry-run mode.
        assert backend.chat_completions_create.call_count == 0


class TestDegradationHeader:
    """Review §11 / P4b: X-MOEPT-Optimization-Degraded surfaces swallowed stage
    failures."""

    def test_degradation_header_lists_failed_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import moeptimizer.app as app_module

        _reset_metrics()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *a, **kw: _mock_backend(cached_tokens=15))

        # Force the context-canonicalization stage to fail so it is recorded in
        # the degradation vector. The pipeline swallows it (logs a warning) and
        # falls back, but the failure must surface in the response header. We use
        # a large input so the stage is actually reached (it is skipped on lean
        # contexts), and patch the canonicalizer's method directly.
        from moeptimizer.context_canonicalizer import get_context_canonicalizer

        real_canonicalize = get_context_canonicalizer().canonicalize

        def _failing_canonicalize(self, messages):
            real_canonicalize(messages)
            raise RuntimeError("injected canonicalization failure")

        monkeypatch.setattr(
            get_context_canonicalizer().__class__,
            "canonicalize",
            _failing_canonicalize,
        )

        # A large enough prompt to exceed the proactive threshold and reach the
        # canonicalization stage. The balanced profile derives its token budget
        # dynamically from the live backend window (budget_window_fraction=0.06 of
        # the 262144-token window -> ~15728 tokens) and applies proactive_trim_ratio
        # 0.6, so the proactive threshold is ~9436 tokens. Size the input well above
        # that so the stage is actually reached even though the tokenizer compresses
        # repetitive "x" text efficiently.
        big = ("Implement a feature. " * 400) + ("x" * 90000)
        app = create_app(AppConfig())
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": big}],
                    "max_tokens": 8192,
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        # The degradation header is present and names the failing stage.
        assert "X-MOEPT-Optimization-Degraded" in resp.headers
        assert "context_canonicalization" in resp.headers["X-MOEPT-Optimization-Degraded"]
        metrics = client.get("/v1/metrics").json()
        assert metrics["degradation_counts"]["context_canonicalization"] >= 1


class TestMetricsDashboard:
    """Review §11 / P4c: /v1/metrics/ui serves a self-contained HTML dashboard."""

    def test_metrics_ui_returns_html(self) -> None:
        app = create_app(AppConfig())
        with TestClient(app) as client:
            resp = client.get("/v1/metrics/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "<html" in body
        assert "/v1/metrics" in body
        # No external assets — fully self-contained.
        assert "src=" not in body
        assert "http://" not in body and "https://" not in body


class TestConfigHotReload:
    """Review §11.5 / C9: live config reload without restart (SIGUSR2 or endpoint)."""

    def test_session_manager_reload_config_swaps_config(self) -> None:
        from moeptimizer.session_manager import SessionManager

        sm = SessionManager(config=AppConfig())
        before = sm._config
        # Mutate the live config object to prove reload replaces the reference.
        new_cfg = sm.reload_config()
        assert new_cfg is not before
        assert sm._config is new_cfg

    def test_reload_endpoint_returns_ok(self) -> None:
        app = create_app(AppConfig())
        with TestClient(app) as client:
            resp = client.post("/v1/config/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "quality_profile" in body

    def test_reload_endpoint_disabled_when_flag_off(self) -> None:
        cfg = AppConfig()
        cfg.agentic.config_hot_reload_enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            resp = client.post("/v1/config/reload")
        assert resp.status_code == 403
        assert resp.json()["status"] == "disabled"

    def test_existing_session_keeps_config_after_reload(self) -> None:
        """In-flight sessions must not race a mid-turn config change (C9)."""
        from moeptimizer.session_manager import SessionManager

        sm = SessionManager(config=AppConfig())
        opt = sm.get_or_create("sess-1")
        old_config = opt._config
        sm.reload_config()
        # The existing session's optimizer still holds its original config object.
        assert sm.get_or_create("sess-1")._config is old_config
        # A brand-new session picks up the reloaded config.
        assert sm.get_or_create("sess-2")._config is sm._config


class TestExpandContentTool:
    """B1 (review §4.1.2): model-facing expand_content tool + fulfilment."""

    def _optimizer_with_content(self, content: str):
        from moeptimizer.optimizer import AgentContextOptimizer

        opt = AgentContextOptimizer(AppConfig())
        handle = opt.content_store.put(content)
        return opt, handle

    def test_expand_content_tool_schema(self) -> None:
        from moeptimizer.content_store import EXPAND_TOOL_NAME, expand_content_tool

        tool = expand_content_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == EXPAND_TOOL_NAME
        assert "handle" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["handle"]

    def test_extract_placeholder_handle(self) -> None:
        from moeptimizer.content_store import extract_placeholder_handle

        placeholder = '[original retained: handle=abc123def456, 5000 chars; call expand_content(handle="abc123def456") to retrieve the full original]'
        assert extract_placeholder_handle(placeholder) == "abc123def456"
        assert extract_placeholder_handle("no placeholder here") is None

    def test_optimizer_expand_content_round_trip(self) -> None:
        original = "def f():\n    return 42\n" * 50
        opt, handle = self._optimizer_with_content(original)
        assert opt.expand_content(handle) == original
        assert opt.expand_content("unknown-handle") is None

    def test_expand_content_retrieval_repeats_until_evicted(self) -> None:
        from moeptimizer.content_store import ContentStore

        opt, handle = self._optimizer_with_content("repeatable content")
        assert opt.expand_content(handle) == "repeatable content"
        assert opt.expand_content(handle) == "repeatable content"

        opt.content_store = ContentStore(max_entries=1)
        opt.content_store.put("new content")
        assert opt.expand_content(handle) is None

    def test_expand_tool_results_fulfils_expand_call(self) -> None:
        from moeptimizer.app import _expand_tool_results

        original = "the full original tool output"
        opt, handle = self._optimizer_with_content(original)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "expand_content", "arguments": f'{{"handle": "{handle}"}}'},
            }],
        }
        results = _expand_tool_results(opt, message)
        assert results is not None
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["content"] == original

    def test_expand_tool_results_none_for_non_expand(self) -> None:
        from moeptimizer.app import _expand_tool_results

        opt, _ = self._optimizer_with_content("x")
        # A non-expand tool call must go to the client -> None.
        message = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "some_other_tool", "arguments": "{}"},
            }],
        }
        assert _expand_tool_results(opt, message) is None
        # No tool calls -> None.
        assert _expand_tool_results(opt, {"role": "assistant", "content": "hi"}) is None
        # No optimizer -> None.
        assert _expand_tool_results(None, message) is None

    def test_expand_tool_results_missing_handle_placeholder(self) -> None:
        from moeptimizer.app import _expand_tool_results

        opt, _ = self._optimizer_with_content("x")
        message = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "expand_content", "arguments": '{"handle": "does-not-exist"}'},
            }],
        }
        results = _expand_tool_results(opt, message)
        assert results is not None
        assert "no longer available" in results[0]["content"]

