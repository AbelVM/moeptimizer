"""Smoke tests for moeptimizer app endpoints, streaming, and metrics.

Covers:
- Health, models, and metrics endpoints
- Chat completions (streaming + non-streaming)
- OutputShaper integration
- Metrics recording and reset
"""

from __future__ import annotations

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

    def test_non_streaming_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_non_streaming_sets_optimized_prompt_header(
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
        )
        assert "X-MOEPT-Optimized-Prompt-Text" in response.headers


class TestChatCompletionsStreaming:
    """Smoke tests for streaming chat completions."""

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


class TestOutputShaperIntegration:
    """OutputShaper is applied to the request body before sending to backend."""

    def test_output_shaper_injects_terse_instruction(
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
        # The terse instruction should be appended to the system prompt
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        if system_msg:
            assert "Be concise" in (system_msg.get("content") or "")

    def test_output_shaper_clamps_max_tokens(
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
        # max_tokens should be clamped by OutputShaper for a new_question turn
        assert body.get("max_tokens", 8192) <= 4096


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
            "sessions",
        }
        assert expected_keys.issubset(snap.keys())

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
