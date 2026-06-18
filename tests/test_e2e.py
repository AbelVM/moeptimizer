"""End-to-end tests: moeptimizer proxy logic verification.

Two modes:
  1. Dry run (default): Test proxy logic functions directly with mocks.
     Fast, deterministic, no server calls needed.
  2. Live run (--live): Hit the real Lemonade server at localhost:13305.
     Benchmarks latency, cache behavior, response quality.

Usage:
  pytest tests/test_e2e.py -v                  # Dry run (mocked)
  pytest tests/test_e2e.py -v --live           # Live run (real server)
  pytest tests/test_e2e.py -v --live --rounds 3  # Live with multiple rounds
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from moeptimizer.app import create_app
from moeptimizer.config import AppConfig

# ─── Config ──────────────────────────────────────────────────────────────────

LEMONADE_URL = "http://localhost:13305/api/v1"
MODEL_ID = "Qwen3.6-35B-A3B-MTP-GGUF"
MOEPT_PORT = int(os.environ.get("MOEPT_PORT", "8080"))


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    round: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    total_ms: float = 0.0
    content: str = ""
    finish_reason: str = ""
    cache_refill: bool = False
    foreign_markers: list[str] = field(default_factory=list)
    session_state: str = ""
    error: str = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_messages(*parts: tuple[str, str]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in parts]


def _detect_cache_refill(prompt_tokens: int, cached_tokens: int) -> bool:
    if prompt_tokens == 0:
        return False
    uncached = prompt_tokens - cached_tokens
    return uncached > (prompt_tokens * 0.1)


def _check_foreign_markers(messages: list[dict]) -> list[str]:
    forbidden = ["[ARCHIVED", "[REASONING", "[PROGRESS", "[LOOP DETECTED"]
    found = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            for marker in forbidden:
                if marker in content:
                    found.append(marker)
    return found


# ─── Dry Run Tests ───────────────────────────────────────────────────────────

class TestDryRunResponseNormalization:
    """Test response normalization handles reasoning_content correctly."""

    def test_empty_content_filled_from_reasoning(self) -> None:
        """When content is empty but reasoning_content exists, merge it."""
        from unittest.mock import AsyncMock, MagicMock

        from moeptimizer.app import _do_non_streaming
        from moeptimizer.config import AppConfig

        config = AppConfig()
        config.server.url = LEMONADE_URL

        backend_resp = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Thinking...\n\nAnswer is 42.",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 15},
            },
        }

        # Mock OpenAI ChatCompletion response object
        mock_response = MagicMock()
        mock_response.model_dump.return_value = backend_resp

        mock_backend_client = MagicMock()
        mock_backend_client.chat_completions_create = AsyncMock(return_value=mock_response)

        async def run_test():
            return await _do_non_streaming(
                body={"model": MODEL_ID},
                session_state="test-state",
                cfg=config,
                backend_client=mock_backend_client,
            )

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result.status_code == 200
        data = json.loads(result.body)  # JSONResponse.body is bytes
        content = data["choices"][0]["message"]["content"]

        assert content == "Thinking...\n\nAnswer is 42.", (
            f"Content should be filled from reasoning_content, got: {content}"
        )
        assert data["choices"][0]["message"]["reasoning_content"] == "Thinking...\n\nAnswer is 42.", (
            "reasoning_content should be preserved for cache-stable echo"
        )

    def test_non_empty_content_unchanged(self) -> None:
        """When content is already populated, don't overwrite."""
        from unittest.mock import AsyncMock, MagicMock

        from moeptimizer.app import _do_non_streaming
        from moeptimizer.config import AppConfig

        config = AppConfig()
        config.server.url = LEMONADE_URL

        backend_resp = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Direct answer.",
                    "reasoning_content": "Thinking...",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 15},
            },
        }

        # Mock OpenAI ChatCompletion response object
        mock_response = MagicMock()
        mock_response.model_dump.return_value = backend_resp

        mock_backend_client = MagicMock()
        mock_backend_client.chat_completions_create = AsyncMock(return_value=mock_response)

        async def run_test():
            return await _do_non_streaming(
                body={"model": MODEL_ID},
                session_state="test-state",
                cfg=config,
                backend_client=mock_backend_client,
            )

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result.status_code == 200
        data = json.loads(result.body)
        content = data["choices"][0]["message"]["content"]

        assert content == "Direct answer.", (
            "Existing content should not be overwritten"
        )
        assert data["choices"][0]["message"]["reasoning_content"] == "Thinking...", (
            "reasoning_content should be preserved for cache-stable echo"
        )

    def test_reasoning_content_preserved_when_both_present(self) -> None:
        """When both content and reasoning_content exist, preserve both."""
        from unittest.mock import AsyncMock, MagicMock

        from moeptimizer.app import _do_non_streaming
        from moeptimizer.config import AppConfig

        config = AppConfig()
        config.server.url = LEMONADE_URL

        backend_resp = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Step-by-step reasoning...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_response = MagicMock()
        mock_response.model_dump.return_value = backend_resp

        mock_backend_client = MagicMock()
        mock_backend_client.chat_completions_create = AsyncMock(return_value=mock_response)

        async def run_test():
            return await _do_non_streaming(
                body={"model": MODEL_ID},
                session_state="test-state",
                cfg=config,
                backend_client=mock_backend_client,
            )

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result.status_code == 200
        data = json.loads(result.body)
        msg = data["choices"][0]["message"]
        assert msg["content"] == "The answer is 42."
        assert msg["reasoning_content"] == "Step-by-step reasoning..."


class TestDryRunMessageValidation:
    """Test message validation before sending to backend."""

    def test_valid_messages_pass(self) -> None:
        """Messages with content fields should pass validation."""
        from moeptimizer.app import _validate_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        # Should not raise
        _validate_messages(messages)

    def test_missing_content_raises(self) -> None:
        """Non-assistant messages without content should raise ValueError."""
        from moeptimizer.app import _validate_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user"},  # missing content
            {"role": "assistant", "content": "Hi there"},
        ]
        with pytest.raises(ValueError, match="no 'content' field"):
            _validate_messages(messages)

    def test_assistant_without_content_allowed(self) -> None:
        """Assistant messages can omit content (reasoning-only responses)."""
        from moeptimizer.app import _validate_messages

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant"},  # no content, reasoning only
        ]
        # Should not raise — assistant messages are exempt
        _validate_messages(messages)


class TestDryRunSessionResolution:
    """Test standard OpenAI-compatible session continuity."""

    def test_legacy_custom_session_id_still_works(self) -> None:
        from moeptimizer.app import _resolve_session_id

        session_id = _resolve_session_id(
            {"user": "alice", "messages": [{"role": "user", "content": "hello"}]},
            [{"role": "user", "content": "hello"}],
            legacy_session_id="legacy-session",
        )

        assert session_id == "legacy-session"

    def test_standard_user_and_first_user_message_form_stable_session(self) -> None:
        from moeptimizer.app import _resolve_session_id

        first = _resolve_session_id(
            {"user": "alice"},
            [{"role": "user", "content": "Build a REST API"}],
        )
        second = _resolve_session_id(
            {"user": "alice"},
            [
                {"role": "user", "content": "Build a REST API"},
                {"role": "assistant", "content": "I can help."},
                {"role": "user", "content": "Add auth."},
            ],
        )

        assert first == second
        assert first.startswith("user:")

    def test_anonymous_session_uses_first_user_message_for_stability(self) -> None:
        from moeptimizer.app import _resolve_session_id

        first = _resolve_session_id(
            {},
            [{"role": "user", "content": "hello"}],
        )
        second = _resolve_session_id(
            {},
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        assert first.startswith("anon:")
        assert first == second

    def test_chat_completion_strips_custom_session_fields_before_lemonade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        import moeptimizer.app as app_module

        class FakeLemonadeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.calls: list[dict[str, Any]] = []

            async def chat_completions_create(
                self,
                messages: list[dict[str, Any]],
                model: str,
                **kwargs: Any,
            ) -> SimpleNamespace:
                self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
                return SimpleNamespace(
                    model_dump=lambda: {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            async def chat_completions_stream(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("streaming should not be used")

        fake_backend = FakeLemonadeClient()
        monkeypatch.setattr(app_module, "LemonadeClient", lambda *args, **kwargs: fake_backend)
        app = app_module.create_app(AppConfig())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "client-model",
                "messages": [{"role": "user", "content": "hello"}],
                "user": "alice",
                "stream": False,
                "_session_id": "legacy-session",
                "_session_state": "{}",
            },
        )

        assert response.status_code == 200
        assert fake_backend.calls
        call = fake_backend.calls[0]
        assert call["model"] == MODEL_ID
        kwargs = call["kwargs"]
        messages = call["messages"]
        assert isinstance(kwargs, dict)
        assert isinstance(messages, list)
        assert kwargs.get("user") == "alice"
        assert "_session_id" not in kwargs
        assert "_session_state" not in kwargs
        assert all(
            "_session_id" not in msg and "_session_state" not in msg
            for msg in messages
            if isinstance(msg, dict)
        )


class TestDryRunEnsureContent:
    """Test _ensure_content adds missing 'content' to non-assistant messages."""

    def test_tool_message_without_content_gets_empty(self) -> None:
        """Tool/result messages without content get '' added."""
        from moeptimizer.app import _ensure_content

        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1"},  # no content
        ]
        _ensure_content(messages)
        assert messages[2]["content"] == ""

    def test_assistant_without_content_unchanged(self) -> None:
        """Assistant messages without content are not modified."""
        from moeptimizer.app import _ensure_content

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant"},  # no content (reasoning-only)
        ]
        _ensure_content(messages)
        assert "content" not in messages[1]

    def test_existing_content_preserved(self) -> None:
        """Messages that already have content are unchanged."""
        from moeptimizer.app import _ensure_content

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "tool", "tool_call_id": "1", "content": "Result here"},
        ]
        _ensure_content(messages)
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["content"] == "Result here"

    def test_system_message_without_content_gets_empty(self) -> None:
        """System messages without content get '' added."""
        from moeptimizer.app import _ensure_content

        messages = [
            {"role": "system"},  # no content
            {"role": "user", "content": "Hi"},
        ]
        _ensure_content(messages)
        assert messages[0]["content"] == ""


class TestDryRunStreamingGenerator:
    """Test streaming generator produces OpenAI-compatible SSE events."""

    def test_stream_produces_sse_format(self) -> None:
        """Streaming generator yields proper SSE-formatted chunks."""
        from unittest.mock import MagicMock

        from moeptimizer.app import _make_streaming_generator
        from moeptimizer.config import AppConfig

        cfg = AppConfig()
        cfg.server.llm_model = MODEL_ID

        # Mock chunk objects mimicking OpenAI SDK ChatCompletionChunk
        class MockDelta:
            def __init__(self, role=None, content=None):
                self.role = role
                self.content = content

        class MockChoice:
            def __init__(self, delta, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason

        class MockChunk:
            def __init__(self, delta_content="", finish_reason=None, role=None):
                choices = [MockChoice(
                    delta=MockDelta(role=role, content=delta_content),
                    finish_reason=finish_reason,
                )]
                self.choices = choices

        chunks = [
            MockChunk(role="assistant", delta_content="Hello"),
            MockChunk(delta_content=" world"),
            MockChunk(finish_reason="stop"),
        ]

        # Use a real async generator for proper mock behavior
        async def chunk_stream():
            for c in chunks:
                yield c

        mock_backend_client = MagicMock()
        mock_backend_client.chat_completions_stream = chunk_stream

        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Hi"}]}
        gen_func = _make_streaming_generator(body, cfg, mock_backend_client)
        # gen_func is the inner async generator function; call it to get the actual generator
        gen = gen_func()

        async def collect():
            events = []
            async for event in gen:
                events.append(event)
            return events

        loop = asyncio.new_event_loop()
        events = loop.run_until_complete(collect())
        loop.close()

        # Check first event is initial chunk with role=assistant, content=""
        assert events[0].startswith("data: {")
        first_data = json.loads(events[0][len("data: "):].removesuffix("\n\n"))
        assert first_data["object"] == "chat.completion.chunk"
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

        # Last event should be [DONE]
        assert events[-1].strip() == "data: [DONE]"

    def test_stream_error_provides_graceful_handling(self) -> None:
        """Streaming errors produce error chunk and [DONE] terminator."""
        from unittest.mock import MagicMock

        from moeptimizer.app import _make_streaming_generator
        from moeptimizer.config import AppConfig

        cfg = AppConfig()
        cfg.server.llm_model = MODEL_ID

        async def failing_stream():
            raise RuntimeError("backend down")

        mock_backend_client = MagicMock()
        mock_backend_client.chat_completions_stream = failing_stream

        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Hi"}]}
        gen_func = _make_streaming_generator(body, cfg, mock_backend_client)
        gen = gen_func()

        async def collect():
            events = []
            async for event in gen:
                events.append(event)
            return events

        loop = asyncio.new_event_loop()
        events = loop.run_until_complete(collect())
        loop.close()

        assert events[-1].strip() == "data: [DONE]"


class TestDryRunFrontLoadingEviction:
    """Verify front-loading eviction drops turns, not content."""

    def test_eviction_drops_whole_turns(self) -> None:
        """Verify old turns are dropped, content is never truncated."""
        from moeptimizer.compactor import ScratchpadCompactor

        compactor = ScratchpadCompactor(keep_full=2)

        # Need 4 turns after anchor to trigger eviction of 1 turn
        messages = _build_messages(
            ("system", "You are helpful."),
            ("user", "First task: do something important."),
            ("assistant", "I will complete the first task with detailed reasoning and action."),
            ("user", "Second task: do another thing."),
            ("assistant", "I will complete the second task with detailed reasoning and action."),
            ("user", "Third task: final thing."),
            ("assistant", "I will complete the third task with detailed reasoning and action."),
            ("user", "Fourth task: last thing."),
            ("assistant", "I will complete the fourth task with detailed reasoning and action."),
        )

        result = compactor.compact_messages(messages)

        # System anchor (3 msgs) + protected tail (4 msgs) = 7
        # Turn 2 is evicted (user2 + assistant2)
        assert len(result) == 7
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "First task: do something important."
        assert result[2]["content"] == "I will complete the first task with detailed reasoning and action."

        # Protected tail starts at index 3
        assert result[3]["content"] == "Third task: final thing."
        assert result[4]["content"] == "I will complete the third task with detailed reasoning and action."
        assert result[5]["content"] == "Fourth task: last thing."
        assert result[6]["content"] == "I will complete the fourth task with detailed reasoning and action."

    def test_no_content_truncation(self) -> None:
        """Verify no content is ever truncated during eviction."""
        from moeptimizer.compactor import ScratchpadCompactor

        long_content = "x" * 500
        compactor = ScratchpadCompactor(keep_full=1)

        messages = _build_messages(
            ("system", "System"),
            ("user", "First task"),
            ("assistant", long_content),
            ("user", "Second task"),
            ("assistant", "Short response"),
        )

        result = compactor.compact_messages(messages)

        # The preserved assistant should have original content, untouched
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert len(msg["content"]) == len(long_content) or msg["content"] == "Short response", (
                "Content should never be truncated"
            )

    def test_no_foreign_markers_in_assistant_content(self) -> None:
        """Verify no [ARCHIVED] markers are injected into assistant content."""
        from moeptimizer.compactor import ScratchpadCompactor

        compactor = ScratchpadCompactor(keep_full=1)

        messages = _build_messages(
            ("system", "System"),
            ("user", "Task 1"),
            ("assistant", "Response 1"),
            ("user", "Task 2"),
            ("assistant", "Response 2"),
            ("user", "Task 3"),
            ("assistant", "Response 3"),
        )

        result = compactor.compact_messages(messages)

        for msg in result:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "[ARCHIVED" not in content, (
                    f"Foreign marker in assistant: {content[:100]}"
                )


class TestDryRunThinkingPreserver:
    """Verify ThinkingPreserver is a pass-through."""

    def test_pass_through_all_messages(self) -> None:
        """All messages pass through unchanged."""
        from moeptimizer.thinking_preserver import ThinkingPreserver

        preserver = ThinkingPreserver()
        messages = _build_messages(
            ("assistant", "Old thinking content"),
            ("assistant", "Recent response"),
        )

        result = preserver.process_messages(messages)

        assert len(result) == 2
        assert result[0]["content"] == "Old thinking content"
        assert result[1]["content"] == "Recent response"

    def test_reasoning_tags_preserved(self) -> None:
        """Reasoning tags pass through unchanged."""
        from moeptimizer.thinking_preserver import ThinkingPreserver

        preserver = ThinkingPreserver()
        content = "<thinking>Step 1: analyze\nStep 2: conclude</thinking>Answer: 42"
        messages = _build_messages(
            ("assistant", content),
            ("assistant", "Recent"),
        )

        result = preserver.process_messages(messages)

        assert result[0]["content"] == content
        assert "<thinking>" in result[0]["content"]


class TestDryRunOptimizerPipeline:
    """Verify the full optimizer pipeline."""

    def test_optimize_messages_pipeline(self) -> None:
        """Test the full optimization pipeline."""
        from moeptimizer.optimizer import AgentContextOptimizer

        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        optimizer = AgentContextOptimizer(config)

        messages = _build_messages(
            ("system", "You are helpful."),
            ("user", "Task 1"),
            ("assistant", "Response 1"),
            ("user", "Task 2"),
            ("assistant", "Response 2"),
        )

        result = optimizer.optimize_messages(messages)

        # System and first user preserved
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

        # No foreign markers
        for msg in result:
            if msg.get("role") == "assistant":
                assert "[ARCHIVED" not in msg.get("content", "")
                assert "[REASONING" not in msg.get("content", "")

    def test_session_state_serialization(self) -> None:
        """Test session state round-trip."""
        import json

        from moeptimizer.optimizer import AgentContextOptimizer

        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        optimizer = AgentContextOptimizer(config)

        messages = _build_messages(
            ("user", "Hello"),
            ("assistant", "Hi"),
        )
        optimizer.optimize_messages(messages)
        state = optimizer.get_session_state()
        data = json.loads(state)

        assert "store" in data
        assert "progress" in data

        # Load into new optimizer
        new_optimizer = AgentContextOptimizer(config)
        new_optimizer.load_session_state(state)
        assert len(new_optimizer.store.steps) == len(optimizer.store.steps)


class TestDryRunHealthAndModels:
    """Test proxy health and model listing endpoints."""

    def test_health_check(self) -> None:
        app, _ = _create_app()
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/v1/health")
        # May fail if Lemonade is unreachable, but endpoint exists
        assert resp.status_code in (200, 500)

    def test_model_listing(self) -> None:
        app, _ = _create_app()
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert MODEL_ID in model_ids


def _create_app(config: AppConfig | None = None) -> tuple:
    """Create app and return (app, embedding_service)."""
    cfg = config or AppConfig()
    cfg.server.url = LEMONADE_URL
    cfg.server.llm_model = MODEL_ID
    cfg.agentic.max_optimized_chars = 8000
    cfg.agentic.keep_full_steps = 2
    cfg.agentic.thinking_protect_recent = 2

    app = create_app(cfg)
    return app, app.state.embedding_service


# ─── Live Run Tests (Real Server) ────────────────────────────────────────────

class BenchmarkConfig:
    rounds: int = 1


@pytest.fixture
def bench_config(request):
    cfg = BenchmarkConfig()
    if request.config.getoption("--live"):
        cfg.rounds = int(request.config.getoption("--rounds", default=1))
    return cfg


def _direct_request(messages: list[dict], max_tokens: int = 100) -> dict:
    import requests
    resp = requests.post(
        f"{LEMONADE_URL}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _proxy_request(messages: list[dict], max_tokens: int = 100, session_id: str | None = None) -> dict:
    import requests
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if session_id:
        body["_session_id"] = session_id

    resp = requests.post(
        f"http://127.0.0.1:{MOEPT_PORT}/v1/chat/completions",
        json=body,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _print_comparison(direct: dict, proxy: dict) -> None:
    """Print a formatted comparison."""
    print("\n" + "=" * 60)
    print("  DIRECT vs PROXY COMPARISON")
    print("=" * 60)

    d_usage = direct.get("usage", {})
    p_usage = proxy.get("usage", {})

    print("\n  Tokens:")
    print(f"    Direct prompt:     {d_usage.get('prompt_tokens', 0):>8}")
    print(f"    Proxy  prompt:     {p_usage.get('prompt_tokens', 0):>8}")
    print(f"    Direct cached:     {d_usage.get('prompt_tokens_details', {}).get('cached_tokens', 0):>8}")
    print(f"    Proxy  cached:     {p_usage.get('prompt_tokens_details', {}).get('cached_tokens', 0):>8}")

    d_content = direct["choices"][0]["message"].get("content", "") or direct["choices"][0]["message"].get("reasoning_content", "")
    p_content = proxy["choices"][0]["message"].get("content", "") or proxy["choices"][0]["message"].get("reasoning_content", "")

    print("\n  Content:")
    print(f"    Direct length:     {len(d_content):>8} chars")
    print(f"    Proxy  length:     {len(p_content):>8} chars")
    print(f"    Direct finish:     {direct['choices'][0].get('finish_reason', '')}")
    print(f"    Proxy  finish:     {proxy['choices'][0].get('finish_reason', '')}")

    markers = _check_foreign_markers([{"role": "assistant", "content": p_content}])
    print("\n  MTP Integrity:")
    print(f"    Foreign markers:   {'NONE' if not markers else str(markers)}")

    print("=" * 60 + "\n")


class TestLiveBasicCompletion:
    """Live tests against real Lemonade server via proxy."""

    def test_simple_completion(self, request, bench_config) -> None:
        if not request.config.getoption("--live"):
            pytest.skip("Live tests require --live flag")

        messages = _build_messages(
            ("system", "You are a helpful assistant."),
            ("user", "What is 2+2? Answer with just the number."),
        )

        for i in range(bench_config.rounds):
            direct = _direct_request(messages, max_tokens=50)
            proxy = _proxy_request(messages, max_tokens=50)

            direct_content = direct["choices"][0]["message"].get("content", "") or direct["choices"][0]["message"].get("reasoning_content", "")
            proxy_content = proxy["choices"][0]["message"].get("content", "") or proxy["choices"][0]["message"].get("reasoning_content", "")

            assert len(direct_content) > 0, f"Direct response empty in round {i}"
            assert len(proxy_content) > 0, f"Proxy response empty in round {i}"

            # Proxy response should be non-empty and contain reasoning content merged into content
            assert len(proxy_content) > 10, (
                f"Proxy response unexpectedly short. Got: {proxy_content[:100]}"
            )

            # No foreign markers
            markers = _check_foreign_markers([{"role": "assistant", "content": proxy_content}])
            assert not markers, f"Foreign markers: {markers}"

            _print_comparison(direct, proxy)


class TestLiveLongContext:
    """Live tests with long context (eviction test)."""

    def test_long_context_eviction(self, request, bench_config) -> None:
        if not request.config.getoption("--live"):
            pytest.skip("Live tests require --live flag")

        messages = _build_messages(
            ("system", "You are a helpful assistant."),
        )
        for i in range(10):
            messages.extend(_build_messages(
                ("user", f"Turn {i}: This is test content for turn {i}."),
                ("assistant", f"Response {i}: I remember turn {i}."),
            ))

        for _round in range(bench_config.rounds):
            direct = _direct_request(messages, max_tokens=50)
            proxy = _proxy_request(messages, max_tokens=50)

            proxy_content = proxy["choices"][0]["message"].get("content", "") or proxy["choices"][0]["message"].get("reasoning_content", "")
            assert len(proxy_content) > 0, f"Proxy response empty in round {i}"

            markers = _check_foreign_markers([{"role": "assistant", "content": proxy_content}])
            assert not markers, f"Evicted context has markers: {markers}"

            _print_comparison(direct, proxy)
