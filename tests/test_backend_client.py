"""Tests for backend client MTP integration."""

from types import SimpleNamespace
from typing import Any

import openai

from moeptimizer.backend_client import (
    LemonadeClient,
)


class _FakeChat:
    def __init__(self, raise_on_create=None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._raise_on_create = raise_on_create

    async def create(self, **kwargs: object) -> SimpleNamespace:
        if self._raise_on_create is not None:
            raise self._raise_on_create
        self.kwargs = kwargs
        return SimpleNamespace(choices=[], usage=None)


class _FakeCompletions:
    def __init__(self, raise_on_create=None) -> None:
        self._chat = _FakeChat(raise_on_create=raise_on_create)
        self.kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        if self._chat._raise_on_create is not None:
            raise self._chat._raise_on_create
        self.kwargs = kwargs
        return await self._chat.create(**kwargs)

    @property
    def chat(self) -> _FakeChat:
        return self._chat


class _FakeCompletionsRoot:
    def __init__(self, raise_on_create=None) -> None:
        self._completions = _FakeCompletions(raise_on_create=raise_on_create)

    @property
    def completions(self) -> _FakeCompletions:
        return self._completions


class _FakeModels:
    def __init__(self, data) -> None:
        self.data = data

    async def list(self):
        return self


class _FakeClient:
    def __init__(self, *, models=None, raise_on_create=None) -> None:
        self._chat = _FakeCompletionsRoot(raise_on_create=raise_on_create)
        self._models = models

    @property
    def chat(self) -> _FakeCompletionsRoot:
        return self._chat

    @property
    def models(self) -> object:
        if self._models is None:
            raise AssertionError("models.list not configured for fake")
        return self._models


class TestLemonadeClient:
    def test_client_creation(self) -> None:
        """Client can be created with base URL."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client is not None

    async def test_chat_completions_create_forwards_standard_extra_body(self) -> None:
        """Chat completion requests forward standard OpenAI-compatible extra_body."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            extra_body={"metadata": {"purpose": "test"}},
        )

        assert fake.chat.completions.kwargs["extra_body"] == {
            "metadata": {"purpose": "test"},
        }

    async def test_chat_completions_create_routes_id_slot_into_extra_body(self) -> None:
        """id_slot is a llama.cpp extension: it must go in extra_body, not as a
        top-level kwarg the OpenAI SDK would reject."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            id_slot=3,
        )

        assert "id_slot" not in fake.chat.completions.kwargs
        assert fake.chat.completions.kwargs["extra_body"] == {"id_slot": 3}

    async def test_chat_completions_create_merges_id_slot_with_extra_body(self) -> None:
        """id_slot merges into an existing extra_body without clobbering it."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            id_slot=1,
            extra_body={"metadata": {"purpose": "test"}},
        )

        assert "id_slot" not in fake.chat.completions.kwargs
        assert fake.chat.completions.kwargs["extra_body"] == {
            "metadata": {"purpose": "test"},
            "id_slot": 1,
        }

    async def test_chat_completions_create_strips_proxy_internal_extra_body(self) -> None:
        """Proxy-internal MTP/expert fields must not reach Lemonade."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            extra_body={
                "metadata": {"purpose": "test"},
                "expert_hints": [{"position": 0, "experts": [1]}],
            },
        )

        assert fake.chat.completions.kwargs["extra_body"] == {
            "metadata": {"purpose": "test"},
        }

    async def test_chat_completions_create_strips_custom_session_fields(self) -> None:
        """Internal session fields must not reach a standard OpenAI backend."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            _session_id="session",
            _session_state="state",
        )

        kwargs = fake.chat.completions.kwargs
        assert "_session_id" not in kwargs
        assert "_session_state" not in kwargs

    async def test_detect_mtp_support_true_when_backend_accepts(self) -> None:
        """Probe returns True when the backend accepts the speculative key."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient(models=_FakeModels([SimpleNamespace(id="local-model")]))
        client._client = fake  # type: ignore[assignment]

        assert await client.detect_mtp_support() is True

    async def test_detect_mtp_support_false_on_reject(self) -> None:
        """Probe returns False when the backend rejects the speculative key."""
        import httpx

        err = openai.APIStatusError(
            "bad request",
            response=httpx.Response(422, request=httpx.Request("POST", "http://x")),
            body=None,
        )
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient(
            models=_FakeModels([SimpleNamespace(id="local-model")]),
            raise_on_create=err,
        )
        client._client = fake  # type: ignore[assignment]

        assert await client.detect_mtp_support() is False

    async def test_detect_mtp_support_false_on_connection_error(self) -> None:
        """Probe returns False (best-effort) when the backend is unreachable."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient(
            models=_FakeModels([SimpleNamespace(id="local-model")]),
            raise_on_create=openai.APIConnectionError(
                message="refused",
                request=SimpleNamespace(method="POST", url="http://x"),
            ),
        )
        client._client = fake  # type: ignore[assignment]

        assert await client.detect_mtp_support() is False

    async def test_detect_mtp_support_false_when_models_unavailable(self) -> None:
        """Probe returns False when the model list cannot be fetched."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        assert await client.detect_mtp_support() is False

    def test_enable_native_mtp_passthrough(self) -> None:
        """Passthrough can be enabled at runtime and is reflected in the flag."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client.native_mtp_passthrough is False
        client.enable_native_mtp_passthrough()
        assert client.native_mtp_passthrough is True

