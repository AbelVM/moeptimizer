"""Tests for backend client MTP integration."""

from types import SimpleNamespace
from typing import Any

from moeptimizer.backend_client import (
    LemonadeClient,
    SpeculativeDecoder,
)


class _FakeChat:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(choices=[], usage=None)


class _FakeCompletions:
    def __init__(self) -> None:
        self._chat = _FakeChat()
        self.kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return await self._chat.create(**kwargs)

    @property
    def chat(self) -> _FakeChat:
        return self._chat


class _FakeCompletionsRoot:
    def __init__(self) -> None:
        self._completions = _FakeCompletions()

    @property
    def completions(self) -> _FakeCompletions:
        return self._completions


class _FakeClient:
    def __init__(self) -> None:
        self._chat = _FakeCompletionsRoot()

    @property
    def chat(self) -> _FakeCompletionsRoot:
        return self._chat


class TestLemonadeClient:
    def test_client_creation(self) -> None:
        """Client can be created with base URL."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client is not None

    def test_speculative_decoder_disabled(self) -> None:
        """Speculative decoder is disabled by default."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client.speculative_decoder is None

    def test_enable_speculative_decoding(self) -> None:
        """Speculative decoding can be enabled."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        client.enable_speculative_decoding()
        assert client.speculative_decoder is not None

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

    async def test_chat_completions_create_strips_proxy_internal_extra_body(self) -> None:
        """Proxy-internal MTP/KV fields must not reach Lemonade."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            extra_body={
                "metadata": {"purpose": "test"},
                "kv_cache_warmup": {"enabled": True},
                "expert_hints": [{"position": 0, "experts": [1]}],
            },
        )

        assert fake.chat.completions.kwargs["extra_body"] == {
            "metadata": {"purpose": "test"},
        }

    async def test_speculative_decoder_does_not_inject_nonstandard_extra_body(self) -> None:
        """Enabled speculative wrapper must not send unsupported Lemonade fields."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        client.enable_speculative_decoding(mtp_lookahead=2)
        fake = _FakeClient()
        client._client = fake  # type: ignore[assignment]

        await client.chat_completions_create(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            extra_body={"expert_hints": [{"position": 0, "experts": [1]}]},
        )

        extra_body = fake.chat.completions.kwargs["extra_body"]
        assert isinstance(extra_body, dict)
        assert "speculative_decoding" not in extra_body
        assert "expert_hints" not in extra_body

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


class TestSpeculativeDecoder:
    def test_get_temperature_for_mtp_confidence(self) -> None:
        """Temperature is adjusted based on MTP confidence.

        For precise coding tasks, target ~0.6 for best results.
        """
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        decoder = SpeculativeDecoder(client)

        # High confidence → precise coding temperature
        assert decoder.get_temperature_for_mtp_confidence(0.9) == 0.5
        # Medium confidence → recommended for coding
        assert decoder.get_temperature_for_mtp_confidence(0.6) == 0.6
        # Low confidence → allow exploration
        assert decoder.get_temperature_for_mtp_confidence(0.3) == 0.7

    def test_get_stats(self) -> None:
        """Stats are tracked correctly."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        decoder = SpeculativeDecoder(client)
        stats = decoder.get_stats()
        assert "accepted" in stats
        assert "rejected" in stats
