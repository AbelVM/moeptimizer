"""Tests for live backend capability auto-detection."""

from types import SimpleNamespace
from unittest.mock import patch

from moeptimizer.backend_capabilities import (
    BackendCapabilities,
    BackendCapabilityProbe,
    _derive_hf_tokenizer_id,
)


class _Resp:
    def __init__(self, json: dict, status: int = 200) -> None:
        self._json = json
        self.status_code = status

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    """Async context manager whose get/post return queued responses in order.

    When the queue is exhausted (e.g. a later tokenize_count call), it falls
    back to ``fallback`` so the fake can be reused across multiple client
    creations (the probe builds a fresh httpx client per call).
    """

    def __init__(self, responses: list[_Resp], fallback: _Resp | None = None) -> None:
        self._responses = list(responses)
        self._fallback = fallback or _Resp({"tokens": [1, 2, 3]})
        self.calls: list[str] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def _next(self) -> _Resp:
        return self._responses.pop(0) if self._responses else self._fallback

    async def get(self, url: str) -> _Resp:
        self.calls.append(url)
        return self._next()

    async def post(self, url: str, json: dict | None = None) -> _Resp:
        self.calls.append(url)
        return self._next()


def _make_probe(
    responses: list[_Resp], ttl: float = 30.0, fallback: _Resp | None = None
) -> BackendCapabilityProbe:
    probe = BackendCapabilityProbe(
        base_url="http://localhost:13305/api/v1",
        model="Qwen3.6-35B-A3B-MTP-GGUF",
        ttl_seconds=ttl,
    )
    probe._fake = _FakeAsyncClient(responses, fallback=fallback)  # type: ignore[attr-defined]
    return probe


def _patch(probe: BackendCapabilityProbe):
    return patch(
        "moeptimizer.backend_capabilities.httpx.AsyncClient",
        return_value=probe._fake,  # type: ignore[attr-defined]
    )


# --- pure helpers ---------------------------------------------------------


def test_native_base_strips_v1() -> None:
    probe = BackendCapabilityProbe(
        base_url="http://localhost:13305/api/v1",
        model="M",
        ttl_seconds=30.0,
    )
    assert probe._native_base(SimpleNamespace(llm_backend_url="http://localhost:8002/v1")) == (
        "http://localhost:8002"
    )
    assert probe._native_base(SimpleNamespace(llm_backend_url="http://host:8002")) == (
        "http://host:8002"
    )
    assert probe._native_base(SimpleNamespace(llm_backend_url="http://host:8002/")) == (
        "http://host:8002"
    )
    # fallback to aggregated base when no per-model url
    assert probe._native_base(SimpleNamespace(llm_backend_url=None)) == (
        "http://localhost:13305/api/v1"
    )


def test_derive_hf_tokenizer_id() -> None:
    assert (
        _derive_hf_tokenizer_id("unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_M")
        == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    )
    assert _derive_hf_tokenizer_id("unsloth/Qwen3.6-35B-A3B-MTP-GGUF") == (
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    )
    assert _derive_hf_tokenizer_id("/local/path/model.gguf") is None
    assert _derive_hf_tokenizer_id("") is None


# --- GPU device (full capabilities) --------------------------------------


async def test_probe_gpu_device_full_caps() -> None:
    health = {
        "all_models_loaded": [
            {
                "type": "llm",
                "model_name": "Qwen3.6-35B-A3B-MTP-GGUF",
                "device": "gpu",
                "backend_url": "http://localhost:8002/v1",
            }
        ]
    }
    slots = [{"id": 0, "n_ctx": 262144, "speculative": True}]
    models = {
        "data": [
            {
                "id": "Qwen3.6-35B-A3B-MTP-GGUF",
                "checkpoint": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_M",
                "labels": ["vision", "tool-calling", "mtp"],
                "max_context_window": 262144,
            }
        ]
    }
    probe = _make_probe(
        [_Resp(models), _Resp(health), _Resp(slots), _Resp({"tokens": [1, 2, 3]})]
    )
    with _patch(probe):
        caps = await probe.get()

    assert caps.device == "gpu"
    assert caps.slot_pinning is True
    assert caps.total_slots == 1
    assert caps.mtp is True
    assert caps.remote_tokenize is True
    assert caps.tokenizer_id == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    assert caps.max_context_window == 262144
    # native endpoints routed to the LLM backend root, not the aggregated /api/v1
    assert any(u.startswith("http://localhost:8002/") for u in probe._fake.calls)
    assert any(u.endswith("/slots") for u in probe._fake.calls)
    assert any(u.endswith("/tokenize") for u in probe._fake.calls)
    with _patch(probe):
        assert await probe.tokenize_count("a b c") == 3


# --- NPU device (no slots, no native tokenize) ---------------------------


async def test_probe_npu_device_no_slots() -> None:
    health = {
        "all_models_loaded": [
            {
                "type": "llm",
                "model_name": "Qwen3.6-35B-A3B-MTP-GGUF",
                "device": "npu",
                "backend_url": "http://localhost:8001/v1",
            }
        ]
    }
    slots = _Resp({"error": "not supported by npu"}, status=400)
    models = {
        "data": [
            {
                "id": "Qwen3.6-35B-A3B-MTP-GGUF",
                "checkpoint": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_M",
                "labels": ["vision", "tool-calling"],
                "max_context_window": 262144,
            }
        ]
    }
    probe = _make_probe(
        [_Resp(models), _Resp(health), slots],
        fallback=_Resp({"error": "not supported by npu"}),
    )
    with _patch(probe):
        caps = await probe.get()

    assert caps.device == "npu"
    assert caps.slot_pinning is False
    assert caps.total_slots == 0
    assert caps.mtp is False  # no "mtp" label, no speculative slot
    assert caps.remote_tokenize is False
    assert caps.tokenizer_id == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"


# --- TTL caching ----------------------------------------------------------


async def test_probe_ttl_caches() -> None:
    health = {
        "all_models_loaded": [
            {
                "type": "llm",
                "model_name": "M",
                "device": "gpu",
                "backend_url": "http://localhost:8002/v1",
            }
        ]
    }
    slots = [{"id": 0, "n_ctx": 4096, "speculative": False}]
    models = {
        "data": [
            {
                "id": "M",
                "checkpoint": "org/M:Q4",
                "labels": [],
                "max_context_window": 4096,
            }
        ]
    }
    probe = _make_probe(
        [_Resp(models), _Resp(health), _Resp(slots), _Resp({"tokens": [1]})],
        ttl=1000.0,
    )
    with _patch(probe):
        first = await probe.get()
        second = await probe.get()  # should hit cache, no new calls
    assert first.device == second.device == "gpu"
    # only the first probe issued network calls
    assert len(probe._fake.calls) == 4


# --- failure handling -----------------------------------------------------


async def test_probe_handles_connection_error() -> None:
    class _BoomClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            raise RuntimeError("connection refused")

        async def post(self, url: str, json: dict | None = None) -> _Resp:
            raise RuntimeError("connection refused")

    probe = BackendCapabilityProbe(
        base_url="http://localhost:13305/api/v1",
        model="M",
        ttl_seconds=30.0,
    )
    with patch(
        "moeptimizer.backend_capabilities.httpx.AsyncClient", return_value=_BoomClient()
    ):
        caps = await probe.get()
    # safe defaults, never raises
    assert caps.device is None
    assert caps.slot_pinning is False
    assert caps.mtp is False
    assert caps.remote_tokenize is False


def test_capabilities_defaults() -> None:
    caps = BackendCapabilities()
    assert caps.device is None
    assert caps.slot_pinning is False
    assert caps.mtp is False
    assert caps.remote_tokenize is False
    assert caps.total_slots == 0
    assert caps.tokenizer_id is None
    assert caps.max_context_window is None
