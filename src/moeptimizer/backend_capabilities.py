"""BackendCapabilities — live, device-aware detection of what the backend supports.

The target backend (Lemonade) hot-swaps the LLM between NPU and GPU (llama.cpp)
runtimes, and the two runtimes expose *different* capabilities:

  - GPU / llama.cpp:  ``/slots`` works (slot pinning + ``speculative`` flag),
                      ``POST /tokenize`` returns exact token ids, ``/props`` is rich.
  - NPU:              ``/slots`` and ``/tokenize`` return "not supported by npu".

Model-identity metadata (``checkpoint``, ``labels``, ``max_context_window``) is
stable across both, but the *functional* capabilities (slot pinning, remote
tokenization) are runtime-dependent and can change while the proxy is running.

This module probes the live backend and caches the result with a short TTL so
capabilities follow the active device automatically, instead of being frozen at
startup or hard-coded in config. Manual config flags act as overrides:
``True``/``False`` force a value, ``None``/"auto" defer to detection.

Signals (verified against a live Lemonade backend):
  - device:            ``GET /health`` -> per-model ``device`` ("npu"|"gpu")
  - slot pinning:      ``GET /slots``  -> list => supported; error/JSON-error => no
  - MTP / speculative: ``/models[].labels`` contains "mtp" AND/OR
                       ``/slots[].speculative`` is true AND/OR
                       ``recipe_options.llamacpp_args`` has ``--spec-type ...mtp``
  - tokenizer id:      ``/models[].checkpoint`` (HF repo id of the GGUF)
  - remote tokenize:   ``POST /tokenize`` -> {"tokens":[...]} => exact counts
  - context window:    ``/models[].max_context_window``
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _derive_hf_tokenizer_id(checkpoint: str | None) -> str | None:
    """Extract a HuggingFace repo id from a Lemonade ``checkpoint`` string.

    Lemonade reports e.g.
        ``unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf``
    The part before the first ``:`` is the HF repo id, which
    ``transformers.AutoTokenizer`` can load (it ships ``tokenizer.json`` for
    GGUF-source repos). Returns None when nothing usable is present.
    """
    if not checkpoint or not isinstance(checkpoint, str):
        return None
    repo = checkpoint.split(":", 1)[0].strip()
    # A valid HF repo id looks like "org/name". Bare local paths / single tokens
    # are not loadable as repo ids, so we reject them here (the caller falls back).
    if "/" not in repo or repo.startswith((".", "/")):
        return None
    return repo


@dataclass
class BackendCapabilities:
    """Snapshot of what the live backend currently supports."""

    device: str | None = None            # "npu" | "gpu" | None (unknown)
    slot_pinning: bool = False           # /slots returned a usable slot list
    total_slots: int = 0
    mtp: bool = False                    # native MTP / speculative decoding active
    remote_tokenize: bool = False        # POST /tokenize returns exact token ids
    tokenizer_id: str | None = None      # HF repo id derived from checkpoint
    max_context_window: int | None = None
    labels: tuple[str, ...] = field(default_factory=tuple)
    llm_backend_url: str | None = None   # per-model backend base (e.g. llama.cpp :8002)
    probed_at: float = 0.0

    def summary(self) -> str:
        return (
            f"device={self.device} slot_pinning={self.slot_pinning} "
            f"total_slots={self.total_slots} mtp={self.mtp} "
            f"remote_tokenize={self.remote_tokenize} "
            f"tokenizer_id={self.tokenizer_id} ctx={self.max_context_window}"
        )


class BackendCapabilityProbe:
    """Probes and caches live backend capabilities with a short TTL.

    Thread-safe. A single probe touches only lightweight metadata endpoints
    (``/health``, ``/models``, ``/slots``, a tiny ``/tokenize``), all bounded by
    a short timeout so a slow or unreachable backend never blocks the caller.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        ttl_seconds: float = 30.0,
        probe_timeout: float = 4.0,
    ) -> None:
        # base_url is the OpenAI-style base, e.g. http://host:13305/api/v1.
        # The Lemonade-native endpoints live under the same prefix
        # (/api/v1/health, /api/v1/slots, /api/v1/tokenize, /api/v1/models).
        self._base = base_url.rstrip("/")
        self._model = model
        self._ttl = max(1.0, ttl_seconds)
        self._probe_timeout = probe_timeout
        self._lock = threading.Lock()
        self._cache: BackendCapabilities | None = None

    async def get(self, force: bool = False) -> BackendCapabilities:
        """Return cached capabilities, re-probing when stale or forced."""
        now = time.time()
        with self._lock:
            cached = self._cache
        if (
            not force
            and cached is not None
            and (now - cached.probed_at) < self._ttl
        ):
            return cached

        caps = await self._probe()
        with self._lock:
            # If the device changed since the last snapshot, log it: capabilities
            # (slot pinning, remote tokenize) flip with the runtime.
            if cached is not None and cached.device != caps.device:
                logger.info(
                    "Backend device changed %s -> %s; capabilities updated (%s)",
                    cached.device,
                    caps.device,
                    caps.summary(),
                )
            self._cache = caps
        return caps

    def cached(self) -> BackendCapabilities | None:
        """Return the last snapshot without probing (may be None/stale)."""
        with self._lock:
            return self._cache

    async def _probe(self) -> BackendCapabilities:
        caps = BackendCapabilities(probed_at=time.time())
        try:
            async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
                # Order matters: /health yields the LLM's own backend_url + device,
                # and the per-model backend is where /slots + /tokenize actually
                # work. The AGGREGATED /api/v1/slots and /api/v1/tokenize route to
                # the default (often NPU embedding) backend and wrongly report
                # "not supported by npu" even when the LLM is on GPU — verified
                # live. So we always prefer the per-model backend_url.
                await self._probe_models(client, caps)
                await self._probe_health(client, caps)
                await self._probe_slots(client, caps)
                await self._probe_tokenize(client, caps)
        except Exception as exc:
            logger.debug("Capability probe failed: %s", exc)
        logger.debug("Backend capabilities: %s", caps.summary())
        return caps

    async def _probe_models(
        self, client: httpx.AsyncClient, caps: BackendCapabilities
    ) -> None:
        """Read stable model-identity metadata (checkpoint, labels, ctx)."""
        try:
            r = await client.get(f"{self._base}/models")
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.debug("Capability probe: /models unavailable (%s)", exc)
            return
        entry = self._select_model_entry(data.get("data", []))
        if not entry:
            return
        caps.labels = tuple(
            str(x).lower() for x in entry.get("labels", []) if isinstance(x, str)
        )
        caps.tokenizer_id = _derive_hf_tokenizer_id(entry.get("checkpoint"))
        mcw = entry.get("max_context_window")
        if isinstance(mcw, int) and mcw > 0:
            caps.max_context_window = mcw
        # MTP can be declared via labels or the launch args llama.cpp was given.
        if "mtp" in caps.labels:
            caps.mtp = True
        recipe_opts = entry.get("recipe_options") or {}
        args = recipe_opts.get("llamacpp_args")
        if isinstance(args, str) and "--spec-type" in args and "mtp" in args:
            caps.mtp = True

    def _select_model_entry(self, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Pick the LLM entry matching our configured model, else first non-embedding."""
        if not entries:
            return None
        for e in entries:
            if e.get("id") == self._model:
                return e
        for e in entries:
            labels = [str(x).lower() for x in e.get("labels", []) if isinstance(x, str)]
            if "embeddings" not in labels and e.get("recipe") != "flm":
                return e
        return entries[0]

    async def _probe_health(
        self, client: httpx.AsyncClient, caps: BackendCapabilities
    ) -> None:
        """Read the active device + per-model backend_url for the LLM.

        The per-model ``backend_url`` (e.g. the llama.cpp server on :8002) is the
        endpoint where ``/slots`` and ``/tokenize`` actually work; the aggregated
        Lemonade endpoints route to the default backend and misreport support.
        """
        try:
            r = await client.get(f"{self._base}/health")
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.debug("Capability probe: /health unavailable (%s)", exc)
            return
        for m in data.get("all_models_loaded", []):
            if not isinstance(m, dict):
                continue
            if m.get("type") == "llm" or m.get("model_name") == self._model:
                dev = m.get("device")
                if isinstance(dev, str):
                    caps.device = dev.lower()
                url = m.get("backend_url")
                if isinstance(url, str) and url:
                    caps.llm_backend_url = url.rstrip("/")
                return

    def _native_base(self, caps: BackendCapabilities) -> str:
        """Return the base URL to use for llama.cpp-native endpoints.

        llama.cpp exposes ``/slots`` and ``/tokenize`` at the SERVER ROOT, not
        under the OpenAI ``/v1`` prefix (verified: ``:8002/slots`` works,
        ``:8002/v1/slots`` 404s). The per-model ``backend_url`` from /health ends
        in ``/v1``, so strip it. Fall back to the aggregated Lemonade base when
        the per-model URL is unknown.
        """
        base = caps.llm_backend_url
        if base:
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            return base.rstrip("/")
        return self._base

    async def _probe_slots(
        self, client: httpx.AsyncClient, caps: BackendCapabilities
    ) -> None:
        """Detect slot pinning support (a JSON list => supported).

        Probes the LLM's per-model backend (llama.cpp) directly so a GPU LLM is
        not misreported as slot-less just because the default backend is NPU.
        """
        base = self._native_base(caps)
        try:
            r = await client.get(f"{base}/slots")
            data = r.json()
        except Exception as exc:
            logger.debug("Capability probe: /slots unavailable (%s)", exc)
            return
        # NPU path returns {"error": {...}}; GPU/llama.cpp returns a list.
        if isinstance(data, list):
            caps.slot_pinning = True
            caps.total_slots = len(data)
            if any(isinstance(s, dict) and s.get("speculative") for s in data):
                caps.mtp = True
        else:
            caps.slot_pinning = False

    async def _probe_tokenize(
        self, client: httpx.AsyncClient, caps: BackendCapabilities
    ) -> None:
        """Detect exact remote tokenization (POST /tokenize returns token ids).

        Probes the LLM's per-model backend directly (see _probe_slots).
        """
        base = self._native_base(caps)
        try:
            r = await client.post(
                f"{base}/tokenize",
                json={"model": self._model, "content": "probe"},
            )
            data = r.json()
        except Exception as exc:
            logger.debug("Capability probe: /tokenize unavailable (%s)", exc)
            return
        caps.remote_tokenize = isinstance(data, dict) and isinstance(
            data.get("tokens"), list
        )

    def tokenize_count_sync(self, text: str) -> int | None:
        """Sync wrapper around ``tokenize_count`` for use from sync code.

        Tries ``asyncio.run()``; if we are already inside a running event loop
        (e.g. called from async app code via a sync optimizer), falls back to
        ``None`` so the caller uses local counting instead.
        """
        try:
            import asyncio

            return asyncio.run(self.tokenize_count(text))
        except RuntimeError:
            # Already in an async context — cannot start a new event loop.
            return None

    async def tokenize_count(self, text: str) -> int | None:
        """Return the exact remote token count for ``text``, or None if unavailable.

        Best-effort: any error (unsupported device, timeout) returns None so the
        caller falls back to local counting. Verified fast on the GPU path
        (~1ms small, ~4ms for ~2k tokens), but still gated behind the caller's
        own fingerprint cache so it is not called per-fragment on the hot path.

        Routes to the LLM's per-model backend when known (see _probe_slots).
        """
        caps = self.cached() or await self.get()
        base = self._native_base(caps)
        try:
            async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
                r = await client.post(
                    f"{base}/tokenize",
                    json={"model": self._model, "content": text},
                )
                data = r.json()
            toks = data.get("tokens") if isinstance(data, dict) else None
            if isinstance(toks, list):
                return len(toks)
        except Exception as exc:
            logger.debug("Remote tokenize failed: %s", exc)
        return None
