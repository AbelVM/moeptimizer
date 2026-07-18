"""FastAPI application — OpenAI-compatible proxy with agentic context optimization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError, APIStatusError, AsyncOpenAI

from moeptimizer.backend_client import LemonadeClient
from moeptimizer.config import AppConfig, get_config
from moeptimizer.embedding import EmbeddingService
from moeptimizer.optimizer import AgentContextOptimizer
from moeptimizer.output_shaper import OutputShaper
from moeptimizer.session_manager import SessionManager

logger = logging.getLogger(__name__)


class _ProxyMetrics:
    """Process-wide aggregate metrics for the proxy (review §11.1).

    Fed from the backend's real ``cached_tokens`` signal on every turn so
    operators can see whether the proxy is actually helping (prefix-cache reuse,
    token savings, latency delta). Cheap, lock-protected counters; no per-turn
    allocation on the request path beyond a couple of integer adds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cached_tokens = 0
        self.total_prompt_tokens = 0
        self.total_saved_tokens = 0
        self.total_latency_ms = 0.0
        # Count of turns where the backend returned an error (e.g. HTTP 500 for a
        # truncated tool call) while streaming/serving. Surfaced in /v1/metrics so
        # operators can distinguish "proxy not helping" from "backend failing".
        self.backend_errors = 0
        # Per-session breakdown, bounded LRU so it can never grow without limit
        # even under a flood of distinct session ids (review §11.1).
        self._per_session: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_sessions_tracked = 512

    def record_backend_error(self, session_id: str | None = None) -> None:
        """Record that the backend failed to serve a turn (best-effort counter)."""
        with self._lock:
            self.backend_errors += 1
            if session_id:
                entry = self._per_session.get(session_id)
                if entry is None:
                    entry = {
                        "requests": 0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "total_cached_tokens": 0,
                        "total_prompt_tokens": 0,
                        "total_saved_tokens": 0,
                        "total_latency_ms": 0.0,
                        "backend_errors": 0,
                    }
                    self._per_session[session_id] = entry
                entry["backend_errors"] = entry.get("backend_errors", 0) + 1
                self._per_session.move_to_end(session_id)
                while len(self._per_session) > self._max_sessions_tracked:
                    self._per_session.popitem(last=False)

    def record_turn(
        self,
        *,
        session_id: str | None = None,
        cached_tokens: int | None = None,
        prompt_tokens: int | None = None,
        saved_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self._lock:
            self.requests += 1
            if cached_tokens is not None:
                if cached_tokens > 0:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                self.total_cached_tokens += max(0, cached_tokens)
            if prompt_tokens is not None:
                self.total_prompt_tokens += max(0, prompt_tokens)
            if saved_tokens is not None:
                self.total_saved_tokens += max(0, saved_tokens)
            if latency_ms is not None:
                self.total_latency_ms += max(0.0, latency_ms)
            if session_id:
                self._record_session_locked(
                    session_id,
                    cached_tokens=cached_tokens,
                    prompt_tokens=prompt_tokens,
                    saved_tokens=saved_tokens,
                    latency_ms=latency_ms,
                )

    def _record_session_locked(
        self,
        session_id: str,
        *,
        cached_tokens: int | None,
        prompt_tokens: int | None,
        saved_tokens: int | None,
        latency_ms: float | None,
    ) -> None:
        """Update the per-session counters. Caller must hold ``self._lock``."""
        entry = self._per_session.get(session_id)
        if entry is None:
            entry = {
                "requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_cached_tokens": 0,
                "total_prompt_tokens": 0,
                "total_saved_tokens": 0,
                "total_latency_ms": 0.0,
                "backend_errors": 0,
            }
        entry["requests"] += 1
        if cached_tokens is not None:
            if cached_tokens > 0:
                entry["cache_hits"] += 1
            else:
                entry["cache_misses"] += 1
            entry["total_cached_tokens"] += max(0, cached_tokens)
        if prompt_tokens is not None:
            entry["total_prompt_tokens"] += max(0, prompt_tokens)
        if saved_tokens is not None:
            entry["total_saved_tokens"] += max(0, saved_tokens)
        if latency_ms is not None:
            entry["total_latency_ms"] += max(0.0, latency_ms)
        # Move-to-end keeps most-recently-active sessions; evict the oldest.
        self._per_session[session_id] = entry
        self._per_session.move_to_end(session_id)
        while len(self._per_session) > self._max_sessions_tracked:
            self._per_session.popitem(last=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = self.requests
            hits = self.cache_hits
            cached = self.total_cached_tokens
            reuse_ratio = (cached / max(1, self.total_prompt_tokens)) if self.total_prompt_tokens else 0.0
            sessions: dict[str, Any] = {}
            for sid, e in self._per_session.items():
                s_req = e["requests"]
                s_cached = e["total_cached_tokens"]
                s_prompt = e["total_prompt_tokens"]
                sessions[sid] = {
                    "requests": s_req,
                    "cache_hits": e["cache_hits"],
                    "cache_misses": e["cache_misses"],
                    "cache_hit_rate": round(e["cache_hits"] / max(1, s_req), 4),
                    "total_cached_tokens": s_cached,
                    "total_prompt_tokens": s_prompt,
                    "prefix_cache_reuse_ratio": round(
                        (s_cached / s_prompt) if s_prompt else 0.0, 4
                    ),
                    "total_saved_tokens": e["total_saved_tokens"],
                    "avg_latency_ms": round(e["total_latency_ms"] / max(1, s_req), 1),
                    "backend_errors": e.get("backend_errors", 0),
                }
            return {
                "requests": requests,
                "cache_hits": hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(hits / max(1, requests), 4),
                "total_cached_tokens": cached,
                "total_prompt_tokens": self.total_prompt_tokens,
                "prefix_cache_reuse_ratio": round(reuse_ratio, 4),
                "total_saved_tokens": self.total_saved_tokens,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "avg_latency_ms": round(self.total_latency_ms / max(1, requests), 1),
                "backend_errors": self.backend_errors,
                "sessions": sessions,
            }

    def reset(self) -> None:
        with self._lock:
            self.__init__()


# Single process-wide metrics instance.
PROXY_METRICS = _ProxyMetrics()


def _explain_header_value(messages: list[dict[str, Any]]) -> str:
    """Serialize the optimized prompt for the explain-mode response header.

    Base64-encoded JSON so the value is header-safe regardless of message
    content (newlines, colons, unicode). Decoded on the client with
    ``json.loads(base64.b64decode(value))``.
    """
    import base64

    payload = json.dumps(messages, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _serialize_messages_text(messages: list[dict[str, Any]]) -> str:
    """Render optimized messages as plain text for the faithfulness header.

    Used by the ``X-MOEPT-Optimized-Prompt-Text`` response header so a
    local benchmark can measure how much of the original context survived
    compaction. Newlines are replaced with ``\\n`` to keep the value
    header-safe; the benchmark reverses this on read.
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts).replace("\n", "\\n")


# Common unicode punctuation that appears in code comments / fixture text but
# is not latin-1-encodable. Folded to ASCII so header values stay valid.
_HEADER_UNICODE_FOLD = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",    # non-breaking space
}


def _header_safe(value: str) -> str:
    """Return ``value`` with any non-latin-1 character folded to a safe ASCII
    substitute so it can be placed in an HTTP response header.

    HTTP header values are latin-1-encoded by Starlette; an un-sanitized unicode
    value (em-dash, smart quotes, non-latin scripts) raises UnicodeEncodeError
    and turns the whole response into a 500. We first apply a small punctuation
    fold, then drop any remaining out-of-range code points.
    """
    if not value:
        return value
    folded = "".join(_HEADER_UNICODE_FOLD.get(ch, ch) for ch in value)
    return "".join(ch if ord(ch) <= 255 else "" for ch in folded)


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate that all non-assistant messages have a 'content' field.

    The Lemonade server requires all non-assistant messages to contain 'content'.
    This function ensures compliance before sending requests to the backend.

    Raises:
        ValueError: If any non-assistant message is missing 'content'
    """
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role != "assistant" and "content" not in msg:
            raise ValueError(
                f"All non-assistant messages must contain 'content'. "
                f"Message {i} has role='{role}' but no 'content' field."
            )


def _ensure_content(messages: list[dict[str, Any]]) -> None:
    """Ensure all non-assistant messages have a 'content' field (set to '' if missing).

    The Lemonade server requires all non-assistant messages to contain 'content'.
    This is applied after optimization because the optimizer/compactor may produce
    tool_result or other non-assistant messages that lack content.
    """
    for msg in messages:
        role = msg.get("role", "")
        if role != "assistant" and "content" not in msg:
            msg["content"] = ""


def _fallback_optimized_messages(messages: list[dict[str, Any]], keep_full_steps: int) -> list[dict[str, Any]]:
    """Return a safe compact fallback when the full optimizer fails.

    This avoids forwarding the full raw conversation to the backend after an
    optimizer exception. It preserves the system prompt and the most recent
    user/assistant turns, which is safer than sending an unbounded raw context.
    """
    if not messages:
        return []

    fallback: list[dict[str, Any]] = []
    if messages[0].get("role") == "system":
        fallback.append(dict(messages[0]))
        start_index = 1
    else:
        start_index = 0

    keep = max(1, keep_full_steps) * 2
    fallback.extend(dict(msg) for msg in messages[max(start_index, len(messages) - keep):])
    return fallback


def _canonicalize(value: Any) -> Any:
    """Return a JSON-stable representation for session fingerprinting."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_text(text: str) -> str:
    """Return a compact stable hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _first_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first user message, used as a stable conversation seed."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg
    return {}


def _resolve_session_id(
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    legacy_session_id: Any = None,
) -> str:
    """Resolve a session id using only standard OpenAI-compatible inputs.

    Legacy custom fields still work for existing integrations. For standard OpenAI
    clients, the proxy uses the standard `user` field plus the first user message
    as the conversation key. If `user` is absent, it fingerprints the message
    history, which is the standard OpenAI mechanism for conversation continuity.
    """
    if isinstance(legacy_session_id, str) and legacy_session_id.strip():
        return legacy_session_id.strip()

    user = body.get("user")
    first_user = _first_user_message(messages)
    if isinstance(user, str) and user.strip():
        seed = _canonical_json(first_user)
        return f"user:{_hash_text(user)}:{_hash_text(seed)}"

    if first_user:
        return f"anon:{_hash_text(_canonical_json(first_user))}"

    return f"anon:{_hash_text(_canonical_json(messages))}"


def _pop_custom_session_fields(body: dict[str, Any]) -> tuple[Any, Any]:
    """Remove internal session fields so they are never forwarded downstream."""
    return body.pop("_session_id", None), body.pop("_session_state", None)


# Session -> stable backend slot mapping for prefix-cache reuse (review §1).
# A process-wide map; slots are assigned lazily and kept for the proxy's lifetime.
_SLOT_MAP: dict[str, int] = {}
_SLOT_LOCK = threading.Lock()
_NEXT_SLOT = 0


def _slot_for_session(
    session_id: str, enabled: bool, total_slots: int = 0
) -> int | None:
    """Return a stable backend slot id for ``session_id`` or ``None``.

    Only assigns a slot when ``enabled`` is True (slot pinning is opt-in so
    non-llama.cpp backends stay OpenAI-transparent). The same session always
    maps to the same slot, which is what lets the backend reuse the whole
    conversation prefix across turns.

    ``enabled`` is resolved per request from live backend capabilities (see
    ``_slot_pinning_active``) so a session is never pinned to a slot the active
    device (e.g. NPU) does not have.

    ``total_slots`` is the backend's real slot count (from the ``/slots`` probe).
    The assigned id is clamped into ``[0, total_slots)`` so we never send an
    out-of-range ``id_slot``: llama.cpp mishandles the KV slot for an unknown id,
    which truncates long generations mid-stream and makes the backend fail to
    parse the (now-unterminated) tool-call arguments as JSON (a 500). When the
    backend exposes only a single slot (``total_slots <= 1``) pinning is skipped
    entirely -- there is nothing to gain from pinning a lone shared slot and it
    only risks cross-session KV collisions.
    """
    if not enabled or not session_id:
        return None
    if total_slots <= 1:
        # Single-slot (or unknown-count) server: pinning cannot isolate sessions
        # and colliding on the one slot corrupts concurrent long generations.
        return None
    with _SLOT_LOCK:
        slot = _SLOT_MAP.get(session_id)
        if slot is None:
            global _NEXT_SLOT
            slot = _NEXT_SLOT % total_slots
            _SLOT_MAP[session_id] = slot
            _NEXT_SLOT += 1
        return slot


def _slot_pinning_active(cfg: AppConfig, probe: Any | None) -> bool:
    """Resolve whether slot pinning should be used for THIS request.

    Precedence:
      1. Manual force-on: ``v050.slot_pinning_enabled`` always wins (operator
         explicitly opted in).
      2. Auto-detect: when ``v050.capability_autodetect`` is on and the live
         backend snapshot reports ``slot_pinning`` (the active device exposes
         ``/slots``, e.g. the GPU/llama.cpp runtime), enable it; when the active
         device has no slots (e.g. NPU), skip it.
      3. Otherwise off.

    Uses only the cached snapshot (no network on the request path); the snapshot
    is refreshed on its own TTL.
    """
    if cfg.v050.slot_pinning_enabled:
        return True
    if not cfg.v050.capability_autodetect or probe is None:
        return False
    caps = probe.cached()
    return bool(caps and caps.slot_pinning)


def _backend_total_slots(probe: Any | None) -> int:
    """Return the backend's reported slot count (0 when unknown).

    Read from the cached capability snapshot only (no network on the request
    path). Used to clamp assigned ``id_slot`` values into the valid range.
    """
    if probe is None:
        return 0
    caps = probe.cached()
    return int(caps.total_slots) if caps else 0


def _first_message_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the first message, for a one-time tokenizer calibration.

    The first message is usually the system prompt: large, stable, and
    representative of the model's real BPE, which makes it a good exact-count
    anchor. Handles both string content and OpenAI structured content parts.
    """
    if not messages:
        return ""
    content = messages[0].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def _normalize_response_choices(data: dict) -> list[dict]:
    """Pass backend response choices through unchanged.

    Qwen/llama.cpp can return explicit `reasoning_content` alongside `content`.
    The proxy must echo BOTH fields exactly as produced. Collapsing
    `reasoning_content` into `content` (the old behavior) made the client persist
    the reasoning as the assistant `content`, so the next turn's prefix differed
    from what the model actually generated — which broke prefix-cache reuse and
    MTP alignment (review §8.3). We therefore never mutate the message here.
    """
    return data.get("choices", [])


def _make_streaming_generator(
    body: dict,
    cfg: AppConfig,
    backend_client: LemonadeClient,
    optimizer: AgentContextOptimizer | None = None,
    id_slot: int | None = None,
    turn_start: float | None = None,
    session_id: str | None = None,
) -> Any:
    """Create an async generator for SSE streaming using OpenAI SDK."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = body.get("model", cfg.server.llm_model)

    async def stream_generator() -> AsyncIterator[str]:
        initial_chunk = json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        })
        yield f"data: {initial_chunk}\n\n"

        cached_tokens: int | None = None
        backend_prompt_tokens: int | None = None
        backend_error = False
        try:
            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.1)
            max_tokens = body.get("max_tokens")
            request_kwargs = {
                key: value
                for key, value in body.items()
                if key not in {"messages", "model", "temperature", "max_tokens", "stream"}
            }

            # Request final-chunk usage (incl. cached_tokens) so the real prefix
            # cache outcome is reported even in streaming (review §8.1). Preserve
            # any caller-provided stream_options but force include_usage on.
            existing = request_kwargs.get("stream_options")
            if isinstance(existing, dict):
                existing = dict(existing)
                existing["include_usage"] = True
                request_kwargs["stream_options"] = existing
            else:
                request_kwargs["stream_options"] = {"include_usage": True}

            # Pin this session to a stable backend slot when slot pinning is on
            # (review §1). id_slot is a llama.cpp extension; it is only injected
            # when explicitly enabled so other backends stay OpenAI-transparent.
            if id_slot is not None:
                request_kwargs["id_slot"] = id_slot

            async for chunk in backend_client.chat_completions_stream(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs,
            ):
                # Extract fields from OpenAI SDK ChatCompletionChunk
                delta = {}
                finish_reason = None

                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta"):
                        d = choice.delta
                        if hasattr(d, "role") and d.role:
                            delta["role"] = d.role
                        if hasattr(d, "content") and d.content is not None:
                            delta["content"] = d.content
                        if hasattr(d, "reasoning_content") and d.reasoning_content is not None:
                            delta["reasoning_content"] = d.reasoning_content
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason

                # Some backends report usage (incl. cached_tokens) on the final
                # chunk. Capture it so we can feed the real cache outcome to the
                # hit-prediction model.
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = chunk.usage
                    details = getattr(usage, "prompt_tokens_details", None)
                    details = details if isinstance(details, dict) else getattr(details, "__dict__", {})
                    cached_tokens = (
                        getattr(usage, "cache_hit_tokens", None)
                        or getattr(usage, "cached_tokens", None)
                        or details.get("cached_tokens")
                        if isinstance(details, dict)
                        else None
                    )
                    # Backend's true prompt token count for the optimized prompt
                    # we sent; used to calibrate the proxy's estimates (#6).
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                        backend_prompt_tokens = prompt_tokens

                if not delta and finish_reason is None:
                    continue

                sse_chunk = json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }],
                })
                yield f"data: {sse_chunk}\n\n"

                # Do NOT break on finish_reason. With stream_options.include_usage=True
                # the backend emits the authoritative usage chunk (incl. cached_tokens)
                # *after* the finish_reason chunk. Breaking here would skip the real
                # prefix-cache outcome, so we let the loop run until the stream ends
                # and capture usage on the trailing chunk below.

        except (APIStatusError, APIError) as e:
            # Backend failed while streaming (e.g. HTTP 500 when the model's
            # tool-call arguments were truncated by max_tokens and llama.cpp
            # could not parse the unterminated JSON). Degrade gracefully: emit a
            # well-formed OpenAI error object + a terminating stop chunk so the
            # client sees a valid, closed stream instead of a broken connection.
            status = getattr(e, "status_code", None)
            logger.warning(
                "Backend error during streaming (status=%s): %s",
                status,
                type(e).__name__,
            )
            with suppress(Exception):
                PROXY_METRICS.record_backend_error(session_id)
            backend_error = True
            error_payload = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
                "error": {
                    "message": f"Backend error while streaming: {type(e).__name__}",
                    "type": "backend_error",
                    "code": status,
                },
            })
            yield f"data: {error_payload}\n\n"

        except Exception as e:
            logger.exception("Streaming error in chat completions")
            with suppress(Exception):
                PROXY_METRICS.record_backend_error(session_id)
            backend_error = True
            error_chunk = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
                "error": {
                    "message": f"Stream interrupted: {type(e).__name__}",
                    "type": "proxy_error",
                    "code": None,
                },
            })
            yield f"data: {error_chunk}\n\n"

        # On a backend error we already recorded the error counter above and must
        # not also record a phantom "successful" turn or calibrate on partial
        # data. Still emit the [DONE] sentinel so the SSE stream closes cleanly.
        if not backend_error:
            if optimizer is not None:
                try:
                    optimizer.record_cache_outcome(cached_tokens)
                except Exception:
                    logger.debug("Failed to record streaming cache outcome", exc_info=True)

            # Aggregate process-wide metrics from the authoritative backend signal.
            PROXY_METRICS.record_turn(
                session_id=session_id,
                cached_tokens=cached_tokens,
                prompt_tokens=(optimizer.last_optimized_token_count if optimizer is not None else None),
                saved_tokens=(optimizer.last_saved_token_count if optimizer is not None else None),
                latency_ms=((time.time() - turn_start) * 1000.0 if turn_start is not None else None),
            )

            # Calibrate the proxy's token estimates against the backend's real
            # tokenizer (review §1/§9, priority fix #6). The backend reports its true
            # `prompt_tokens` for the optimized prompt we sent; the ratio between that
            # and our tiktoken estimate lets the budget be enforced on true token
            # counts instead of an estimate that diverges for code-heavy prompts.
            if optimizer is not None and isinstance(backend_prompt_tokens, int) and backend_prompt_tokens > 0:
                try:
                    # Calibrate against the OPTIMIZED prompt we actually sent
                    # (optimizer._last_optimized), not the raw incoming messages, so
                    # the ratio reflects the true backend/proxy token gap (#6).
                    proxy_estimated_msgs = getattr(optimizer, "_last_optimized", None) or messages
                    proxy_estimated = optimizer.token_counter.count_messages(proxy_estimated_msgs)
                    if proxy_estimated > 0:
                        optimizer.set_token_calibration(backend_prompt_tokens / proxy_estimated)
                except Exception:
                    logger.debug("Streaming token calibration failed", exc_info=True)

            # HTTP response headers are already sent when streaming begins, so the
            # real cache-hit signal cannot be exposed as an X- header here. Emit it
            # as an SSE comment line instead (valid SSE, ignored by clients but
            # visible to tooling) so the streaming path also surfaces reuse (review §8.2).
            if cached_tokens is not None:
                yield f": X-Prefix-Cache-Hit-Tokens: {cached_tokens}\n\n"

        yield "data: [DONE]\n\n"

    return stream_generator


async def _do_non_streaming(
    body: dict,
    session_state: str,
    cfg: AppConfig,
    backend_client: LemonadeClient,
    response_headers: dict[str, str] | None = None,
    optimization_error: str | None = None,
    optimizer: AgentContextOptimizer | None = None,
    id_slot: int | None = None,
    turn_start: float | None = None,
    session_id: str | None = None,
) -> JSONResponse:
    """Execute non-streaming backend call using LemonadeClient (OpenAI SDK)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = body.get("model", cfg.server.llm_model)

    try:
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.1)
        max_tokens = body.get("max_tokens")
        request_kwargs = {
            key: value
            for key, value in body.items()
            if key not in {"messages", "model", "temperature", "max_tokens", "stream"}
        }

        response = await backend_client.chat_completions_create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            stream=False,
            max_tokens=max_tokens,
            id_slot=id_slot,
            **request_kwargs,
        )

        # Convert OpenAI SDK ChatCompletion to dict format
        backend_data = response.model_dump() if hasattr(response, "model_dump") else dict(response)

        # Normalize choices for OpenAI compatibility
        _normalize_response_choices(backend_data)

        usage = backend_data.get("usage", {})

        # Log response details
        choices = backend_data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "")
            logger.debug(
                "Lemonade non-streaming response: content_len=%d, finish_reason=%s, usage=%s",
                len(content),
                finish_reason,
                usage,
            )
            if not content and finish_reason != "length":
                logger.warning(
                    "Lemonade returned empty content for %d messages (finish_reason=%s)",
                    len(messages),
                    finish_reason,
                )

        # Record cache hit for cache registry. Lemonade may expose cached tokens
        # either as top-level cache_hit_tokens or inside prompt_tokens_details.
        usage_dict = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
        prompt_details = usage_dict.get("prompt_tokens_details", {}) or {}
        cache_hit_tokens = (
            usage_dict.get("cache_hit_tokens")
            or usage_dict.get("cached_tokens")
            or prompt_details.get("cached_tokens")
            or prompt_details.get("cache_hit_tokens")
        )
        if isinstance(cache_hit_tokens, int) and cache_hit_tokens > 0:
            from moeptimizer.cache_registry import get_cache_registry
            registry = get_cache_registry()
            registry.record_cache_hit(messages, cache_hit_tokens)

        # Feed the real backend cache outcome to the hit-prediction model so it
        # learns from actual reuse instead of a constant hit=True label.
        if optimizer is not None:
            try:
                optimizer.record_cache_outcome(cache_hit_tokens)
            except Exception:
                logger.debug("Failed to record cache outcome", exc_info=True)

        # Aggregate process-wide metrics from the authoritative backend signal.
        PROXY_METRICS.record_turn(
            session_id=session_id,
            cached_tokens=cache_hit_tokens if isinstance(cache_hit_tokens, int) else None,
            prompt_tokens=(optimizer.last_optimized_token_count if optimizer is not None else None),
            saved_tokens=(optimizer.last_saved_token_count if optimizer is not None else None),
            latency_ms=((time.time() - turn_start) * 1000.0 if turn_start is not None else None),
        )

        # Calibrate the proxy's token estimates against the backend's real
        # tokenizer (review §1/§9, priority fix #6). The backend reports its true
        # `prompt_tokens` for the optimized prompt we sent; the ratio between that
        # and our tiktoken estimate lets the budget be enforced on true token
        # counts instead of an estimate that diverges for code-heavy prompts.
        if optimizer is not None:
            try:
                backend_prompt_tokens = usage_dict.get("prompt_tokens")
                if isinstance(backend_prompt_tokens, int) and backend_prompt_tokens > 0:
                    # Calibrate against the OPTIMIZED prompt we actually sent
                    # (optimizer._last_optimized), not the raw incoming messages,
                    # so the ratio reflects the true backend/proxy token gap (#6).
                    proxy_estimated_msgs = getattr(optimizer, "_last_optimized", None) or messages
                    proxy_estimated = optimizer.token_counter.count_messages(proxy_estimated_msgs)
                    if proxy_estimated > 0:
                        optimizer.set_token_calibration(
                            backend_prompt_tokens / proxy_estimated
                        )
            except Exception:
                logger.debug("Token calibration failed", exc_info=True)

        response_headers = dict(response_headers or {})
        if isinstance(cache_hit_tokens, int):
            response_headers["X-Prefix-Cache-Hit-Tokens"] = str(cache_hit_tokens)

        return JSONResponse(
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": backend_data.get("choices", []),
                "usage": usage,
            },
            headers={
                **dict(response_headers or {}),
                "_session_state": session_state[:64000] if session_state and len(session_state) > 64000 else (session_state or ""),
            },
        )

    except (APIStatusError, APIError) as e:
        # Backend returned an error (e.g. HTTP 500 for a truncated tool call).
        # Surface it as a well-formed OpenAI error object, preserving the backend
        # status code where available, and record the backend_error metric.
        status = getattr(e, "status_code", None) or 502
        logger.warning(
            "Backend error in non-streaming completion (status=%s): %s",
            status,
            type(e).__name__,
        )
        with suppress(Exception):
            PROXY_METRICS.record_backend_error(session_id)
        response_headers = {}
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": f"Backend error: {type(e).__name__}",
                    "type": "backend_error",
                    "param": None,
                    "code": getattr(e, "status_code", None),
                }
            },
            headers=response_headers,
        )

    except Exception as e:
        logger.exception("Non-streaming chat completion error")
        with suppress(Exception):
            PROXY_METRICS.record_backend_error(session_id)
        response_headers = {}
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Internal error: {type(e).__name__}: {e}",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            },
            headers=response_headers,
        )


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or get_config()
    # Layer the selected quality preset (quality/balanced/aggressive) onto the
    # agentic config before any optimizer is built (review03.md §10).
    from moeptimizer.config import apply_quality_profile

    apply_quality_profile(cfg)
    # Live, device-aware capability probe (NPU<->GPU aware). Detects slot
    # pinning, native MTP, exact remote tokenization, and the tokenizer id from
    # the backend's own metadata, refreshed on a TTL so capabilities follow the
    # active device. Constructed even when autodetect is off (it is only *used*
    # when cfg.v050.capability_autodetect is on) so tests can inspect it.
    from moeptimizer.backend_capabilities import BackendCapabilityProbe

    capability_probe = BackendCapabilityProbe(
        base_url=cfg.server.url,
        model=cfg.server.llm_model,
        ttl_seconds=cfg.v050.capability_probe_ttl_seconds,
    )
    session_manager = SessionManager(
        config=cfg,
        capability_probe=capability_probe,
    )
    embedding_service = EmbeddingService()
    backend_client = LemonadeClient(
        base_url=cfg.server.url,
        api_key="lemonade",
        timeout=cfg.server.timeout,
        native_mtp_passthrough=cfg.v050.native_mtp_passthrough,
    )
    output_shaper = OutputShaper(
        enabled=cfg.agentic.tool_output_compression_enabled,
    )
    embed_client = AsyncOpenAI(
        base_url=cfg.server.embed_url,
        api_key="lemonade",
        timeout=cfg.server.timeout,
    )

    # Lemonade exposes a standard OpenAI API. Do not enable proxy-level
    # speculative decoding wrappers here: the current backend does not expose
    # native MTP/speculative endpoints, and custom extra_body fields are not
    # part of the standard OpenAI chat-completions contract.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await embedding_service.initialize()
        # Live capability detection (review: NPU<->GPU aware). A single probe
        # reads the backend's own metadata (active device, /slots, native MTP,
        # exact /tokenize, tokenizer id) instead of guessing. This drives slot
        # pinning, MTP passthrough, and tokenizer selection, and is refreshed on
        # a TTL per request so device hot-swaps are picked up without a restart.
        caps = None
        if cfg.v050.capability_autodetect:
            try:
                caps = await capability_probe.get(force=True)
                logger.info("Detected backend capabilities: %s", caps.summary())
            except Exception as exc:
                logger.warning("Capability detection failed: %s", exc)

        # Resolve MTP passthrough. Metadata (labels=['...','mtp'] or an active
        # speculative slot, or --spec-type ...mtp in the launch args) is a
        # reliable, non-invasive signal; prefer it over the chat probe.
        if not backend_client.native_mtp_passthrough:
            enabled_via_meta = bool(caps and caps.mtp)
            if enabled_via_meta:
                backend_client.enable_native_mtp_passthrough()
                logger.info(
                    "Backend metadata declares native MTP; enabling MTP "
                    "extra_body passthrough."
                )
            elif cfg.v050.native_mtp_autodetect:
                # Fallback: only chat-probe when metadata was inconclusive.
                try:
                    if await backend_client.detect_mtp_support():
                        backend_client.enable_native_mtp_passthrough()
                        logger.info(
                            "Backend chat-probe confirms native MTP; enabling "
                            "MTP extra_body passthrough."
                        )
                except Exception as exc:
                    logger.warning("MTP support auto-detection failed: %s", exc)

        # Derive the tokenizer from the backend's model checkpoint when the
        # operator left it on 'auto' and metadata gave us a concrete id. This
        # makes budget counts use the model's real BPE without manual config.
        # Session optimizers are built lazily after startup and read
        # cfg.server.tokenizer, so setting it here propagates to all sessions.
        if (
            cfg.v050.capability_autodetect
            and cfg.server.tokenizer == "auto"
            and caps
            and caps.tokenizer_id
        ):
            cfg.server.tokenizer = caps.tokenizer_id
            app.state.detected_tokenizer_id = caps.tokenizer_id
            logger.info(
                "Using tokenizer '%s' derived from backend model checkpoint; "
                "sessions will load it (falling back to tiktoken if unavailable "
                "locally). Backend prompt_tokens still calibrates the residual.",
                caps.tokenizer_id,
            )

        logger.info(
            "Resolved native_mtp_passthrough=%s (autodetect=%s, capability_autodetect=%s); "
            "slot_pinning force=%s; the only functional speculative-decoding path is a "
            "backend with native MTP support.",
            backend_client.native_mtp_passthrough,
            cfg.v050.native_mtp_autodetect,
            cfg.v050.capability_autodetect,
            cfg.v050.slot_pinning_enabled,
        )
        yield
        await embedding_service.close()
        await embed_client.close()

    app = FastAPI(
        title="Lemonade MoE Agentic Optimizer",
        description=(
            "Production-ready middleware for Qwen3.6-35B-A3B-MTP with agentic "
            "context management. Features: scratchpad compaction, thinking "
            "preservation, state-based RAG, LanceDB semantic index."
        ),
        lifespan=lifespan,
    )

    # Expose services for direct access by endpoints
    app.state.embedding_service = embedding_service
    app.state.backend_client = backend_client
    app.state.embed_client = embed_client
    app.state.capability_probe = capability_probe
    app.state.output_shaper = output_shaper

    @app.post("/v1/chat/completions")
    async def chat_completions_proxy(request: Request):
        """
        OpenAI-compatible chat completions proxy.

        Request schema (OpenAI):
          { model, messages, temperature, top_p, n, stream, stop, max_tokens,
            presence_penalty, frequency_penalty, logit_bias, user,
            tools, tool_choice, response_format }

        Conversation continuity:
          Existing `_session_id` / `_session_state` fields are still accepted,
          but standard OpenAI clients do not need them. The proxy derives the
          session key from the standard `user` field plus the first user message,
          or from a fingerprint of the message history when `user` is absent.
          Custom session fields are stripped before forwarding to Lemonade.
        """
        _turn_start = time.time()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

        messages = list(body.get("messages", []))
        if not messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid payload: no messages",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": None,
                    }
                },
            )

        # Validate all non-assistant messages have 'content' field
        try:
            _validate_messages(messages)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": None,
                    }
                },
            )

        legacy_session_id, session_state = _pop_custom_session_fields(body)
        session_id = _resolve_session_id(body, messages, legacy_session_id)

        optimizer = session_manager.get_or_create(session_id)

        if session_state:
            if session_id:
                session_manager.load_state(session_id, session_state)
                optimizer = session_manager.get_or_create(session_id)
            else:
                with suppress(Exception):
                    optimizer.load_session_state(session_state)

        optimized_messages = messages
        optimization_error: str | None = None
        try:
            # Run the (CPU-bound, synchronous) optimizer in a worker thread so the
            # asyncio event loop stays free for concurrent sessions. Previously the
            # optimizer ran inline on the event loop, so one long session blocked
            # all others (review §2/§4/§5).
            optimized_messages = await asyncio.get_running_loop().run_in_executor(
                None, optimizer.optimize_messages, messages
            )
        except Exception as e:
            logger.exception("Context optimization failed, falling back to recent-turn context")
            # Fold to header-safe ASCII: this string is surfaced verbatim in the
            # X-Optimization-Error response header, which must be latin-1-encodable.
            optimization_error = _header_safe(f"{type(e).__name__}: {e}")
            optimized_messages = _fallback_optimized_messages(messages, cfg.agentic.keep_full_steps)

        # Refresh live backend capabilities on their TTL (cheap: a no-op when the
        # cached snapshot is fresh). This is what lets slot pinning / MTP / remote
        # tokenization follow the active device when the backend hot-swaps between
        # NPU and GPU without restarting the proxy.
        if cfg.v050.capability_autodetect:
            with suppress(Exception):
                await capability_probe.get()

        # One-time exact-tokenizer calibration seed (review §1/§9, #6). Before this
        # session has ever seen a backend `prompt_tokens` response, anchor the
        # local-count->true-count ratio using the backend's own native /tokenize on
        # a representative sample (the system prompt / first message). This removes
        # turn-1 budget error even when the local tokenizer is the tiktoken
        # fallback. Runs at most once per session, only when the active device
        # exposes remote tokenization; best-effort and never blocks the request.
        if (
            cfg.v050.capability_autodetect
            and cfg.v050.remote_tokenize_enabled
            and not getattr(optimizer, "_calibration_seeded", False)
        ):
            caps = capability_probe.cached()
            if caps and caps.remote_tokenize:
                sample = _first_message_text(optimized_messages)
                if sample:
                    with suppress(Exception):
                        exact = await capability_probe.tokenize_count(sample)
                        if isinstance(exact, int) and exact > 0:
                            optimizer.seed_token_calibration(sample, exact)

        # Pin this session to a stable backend slot when slot pinning is active
        # for the CURRENT device (review §1). Resolved from live capabilities so a
        # session is never pinned to a slot an NPU device does not have. A stable
        # slot lets the backend reuse the whole conversation prefix across turns
        # instead of re-prefilling every turn.
        id_slot = _slot_for_session(
            session_id,
            _slot_pinning_active(cfg, capability_probe),
            _backend_total_slots(capability_probe),
        )

        # Debug logging for long contexts
        if len(optimized_messages) > 10:
            logger.info(
                "[Proxy] Turn with %d messages (original: %d), optimization_error=%s",
                len(optimized_messages),
                len(messages),
                optimization_error,
            )
            # Log message roles and content lengths
            for i, msg in enumerate(optimized_messages[:5]):
                logger.info(
                    "[Proxy] Message %d: role=%s, content_len=%d, preview=%s",
                    i,
                    msg.get("role"),
                    len(msg.get("content") or ""),
                    (msg.get("content") or "")[:100],
                )
            if len(optimized_messages) > 5:
                logger.info(
                    "[Proxy] ... and %d more messages",
                    len(optimized_messages) - 5,
                )

        # Ensure all non-assistant messages have 'content' for Lemonade compatibility.
        # The optimizer/compactor may produce tool_result or other non-assistant
        # messages that lack a content field from the original request.
        _ensure_content(optimized_messages)

        response_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Optimized-Prompt-Tokens": str(
                optimizer.last_optimized_token_count
                if optimizer.last_optimized_token_count is not None
                else optimizer.token_counter.count_messages(optimized_messages)
            ),
        }

        # Expose the exact optimized prompt TEXT the proxy sends to the backend.
        # This is the proxy's one job (it compacts ONLY the input context), so
        # downstream tooling (e.g. the benchmark's prompt-faithfulness metric)
        # can measure how much of the original context survived compaction.
        # Gated to a sane size: full prompts can be huge, and the header is
        # only consumed by local benchmarking, not production clients.
        _opt_text = _serialize_messages_text(optimized_messages)
        if _opt_text and len(_opt_text) <= 32000:
            # HTTP headers must be latin-1-encodable. Optimized prompt text can
            # contain unicode (em-dash, smart quotes, non-latin scripts from code
            # comments / fixture content); an un-sanitized value makes Starlette's
            # StreamingResponse header encoding raise UnicodeEncodeError -> HTTP
            # 500 for that turn. Fold non-latin-1 chars to safe ASCII substitutes
            # so the header (consumed only by local benchmarking) stays valid.
            response_headers["X-MOEPT-Optimized-Prompt-Text"] = _header_safe(_opt_text)

        # Dry-run / explain mode (review03.md §10): expose the exact optimized
        # prompt the proxy would send to the backend so operators can inspect
        # what changed. Opt-in per request via the X-MOEPT-Explain header, or
        # globally via agentic.explain_mode_enabled.
        explain_on = cfg.agentic.explain_mode_enabled or str(
            request.headers.get("X-MOEPT-Explain", "")
        ).strip().lower() in ("1", "true", "yes") or bool(body.get("_explain"))
        if explain_on:
            response_headers["X-MOEPT-Explain"] = "true"
            response_headers["X-MOEPT-Optimized-Messages"] = _explain_header_value(
                optimized_messages
            )

        session_state = optimizer.get_session_state()
        existing_extra_body = body.get("extra_body")
        if existing_extra_body is not None and not isinstance(existing_extra_body, dict):
            logger.warning("Ignoring invalid extra_body value: %s", type(existing_extra_body).__name__)
            existing_extra_body = None
        backend_extra_body = optimizer.get_backend_extra_body(
            optimized_messages,
            existing_extra_body,
        )
        if backend_extra_body:
            body["extra_body"] = backend_extra_body
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error

        body["model"] = cfg.server.llm_model
        body["messages"] = optimized_messages
        body.setdefault("temperature", 0.1)
        body.setdefault("stream", True)

        # Step: shape the backend request for output length (review §2.4 / P1).
        # Applies cache-safe system-prompt tail instruction + per-turn-class
        # max_tokens / reasoning_effort clamping. Does not touch the input path.
        output_shaper = getattr(app.state, "output_shaper", None)
        if output_shaper is not None:
            try:
                body = output_shaper.shape_request(body)
            except Exception as e:
                logger.debug("Output shaping failed: %s", e)

        is_streaming = body.get("stream", True)

        # Only include session state in header if it's reasonably sized
        # (full state is maintained server-side via session_id)
        if session_state and len(session_state) <= 64000:
            response_headers["_session_state"] = session_state
        elif session_state:
            logger.warning(
                "Session state too large for header (%d bytes), omitting from response",
                len(session_state),
            )

        if is_streaming:
            # _make_streaming_generator returns a factory; invoke it to get the
            # async generator (an async iterator) that StreamingResponse expects.
            # Passing the bare factory function made Starlette do
            # `async for chunk in <function>` -> TypeError: 'function' object is
            # not iterable, which killed the stream and made clients see a
            # truncated response ("Response ended prematurely").
            return StreamingResponse(
                _make_streaming_generator(body, cfg, backend_client, optimizer, id_slot, _turn_start, session_id)(),
                media_type="text/event-stream",
                headers=response_headers,
            )
        else:
            return await _do_non_streaming(
                body, session_state, cfg, backend_client, response_headers, optimization_error, optimizer, id_slot, _turn_start, session_id
            )

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible models list endpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": cfg.server.llm_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                },
                {
                    "id": cfg.server.embed_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "gemma",
                },
            ],
        }

    @app.get("/v1/metrics")
    async def proxy_metrics():
        """Process-wide proxy effectiveness metrics (review §11.1, fix #10).

        Surfaces whether the proxy is actually helping: real prefix-cache reuse
        (from the backend's authoritative ``cached_tokens``), token savings from
        optimization, cache-hit rate, and average latency. Not part of the OpenAI
        contract; purely observational for operators.
        """
        return {"object": "proxy.metrics", **PROXY_METRICS.snapshot()}

    @app.post("/v1/metrics/reset")
    async def proxy_metrics_reset():
        """Reset the process-wide proxy metrics counters."""
        PROXY_METRICS.reset()
        return {"object": "proxy.metrics", "status": "reset"}

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        """OpenAI-compatible embeddings endpoint (proxied to Lemonade)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

        input_data = body.get("input")
        if input_data is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Missing required field: input",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    }
                },
            )

        model = body.get("model", cfg.server.embed_model)

        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list):
            if input_data and isinstance(input_data[0], dict):
                input_list = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in input_data
                ]
            else:
                input_list = [str(item) for item in input_data]
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid input format",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    }
                },
            )

        try:
            embed_client = getattr(app.state, "embed_client", backend_client._client)
            result = await embed_client.embeddings.create(
                model=model,
                input=input_list,
            )
            resp_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            embeddings_data = resp_dict.get("data", [])
            usage = resp_dict.get("usage", {})

            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": emb.get("embedding", []),
                        }
                        for i, emb in enumerate(embeddings_data)
                    ],
                    "model": model,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                },
            )
        except Exception as e:
            logger.exception("Embedding error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Embedding error: {e}",
                        "type": "api_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

    @app.get("/v1/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "lemonade": "not_checked"}

    @app.post("/v1/agent/state")
    async def get_agent_state(request: Request):
        """Return current agent session state for persistence."""
        body = await request.json() if await request.body() else {}
        session_id = body.get("_session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        optimizer = session_manager.get_or_create(session_id)
        progress = optimizer.progress_tracker.get_progress()

        return {
            "session_id": session_id,
            "session_state": optimizer.get_session_state(),
            "step_count": len(optimizer.store.steps),
            "goal": optimizer.store.get_goal().original_prompt if optimizer.store.get_goal() is not None else None,
            "progress": progress.to_dict(),
            "loop_warnings": [
                {"type": w.loop_type, "message": w.message}
                for w in optimizer.loop_detector.get_recent_warnings()
            ],
        }

    @app.post("/v1/agent/state/reset")
    async def reset_agent_state(request: Request):
        """Reset agent session state."""
        body = await request.json() if await request.body() else {}
        session_id = body.get("_session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        session_manager.reset_session(session_id)
        return {"status": "ok", "message": "Agent state reset", "session_id": session_id}

    @app.get("/v1/agent/sessions")
    async def list_sessions():
        """List all active agent sessions."""
        sessions = session_manager.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}

    @app.get("/v1/agent/sessions/{session_id}/debug")
    async def session_debug(session_id: str):
        """Per-session debug dashboard (review §10, P4).

        Exposes the live-zone boundary (stable prefix vs. live zone), the real
        prefix-cache outcome, token savings, and the embedding circuit-breaker
        state so operators can see why a session is (or is not) reusing its KV
        cache and whether the embedding dependency is healthy. Read-only.
        """
        optimizer = session_manager.get_or_create(session_id)
        try:
            debug = optimizer.get_debug_info()
        except Exception as e:  # Never let a debug read crash the request path
            logger.debug("Failed to build session debug info: %s", e)
            debug = {"error": f"{type(e).__name__}: {e}"}
        debug["session_id"] = session_id
        debug["metrics"] = PROXY_METRICS.snapshot().get("sessions", {}).get(session_id)
        return {"object": "agent.session.debug", **debug}

    @app.delete("/v1/agent/session/{session_id}")
    async def delete_session(session_id: str):
        """Delete an agent session."""
        deleted = session_manager.delete_session(session_id)
        return {"status": "ok" if deleted else "not_found", "session_id": session_id}

    @app.post("/v1/cache/clear")
    async def clear_caches():
        """Clear all caches."""
        embedding_service._embed_cache.clear()
        return {
            "status": "ok",
            "embed_cache_size": 0,
        }

    return app
