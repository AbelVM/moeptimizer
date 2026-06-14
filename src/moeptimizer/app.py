"""FastAPI application — OpenAI-compatible proxy with agentic context optimization."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from moeptimizer.backend_client import LemonadeClient
from moeptimizer.config import AppConfig, get_config
from moeptimizer.embedding import EmbeddingService
from moeptimizer.session_manager import SessionManager

logger = logging.getLogger(__name__)


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


def _normalize_response_choices(data: dict) -> list[dict]:
    """Normalize backend response choices for OpenAI API compatibility.

    Merges reasoning_content into content when content is empty,
    and always strips reasoning_content from the output.
    """
    choices = data.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        reasoning = message.pop("reasoning_content", "") or ""
        if reasoning and not message.get("content"):
            message["content"] = reasoning
    return choices


def _make_streaming_generator(
    body: dict,
    cfg: AppConfig,
    backend_client: LemonadeClient,
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

        try:
            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.1)
            max_tokens = body.get("max_tokens")

            async for chunk in backend_client.chat_completions_stream(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
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
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason

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

                if finish_reason is not None:
                    break

        except Exception as e:
            logger.exception("Streaming error in chat completions")
            error_chunk = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[Stream interrupted: {type(e).__name__}]"},
                    "finish_reason": None,
                }],
            })
            yield f"data: {error_chunk}\n\n"

        yield "data: [DONE]\n\n"

    return stream_generator


async def _do_non_streaming(
    body: dict,
    session_state: str,
    cfg: AppConfig,
    backend_client: LemonadeClient,
) -> JSONResponse:
    """Execute non-streaming backend call using LemonadeClient (OpenAI SDK)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = body.get("model", cfg.server.llm_model)

    try:
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.1)
        max_tokens = body.get("max_tokens")

        response = await backend_client.chat_completions_create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            stream=False,
            max_tokens=max_tokens,
        )

        # Convert OpenAI SDK ChatCompletion to dict format
        if hasattr(response, "model_dump"):
            backend_data = response.model_dump()
        else:
            backend_data = dict(response)

        # Normalize choices for OpenAI compatibility
        _normalize_response_choices(backend_data)

        usage = backend_data.get("usage", {})

        # Record cache hit for cache registry
        # Extract cache hit tokens if available
        cache_hit_tokens = getattr(usage, "cache_hit_tokens", None) if hasattr(usage, "cache_hit_tokens") else usage.get("cache_hit_tokens", None)
        if cache_hit_tokens is not None and cache_hit_tokens > 0:
            # We have cache hits - record them
            from moeptimizer.cache_registry import get_cache_registry
            registry = get_cache_registry()
            registry.record_cache_hit(messages, cache_hit_tokens)

        return JSONResponse(
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": backend_data.get("choices", []),
                "usage": usage,
            },
            headers={"_session_state": session_state},
        )

    except Exception as e:
        logger.exception("Non-streaming chat completion error")
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
        )


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or get_config()
    session_manager = SessionManager()
    embedding_service = EmbeddingService()
    backend_client = LemonadeClient(
        base_url=cfg.server.url,
        api_key="lemonade",
        timeout=cfg.server.timeout,
    )

    # Enable speculative decoding if configured
    if cfg.speculative.enabled:
        backend_client.enable_speculative_decoding(
            mtp_lookahead=cfg.speculative.mtp_lookahead,
            confidence_threshold=cfg.speculative.confidence_threshold,
        )
        logger.info("Speculative decoding enabled: mtp_lookahead=%d, confidence_threshold=%.2f",
                    cfg.speculative.mtp_lookahead, cfg.speculative.confidence_threshold)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await embedding_service.initialize()
        yield
        await embedding_service.close()

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

    @app.post("/v1/chat/completions")
    async def chat_completions_proxy(request: Request):
        """
        OpenAI-compatible chat completions proxy.

        Request schema (OpenAI):
          { model, messages, temperature, top_p, n, stream, stop, max_tokens,
            presence_penalty, frequency_penalty, logit_bias, user,
            tools, tool_choice, response_format }

        Custom fields (transparently handled, not forwarded to backend):
          _session_id, _session_state
        """
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

        messages = body.get("messages", [])
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

        session_id = body.pop("_session_id", None)
        session_state = body.pop("_session_state", None)

        optimizer = session_manager.get_or_create(session_id)

        if session_state:
            if session_id:
                session_manager.load_state(session_id, session_state)
                optimizer = session_manager.get_or_create(session_id)
            else:
                with suppress(Exception):
                    optimizer.load_session_state(session_state)

        optimized_messages = optimizer.optimize_messages(messages)

        # Ensure all non-assistant messages have 'content' for Lemonade compatibility.
        # The optimizer/compactor may produce tool_result or other non-assistant
        # messages that lack a content field from the original request.
        _ensure_content(optimized_messages)

        session_state = optimizer.get_session_state()

        body["model"] = cfg.server.llm_model
        body["messages"] = optimized_messages
        body.setdefault("temperature", 0.1)
        body.setdefault("stream", True)

        is_streaming = body.get("stream", True)

        if is_streaming:
            return StreamingResponse(
                _make_streaming_generator(body, cfg, backend_client),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _do_non_streaming(
                body, session_state, cfg, backend_client
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
            result = await backend_client._client.embeddings.create(
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
        try:
            resp = await backend_client._client.models.list()
            status = "ok" if resp else "degraded"
            return {"status": status, "lemonade": status}
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return {"status": "unhealthy", "lemonade": "error", "detail": str(e)}

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
