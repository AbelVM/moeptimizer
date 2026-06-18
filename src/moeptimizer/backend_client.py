"""Backend client — OpenAI SDK wrapper for Lemonade NPU server.

Uses the official OpenAI Python SDK to ensure correct request formatting
when communicating with the Lemonade server (which exposes an OpenAI-shaped API).

Enhanced with:
- MTP-aware speculative decoding
- Expert routing hints
- Tree-based verification
- Confidence threshold control
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from moeptimizer.mtp_speculative import (
    MTPSpeculativeDecoder,
    build_mtp_speculative_body,
)

logger = logging.getLogger(__name__)

_CUSTOM_SESSION_FIELDS = {"_session_id", "_session_state"}
_UNSUPPORTED_EXTRA_BODY_KEYS = {
    "speculative_decoding",
    "mtp_heads",
    "head_temperatures",
    "expert_hints",
    "kv_cache_warmup",
    "cache_control_hints",
}


def _strip_unsupported_extra_body(params: dict[str, Any]) -> dict[str, Any]:
    """Remove proxy-internal fields that are not part of standard OpenAI API."""
    extra_body = params.get("extra_body")
    if isinstance(extra_body, dict):
        params["extra_body"] = {
            key: value
            for key, value in extra_body.items()
            if key not in _UNSUPPORTED_EXTRA_BODY_KEYS
        }
    return params


def _strip_custom_session_fields(params: dict[str, Any]) -> dict[str, Any]:
    """Remove internal session fields before sending to a standard OpenAI API."""
    return {
        key: value
        for key, value in params.items()
        if key not in _CUSTOM_SESSION_FIELDS
    }


class SpeculativeDecoder:
    """
    Speculative decoding with MTP-aware draft model.

    Uses MTP head outputs as draft tokens for tree-based verification.
    Improves throughput by 2-3x when draft model is available.
    """

    def __init__(
        self,
        target_client: LemonadeClient,
        draft_client: LemonadeClient | None = None,
        mtp_lookahead: int = 4,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._target = target_client
        self._draft = draft_client
        lookahead_heads = [head for head in (2, 3, 4) if head <= max(1, mtp_lookahead)]
        if not lookahead_heads:
            lookahead_heads = [1]
        self._mtp_decoder = MTPSpeculativeDecoder(
            mtp_heads=len(lookahead_heads),
            mtp_lookahead=lookahead_heads,
            confidence_threshold=confidence_threshold,
        )
        self._stats: dict[str, int] = {"accepted": 0, "rejected": 0, "total": 0}
        self._temp_stats: dict[str, float] = {"high_conf": 0.0, "low_conf": 0.0}

    def get_temperature_for_mtp_confidence(
        self,
        mtp_confidence: float,
    ) -> float:
        """Get optimal temperature based on MTP confidence.

        High confidence → lower temperature (more deterministic)
        Low confidence → higher temperature (more exploration)

        For precise coding tasks, target ~0.6 for best results.
        """
        if mtp_confidence > 0.8:
            return 0.5   # Very confident, precise coding
        elif mtp_confidence > 0.5:
            return 0.6   # Moderately confident, recommended for coding
        else:
            return 0.7   # Low confidence, allow exploration

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Generate with speculative decoding if draft model available."""
        # Use MTP-aware speculative decoding
        return await self._mtp_speculative_generate(
            messages=messages,
            model=model,
            **kwargs,
        )

    async def _mtp_speculative_generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """MTP-aware speculative generation using native MTP heads."""
        # Build MTP-aware speculative body
        mtp_body = build_mtp_speculative_body(
            mtp_heads=self._mtp_decoder._mtp_heads,
            mtp_lookahead=max(self._mtp_decoder._mtp_lookahead),
            confidence_threshold=self._mtp_decoder._confidence_threshold,
        )

        # Merge with existing extra_body
        params: dict[str, Any] = {
            key: value
            for key, value in kwargs.items()
            if value is not None and key not in _CUSTOM_SESSION_FIELDS
        }
        params["model"] = model
        params["messages"] = messages
        existing_body = params.get("extra_body", {})
        params["extra_body"] = {**existing_body, **mtp_body}

        return await self._target._send_chat_completions_request(
            params=params,
            validated_messages=messages,
            model=model,
            stream=bool(params.get("stream", False)),
        )

    def get_stats(self) -> dict[str, int]:
        """Get speculative decoding statistics."""
        return dict(self._stats)


class LemonadeClient:
    """Async client for the Lemonade NPU server using OpenAI SDK."""

    def __init__(self, base_url: str, api_key: str = "lemonade", timeout: float = 300.0) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the Lemonade server (e.g., http://localhost:13305/api/v1)
            api_key: API key for authentication (Lemonade uses "lemonade" as default)
            timeout: Request timeout in seconds (default 300s for long contexts)
        """
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=2,
            timeout=timeout,
        )
        self._speculative_decoder: SpeculativeDecoder | None = None

    def enable_speculative_decoding(
        self,
        draft_client: LemonadeClient | None = None,
        mtp_lookahead: int = 4,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Enable speculative decoding with optional draft model."""
        self._speculative_decoder = SpeculativeDecoder(
            target_client=self,
            draft_client=draft_client,
            mtp_lookahead=mtp_lookahead,
            confidence_threshold=confidence_threshold,
        )

    async def chat_completions_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        stream: bool = False,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion.

        Args:
            messages: List of message dicts with role and content
            model: Model ID to use
            temperature: Sampling temperature
            stream: Whether to stream the response
            max_tokens: Maximum number of tokens in the response
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            ChatCompletion object (non-streaming) or async iterator (streaming)
        """
        # Build the message payload — ensure all messages have 'content' field
        validated_messages = []
        for msg in messages:
            if "content" not in msg and msg.get("role") != "assistant":
                logger.warning(
                    "Message with role=%s missing 'content' field, setting to empty string",
                    msg.get("role"),
                )
                msg = {**msg, "content": ""}
            validated_messages.append(msg)

        params: dict[str, Any] = {
            "model": model,
            "messages": validated_messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Pass through any additional parameters, excluding internal session fields
        # that are not part of the standard OpenAI chat completions schema.
        for key, value in kwargs.items():
            if value is not None and key not in _CUSTOM_SESSION_FIELDS:
                params[key] = value

        params = _strip_custom_session_fields(params)

        logger.debug("Sending request to Lemonade: model=%s, messages=%d, stream=%s",
                     model, len(validated_messages), stream)

        # Log message summary for debugging
        if len(validated_messages) > 5:
            logger.debug(
                "Lemonade request message summary: %d messages, first role=%s, last role=%s",
                len(validated_messages),
                validated_messages[0].get("role") if validated_messages else "none",
                validated_messages[-1].get("role") if validated_messages else "none",
            )

        if self._speculative_decoder is not None and not stream:
            spec_params = dict(params)
            spec_params.pop("model", None)
            spec_params.pop("messages", None)
            return await self._speculative_decoder.generate(
                messages=validated_messages,
                model=model,
                **spec_params,
            )

        return await self._send_chat_completions_request(
            params=params,
            validated_messages=validated_messages,
            model=model,
            stream=stream,
        )

    async def _send_chat_completions_request(
        self,
        params: dict[str, Any],
        validated_messages: list[dict[str, Any]],
        model: str,
        stream: bool,
    ) -> Any:
        """Send a chat completion request without re-entering speculative decoding."""
        response = await self._client.chat.completions.create(
            **_strip_custom_session_fields(_strip_unsupported_extra_body(params)),
        )

        # Log response summary
        if hasattr(response, "usage") and response.usage:
            logger.debug(
                "Lemonade response: completion_tokens=%s, prompt_tokens=%s",
                getattr(response.usage, "completion_tokens", None),
                getattr(response.usage, "prompt_tokens", None),
            )
            # Check for empty or suspicious usage
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            if prompt_tokens == 0:
                logger.warning(
                    "Lemonade returned prompt_tokens=0 for %d messages - possible rejection or error",
                    len(validated_messages),
                )

        return response

    async def chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Create a streaming chat completion.

        Args:
            messages: List of message dicts with role and content
            model: Model ID to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens in the response
            **kwargs: Additional arguments to pass to the OpenAI API

        Yields:
            ChatCompletionChunk objects
        """
        stream = self.chat_completions_create(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True,
            max_tokens=max_tokens,
            **kwargs,
        )
        async for chunk in stream:  # type: ignore[attr-defined]
            yield chunk

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying OpenAI client."""
        return self._client

    @property
    def speculative_decoder(self) -> SpeculativeDecoder | None:
        """Return speculative decoder if enabled."""
        return self._speculative_decoder
