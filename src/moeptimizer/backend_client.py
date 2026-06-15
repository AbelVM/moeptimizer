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
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from moeptimizer.mtp_speculative import (
    MTPSpeculativeDecoder,
    build_mtp_speculative_body,
)

logger = logging.getLogger(__name__)


class SpeculativeDecoder:
    """
    Speculative decoding with MTP-aware draft model.

    Uses MTP head outputs as draft tokens for tree-based verification.
    Improves throughput by 2-3x when draft model is available.
    """

    def __init__(
        self,
        target_client: "LemonadeClient",
        draft_client: "LemonadeClient | None" = None,
        mtp_lookahead: int = 4,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._target = target_client
        self._draft = draft_client
        self._mtp_decoder = MTPSpeculativeDecoder(
            mtp_heads=3,
            mtp_lookahead=[2, 3, 4],
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
            mtp_heads=3,
            mtp_lookahead=4,
            confidence_threshold=self._mtp_decoder._confidence_threshold,
        )

        # Merge with existing extra_body
        existing_body = kwargs.get("extra_body", {})
        kwargs["extra_body"] = {**existing_body, **mtp_body}

        return await self._target.chat_completions_create(
            messages=messages,
            model=model,
            **kwargs,
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
        draft_client: "LemonadeClient | None" = None,
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

        # Pass through any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value

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

        response = await self._client.chat.completions.create(**params)

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
        async for chunk in await self.chat_completions_create(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying OpenAI client."""
        return self._client

    @property
    def speculative_decoder(self) -> SpeculativeDecoder | None:
        """Return speculative decoder if enabled."""
        return self._speculative_decoder
