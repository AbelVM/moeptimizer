"""Backend client — OpenAI SDK wrapper for Lemonade NPU server.

Uses the official OpenAI Python SDK to ensure correct request formatting
when communicating with the Lemonade server (which exposes an OpenAI-shaped API).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LemonadeClient:
    """Async client for the Lemonade NPU server using OpenAI SDK."""

    def __init__(self, base_url: str, api_key: str = "lemonade") -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the Lemonade server (e.g., http://localhost:13305/api/v1)
            api_key: API key for authentication (Lemonade uses "lemonade" as default)
        """
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=2,
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

        response = await self._client.chat.completions.create(**params)
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
