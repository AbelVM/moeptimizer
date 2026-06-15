"""TokenCounter — Estimate token usage for context budget management.

Uses tiktoken for accurate token counting with the model's actual tokenizer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter for context budget management.

    Uses tiktoken for accurate token counting with the model's actual tokenizer.
    """

    CHARS_PER_TOKEN: ClassVar[dict[str, float]] = {
        "python": 4.0,
        "javascript": 3.8,
        "typescript": 3.7,
        "go": 3.9,
        "rust": 3.6,
        "cpp": 3.5,
        "java": 3.6,
        "c_sharp": 3.5,
        "php": 3.8,
        "ruby": 4.1,
        "html": 2.5,
        "css": 2.8,
        "json": 2.0,
        "generic": 3.5,
    }

    def __init__(self, model_name: str = "gpt-4") -> None:
        """Initialize with optional model name for tokenizer selection."""
        self._model_name = model_name
        self._encoder: tiktoken.Encoding | None = None
        try:
            # Use a tokenizer close to Qwen's (GPT-4 is a reasonable approximation)
            self._encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            # Fall back to cl100k_base (GPT-4 tokenizer)
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize tiktoken encoder. "
                    "Ensure tiktoken is installed: pip install tiktoken"
                ) from e

    def count(self, text: str, lang: str = "generic") -> int:
        """Estimate token count for the given text."""
        if not text:
            return 0

        non_ws = len(text.strip())
        if non_ws == 0:
            return 0

        # Use actual tokenizer
        try:
            return len(self._encoder.encode(text))  # type: ignore[union-attr]
        except Exception:
            # Fallback to character-based estimation
            cpt = self.CHARS_PER_TOKEN.get(lang, 3.5)
            return max(1, int(len(text) / cpt))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += self.count(part.get("text", ""))
            total += 5  # Per-message overhead
        return total

    def count_tokens_precise(self, text: str) -> int:
        """Get precise token count using the model tokenizer."""
        try:
            return len(self._encoder.encode(text))  # type: ignore[union-attr]
        except Exception:
            # Fallback
            return self.count(text)

    def estimate_kv_cache_usage(self, token_count: int) -> str:
        """Convert token count to a human-readable KV-cache estimate."""
        slots = token_count * 4
        if token_count < 10000:
            return f"{token_count:,} tokens (~{slots:,} KV slots)"
        return f"{token_count:,} tokens (~{slots:,} KV slots — near context limit)"