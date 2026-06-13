"""TokenCounter — Estimate token usage for context budget management."""

from __future__ import annotations

from typing import Any, ClassVar


class TokenCounter:
    """
    Lightweight token counter for context budget management.

    Uses character-based estimation with language-aware adjustments.
    For code: ~4 chars/token. For prose: ~3.5 chars/token.
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

    def count(self, text: str, lang: str = "generic") -> int:
        """Estimate token count for the given text."""
        if not text:
            return 0

        non_ws = len(text.strip())
        if non_ws == 0:
            return 0

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

    def estimate_kv_cache_usage(self, token_count: int) -> str:
        """Convert token count to a human-readable KV-cache estimate."""
        slots = token_count * 4
        if token_count < 10000:
            return f"{token_count:,} tokens (~{slots:,} KV slots)"
        return f"{token_count:,} tokens (~{slots:,} KV slots — near context limit)"
