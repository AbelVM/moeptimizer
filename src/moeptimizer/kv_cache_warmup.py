"""KV-Cache Warm-Up for MTP Heads.

Runs a cheap forward pass on static layers to pre-populate KV-cache
before the first token generation, reducing first-token latency.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class KVCacheWarmup:
    """
    Pre-populates KV-cache for static layers before first token generation.

    For MoE models, the first token generation is slow because the KV-cache
    must be filled. This module runs a cheap forward pass on the static
    layer (system prompt + first user message) to pre-populate the cache,
    reducing first-token latency on subsequent requests with the same prefix.
    """

    def __init__(
        self,
        max_warmups: int = 32,
        warmup_timeout_ms: float = 500.0,
    ) -> None:
        self._max_warmups = max_warmups
        self._warmup_timeout_ms = warmup_timeout_ms
        self._warmup_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stats: dict[str, int] = {
            "warmups_performed": 0,
            "warmup_hits": 0,
            "warmup_misses": 0,
            "warmup_time_ms": 0,
        }

    def _static_prefix_hash(self, messages: list[dict[str, Any]]) -> str:
        """Generate a hash for the static prefix of messages."""
        static_parts: list[str] = []
        found_first_user = False

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                static_parts.append(f"system:{content}")
            elif role == "user" and not found_first_user:
                static_parts.append(f"user:{content}")
                found_first_user = True
                break

        prefix = "\n".join(static_parts)
        return hashlib.md5(prefix.encode("utf8")).hexdigest()[:32]

    def get_warmup_key(self, messages: list[dict[str, Any]]) -> str:
        """Get the warmup cache key for a message list."""
        return self._static_prefix_hash(messages)

    def has_warmup(self, messages: list[dict[str, Any]]) -> bool:
        """Check if a warmup exists for this message list's static prefix."""
        key = self.get_warmup_key(messages)
        if key in self._warmup_cache:
            self._stats["warmup_hits"] += 1
            self._warmup_cache.move_to_end(key)
            return True
        self._stats["warmup_misses"] += 1
        return False

    def store_warmup(
        self,
        messages: list[dict[str, Any]],
        warmup_data: dict[str, Any],
    ) -> str:
        """Store warmup data for a static prefix.

        Args:
            messages: The message list
            warmup_data: The warmup data from the forward pass

        Returns:
            The warmup cache key
        """
        key = self.get_warmup_key(messages)
        self._warmup_cache[key] = warmup_data
        self._warmup_cache.move_to_end(key)

        while len(self._warmup_cache) > self._max_warmups:
            self._warmup_cache.popitem(last=False)

        self._stats["warmups_performed"] += 1
        return key

    def get_warmup_data(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Get warmup data for a message list's static prefix."""
        key = self.get_warmup_key(messages)
        if key in self._warmup_cache:
            self._warmup_cache.move_to_end(key)
            return self._warmup_cache[key]
        return None

    def record_warmup_time(self, elapsed_ms: float) -> None:
        """Record the time taken for a warmup forward pass."""
        self._stats["warmup_time_ms"] += int(elapsed_ms)

    def should_warmup(
        self,
        messages: list[dict[str, Any]],
        force: bool = False,
    ) -> bool:
        """Determine if a warmup should be performed.

        Args:
            messages: The message list
            force: Force warmup regardless of cache state

        Returns:
            True if warmup should be performed
        """
        if force:
            return True

        # Only warmup if we don't have cached data
        return not self.has_warmup(messages)

    def get_warmup_payload(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Get the warmup payload to send with the request.

        This payload tells the backend to use pre-computed KV-cache
        for the static prefix.

        Args:
            messages: The message list

        Returns:
            Warmup payload dict or None
        """
        warmup_data = self.get_warmup_data(messages)
        if warmup_data is None:
            return None

        return {
            "kv_cache_warmup": {
                "enabled": True,
                "prefix_hash": self.get_warmup_key(messages),
                "static_layer_tokens": warmup_data.get("static_tokens", 0),
            }
        }

    def get_stats(self) -> dict[str, int]:
        """Get warmup statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all warmup data."""
        self._warmup_cache.clear()
        self._stats = {
            "warmups_performed": 0,
            "warmup_hits": 0,
            "warmup_misses": 0,
            "warmup_time_ms": 0,
        }


# Global instance
_kv_warmup: KVCacheWarmup | None = None


def get_kv_cache_warmup() -> KVCacheWarmup:
    """Get or create the global KV-cache warmup manager."""
    global _kv_warmup
    if _kv_warmup is None:
        _kv_warmup = KVCacheWarmup()
    return _kv_warmup
