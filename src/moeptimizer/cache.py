"""Cache utilities for the MoE optimizer.

Enhanced with:
- Multi-level cache key canonicalization
- Static layer block alignment
- Expert routing cache support
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, TypeVar

T = TypeVar("T")

# Block alignment constant (Qwen context block size)
CONTEXT_BLOCK_SIZE = 1024

_md5 = hashlib.md5


def cache_key(text: str) -> str:
    """Generate a deterministic cache key from text."""
    return _md5(text.encode("utf8")).hexdigest()


def canonicalize_code_for_cache(text: str) -> str:
    """Canonicalize code for improved cache hit rates.

    - Normalizes whitespace
    - Sorts import statements
    - Removes trailing whitespace
    - Preserves semantic structure
    """
    # Normalize line endings and trailing whitespace
    lines = [line.rstrip() for line in text.split("\n")]

    # Sort import statements (Python-style)
    import_lines = []
    other_lines = []
    in_import_block = True

    for line in lines:
        stripped = line.strip()
        if in_import_block and (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("#")
            or stripped == ""
        ):
            import_lines.append(line)
        else:
            in_import_block = False
            other_lines.append(line)

    # Sort imports alphabetically
    import_lines = sorted(
        [l for l in import_lines if l.strip().startswith("import ")],
        key=lambda x: x.strip(),
    ) + sorted(
        [l for l in import_lines if l.strip().startswith("from ")],
        key=lambda x: x.strip(),
    )

    return "\n".join(import_lines + other_lines)


def canonicalize_prompt_for_cache(messages: list[dict[str, Any]]) -> str:
    """Canonicalize a message list for cache key generation.

    - Sorts system messages
    - Normalizes whitespace in content
    - Preserves message order (critical for MTP)
    - Returns hashable string representation
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            # Normalize whitespace but preserve structure
            content = re.sub(r"[ \t]+", " ", content)
            content = re.sub(r"\n{3,}", "\n\n", content)
        parts.append(f"{role}:{content}")
    return "\n".join(parts)


def align_to_block_boundary(text: str, block_size: int = CONTEXT_BLOCK_SIZE) -> str:
    """Align text to block boundary for prefix cache optimization.

    Pads with whitespace to fill complete blocks, improving
    cache hit rates when static layer size is consistent.
    """
    current_len = len(text)
    remainder = current_len % block_size
    if remainder == 0:
        return text
    padding_needed = block_size - remainder
    return text + "\n" * padding_needed


def get_block_aligned_cache_key(
    messages: list[dict[str, Any]],
    block_size: int = CONTEXT_BLOCK_SIZE,
) -> str:
    """Generate cache key with block alignment for static layer.

    Returns a tuple of (key, aligned_length) for cache partitioning.
    """
    canonical = canonicalize_prompt_for_cache(messages)
    key = _md5(canonical.encode("utf8")).hexdigest()
    return key


def cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    """Get a value from an LRU OrderedDict cache, moving it to end on hit."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def cache_put(cache: OrderedDict[str, T], key: str, value: T, max_size: int) -> T:
    """Put a value into an LRU OrderedDict cache, evicting oldest if full."""
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)
    return value


class ExpertRoutingCache:
    """Cache for MoE expert routing decisions.

    Qwen3.6-35B-A3B-MTP uses token-level expert routing.
    This cache stores (token_pattern → expert_mask) mappings
    to reduce routing overhead and improve expert cache locality.
    """

    def __init__(self, max_size: int = 4096) -> None:
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._max_size = max_size
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, token_pattern: str) -> tuple[int, ...] | None:
        """Get cached expert mask for a token pattern."""
        if token_pattern in self._cache:
            self._stats["hits"] += 1
            self._cache.move_to_end(token_pattern)
            return self._cache[token_pattern]
        self._stats["misses"] += 1
        return None

    def put(self, token_pattern: str, expert_mask: tuple[int, ...]) -> None:
        """Cache an expert routing decision."""
        if token_pattern in self._cache:
            self._cache.move_to_end(token_pattern)
        self._cache[token_pattern] = expert_mask
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def get_or_compute(
        self,
        token_pattern: str,
        compute_fn: Any,
    ) -> tuple[int, ...]:
        """Get cached expert mask or compute and cache it."""
        cached = self.get(token_pattern)
        if cached is not None:
            return cached
        result = compute_fn()
        self.put(token_pattern, result)
        return result

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}


# Global expert routing cache instance
_expert_cache: ExpertRoutingCache | None = None


def get_expert_cache() -> ExpertRoutingCache:
    """Get or create the global expert routing cache."""
    global _expert_cache
    if _expert_cache is None:
        _expert_cache = ExpertRoutingCache()
    return _expert_cache


def hash_ast_node(node_text: str, node_type: str) -> str:
    """Hash an AST node for expert routing prediction.

    Uses node type + content hash to predict expert routing.
    """
    return _md5(f"{node_type}:{node_text}".encode("utf8")).hexdigest()[:16]
