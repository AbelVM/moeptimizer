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
# Note: llama.cpp uses 128-token blocks, not 1024
# This can be overridden by querying the model
DEFAULT_CONTEXT_BLOCK_SIZE = 128
CONTEXT_BLOCK_SIZE = DEFAULT_CONTEXT_BLOCK_SIZE

# Global block size that can be updated from model config
_block_size: int = DEFAULT_CONTEXT_BLOCK_SIZE


def set_block_size(size: int) -> None:
    """Set the context block size from model configuration."""
    global _block_size
    _block_size = size


def get_block_size() -> int:
    """Get the current context block size."""
    return _block_size

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


def align_to_block_boundary(text: str, block_size: int | None = None) -> str:
    """Align text to block boundary for prefix cache optimization.

    Pads with whitespace to fill complete blocks, improving
    cache hit rates when static layer size is consistent.
    """
    if block_size is None:
        block_size = get_block_size()
    current_len = len(text)
    remainder = current_len % block_size
    if remainder == 0:
        return text
    padding_needed = block_size - remainder
    return text + "\n" * padding_needed


def get_block_aligned_cache_key(
    messages: list[dict[str, Any]],
    block_size: int | None = None,
) -> str:
    """Generate cache key with block alignment for static layer.

    Returns a tuple of (key, aligned_length) for cache partitioning.
    Uses 32 hex chars (128 bits) to minimize collision risk.
    """
    if block_size is None:
        block_size = get_block_size()
    canonical = canonicalize_prompt_for_cache(messages)
    key = _md5(canonical.encode("utf8")).hexdigest()[:32]
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


def hash_ast_node(node_text: str, node_type: str) -> str:
    """Hash an AST node for expert routing prediction.

    Uses node type + content hash to predict expert routing.
    """
    return _md5(f"{node_type}:{node_text}".encode("utf8")).hexdigest()[:16]
