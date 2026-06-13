"""Cache utilities for the MoE optimizer."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, TypeVar

T = TypeVar("T")

_md5 = hashlib.md5


def cache_key(text: str) -> str:
    """Generate a deterministic cache key from text."""
    return _md5(text.encode("utf8")).hexdigest()


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
