"""Content-addressed store for reversible compression (review §4.1.2 / Forward plan B1).

When the proxy compresses a large tool output / code block, the original bytes are
kept here keyed by their SHA-256 hash and the context carries only a compact
placeholder + handle. Compression is therefore *reversible*: the original can be
retrieved by handle (HTTP ``GET /v1/content/{handle}`` today; a model-facing
``expand(id)`` tool is the planned follow-up) instead of being lost forever, which is
a driver of the low semantic-similarity / ``code_block_loss`` signals.

The handle is the content hash, so the placeholder is deterministic — the same
content always yields the same placeholder, which keeps the optimized prefix
byte-stable (cache-safe) when the transform is applied on first appearance.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict


class ContentStore:
    """Bounded, thread-safe, content-addressed store keyed by SHA-256.

    Per-optimizer (one per session) so concurrent sessions never share stored
    content (same isolation guarantee as the per-session summarizer, review §4.8.2).
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._lock = threading.Lock()
        self._items: OrderedDict[str, str] = OrderedDict()
        self._max_entries = max_entries

    @staticmethod
    def handle(content: str) -> str:
        """Stable handle for ``content`` (first 16 hex chars of its SHA-256)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def put(self, content: str) -> str:
        """Store ``content`` and return its handle (idempotent)."""
        h = self.handle(content)
        with self._lock:
            self._items[h] = content
            self._items.move_to_end(h)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
        return h

    def get(self, handle: str) -> str | None:
        """Return the stored content for ``handle``, or None if evicted/unknown."""
        with self._lock:
            content = self._items.get(handle)
            if content is not None:
                self._items.move_to_end(handle)
            return content

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
