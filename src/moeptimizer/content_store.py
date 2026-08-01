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
import re
import threading
from collections import OrderedDict
from typing import Any

# Name of the model-facing tool that retrieves a compressed tool output's original
# by handle (review §4.1.2 / Forward plan B1). The proxy injects this tool into the
# schema when reversible compression is on and fulfils it from the ContentStore.
EXPAND_TOOL_NAME = "expand_content"

# Matches the handle embedded in a reversible-compression placeholder, e.g.
# "[original retained: handle=abc123def456, 5000 chars; ...]".
_HANDLE_RE = re.compile(r"\[original retained: handle=([0-9a-f]+)")


def expand_content_tool() -> dict[str, Any]:
    """OpenAI tool schema for ``expand_content(handle)``.

    Advertised to the model so it can request the full original of a tool output
    that reversible compression replaced with a ``[original retained: handle=...]``
    placeholder. The proxy fulfils the call from the per-session ContentStore.
    """
    return {
        "type": "function",
        "function": {
            "name": EXPAND_TOOL_NAME,
            "description": (
                "Retrieve the full original text of a tool output that was compressed "
                "to a '[original retained: handle=...]' placeholder. Pass the handle "
                "from the placeholder; returns the complete uncompressed content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "The handle from the '[original retained: handle=...]' placeholder.",
                    },
                },
                "required": ["handle"],
            },
        },
    }


def extract_placeholder_handle(content: str) -> str | None:
    """Return the handle embedded in a reversible-compression placeholder, or None."""
    m = _HANDLE_RE.search(content)
    return m.group(1) if m else None


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
