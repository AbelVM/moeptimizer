"""Incremental updater for context optimization.

Only appends new content to preserve cache hits.
"""

from __future__ import annotations

import hashlib
from typing import Any


class IncrementalUpdater:
    """
    Updates context incrementally to preserve cache hits.

    - Only append new content
    - Never modify middle of cached context
    - Track context version for cache validation
    """

    def __init__(self) -> None:
        self._context_versions: dict[str, int] = {}
        self._last_context_key: str | None = None

    def update_context(
        self,
        messages: list[dict[str, Any]],
        new_content: str,
    ) -> list[dict[str, Any]]:
        """Update context by appending new content.

        If this is a known context (key exists), we can append to the last message
        to preserve cache. Otherwise, we just return messages unchanged.
        """
        if not messages:
            return messages

        # Get current context key
        key = self._get_context_key(messages)

        # If this is a known context, we can potentially append
        if key in self._context_versions and new_content:
                result = [dict(m) for m in messages]
                result[-1] = {
                    **result[-1],
                    "content": result[-1].get("content", "") + "\n" + new_content,
                }
                self._context_versions[key] = self._context_versions.get(key, 0) + 1
                return result

        # Register this context for future reference
        self._context_versions[key] = 1
        self._last_context_key = key

        return messages

    def _get_context_key(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Get key for current context."""
        content = "".join(
            m.get("content", "") for m in messages
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _can_append(
        self,
        key: str,
    ) -> bool:
        """Check if we can append to this context."""
        return key in self._context_versions

    def get_version(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Get current context version."""
        key = self._get_context_key(messages)
        return self._context_versions.get(key, 0)

    def reset_version(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        """Reset version for context (cache invalidated)."""
        key = self._get_context_key(messages)
        self._context_versions.pop(key, None)

    def should_preserve_cache(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
    ) -> bool:
        """Check if cache should be preserved.

        Uses proper prefix detection: old context must be a prefix of new context.
        """
        # Check if old context is a prefix of new context
        # by comparing the actual content, not the hash
        old_content = "".join(m.get("content", "") for m in old_messages)
        new_content = "".join(m.get("content", "") for m in new_messages)

        return new_content.startswith(old_content)


def get_incremental_updater() -> IncrementalUpdater:
    """Get an incremental updater instance."""
    return IncrementalUpdater()
