"""Thinking/reasoning preservation methods extracted from AgentContextOptimizer (E1 god-object decomposition)."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from moeptimizer.optimizer import AgentContextOptimizer

logger = logging.getLogger(__name__)


class ThinkingOpsMixin:
    """Thinking/reasoning preservation methods (see AgentContextOptimizer)."""

    @staticmethod
    def _thinking_key(content: str) -> str:
        """Stable key for an assistant message: hash of its ``content``."""
        return hashlib.md5((content or "").encode("utf-8", "replace")).hexdigest()[:16]

    def pin_tools(self: AgentContextOptimizer, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Pin and re-emit the `tools` schema verbatim (review P1.2 / DO #5).

        The backend's prefix cache includes the serialized `tools` array. If the
        client re-sends tools in a different order or with a different dict layout
        turn-to-turn, the cached prefix no longer matches and the backend must
        re-prefill. We cache the first-seen schema and return it unchanged on
        every subsequent turn, ignoring client reordering. Returns ``None`` when
        no tools were ever seen (the caller should then leave `tools` untouched).
        """
        if tools:
            # Normalize once: stable, sorted-by-name ordering so the serialized
            # bytes are deterministic regardless of client input order.
            normalized = sorted(
                (dict(t) for t in tools if isinstance(t, dict)),
                key=lambda t: str(t.get("function", {}).get("name", t.get("name", ""))),
            )
            if self._pinned_tools is None:
                self._pinned_tools = normalized
            return self._pinned_tools
        return self._pinned_tools

    def capture_thinking(self: AgentContextOptimizer, content: str, reasoning: str | None) -> None:
        """Store the thinking block observed for an assistant ``content``.

        Called by the app layer after a streaming response completes, so the
        proxy remembers the reasoning block the backend cached alongside this
        assistant message. Bounded LRU.
        """
        if not content or not reasoning:
            return
        key = self._thinking_key(content)
        # Route through the optimizer lock (review §4.8.7 / E3): called from the app
        # layer after the stream, concurrently with optimize_messages (which reads the
        # store via _restore_thinking); the RLock keeps the LRU from tearing.
        with self._lock:
            if key in self._thinking_store:
                self._thinking_order.remove(key)
            self._thinking_store[key] = reasoning
            self._thinking_order.append(key)
            while len(self._thinking_order) > 32:
                old = self._thinking_order.pop(0)
                self._thinking_store.pop(old, None)

    def _restore_thinking(self: AgentContextOptimizer, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-inject stored thinking blocks into assistant messages (review P1.1).

        If a client stripped ``reasoning_content`` from an assistant message we
        previously saw WITH thinking, re-add it so the message we send to the
        backend byte-matches what the backend cached — avoiding a forced re-prefill.
        Only adds; never removes existing reasoning the client did echo.
        """
        if not self._thinking_store:
            return messages
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                content = msg.get("content") or ""
                key = self._thinking_key(content)
                reasoning = self._thinking_store.get(key)
                if reasoning:
                    new_msg = dict(msg)
                    new_msg["reasoning_content"] = reasoning
                    out.append(new_msg)
                    continue
            out.append(msg)
        return out

    def _preseed_reasoning(
        self: AgentContextOptimizer,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pre-seed reasoning prefix for MTP optimization.

        Adds task-specific reasoning scaffolding to improve MTP convergence.
        Only applies if there's sufficient budget headroom.
        """
        if not messages:
            return messages

        # Check if we have budget headroom for preseeding
        total_tokens = self.token_counter.count_messages(messages)
        max_tokens = self._budget_tokens()
        if total_tokens > int(max_tokens * 0.9):
            # Too close to budget, skip preseeding
            return messages

        # Find the last user message
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx < 0:
            return messages

        # Pre-seed reasoning for the last user message
        user_msg = messages[last_user_idx]
        content = user_msg.get("content") or ""

        if isinstance(content, str):
            # Add reasoning pre-seed
            preseeded = self.thinking_preserver.preseed_reasoning_prefix(
                content,
                self._task_type,
            )
            # Return a new list with the modified message
            result = [dict(m) for m in messages]
            result[last_user_idx] = {
                **user_msg,
                "content": preseeded,
            }
            return result

        return messages
