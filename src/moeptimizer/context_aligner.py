"""Context aligner for cache block optimization.

Aligns context to cache block boundaries to maximize prefix cache hits.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Cache block size for Qwen models (default, can be overridden)
CACHE_BLOCK_SIZE = 128


class ContextAligner:
    """
    Aligns context to cache block boundaries.

    For MoE models, KV-cache fill is extremely slow. This module
    optimizes context structure to maximize cache hit rates.
    """

    def __init__(
        self,
        block_size: int | None = None,
    ) -> None:
        # ``block_size`` is retained for API compatibility; the live pipeline
        # freezes the stable prefix verbatim (freeze_static_prefix) rather than
        # padding to a block boundary, so it is no longer used internally.
        self._block_size = block_size or CACHE_BLOCK_SIZE

    def _find_static_layer_end(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Find the end of the static layer (system + first user)."""
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif (msg.get("role") == "user" and static_end > 0) or (msg.get("role") == "user" and static_end == 0):
                static_end = i + 1
                break
        return static_end

    def prefix_signature(self, messages: list[dict[str, Any]]) -> str:
        """Return a stable hash of the static prefix (system + first user).

        Used to monitor prefix stability across turns: if the signature is
        identical between turns, the backend's automatic prefix cache can reuse
        the prefix. Returns "" when no static prefix exists.
        """
        n = self._find_static_layer_end(messages)
        if n == 0:
            return ""
        payload = json.dumps(
            [{"role": m.get("role"), "content": m.get("content")} for m in messages[:n]],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:32]

    def frozen_prefix_end(
        self,
        messages: list[dict[str, Any]],
        frozen_prefix_turns: int,
    ) -> int:
        """Return the index just past the stable prefix block.

        The stable prefix is ``system`` messages, then the first ``user``
        message, then the next ``frozen_prefix_turns`` complete user-led turns
        (each user message plus the assistant/tool messages that follow it until
        the next user). When ``frozen_prefix_turns <= 0`` this falls back to the
        system-only boundary, preserving the old behavior.
        """
        if frozen_prefix_turns <= 0:
            return self._find_system_end(messages)

        n = len(messages)
        i = 0
        while i < n and messages[i].get("role") == "system":
            i += 1
        # First user message is always part of the stable prefix.
        if i < n and messages[i].get("role") == "user":
            i += 1
        else:
            return i

        # Skip assistant/tool responses from the first user message before
        # counting frozen turns so we don't accidentally consume the first
        # complete turn as part of the anchor.
        while i < n and messages[i].get("role") != "user":
            i += 1

        turns = 0
        while turns < frozen_prefix_turns and i < n:
            # Include this user message and everything until the next user
            # (its assistant/tool responses), which completes one turn.
            i += 1
            while i < n and messages[i].get("role") != "user":
                i += 1
            turns += 1
        return i

    def freeze_static_prefix(
        self,
        original: list[dict[str, Any]],
        optimized: list[dict[str, Any]],
        frozen_prefix_turns: int = 0,
    ) -> list[dict[str, Any]]:
        """Guarantee the stable prefix of ``optimized`` is byte-identical to
        ``original``'s stable prefix.

        When ``frozen_prefix_turns <= 0`` (legacy behavior) only the system
        prompt is frozen verbatim. When ``frozen_prefix_turns > 0`` (cache-stable
        mode, review §1/§3/§7) the system prompt and the next
        ``frozen_prefix_turns`` complete turns are frozen verbatim, while the
        first user message is kept in its (deterministic, stable) compressed form
        so the proxy still saves tokens on the largest message.

        The frozen block is sourced from ``original`` (which the caller passes as
        the already-optimized messages, i.e. *after* tool-output compression at
        step 11.6). Sourcing the frozen prefix from the optimized messages — not
        the raw client payload — is what lets the proxy's boundary compression
        actually take effect on benchmark/agentic traffic: the compressed tool
        output is deterministic and idempotent, so it is byte-stable across turns
        and safe to freeze, while still saving the tokens the compressor recovered.

        Freezing the early turns is what makes the backend's automatic prefix
        cache reusable across turns: the proxy's front-eviction otherwise drops
        the oldest turn every turn, shifting the serialized prefix and forcing a
        full re-prefill. The early turns are kept uncompressed (the client sends
        them identically every turn) so the bytes are guaranteed stable; the
        token cost is bounded and is repaid many times over by avoiding
        re-prefill. Returns ``optimized`` unchanged when the structures are
        incompatible or no stable prefix exists.
        """
        if not original or not optimized:
            return optimized
        if frozen_prefix_turns <= 0:
            n = self._find_system_end(original)
            if n == 0 or len(optimized) < n:
                return optimized
            if self._prefix_equal(original[:n], optimized[:n]):
                return optimized
            frozen = [self._strip_internal(m) for m in original[:n]]
            return frozen + [dict(m) for m in optimized[n:]]

        # Cache-stable mode: freeze system + early turns verbatim, but keep the
        # first user message compressed (it is deterministic and therefore
        # already stable across turns, and freezing it uncompressed would undo
        # the proxy's token savings on the single largest message).
        n = self.frozen_prefix_end(original, frozen_prefix_turns)
        if n == 0 or len(optimized) < n:
            return optimized
        frozen = [self._strip_internal(m) for m in original[:n]]
        first_user = self._find_first_user(original)
        if first_user is not None and 0 <= first_user < len(optimized):
            frozen[first_user] = dict(optimized[first_user])
        return frozen + [dict(m) for m in optimized[n:]]

    @staticmethod
    def _find_first_user(messages: list[dict[str, Any]]) -> int | None:
        """Return the index of the first user message, or None."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                return i
        return None

    @staticmethod
    def _strip_internal(msg: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``msg`` with proxy-internal ``_``-prefixed keys removed."""
        return {key: value for key, value in msg.items() if not key.startswith("_")}

    @staticmethod
    def _find_system_end(messages: list[dict[str, Any]]) -> int:
        """Return the index just past the leading run of system messages."""
        n = 0
        for msg in messages:
            if msg.get("role") == "system":
                n += 1
            else:
                break
        return n

    @staticmethod
    def _prefix_equal(
        a: list[dict[str, Any]],
        b: list[dict[str, Any]],
    ) -> bool:
        """Compare two prefix slices by role + content only."""
        def norm(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"role": m.get("role"), "content": m.get("content")} for m in msgs]

        return norm(a) == norm(b)


def get_context_aligner(
    block_size: int = CACHE_BLOCK_SIZE,
) -> ContextAligner:
    """Get a context aligner instance."""
    return ContextAligner(block_size=block_size)
