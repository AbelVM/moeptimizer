"""Quality-anchor / volatile-context methods extracted from AgentContextOptimizer (E1 god-object decomposition)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from moeptimizer.optimizer import AgentContextOptimizer

logger = logging.getLogger(__name__)


class AnchorOpsMixin:
    """Quality-anchor / volatile-context methods (see AgentContextOptimizer)."""

    def _append_volatile_context(
        self: AgentContextOptimizer,
        messages: list[dict[str, Any]],
        anchor: str,
        rag_context: str,
        warning_lines: list[str],
        proactive_threshold_tokens: int,
    ) -> list[dict[str, Any]]:
        """Append volatile (derived) context as ONE trailing user turn.

        KV-cache stability (review §1/§9, priority fix #1): the quality anchor,
        RAG context, and loop warnings are volatile — they change every turn. If
        they were appended to the active (last) user turn, that turn's content
        would differ between the turn it was generated and the next turn (when it
        becomes historical), shifting the token boundary the backend hashes and
        defeating prefix-cache reuse for everything up to that turn. Appending
        them as a single trailing user turn keeps every historical turn
        byte-identical across turns, so the backend reuses its cached KV for the
        whole stable leading prefix instead of re-prefilling.

        Only injected when the context is already over the proactive threshold
        (matching the prior behavior): lean contexts stay untouched.
        """
        if not messages:
            return messages
        if self.token_counter.count_messages(messages) <= proactive_threshold_tokens:
            return messages
        if self._budget_tokens() <= 100:
            return messages

        parts: list[str] = []
        if anchor:
            parts.append(f"# Conversation Quality Anchor\n{anchor}")
        if warning_lines:
            parts.append("\n\n".join(warning_lines))
        if rag_context:
            parts.append(f"# Relevant Context\n{rag_context}")
        if not parts:
            return messages

        content = "\n\n".join(parts)
        # F1 (review §4.6.2): never let injected context leave a code fence open.
        content = self._balance_code_fences(content)
        # Stable tag so we can find and REMOVE any prior volatile turn from a
        # previous pass before appending a fresh one. Without this, the prior
        # volatile turn becomes a historical user turn on the next request and a
        # new one is appended after it, so the context accumulates one extra
        # volatile turn every turn until eviction (review §8).
        result = [
            dict(msg)
            for msg in messages
            if not msg.get("_volatile_turn")
        ]
        # Avoid duplicating an identical trailing volatile turn from a prior pass.
        if result and result[-1].get("role") == "user" and result[-1].get("content") == content:
            return messages
        result.append({"role": "user", "content": content, "_volatile_turn": True})
        return result

    def _build_quality_anchor(self: AgentContextOptimizer, messages: list[dict[str, Any]]) -> str:
        """Build a compact, **monotonic** anchor from the original request and constraints.

        KV-cache stability (review §5, C5): the anchor is appended as the trailing
        volatile user turn. If its content *churns* turn-to-turn (e.g. the oldest
        constraints drop off a ``[-5:]`` slice while newer ones shift position), the
        backend's prefix cache for that turn is invalidated every turn. To keep the
        trailing turn byte-stable across turns we accumulate constraints **append-only**
        in ``self._anchor_constraints`` and only ever drop from the FRONT (oldest) when
        the cap is exceeded — the most-recent tail therefore stays identical between
        turns. The first request is captured once and never rewritten.

        Only scans real user turns (volatile trailing turns are tagged ``_volatile_turn``
        and stripped by the caller before this runs), so a prior anchor can never leak
        into the next anchor's source text.
        """
        marker = "# Conversation Quality Anchor\n"
        user_messages = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content") or "", str):
                content = msg.get("content") or ""
                if marker in content:
                    content = content.split(marker, 1)[0]
                user_messages.append(content)
        if not user_messages:
            return ""

        # First request is captured once and frozen (monotonic anchor head).
        if not self._anchor_first_request:
            self._anchor_first_request = self._compact_anchor_text(
                self._placeholder_code_blocks(user_messages[0]),
                max_chars=700,
            )
        first_request = self._anchor_first_request

        # Append-only constraint accumulation: only NEW constraints are added; the
        # existing tail is never reordered or rewritten. Dedup against the running
        # set so repeated user turns don't grow the anchor.
        seen = {c[len("- "):] if c.startswith("- ") else c for c in self._anchor_constraints}
        for content in user_messages[1:]:
            compact = self._compact_anchor_text(self._placeholder_code_blocks(content), max_chars=160)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            self._anchor_constraints.append(f"- {compact}")

        # Cap by dropping from the FRONT (oldest), keeping the recent tail stable.
        max_constraints = self._dynamic_max_anchor_constraints()
        if len(self._anchor_constraints) > max_constraints:
            self._anchor_constraints = self._anchor_constraints[-max_constraints:]

        lines = [f"Original request:\n{first_request}"]
        if self._anchor_constraints:
            lines.append("Accumulated constraints:")
            lines.extend(self._anchor_constraints)

        anchor = "\n".join(lines)
        return self._compact_anchor_text(anchor, max_chars=900)

    def _placeholder_code_blocks(self: AgentContextOptimizer, text: str) -> str:
        """Replace large code fences with a compact placeholder for anchors."""
        return re.sub(r"```[\s\S]*?```", "[code block]", text)

    @staticmethod
    def _balance_code_fences(text: str) -> str:
        """Ensure ``text`` has balanced ``` code fences (F1, review §4.6.2).

        Injected volatile context (anchor / RAG / loop warnings) is assembled from
        arbitrary source text that may carry a dangling fence —
        ``_placeholder_code_blocks`` only strips *balanced* ```` ```...``` ```` pairs,
        so an odd fence survives. An unterminated fence makes the backend treat
        everything after it as code, bleeding past the injected message and degrading
        MTP draft acceptance. Count line-start fence delimiters; an odd count is a
        dangling opener, closed by appending a fence so the block terminates within this
        message. Deterministic ⇒ cache-stable, and the volatile turn is trailing so
        nothing before it shifts.
        """
        if len(re.findall(r"(?m)^[ \t]*```", text)) % 2 == 0:
            return text
        sep = "" if text.endswith("\n") else "\n"
        return f"{text}{sep}```"

    def _compact_anchor_text(self: AgentContextOptimizer, text: str, max_chars: int) -> str:
        """Compact whitespace and truncate text for quality anchors."""
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3].rstrip()}..."
