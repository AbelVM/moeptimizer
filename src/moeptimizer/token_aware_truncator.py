"""Token-Aware Truncation using tiktoken.

Uses tiktoken to trim at true token boundaries, preserving whole-token
alignment and avoiding partial token truncation.
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

from moeptimizer.context_aligner import ContextAligner, get_context_aligner

logger = logging.getLogger(__name__)


class TokenAwareTruncator:
    """
    Truncates text at true token boundaries using tiktoken.

    Unlike character-based truncation which can split tokens, this ensures
    truncation happens only at token boundaries, preserving model input
    integrity and avoiding partial-token artifacts.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        cache_stable_mode: bool = False,
        frozen_prefix_turns: int = 0,
        context_aligner: ContextAligner | None = None,
        token_calibration: float = 1.0,
    ) -> None:
        self._model_name = model_name
        self._encoder: tiktoken.Encoding | None = None
        try:
            self._encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize tiktoken encoder. "
                    "Ensure tiktoken is installed: pip install tiktoken"
                ) from e
        # Cache-stable mode (review §1/§3/§7): freeze the early complete turns as
        # part of the immutable anchor so budget trimming never shifts the stable
        # prefix the backend caches.
        self._cache_stable_mode = cache_stable_mode
        self._frozen_prefix_turns = frozen_prefix_turns
        self._context_aligner = context_aligner or get_context_aligner()
        # Token-count calibration (review §1/§9, priority fix #6). tiktoken's
        # cl100k_base BPE diverges from the backend's real tokenizer (Qwen), so
        # raw counts are wrong for code-heavy prompts. The proxy learns a ratio
        # from the backend's actual `prompt_tokens` on the previous turn and scales
        # its counts so the budget is enforced against the backend's true token
        # count instead of an estimate.
        self._token_calibration = max(0.5, min(2.0, float(token_calibration)))

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Return text unchanged.

        The KV-cache stability guide forbids chopping historical message content.
        This method is kept for API compatibility, but budget enforcement now
        drops whole turns/messages from the top instead of slicing text.
        """
        return text

    def truncate_message(
        self,
        message: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        """Return message unchanged.

        Content-level truncation changes token IDs for preserved history and can
        invalidate downstream KV-cache matching. Use `trim_messages_to_budget`
        to evict whole turns instead.
        """
        return message

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: The text to count tokens for

        Returns:
            Token count
        """
        if not text:
            return 0

        return len(self._encoder.encode(text))

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """Count tokens in a message dict.

        Args:
            message: The message dict

        Returns:
            Token count including message overhead
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return self.count_tokens(content) + 5  # Per-message overhead
        elif isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += self.count_tokens(part.get("text", ""))
            return total + 5
        return 5

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count total tokens across all messages.

        The raw tiktoken count is scaled by ``_token_calibration`` so the budget
        is enforced against the backend's true tokenizer (review §1/§9, #6).

        Args:
            messages: List of message dicts

        Returns:
            Calibrated total token count
        """
        raw = sum(self.count_message_tokens(m) for m in messages)
        return round(raw * self._token_calibration)

    def trim_messages_to_budget(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Trim messages to fit within a token budget.

        Uses front-loading eviction: drops complete user-assistant pairs
        from the front of the evictable body. Truncates individual messages
        only at token boundaries.

        Args:
            messages: The message list to trim
            max_tokens: Maximum total tokens

        Returns:
            Trimmed message list
        """
        if not messages:
            return messages

        total = self.count_messages_tokens(messages)
        if total <= max_tokens:
            return messages

        # Partition into zones
        system_anchor, evictable_body, protected_tail = self._partition_for_budget(
            messages
        )

        # Reserve space for non-evictable zones
        reserved = self.count_messages_tokens(system_anchor) + self.count_messages_tokens(
            protected_tail
        )
        evictable_budget = max(0, max_tokens - reserved)

        # Evict from front of evictable body
        evictable_body = self._evict_for_budget(evictable_body, evictable_budget)

        result = system_anchor + evictable_body + protected_tail

        # If still over budget, drop whole non-system messages from the front of
        # the dynamic layer. Do not truncate content or remove the active last
        # user turn unless the static prefix itself exceeds the budget.
        total = self.count_messages_tokens(result)
        if total > max_tokens:
            result = self._drop_whole_messages_from_front(result, max_tokens)

        return result

    def _partition_for_budget(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Partition messages into three zones for budget trimming."""
        system_anchor: list[dict[str, Any]] = []
        i = 0

        if i < len(messages) and messages[i].get("role") == "system":
            system_anchor.append(messages[i])
            i += 1

        summary_messages: list[dict[str, Any]] = []
        while i < len(messages) and messages[i].get("_summary_id"):
            summary_messages.append(messages[i])
            i += 1

        if i < len(messages) and messages[i].get("role") == "user":
            system_anchor.append(messages[i])
            i += 1

        # Cache-stable mode (review §1/§3/§7): also freeze the early complete
        # turns as part of the immutable anchor so budget trimming never shifts
        # the stable prefix. Mirrors optimizer._partition_for_budget.
        if self._cache_stable_mode and self._frozen_prefix_turns > 0:
            frozen_end = self._context_aligner.frozen_prefix_end(
                messages, self._frozen_prefix_turns
            )
            if frozen_end > i:
                system_anchor.extend(dict(m) for m in messages[i:frozen_end])
                i = frozen_end

        # Group remaining into user-assistant pairs
        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            if role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [dict(msg)]
            else:
                if not current_turn:
                    current_turn = [{"_orphan": True}]
                    current_turn.append(dict(msg))
                else:
                    current_turn.append(dict(msg))
            i += 1

        if current_turn and not any(m.get("_orphan", False) for m in current_turn):
            turns.append(current_turn)
        elif current_turn:
            if turns:
                turns[-1].extend([m for m in current_turn if not m.get("_orphan")])
            else:
                turns.append(current_turn)

        complete_turns = [t for t in turns if any(m.get("role") == "assistant" for m in t)]
        pending_turns = [t for t in turns if not any(m.get("role") == "assistant" for m in t)]

        keep = 3  # Default keep_full_steps
        if len(complete_turns) > keep:
            evictable = [m for t in complete_turns[:-keep] for m in t]
            protected = [m for t in complete_turns[-keep:] for m in t]
        else:
            evictable = []
            protected = [m for t in complete_turns for m in t]

        for t in pending_turns:
            protected.extend(t)

        protected.extend(summary_messages)

        return system_anchor, evictable, protected

    def _evict_for_budget(
        self,
        evictable_body: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        """Drop pairs from front of evictable body until under budget."""
        if not evictable_body:
            return evictable_body

        pairs: list[list[dict[str, Any]]] = []
        current_pair: list[dict[str, Any]] = []

        for msg in evictable_body:
            role = msg.get("role", "")
            if role == "user":
                if current_pair:
                    pairs.append(current_pair)
                current_pair = [dict(msg)]
            else:
                current_pair.append(dict(msg))

        if current_pair:
            pairs.append(current_pair)

        total_tokens = sum(self.count_messages_tokens(pair) for pair in pairs)

        while total_tokens > budget and pairs:
            pair_tokens = self.count_messages_tokens(pairs[0])
            total_tokens -= pair_tokens
            pairs = pairs[1:]

        return [m for pair in pairs for m in pair]

    def _drop_whole_messages_from_front(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Drop whole messages from the front after the system anchor."""
        if not messages:
            return messages

        # Cache-stable mode (review §1/§3/§7): the frozen early-turn prefix is
        # immutable, so it is always kept even in the last-resort fallback.
        frozen_end = 0
        if self._cache_stable_mode and self._frozen_prefix_turns > 0:
            frozen_end = self._context_aligner.frozen_prefix_end(
                messages, self._frozen_prefix_turns
            )

        # Preserve the last user turn as the active request whenever possible.
        last_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        protected_tail: list[dict[str, Any]] = []
        if last_user_idx >= 0:
            protected_tail = [dict(msg) for msg in messages[last_user_idx:]]
            dynamic_middle = messages[frozen_end:last_user_idx]
        else:
            dynamic_middle = messages[frozen_end:]

        result = [dict(msg) for msg in messages[:frozen_end]]
        for msg in dynamic_middle:
            if self.count_messages_tokens([*result, msg, *protected_tail]) <= max_tokens:
                result.append(dict(msg))
            else:
                break

        result.extend(protected_tail)
        return result

    def _truncate_individual_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """No-op for API compatibility; content truncation is disabled."""
        return messages
