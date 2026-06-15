"""Token-Aware Truncation using tiktoken.

Uses tiktoken to trim at true token boundaries, preserving whole-token
alignment and avoiding partial token truncation.
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class TokenAwareTruncator:
    """
    Truncates text at true token boundaries using tiktoken.

    Unlike character-based truncation which can split tokens, this ensures
    truncation happens only at token boundaries, preserving model input
    integrity and avoiding partial-token artifacts.
    """

    def __init__(self, model_name: str = "gpt-4") -> None:
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

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to fit within max_tokens, cutting only at token boundaries.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens to keep

        Returns:
            Truncated text that fits within the token limit
        """
        if not text:
            return text

        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self._encoder.decode(truncated_tokens)

    def truncate_message(
        self,
        message: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        """Truncate a single message to fit within max_tokens.

        Args:
            message: The message dict to truncate
            max_tokens: Maximum tokens for this message

        Returns:
            Truncated message dict
        """
        content = message.get("content", "")
        if not isinstance(content, str):
            return message

        truncated = self.truncate_to_token_limit(content, max_tokens)
        if truncated != content:
            result = dict(message)
            result["content"] = truncated
            return result
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

        Args:
            messages: List of message dicts

        Returns:
            Total token count
        """
        return sum(self.count_message_tokens(m) for m in messages)

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

        # If still over budget, truncate individual messages at token boundaries
        total = self.count_messages_tokens(result)
        if total > max_tokens:
            result = self._truncate_individual_messages(result, max_tokens)

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

        if i < len(messages) and messages[i].get("role") == "user":
            system_anchor.append(messages[i])
            i += 1

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

    def _truncate_individual_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Truncate individual messages at token boundaries to fit budget."""
        result: list[dict[str, Any]] = []
        total = self.count_messages_tokens(messages)

        # Start from the front (after system anchor) and truncate
        for msg in messages:
            if total <= max_tokens:
                result.append(msg)
                continue

            role = msg.get("role", "")

            # Never truncate system or assistant messages
            if role in ("system", "assistant"):
                result.append(msg)
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                result.append(msg)
                continue

            # Calculate how much we need to reduce
            msg_tokens = self.count_message_tokens(msg)
            remaining_budget = max_tokens - (total - msg_tokens)

            if remaining_budget <= 0:
                # Skip this message entirely
                total -= msg_tokens
                continue

            # Truncate to remaining budget
            truncated = self.truncate_to_token_limit(content, remaining_budget)
            new_tokens = self.count_tokens(truncated) + 5

            if new_tokens < msg_tokens:
                result.append({**msg, "content": truncated})
                total -= msg_tokens - new_tokens
            else:
                result.append(msg)

        return result
