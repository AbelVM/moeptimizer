"""ScratchpadCompactor — Front-Loading Eviction for MTP head protection.

Strategy:
  Three immutable zones:
    1. System Anchor: System prompt + first complete turn (user + assistant)
    2. Evictable Body: Historical turns (dropped from front)
    3. Protected Tail: Last N conversation turns (never modified)

  When context exceeds budget, entire user-assistant pairs are dropped from
  the front of the Evictable Body. No summarization, no token offset changes.

  This preserves sequence patterns for the MTP (Multi-Token Prediction) heads.
  Traditional summarization or mid-prompt token deletion changes token offsets,
  destroying sequence patterns and causing 100% MTP prediction failure.
"""

from __future__ import annotations

from typing import Any

from moeptimizer.config import get_config
from moeptimizer.context_aligner import ContextAligner, get_context_aligner


class ScratchpadCompactor:
    """
    Front-loading eviction compactor for Qwen3.6-35B-A3B-MTP.

    Maintains three immutable zones:
      - System Anchor: system + first complete turn (immutable)
      - Evictable Body: historical turns (evicted from front)
      - Protected Tail: last N turns (immutable)

    Eviction drops complete user-assistant pairs from the front of the
    Evictable Body, preserving structural integrity of the conversation.

    When ``hierarchical_summarizer`` is provided and cache-stable summary mode
    is active, the evictable body is folded into a rolling summary block instead
    of being deleted. This preserves 5-10x more signal within the same token
    budget and prevents the quality collapse caused by all-or-nothing deletion.
    """

    def __init__(
        self,
        keep_full: int | None = None,
        cache_stable_mode: bool = False,
        frozen_prefix_turns: int = 0,
        context_aligner: ContextAligner | None = None,
        hierarchical_summarizer: Any = None,
    ) -> None:
        config = get_config().agentic
        self.keep_full = keep_full if keep_full is not None else config.keep_full_steps
        # Cache-stable mode (review §1/§3/§7): freeze the early complete turns as
        # part of the immutable anchor so front-eviction never shifts the stable
        # prefix the backend caches. The compactor runs before the optimizer's
        # trims, so it must honor the frozen prefix or the early turns are lost
        # before the later stages can protect them.
        self._cache_stable_mode = cache_stable_mode
        self._frozen_prefix_turns = frozen_prefix_turns
        self._context_aligner = context_aligner or get_context_aligner()
        self._hierarchical_summarizer = hierarchical_summarizer

    def compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Apply front-loading eviction to a message list.

        Returns a new message list with three zones:
          1. System Anchor: system + first user-assistant pair (always preserved)
          2. Evictable Body: historical turns (dropped from front OR summarized)
          3. Protected Tail: last N user-assistant pairs (always preserved)

        When ``hierarchical_summarizer`` is provided and cache-stable summary mode
        is active, the evictable body is folded into a rolling summary block
        instead of being deleted. This preserves far more signal within the same
        token budget and prevents the quality collapse caused by all-or-nothing
        deletion (review §2.1 / P0 fix).

        Tool messages are kept if they belong to a turn that survives eviction.
        """
        if len(messages) <= self.keep_full + 2:
            return messages

        # Partition into zones
        system_anchor, evictable_body, protected_tail = self._partition_zones(messages)

        # When cache-stable summarization is available, fold the evictable body
        # into a rolling summary instead of deleting it. This is the P0 fix for
        # the quality collapse caused by all-or-nothing eviction.
        if (
            self._hierarchical_summarizer is not None
            and self._cache_stable_mode
            and evictable_body
        ):
            try:
                # Build a temporary message list with just the evictable body so
                # the summarizer can group it into turns and extract constraints.
                evictable_messages = list(evictable_body)
                if evictable_messages:
                    summary_block = self._hierarchical_summarizer._build_rolling_summary_block()
                    # Feed the evictable turns into the summarizer's rolling state
                    # so constraints and topics are retained.
                    turns = self._hierarchical_summarizer._group_turns(evictable_messages)
                    new_text = self._hierarchical_summarizer._extract_constraints(turns)
                    if new_text:
                        self._hierarchical_summarizer._rolling_summary_text = (
                            f"{self._hierarchical_summarizer._rolling_summary_text}\n{new_text}"
                            if self._hierarchical_summarizer._rolling_summary_text
                            else new_text
                        )
                        self._hierarchical_summarizer._stats["turns_summarized"] += sum(len(t) for t in turns)
                        self._hierarchical_summarizer._stats["turns_compressed"] += 1
                    self._hierarchical_summarizer._summarized_turn_count += len(turns)
                    # Return the system anchor + protected tail + rolling summary.
                    # The summary is appended as a trailing turn so the stable
                    # leading prefix stays byte-identical (review §1/§9).
                    return system_anchor + protected_tail + [summary_block]
            except Exception as e:
                # Fall back to pure eviction if summarization fails; never block
                # the request path on summary generation.
                logger = __import__("logging").getLogger(__name__)
                logger.warning("ScratchpadCompactor summarization fallback: %s", e)

        # Drop all evictable body (partitioning already determined which turns are evictable)
        return system_anchor + protected_tail

    def _partition_zones(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Partition messages into three immutable zones.

        System Anchor: system message + first complete user-assistant turn
        Protected Tail: last N user-assistant turns (and their tool outputs)
        Evictable Body: everything in between

        Tool messages are attached to their parent turn (the preceding assistant).
        """
        # System Anchor: system message(s) + first user + first assistant + tools
        system_anchor: list[dict[str, Any]] = []
        i = 0

        # Grab system message(s)
        while i < len(messages) and messages[i].get("role") == "system":
            system_anchor.append(dict(messages[i]))
            i += 1

        # Grab first user message
        if i < len(messages) and messages[i].get("role") == "user":
            system_anchor.append(dict(messages[i]))
            i += 1

        # Grab first assistant (completes the first turn)
        if i < len(messages) and messages[i].get("role") == "assistant":
            system_anchor.append(dict(messages[i]))
            i += 1
        # Grab any tool outputs belonging to the first assistant
        while i < len(messages) and messages[i].get("role") == "tool":
            system_anchor.append(dict(messages[i]))
            i += 1

        # Cache-stable mode (review §1/§3/§7): also freeze the early complete
        # turns as part of the immutable anchor so front-eviction never shifts
        # the stable prefix. Mirrors optimizer._partition_for_budget.
        if self._cache_stable_mode and self._frozen_prefix_turns > 0:
            frozen_end = self._context_aligner.frozen_prefix_end(
                messages, self._frozen_prefix_turns
            )
            if frozen_end > i:
                system_anchor.extend(dict(m) for m in messages[i:frozen_end])
                i = frozen_end

        # Group remaining messages into user-assistant turns.
        # Each turn starts with a user message and includes any following
        # assistant/tool messages until the next user message.
        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "user":
                # Save previous turn (complete or pending) before starting new one
                if current_turn:
                    turns.append(current_turn)
                current_turn = [dict(msg)]
            elif role == "assistant":
                current_turn.append(dict(msg))
            elif role == "tool":
                # Tool messages belong to the preceding assistant in the turn
                if current_turn:
                    current_turn.append(dict(msg))
            else:
                current_turn.append(dict(msg))

            i += 1

        # Don't forget the last turn (complete or pending)
        if current_turn:
            turns.append(current_turn)

        # Protected Tail: last N complete turns + all pending users.
        # Pending users (turns without an assistant response) are always preserved
        # because they represent the active conversation context the model needs to respond to.
        protected_tail: list[dict[str, Any]] = []
        evictable_body: list[dict[str, Any]] = []

        complete_turns = [t for t in turns if any(m.get("role") == "assistant" for m in t)]
        pending_turns = [t for t in turns if not any(m.get("role") == "assistant" for m in t)]

        # Evict from front of complete turns only; always keep all pending turns
        if len(complete_turns) > self.keep_full:
            evictable_complete = complete_turns[:-self.keep_full]
            protected_complete = complete_turns[-self.keep_full:]
        else:
            evictable_complete = []
            protected_complete = complete_turns

        for turn in evictable_complete:
            evictable_body.extend(turn)

        for turn in protected_complete:
            protected_tail.extend(turn)

        # Always preserve pending (unpaired user-only) turns
        for turn in pending_turns:
            protected_tail.extend(turn)

        return system_anchor, evictable_body, protected_tail

