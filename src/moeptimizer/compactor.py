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

    The evicted turns are folded into the cache-stable rolling summary by the
    optimizer's single summary step (Step 8.5), not here. Pure front-eviction
    keeps the stable prefix byte-identical for backend prefix-cache reuse.
    """

    def __init__(
        self,
        keep_full: int | None = None,
        cache_stable_mode: bool = False,
        frozen_prefix_turns: int = 0,
        context_aligner: ContextAligner | None = None,
        hierarchical_summarizer: Any = None,
        token_counter: Any = None,
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
        # Optional token counter (P0.6): used to honor a per-turn shrink floor so
        # the compactor never drops the body below min_keep_tokens in one call.
        self._token_counter = token_counter

    def compact_messages(
        self,
        messages: list[dict[str, Any]],
        min_keep_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Apply front-loading eviction to a message list.

        Returns a new message list with three zones:
          1. System Anchor: system + first user-assistant pair (always preserved)
          2. Evictable Body: historical turns (dropped from front)
          3. Protected Tail: last N user-assistant pairs (always preserved)

        Pure front-eviction only. The evicted turns are folded into the
        cache-stable rolling summary by the optimizer's single summary step
        (Step 7 pre-compaction), NOT here — having two summary paths double-folds
        content and was a root cause of the quality collapse (review P0.4). Tool
        messages are kept if they belong to a turn that survives eviction.

        The rolling-summary block (marked ``_summary_id`` / ``_rolling_summary``)
        is NEVER evicted: it is the append-only, byte-stable store of all the task
        state the front-eviction would otherwise discard, so dropping it would
        throw away exactly the context the summary exists to preserve. The
        optimizer's ``_partition_for_budget`` already protects this block; the
        compactor must do the same or the folded state is lost on every turn
        (the turn-10+ faithfulness/recall collapse).

        P0.6 (per-turn shrink cap): when ``min_keep_tokens`` is set, the compactor
        retains enough of the evictable body that the resulting context is never
        smaller than that floor in a single call. This bounds the one-shot
        front-eviction rate so the backend's cached KV for the body head stays
        valid (the v0.7.21 turn-13 break was an 8.5K->2K tok collapse that
        invalidated the whole cached body). When honoring the floor leaves the
        context over the hard budget, the remaining over-budget tokens are shed
        gradually on later turns.
        """
        if len(messages) <= self.keep_full + 2:
            return messages

        # Pull out the rolling-summary block(s) so front-eviction can never drop
        # them. They are re-appended to the protected tail below.
        summary_blocks = [dict(m) for m in messages if m.get("_summary_id") or m.get("_rolling_summary")]
        if summary_blocks:
            messages = [m for m in messages if not (m.get("_summary_id") or m.get("_rolling_summary"))]

        # Partition into zones
        system_anchor, _, protected_tail = self._partition_zones(messages)

        # P0.6: decide how much of the evictable body to keep. By default the
        # compactor drops the entire evictable body (system_anchor + protected_tail
        # only). When a shrink floor is requested, retain evictable pairs from the
        # front until the kept context reaches the floor, so the body never
        # collapses below min_keep_tokens in one call.
        kept_evictable: list[dict[str, Any]] = []
        if min_keep_tokens is not None and self._token_counter is not None:
            # Group the evictable body into complete user-led turns (pairs) so we
            # can retain whole turns, not split mid-turn.
            pairs = self._group_pairs(messages)
            # The non-evictable baseline (system anchor + protected tail) in tokens.
            base_tokens = self._token_counter.count_messages(system_anchor)
            base_tokens += self._token_counter.count_messages(protected_tail)
            # Retain pairs from the front until we reach the floor (or run out).
            running = base_tokens
            for pair in pairs:
                pair_tokens = self._token_counter.count_messages(pair)
                if running + pair_tokens > min_keep_tokens and kept_evictable:
                    # Adding this pair would exceed the floor and we already kept
                    # at least one evictable turn — stop (gradual shrink).
                    break
                kept_evictable.extend(pair)
                running += pair_tokens
        # else: drop all evictable body (legacy behavior).

        # Build the result: system anchor + retained evictable body + protected tail.
        result = list(system_anchor) + list(kept_evictable) + list(protected_tail)
        # Re-attach the protected rolling-summary block(s) IMMEDIATELY AFTER the
        # system anchor (frozen prefix), NOT at the tail. The summary block must
        # stay right after the frozen prefix so the leading bytes
        # [frozen][summary] remain byte-stable for backend prefix-cache reuse
        # (REVIEW.md P0.5). Placing it at the tail would put [frozen][keep_recent]
        # as the leading bytes, which change every turn and break the cache.
        if summary_blocks:
            result = list(system_anchor) + summary_blocks + list(kept_evictable) + list(protected_tail)
        return result

    def _group_pairs(self, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group messages into complete user-led turns (pairs) for retention."""
        pairs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                if current:
                    pairs.append(current)
                current = [dict(msg)]
            else:
                current.append(dict(msg))
        if current:
            pairs.append(current)
        return pairs

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

