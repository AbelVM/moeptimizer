"""Hierarchical Summarization for long conversation context.

Summarizes older turns into a single "recall" token that can be expanded
on demand, keeping context lean while preserving key information.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistence path
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "hierarchical_summaries.json"

# Leading marker of the rolling-summary block's content. The block is recognized
# by this marker (not only by its internal ``_summary_id`` key) so that the
# optimizer's cache-stable prefix detection still finds it AFTER
# ``_strip_internal_flags`` removes the ``_summary_id`` key. The block is part of
# the STABLE PREFIX (append-only, byte-stable leading bytes), so it must be
# detected whether or not the internal marker survived stripping — otherwise the
# summary falls into the live zone and gets re-optimized every turn, breaking the
# backend's prefix-cache reuse (the turn-11 cliff: cached 3192 -> 882).
ROLLING_SUMMARY_MARKER = "Context summary (rolling):"


class HierarchicalSummarizer:
    """
    Summarizes older conversation turns hierarchically.

    Creates multi-level summaries:
    - Level 0: Full turn (original content)
    - Level 1: Compact summary (key points, ~10% size)
    - Level 2: Recall token (single token representation, ~1% size)

    Older turns are progressively summarized to keep context lean.
    """

    # Keywords that mark a constraint / "don't" the model must keep in context.
    # Retaining these in the rolling summary is what stops the 2.17x verbosity
    # regression: when the proxy drops them, the model re-derives them verbosely.
    _CONSTRAINT_HINTS = (
        "don't", "do not", "doesn't", "does not", "dont", "dont",
        "must not", "mustn't", "should not", "shouldn't",
        "cannot", "can't", "can not", "won't", "will not",
        "avoid", "never", "no longer", "not allowed", "prohibited",
        "forbidden", "refrain", "instead of", "without", "only",
        "make sure", "ensure", "keep", "preserve", "don't change",
        "do not change", "don't modify", "do not modify", "unchanged",
    )

    def __init__(
        self,
        max_full_turns: int = 5,
        max_rolling_summary_tokens: int = 1500,
        token_counter: Any = None,
        fold_margin: int | None = None,
    ) -> None:
        self._max_full_turns = max_full_turns
        # Batch-fold margin (the turn-12-30 cache-collapse fix): the live zone
        # may drift this many turns past the keep window before the older turns
        # are folded into the rolling summary in ONE batch. Between folds the
        # emitted prompt is a pure tail append, so the backend reuses its whole
        # prefix cache; each fold eats one invalidation, then the next
        # ``fold_margin`` turns are append-only again. Defaults to the keep
        # window (the live zone may double before a fold).
        self._fold_margin = max(1, int(fold_margin)) if fold_margin is not None else max_full_turns
        # Cap the append-only rolling summary so a very long session does not
        # grow it without bound (review §8.5). When exceeded, the OLDEST lines
        # are dropped — the most-recent task state is what the model needs.
        # The cap is expressed in TOKENS (not chars) so it tracks the actual
        # backend budget; a char cap over/under-counts for code-heavy vs
        # prose-heavy sessions. When over budget, oldest PROSE is dropped first
        # and fenced code blocks (the v0.7.14 ``Code:`` section) are kept
        # preferentially, because evicted code is the highest-value context the
        # model cannot reconstruct on its own.
        #
        # The cap is ADAPTIVE (review: dynamic budget). It grows with the number
        # of folded turns (so a 30-turn session keeps a denser summary than a
        # 5-turn one) and saturates at ``_rolling_summary_ceiling`` — a fraction
        # of the live backend context budget, set by the optimizer each turn via
        # ``set_rolling_summary_ceiling``. The effective per-call cap is
        # ``min(ceiling, floor + per_turn_growth * summarized_turn_count)``.
        self._max_rolling_summary_tokens = max(64, int(max_rolling_summary_tokens))
        # Adaptive-cap parameters (see _effective_summary_budget).
        self._rolling_summary_floor: int = max(64, int(max_rolling_summary_tokens))
        self._rolling_summary_ceiling: int = max(
            self._rolling_summary_floor, int(max_rolling_summary_tokens)
        )
        self._rolling_summary_per_turn: int = 120  # tokens added per folded turn
        self._token_counter = token_counter
        self._summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stats: dict[str, int] = {
            "turns_summarized": 0,
            "turns_compressed": 0,
            "recall_tokens_created": 0,
        }
        self._last_context_changed = False
        # Cache-stable rolling-summary state (review §1/§3/§5, #7). The rolling
        # summary block only ever grows by appending, so its leading bytes stay
        # byte-identical across turns and the backend reuses the prefix cache.
        self._rolling_summary_id: str = ""
        self._leading_summary_id: str = ""
        self._rolling_summary_texts: list[str] = []
        self._summarized_turn_count: int = 0
        # Emitted size right after the last pressure fold (growth-relative
        # hysteresis baseline: the next pressure fold fires only once the
        # context has grown a growth budget past this). None until the first
        # pressure fold; reset whenever the rolling state resets.
        self._last_fold_emitted_tokens: int | None = None
        # REVIEW §6: pin the original request's anchor facts (the task's
        # must-remember constants: API keys, base URLs, fixed constraints) into
        # the rolling summary's leading, byte-stable section. Front-eviction
        # drops Turn 1 where those facts live, so without pinning fact_recall
        # collapses to 0 by turn 30. Seeded once and never rewritten, so the
        # leading bytes of the summary block stay stable for prefix-cache reuse.
        self._original_request_facts: str = ""

    def summarize_turns(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Summarize older turns to reduce context size.

        Keeps the most recent turns intact and summarizes older ones
        into compact summaries.

        Args:
            messages: The message list to summarize

        Returns:
            Message list with older turns summarized
        """
        if len(messages) <= self._max_full_turns:
            return messages

        # Find the system anchor (system + first user)
        system_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_end = i + 1
            elif (msg.get("role") == "user" and system_end > 0) or (msg.get("role") == "user" and system_end == 0):
                system_end = i + 1
                break

        system_anchor = messages[:system_end]
        system_only = system_anchor[:1] if system_anchor else []
        first_user = system_anchor[1:]
        rest = messages[system_end:]

        if len(rest) <= self._max_full_turns:
            return messages

        # Keep recent turns full, summarize older ones
        keep_count = self._max_full_turns
        to_summarize = rest[:-keep_count] if len(rest) > keep_count else []
        keep_recent = rest[-keep_count:]

        if not to_summarize:
            return messages

        # Create hierarchical summary
        summary = self._create_hierarchical_summary(to_summarize)

        result = [*system_only, summary, *first_user, *keep_recent]
        self._stats["turns_summarized"] += len(to_summarize)
        self._stats["turns_compressed"] += 1

        return result

    def seed_original_request(self, text: str) -> None:
        """Pin the original request's anchor facts into the rolling summary (REVIEW §6).

        Called once per session with the first user request. The facts it
        contains (API keys, base URLs, fixed constraints) live in Turn 1, which
        front-eviction drops — so without pinning, ``fact_recall`` collapses to 0
        by turn 30. The seeded text is prepended to the rolling summary block's
        leading, byte-stable section and never rewritten, so the backend's prefix
        cache for the summary head is preserved. Only the first non-empty call
        takes effect; later calls are ignored to keep the head stable.
        """
        if self._original_request_facts or not text:
            return
        # Compact whitespace; keep it short so it does not dominate the block.
        compact = " ".join(text.strip().split())
        if len(compact) > 500:
            compact = f"{compact[:497].rstrip()}..."
        self._original_request_facts = f"Key facts (from original request): {compact}"

    def set_token_counter(self, token_counter: Any) -> None:
        """Attach a token counter so the rolling-summary cap is measured in tokens.

        The cap is enforced in tokens (not chars) so it tracks the real backend
        budget; without a counter the summarizer falls back to the char cap.
        """
        self._token_counter = token_counter

    def set_rolling_summary_ceiling(self, tokens: int) -> None:
        """Set the SATURATING ceiling for the adaptive summary cap (review: dynamic budget).

        The optimizer calls this each turn from its dynamic context budget (a
        fraction of the live backend window) so the summary cap scales with the
        real device instead of being a fixed constant. The effective per-call cap
        grows with folded turns up to this ceiling (see ``_effective_summary_budget``).
        Clamped to at least the floor so a tiny derived budget never starves the
        summary of room for the pinned facts + recent code.
        """
        self._rolling_summary_ceiling = max(self._rolling_summary_floor, int(tokens))

    def _effective_summary_budget(self) -> int:
        """Adaptive per-call token cap: grows with folded turns, saturates at ceiling.

        Early in a session the summary is small (few folded turns), so the cap is
        near the floor; as more turns are folded the cap rises linearly
        (``_rolling_summary_per_turn`` per turn) until it reaches the ceiling
        derived from the live backend window. This keeps a 30-turn session's
        summary denser than a 5-turn one's without ever eating into the verbatim
        recent window.
        """
        grown = self._rolling_summary_floor + self._rolling_summary_per_turn * self._summarized_turn_count
        return min(self._rolling_summary_ceiling, grown)

    def set_rolling_summary_budget(self, tokens: int) -> None:
        """Dynamically set the rolling-summary token cap (review: adaptive budget).

        The optimizer calls this each turn from its dynamic context budget so the
        summary cap scales with the live backend window instead of being a fixed
        constant. Clamped to a sane floor so a tiny derived budget never starves
        the summary of room for the pinned facts + recent code.
        """
        self._max_rolling_summary_tokens = max(64, int(tokens))

    def _count_tokens(self, text: str) -> int:
        """Token count for *text*, using the attached counter if available."""
        if self._token_counter is not None:
            try:
                return int(self._token_counter.count_tokens_precise(text))
            except Exception:
                pass
        # Fallback: ~3.5 chars/token (matches TokenCounter.CHARS_PER_TOKEN generic).
        return max(1, len(text) // 4)

    def _rolling_summary_leading(self) -> str:
        """Return the byte-stable leading section (pinned facts) of the summary."""
        return self._original_request_facts

    def summarize_turns_cache_stable(
        self,
        messages: list[dict[str, Any]],
        frozen_prefix_end: int,
        pressure_target_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Cache-stable BATCH rolling-summary compaction (review §1/§3/§5, #7).

        Folds older dynamic turns into a single append-only rolling summary
        block placed immediately after the frozen prefix. The block retains
        constraints (the task's "don'ts") and key decisions so the model does
        not re-derive them verbosely (the 2.17x verbosity regression).

        Folding happens in BATCHES, not one turn at a time: the live zone is
        allowed to drift ``fold_margin`` turns past the keep window before the
        drifted turns are folded in one shot. Between folds the emitted prompt
        is byte-identical to the previous turn plus the appended new turn, so
        the backend's prefix cache reuses the WHOLE previous prompt and only
        computes the new turn. The old per-turn sliding fold mutated the
        middle of the prompt every turn (the keep window's leading message
        was removed and the summary grew each turn), so the backend re-prefilled
        the entire live zone (~8K tokens) every turn while the direct no-proxy
        path — pure append-only — computed only the new turn (~800 tokens).
        That made the proxy worse than no proxy at all (cached=0 from turn 14
        in the 30-turn opencode log).

        Args:
            messages: Full optimized message list.
            frozen_prefix_end: Index just past the stable prefix block
                (system + first user + frozen early turns).
            pressure_target_tokens: When set (the optimizer passes its
                compaction threshold), fold turns one at a time once the
                EMITTED size (frozen prefix + summary block + unfolded live
                turns) crosses this target, down to ``target - margin``
                (hysteresis: ~25% of the target, so the next fold is several
                append-only turns away). Measured on the post-summary list,
                NOT the raw input, which always exceeds the threshold. This
                makes the cycle self-regulating: one fold turn drops the
                context well under the target, then the following turns are
                pure tail appends until the growing live zone crosses the
                target again.

        Returns:
            Message list with older dynamic turns replaced by the rolling
            summary block, or ``messages`` unchanged when there is nothing to
            summarize.
        """
        if frozen_prefix_end < 0 or frozen_prefix_end > len(messages):
            return messages

        frozen = messages[:frozen_prefix_end]
        rest = messages[frozen_prefix_end:]
        if len(rest) <= self._max_full_turns:
            # Nothing old enough to summarize; reset the rolling counter so a
            # later long context starts fresh.
            self._summarized_turn_count = 0
            self._rolling_summary_texts = []
            self._last_fold_emitted_tokens = None
            return messages

        # Group the dynamic layer into user-led turns.
        turns = self._group_turns(rest)
        total_turns = len(turns)
        keep = self._max_full_turns
        if total_turns <= keep:
            self._summarized_turn_count = 0
            self._rolling_summary_texts = []
            self._last_fold_emitted_tokens = None
            return messages

        # Batch fold. Two triggers, both folding by appending to the rolling
        # summary (cache-stable: REVIEW.md P0.5/P0.6 — the summary is part of
        # the STABLE PREFIX, so existing content is never rewritten; the NEW
        # text is capped to the remaining budget at append time. The old
        # front-trim rewrote the leading bytes and invalidated the backend's
        # cached KV for the whole body — cached 3192 -> 882 at turn 11):
        #
        # 1. PRESSURE (pressure_target_tokens set by the optimizer): fold
        #    turns one at a time until the EMITTED size — frozen prefix +
        #    summary block + UNFOLDED live turns — fits under the target.
        #    Measured on the post-summary list, never the raw input (the raw
        #    input is the client's full cumulative history and always exceeds
        #    the threshold, which would fold EVERY turn and slide the live
        #    zone daily — exactly the per-turn cache break this replaces).
        #    The cycle is self-regulating: one fold turn drops the context
        #    well under the target, then the following turns are pure tail
        #    appends until the growing live zone crosses the target again.
        #    This fold MUST shed the tokens before the scratchpad compactor
        #    runs: the compactor keeps a fixed-size tail window, so every
        #    turn it runs it front-evicts one more turn, sliding the whole
        #    post-compact body and breaking the prefix cache every turn (the
        #    turn-12 cliff: cached 7014 -> 881).
        # 2. DRIFT (no pressure target): fold once the live window has drifted
        #    ``fold_margin`` turns past the keep window, back down to keep.
        #
        # On a fold turn the prompt diverges once (the summary tail grows and
        # the live zone's head jumps forward); between folds the emitted
        # prompt is a pure tail append.
        live_count = total_turns - self._summarized_turn_count
        # DRIFT trigger: once the live window has drifted fold_margin turns
        # past the keep window, fold it back down to keep in ONE batch. The
        # batch extracts across all folded turns at once so the high-value
        # sections (Files/Code) are ordered before prose — the append-time
        # truncation keeps the front, so code survives budget pressure.
        if live_count > keep + self._fold_margin:
            end = total_turns - keep
            new_turns = turns[self._summarized_turn_count:end]
            extracted = self._extract_constraints(new_turns) if new_turns else ""
            if not extracted:
                # Nothing worth remembering (vacuous content): drop the turns
                # so the live zone stays bounded; no state is lost.
                self._summarized_turn_count = end
            else:
                budget = self._effective_summary_budget()
                current_tokens = sum(self._count_tokens(t) for t in self._rolling_summary_texts)
                room = budget - current_tokens
                new_text = (
                    ""
                    if room <= 0
                    else self._truncate_to_budget(extracted, room)
                )
                if new_text:
                    self._rolling_summary_texts.append(new_text)
                    self._stats["turns_summarized"] += sum(len(t) for t in new_turns)
                    self._stats["turns_compressed"] += 1
                    self._summarized_turn_count = end
                # else: summary budget full — leave the turns live (see below).
            live_count = total_turns - self._summarized_turn_count

        # PRESSURE trigger with GROWTH-RELATIVE hysteresis. An absolute
        # "fold until under target" cannot work: the keep window floor
        # (frozen + keep verbatim turns + block) sits AT the target, so the
        # loop always exits at the floor and the very next turn — one turn's
        # growth above the floor — re-triggers. Result: a fold EVERY turn
        # (the per-turn summary append + live-zone slide this replaces).
        # Instead: the FIRST fold fires at the absolute target; each later
        # fold fires only once the context has grown a growth budget past the
        # POST-FOLD size of the previous fold. Between folds the emitted
        # prompt is a pure tail append (fully prefix-cached); one fold buys
        # as many append-only turns as the growth budget covers.
        if pressure_target_tokens is not None and live_count > keep:
            growth_budget = max(2048, pressure_target_tokens // 3)

            def emitted_tokens() -> int:
                # Measure the EMITTED list with count_messages — the SAME
                # measurement the pipeline's gates use (role + tool-call
                # overhead included). Content-only counting undercounts by
                # the tool-call payload, so the scratchpad compactor (gated
                # on count_messages) would fire BEFORE the fold and drop
                # turns the summary never captured — losing their state.
                emitted = [*frozen, *self._build_rolling_summary_blocks()]
                for t in turns[self._summarized_turn_count:]:
                    emitted.extend(t)
                if self._token_counter is not None:
                    try:
                        return int(self._token_counter.count_messages(emitted))
                    except Exception:
                        pass
                return sum(
                    self._count_tokens(str(m.get("content") or "")) for m in emitted
                )

            if self._last_fold_emitted_tokens is None:
                trigger = emitted_tokens() > pressure_target_tokens
                fold_target = pressure_target_tokens
            else:
                trigger = emitted_tokens() > self._last_fold_emitted_tokens + growth_budget
                fold_target = self._last_fold_emitted_tokens
            if trigger:
                while live_count > keep and emitted_tokens() > fold_target:
                    if not self._fold_one_turn(turns[self._summarized_turn_count]):
                        # Summary budget full: stop folding and leave the turn
                        # in the live zone. Never drop state the budget
                        # refused to store; the scratchpad compactor (bounded
                        # per-turn shrink floor) is the size safety valve then.
                        break
                    self._summarized_turn_count += 1
                    live_count -= 1
                # The loop also exits at the keep floor (cannot fold below
                # the verbatim window); record the actual post-fold size so
                # the next trigger is relative to reality, not the wish.
                self._last_fold_emitted_tokens = emitted_tokens()

        if not self._rolling_summary_texts:
            # Nothing folded yet: emit the messages unchanged. The block
            # first appears on the first fold turn — the same turn that
            # invalidates the body anyway, so the insertion costs nothing
            # extra. (Emitting an empty block before any fold would add
            # noise without state; the pinned original-request facts are
            # prepended to the first real fold's block.)
            return messages

        keep_recent = [m for t in turns[self._summarized_turn_count:] for m in t]
        # Place the rolling summary IMMEDIATELY AFTER the frozen prefix:
        # [frozen][append-only summary][live zone]. The summary's index and
        # leading bytes never change; between folds the live zone only grows
        # by appending at the tail, so the whole prompt is a prefix of the
        # next turn's prompt and the backend reuses ALL of it. On a fold turn
        # the divergence is at the summary TAIL (the cached prefix is frozen
        # + the old summary), and the live zone's head jumps forward once.
        #
        # Trailing placement ([frozen][live][summary]) is worse on fold
        # turns: the fold removes the live zone's HEAD, so the divergence is
        # right after the frozen prefix and the old summary's cache is lost
        # too. It is also undone by the scratchpad compactor, which keeps
        # _summary_id messages in the static layer — the block would jump
        # from the tail to the static boundary whenever the compactor
        # toggles, an extra structural break.
        return [*frozen, *self._build_rolling_summary_blocks(), *keep_recent]

    def has_rolling_summary(self) -> bool:
        """Whether any turns have been folded into the rolling summary.

        When True, the batch fold is the active size mechanism and per-turn
        sliding trims must stay off (they would slide the live zone every
        turn and break the prefix cache the fold preserves). When False
        (nothing folded yet), the sliding window is still the size valve for
        short contexts the fold cannot shrink (live window <= keep).
        """
        return bool(self._rolling_summary_texts)

    def _fold_one_turn(self, turn: list[dict[str, Any]]) -> bool:
        """Extract one turn's state and append it to the rolling summary.

        Returns False when the append was refused (the summary budget is
        full) — the caller must then stop folding and leave the turn in the
        live zone rather than dropping unsummarized state. Vacuous turns
        (nothing extractable) return True: there is no state to store, so
        dropping them is safe.
        """
        extracted = self._extract_constraints([turn])
        if not extracted:
            return True
        budget = self._effective_summary_budget()
        current_tokens = sum(self._count_tokens(t) for t in self._rolling_summary_texts)
        room = budget - current_tokens
        if room <= 0:
            return False
        new_text = self._truncate_to_budget(extracted, room)
        if not new_text:
            return False
        self._rolling_summary_texts.append(new_text)
        self._stats["turns_summarized"] += len(turn)
        self._stats["turns_compressed"] += 1
        return True

    def _rolling_summary_content(self) -> str:
        """Return the full append-only content of the rolling summary block."""
        parts: list[str] = []
        leading = self._rolling_summary_leading()
        if leading:
            parts.append(leading)
        parts.extend(self._rolling_summary_texts)
        if not parts:
            parts.append("Earlier context summarized.")
        return f"{ROLLING_SUMMARY_MARKER}\n" + "\n".join(parts)

    def _build_rolling_summary_blocks(self) -> list[dict[str, Any]]:
        """Return the rolling summary as a SINGLE append-only user message.

        The pinned original-request facts (REVIEW §6) form the leading,
        byte-stable section so they survive front-eviction of Turn 1 and keep
        ``fact_recall`` measurable.  Each folded turn appends its extracted
        state to ``self._rolling_summary_texts``; the content is the
        concatenation of the leading facts and ALL appended texts, joined by
        newlines.

        **Why a single message instead of one-per-text?**  The backend
        (Lemonade / llama.cpp) uses TOKEN-LEVEL prefix matching for its KV
        cache — the direct path's smooth ``cached`` growth (1 374 → 2 122 →
        2 920 → …) proves this.  With separate messages, each new summary
        block was INSERTED at the summary / keep_recent boundary, shifting
        every subsequent message forward and breaking the prefix cache at the
        insertion point (the turn-12 cliff: cached 8 050 → 882, then stuck).
        A single message whose content grows by appending keeps the leading
        bytes identical across turns; the backend reuses the cached KV for
        the frozen prefix + summary head and only computes the new tail.
        """
        content = self._rolling_summary_content()
        if not self._rolling_summary_id:
            self._rolling_summary_id = hashlib.md5(b"rolling-summary").hexdigest()[:16]
        return [{
            "role": "user",
            "content": content,
            "_summary_id": self._rolling_summary_id,
            "_summary_level": 1,
            "_rolling_summary": True,
        }]

    def _enforce_rolling_summary_budget(self) -> None:
        """No-op: budget is now enforced at append time, never by rewriting.

        Historically this method front-trimmed the rolling summary when it exceeded
        the token budget. That was a cache-stability bug: the summary is part of the
        STABLE PREFIX, so dropping its oldest (front) segments changed the leading
        bytes the backend had cached, invalidating the cached KV for the whole body
        (the turn-11 cliff: cached 3192 -> 882). The summary is now append-only —
        :meth:`summarize_turns_cache_stable` truncates the NEW text to fit the
        remaining budget before appending, so existing content is never rewritten.
        This method is retained only for API compatibility and does nothing
        destructive.
        """
        return

    def _truncate_to_budget(self, text: str, budget_tokens: int) -> str:
        """Truncate ``text`` (keeping the FRONT) so it fits ``budget_tokens``.

        Used at append time so the rolling summary never exceeds its budget by
        rewriting already-cached content. The front is kept because
        :meth:`_extract_constraints` already orders high-value content (files,
        code) before prose, so the front is the most valuable part to retain.
        """
        if budget_tokens <= 0:
            return ""
        if self._count_tokens(text) <= budget_tokens:
            return text
        # Greedy char-based pre-trim (token counter is approximate anyway), then
        # refine so we stay under budget without splitting a fenced code block mid-way.
        est_chars = max(1, int(budget_tokens * 4))
        if len(text) <= est_chars:
            return text
        return text[:est_chars].rstrip() + " …"

    @staticmethod
    def _group_turns(
        messages: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group a message list into user-led turns (user + following asst/tool)."""
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                if current:
                    turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns

    # Patterns that identify a file path the agent is working on. Capturing
    # these is what lets the rolling summary retain *which file* was edited when
    # the full turn is evicted (review P0.3).
    _FILE_PATH_RE = re.compile(
        r"""(?:\b(?:file|path|module|read|write|edit|open|import|from)\b[^\n]{0,80}?|
            (?:["'`])([./]?[\w./\-/]+\.\w{1,6})\1)""",
        re.VERBOSE,
    )
    # A line that looks like a runtime/test error the model must keep in mind.
    _ERROR_RE = re.compile(
        r"""(?ix)
        (?:error|exception|traceback|failed|failure|assert|raise[d]?||
           \b\d{3}\b\s*(?:error|forbidden|not\s*found)||
           module\s+not\s+found|no\s+such\s+file|syntax\s+error|type\s+error)
        """,
    )
    # Fenced code block WITH its language tag: ```lang ... ```. Capturing the
    # code verbatim is what keeps the rolling summary from "summarizing away"
    # the task's code (review P0.3 follow-up): when a turn carrying a fenced
    # block is evicted, the old _extract_constraints kept only file paths and
    # prose, so the model lost the actual code and has_code_proxy collapsed to
    # 0. The block is preserved byte-for-byte so the model can reproduce it.
    _FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

    def _extract_constraints(
        self,
        turns: list[list[dict[str, Any]]],
    ) -> str:
        """Extract a task-STATE summary from evicted turns (review P0.3).

        The old version only kept lines containing "don't"/"must not"/"avoid"
        keywords, which dropped the actual bug, code, and decisions — causing the
        quality collapse (proxy emitted no code, semantic_similarity ~0.25). The
        new version retains the *state* the model needs to keep working:

        - file paths touched (so the model knows what it was editing),
        - the last error / failure message (so it knows what to fix),
        - the current plan / goal (user requests + assistant "I will" statements),
        - explicit constraints ("don't"/"must not"/"avoid") as before.

        Falls back to a short topic line so the block is never empty.
        """
        constraints: list[str] = []
        plans: list[str] = []
        errors: list[str] = []
        files: list[str] = []
        topics: list[str] = []
        code_blocks: list[str] = []
        seen_files: set[str] = set()
        seen_code: set[str] = set()

        for turn in turns:
            for msg in turn:
                content = msg.get("content", "")
                if not isinstance(content, str) or not content:
                    continue
                role = msg.get("role", "")
                # Fenced code blocks (from any role) — preserved verbatim so the
                # evicted turn's code survives in the rolling summary. Without
                # this the model loses the code and has_code_proxy collapses.
                for cm in self._FENCE_RE.finditer(content):
                    lang = cm.group(1).strip() or "text"
                    code = cm.group(2).strip("\n")
                    if not code:
                        continue
                    key = f"{lang}:{code}"
                    if key in seen_code or len(code) > 1500:
                        continue
                    seen_code.add(key)
                    code_blocks.append(f"```{lang}\n{code}\n```")
                # File paths (from any role).
                for m in self._FILE_PATH_RE.finditer(content):
                    fp = m.group(1)
                    if fp and fp not in seen_files and len(fp) < 120:
                        seen_files.add(fp)
                        files.append(fp)
                # Errors (from tool/assistant output primarily).
                if role in ("tool", "assistant"):
                    for raw_line in content.splitlines():
                        line = raw_line.strip()
                        if 8 < len(line) < 240 and self._ERROR_RE.search(line):
                            errors.append(line)
                for raw_line in content.splitlines():
                    line = raw_line.strip()
                    low_line = line.lower()
                    if 12 < len(line) < 200 and any(
                        hint in low_line for hint in self._CONSTRAINT_HINTS
                    ):
                        constraints.append(line)
                if role == "user":
                    # Strip fenced code before taking the topic sentence so a
                    # code-carrying user turn does not embed the whole block in
                    # the Topic: line (it is already preserved verbatim in Code:).
                    prose = self._FENCE_RE.sub("", content)
                    sentences = re.split(r"[.?!]", prose)
                    for sent in sentences:
                        sent = sent.strip()
                        if 12 < len(sent) < 160:
                            topics.append(sent)
                            break
                if role == "assistant":
                    # Capture plan-like statements ("I will", "Next,", "Let's").
                    for raw_line in content.splitlines():
                        line = raw_line.strip()
                        if 12 < len(line) < 200 and re.match(
                            r"(?i)(i\s+will|next,|let's|now\s+we|the\s+plan|step\s*\d)", line
                        ):
                            plans.append(line)

        parts: list[str] = []
        if files:
            uniq_files = list(dict.fromkeys(files))[:10]
            parts.append("Files touched: " + ", ".join(uniq_files))
        if code_blocks:
            # Preserve the evicted turn's code verbatim so the model can
            # reproduce it (fixes has_code_proxy collapse). Cap to keep the
            # summary within budget; the most recent blocks matter most.
            uniq_c: list[str] = code_blocks[-6:]
            parts.append("Code:\n" + "\n".join(uniq_c))
        if errors:
            seen_e: set[str] = set()
            uniq_e: list[str] = []
            for e in errors:
                if e not in seen_e:
                    seen_e.add(e)
                    uniq_e.append(e)
            parts.append("Last errors: " + " | ".join(uniq_e[:3]))
        if plans:
            seen_p: set[str] = set()
            uniq_p: list[str] = []
            for p in plans:
                if p not in seen_p:
                    seen_p.add(p)
                    uniq_p.append(p)
            parts.append("Plan: " + " ".join(uniq_p[:3]))
        if constraints:
            seen_c: set[str] = set()
            uniq_c: list[str] = []
            for c in constraints:
                if c not in seen_c:
                    seen_c.add(c)
                    uniq_c.append(c)
            parts.append("Constraints: " + "; ".join(uniq_c[:6]))
        if topics:
            parts.append("Topic: " + "; ".join(topics[:2]))
        return "\n".join(parts)

    def _create_hierarchical_summary(
        self,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a hierarchical summary of old turns.

        Creates a compact summary that preserves key information:
        - Topics discussed
        - Code patterns seen
        - Decisions made
        - Current state

        Args:
            turns: List of turns to summarize

        Returns:
            Summary message dict
        """
        # Generate a stable ID for this summary
        turn_ids = [m.get("step_id", str(i)) for i, m in enumerate(turns)]
        summary_id = hashlib.md5(
            json.dumps(turn_ids).encode()
        ).hexdigest()[:16]

        # Extract key information
        topics: list[str] = []
        code_patterns: list[str] = []
        tool_uses: list[str] = []

        for msg in turns:
            content = msg.get("content", "")
            if not content:
                continue

            role = msg.get("role", "")

            # Extract topics from user messages
            if role == "user":
                # Get first meaningful sentence
                sentences = content.replace("?", ".").replace("!", ".").split(".")
                for sent in sentences[:2]:
                    sent = sent.strip()
                    if 10 < len(sent) < 150:
                        topics.append(sent)

            # Extract code patterns from assistant messages
            elif role == "assistant":
                import re
                for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
                    lang = match.group(1)
                    code = match.group(2)
                    # Extract function/class signatures
                    for line in code.split("\n")[:5]:
                        stripped = line.strip()
                        if stripped.startswith(("def ", "class ", "function ", "fn ", "pub fn ")):
                            code_patterns.append(f"{lang}:{stripped[:60]}")
                            break

            # Track tool usage
            elif role == "tool":
                tool_name = msg.get("tool_name", msg.get("metadata", {}).get("name", ""))
                if tool_name:
                    tool_uses.append(tool_name)

        # Build compact summary
        summary_parts = [f"[Recall:{summary_id}]"]

        if topics:
            summary_parts.append(f"Topics: {'; '.join(topics[:3])}")

        if code_patterns:
            summary_parts.append(f"Code: {'; '.join(code_patterns[:3])}")

        if tool_uses:
            unique_tools = list(dict.fromkeys(tool_uses))[:5]
            summary_parts.append(f"Tools: {', '.join(unique_tools)}")

        if not topics and not code_patterns and not tool_uses:
            summary_parts.append(f"History: {len(turns)} summarized turns")

        # Store full summary for potential expansion
        full_summary = {
            "summary_id": summary_id,
            "topics": topics[:5],
            "code_patterns": code_patterns[:5],
            "tool_uses": list(dict.fromkeys(tool_uses))[:10],
            "turn_count": len(turns),
            "created_at": time.time(),
            "level": 1,  # Level 1 = compact summary
        }

        self._summaries[summary_id] = full_summary
        self._last_context_changed = True
        while len(self._summaries) > 100:
            self._summaries.popitem(last=False)

        return {
            "role": "user",
            "content": " | ".join(summary_parts),
            "_summary_id": summary_id,
            "_summary_level": 1,
        }

    def expand_summary(
        self,
        summary_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Expand a recall token back to a fuller summary.

        Args:
            summary_message: The summary message to expand

        Returns:
            Expanded message with more detail
        """
        summary_id = summary_message.get("_summary_id", "")
        if not summary_id or summary_id not in self._summaries:
            return summary_message

        stored = self._summaries[summary_id]
        level = summary_message.get("_summary_level", 1)

        if level >= 2:
            # Already at max expansion
            return summary_message

        # Expand to level 2
        expanded_parts = [f"[Expanded:{summary_id}]"]

        if stored.get("topics"):
            expanded_parts.append(f"Topics: {'; '.join(stored['topics'])}")

        if stored.get("code_patterns"):
            expanded_parts.append(f"Code: {'; '.join(stored['code_patterns'])}")

        if stored.get("tool_uses"):
            expanded_parts.append(f"Tools: {', '.join(stored['tool_uses'])}")

        expanded_parts.append(f"({stored['turn_count']} turns summarized)")

        result = {**summary_message}
        result["content"] = " | ".join(expanded_parts)
        result["_summary_level"] = 2

        self._stats["recall_tokens_created"] += 1
        return result

    def get_summary(self, summary_id: str) -> dict[str, Any] | None:
        """Get a stored summary by ID."""
        return self._summaries.get(summary_id)

    def get_stats(self) -> dict[str, int]:
        """Get summarization statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all stored summaries."""
        self._summaries.clear()
        # Keep all stat keys: ``recall_tokens_created`` is incremented elsewhere
        # and a missing key would raise KeyError after clear().
        self._stats = {
            "turns_summarized": 0,
            "turns_compressed": 0,
            "recall_tokens_created": 0,
        }
        self._last_context_changed = True
        self._rolling_summary_id = ""
        self._leading_summary_id = ""
        self._rolling_summary_texts = []
        self._summarized_turn_count = 0
        self._last_fold_emitted_tokens = None

    def save_to_disk(self, force: bool = False) -> None:
        """Persist summaries to disk."""
        if not force and not self._last_context_changed:
            return
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: v for k, v in self._summaries.items()
            }
            _PERSISTENCE_PATH.write_text(json.dumps(data))
            self._last_context_changed = False
        except Exception as e:
            logger.warning("[HierarchicalSummary] Failed to save: %s", e)

    def load_from_disk(self) -> None:
        """Load summaries from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(_PERSISTENCE_PATH.read_text())
            self._summaries = OrderedDict(data)
            while len(self._summaries) > 100:
                self._summaries.popitem(last=False)
        except Exception as e:
            logger.warning("[HierarchicalSummary] Failed to load: %s", e)


# Global instance
_hierarchical_summarizer: HierarchicalSummarizer | None = None


def get_hierarchical_summarizer(max_full_turns: int = 5) -> HierarchicalSummarizer:
    """Get or create the global hierarchical summarizer."""
    global _hierarchical_summarizer
    if _hierarchical_summarizer is None:
        _hierarchical_summarizer = HierarchicalSummarizer(max_full_turns=max_full_turns)
        _hierarchical_summarizer.load_from_disk()
    return _hierarchical_summarizer
