"""BudgetGovernor — context-budget computation extracted from the optimizer.

Phase 4 god-object decomposition: this module OWNS the token-budget state and
math that used to live on ``AgentContextOptimizer``. The behavior is identical
to the original inlined methods — the logic was moved verbatim, with only the
backend-window lookup (``self._backend_context_window()`` -> ``self._backend_window()``)
and the budget-state references (now governor-owned fields instead of
``getattr`` guards) adapted. The governor does not own a token counter; callers
pass raw counts into :meth:`BudgetGovernor.calibrated_token_count`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from moeptimizer.config import AppConfig


class BudgetGovernor:
    """Owns the context-budget state and computation for one optimizer session.

    Holds the token-calibration ratio, the conversation horizon (turns seen),
    the current request's code volume, and the last optimized token count, and
    derives the static / adaptive / effective budgets and the per-turn growth
    and shrink ceilings from them. The live backend window is supplied via the
    ``backend_window`` callable so the governor never touches the capability
    probe directly.
    """

    def __init__(
        self,
        config: AppConfig,
        backend_window: Callable[[], int | None],
    ) -> None:
        self._config = config
        self._backend_window = backend_window
        # Token-count calibration (review §1/§9, priority fix #6). Initialized
        # here so budget_tokens() can be called as soon as the governor exists
        # (e.g. to seed the adaptive summary-cap ceiling during the optimizer's
        # __init__) before any later detailed setup.
        self._token_calibration: float = 1.0
        # Conversation horizon (turns optimized by this per-session optimizer).
        # Drives the adaptive budget's horizon term (review §3.1): a longer session
        # gets a larger cached working set. Incremented once per top-level optimize.
        self._turns_seen: int = 0
        # Code volume (tokens) in the current request; the adaptive budget's
        # task-complexity term (review §3.1). Recomputed each optimize.
        self._recent_code_tokens: int = 0
        # Token count of the most recent optimized prompt; the monotonic floor of
        # the adaptive budget and the baseline for the growth/shrink ceilings.
        self._last_optimized_token_count: int | None = None
        # Mirrors the optimizer's ``_cache_stable_summary`` (purely config-derived):
        # the optimizer constructs ``hierarchical_summarizer`` non-None exactly when
        # this flag is set, so effective_budget_tokens()'s original two-part check
        # (``_cache_stable_summary and hierarchical_summarizer is not None``) reduces
        # to this single flag, which the governor can own without the summarizer.
        v050 = config.v050
        self._cache_stable_summary: bool = (
            v050.cache_stable_summary_enabled or v050.hierarchical_summary_enabled
        )

    # -- state accessors / mutators -----------------------------------------

    @property
    def token_calibration(self) -> float:
        """The learned backend-tokenizer ratio (clamped to [0.5, 2.0])."""
        return self._token_calibration

    @property
    def last_optimized_token_count(self) -> int | None:
        """Token count of the most recent optimized prompt, if known."""
        return self._last_optimized_token_count

    def advance_turn(self) -> None:
        """Advance the conversation horizon by one top-level optimize."""
        self._turns_seen += 1

    def set_recent_code_tokens(self, messages: list[dict[str, Any]]) -> None:
        """Recompute the current request's code volume for the adaptive budget."""
        self._recent_code_tokens = self.count_code_tokens(messages)

    def set_token_calibration(self, ratio: float) -> None:
        """Set the token-calibration ratio, clamped to [0.5, 2.0].

        Mirrors the optimizer's clamping logic but WITHOUT the
        ``token_aware_truncator`` side-effect (that stays on the optimizer).
        """
        self._token_calibration = max(0.5, min(2.0, float(ratio)))

    def set_last_optimized_token_count(self, n: int | None) -> None:
        """Record the token count of the most recent optimized prompt."""
        self._last_optimized_token_count = n

    # -- budget computation -------------------------------------------------

    def budget_tokens(self) -> int:
        """Return the effective token budget for the optimized context.

        When ``dynamic_budget_enabled`` is on and the live backend window is known,
        the budget is derived from the REAL context window
        (``max(window * budget_window_fraction, max_optimized_tokens)``) and scaled
        by the learned token-calibration ratio, so it is enforced against the
        backend's true token count rather than a static guess. This adapts the cap
        to the actual device (e.g. a 262K window yields ~15.7K vs the old fixed
        12K) and keeps headroom for generation + the cache-stable prefix. Falls back
        to the static ``max_optimized_tokens`` (floored by the char budget) when the
        window is unknown or dynamic budgeting is disabled.
        """
        cfg = self._config.agentic
        base = self.static_budget_tokens()
        if not cfg.adaptive_budget_enabled:
            return base
        return self.adaptive_budget_tokens(base)

    def static_budget_tokens(self) -> int:
        """The legacy fixed budget: ``max(window * budget_window_fraction,
        max_optimized_tokens, char_budget)`` scaled by calibration.

        Used as the base/floor of the adaptive budget and as the full budget when
        ``adaptive_budget_enabled`` is off (review §4.2.2).
        """
        cfg = self._config.agentic
        char_budget = max(1, cfg.max_optimized_chars // 4)
        static = char_budget if cfg.max_optimized_tokens <= 0 else min(char_budget, cfg.max_optimized_tokens)

        if not cfg.dynamic_budget_enabled:
            return static

        window = self._backend_window()
        if window is None or window <= 0:
            return static

        derived = int(window * cfg.budget_window_fraction)
        # Never go below the configured floor, and never below the char budget.
        budget = max(derived, cfg.max_optimized_tokens, char_budget)
        # Scale by the learned backend-tokenizer ratio so the cap is enforced
        # against true backend tokens (clamped to [0.5, 2.0] upstream).
        return max(1, round(budget * self._token_calibration))

    def adaptive_budget_tokens(self, base: int) -> int:
        """Adaptive, horizon-growing budget (review §3.1 / §4.2.2).

        The fixed cap was pinned to ``max_optimized_tokens`` (12K) while the
        immutable zones need ~16K, so ``evictable_budget`` was 0, the cap was
        ignored (context ran 16-18K), and the fold fired every 2-3 turns against a
        93%-idle window. This replaces the constant with a GROWING ceiling:

        - **horizon**: ``base + adaptive_horizon_growth_tokens * turns_seen`` — a
          longer session carries a larger cached working set;
        - **window ceiling**: capped at ``window * adaptive_window_fraction`` so a
          long session grows toward a sane fraction of the real window, not a guess;
        - **monotonic floor**: never below the previous turn's optimized size, so the
          budget is a ceiling the context grows into — never a target to evict down
          to. This keeps ``reserved < budget`` (the governors stop fighting) and the
          prefix cache stable (no forced mid-body shrinkage).

        The result is scaled by the learned calibration like the static budget.
        """
        cfg = self._config.agentic
        budget = base + cfg.adaptive_horizon_growth_tokens * self._turns_seen

        # Task-complexity term (review §3.1): a code-heavy request grows the budget
        # so more verbatim code survives (and the fold pressure target rises, so
        # code-heavy sessions fold less often). The field is owned by the governor
        # and initialized in its __init__, so it is always available here.
        code_tokens = self._recent_code_tokens
        if cfg.adaptive_code_density_factor and code_tokens:
            budget += int(cfg.adaptive_code_density_factor * code_tokens)

        window = self._backend_window()
        if window is not None and window > 0:
            ceiling_frac = cfg.adaptive_window_fraction
            # Space-based folding (review §4.7): the budget ceiling must reach the
            # fold pressure target (fold_window_fraction) so the budget gates
            # (_trim_to_budget & co.) don't front-evict — and break the prefix
            # cache — before the fold fires on real space pressure near the window.
            fold_frac = self._config.v050.fold_window_fraction
            if fold_frac > 0:
                ceiling_frac = max(ceiling_frac, fold_frac)
            ceiling = int(window * ceiling_frac)
            budget = min(budget, ceiling)

        # The governor owns this field (initialized in its __init__), so it is
        # always available — even when budget_tokens() seeds the summary ceiling
        # during the optimizer's __init__.
        last = self._last_optimized_token_count
        if last:
            budget = max(budget, last)

        return max(1, round(budget * self._token_calibration))

    @staticmethod
    def count_code_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate tokens inside fenced code blocks across ``messages``.

        Cheap char-based estimate (chars/4) of code-fence content only — used as
        the adaptive budget's task-complexity signal (review §3.1), not for exact
        counting. Returns 0 when there is no fenced code.
        """
        code_chars = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, str) or "```" not in content:
                continue
            for match in re.finditer(r"```[\s\S]*?```", content):
                code_chars += len(match.group(0))
        return code_chars // 4

    def effective_budget_tokens(self) -> int:
        """Return the budget actually enforced this turn, with the growth ceiling.

        Wraps :meth:`budget_tokens` with the per-turn growth cap
        (``max_context_growth_per_turn``). The dynamic budget can be much larger
        than the previous turn's optimized context (e.g. ~6.5K on a 262K window);
        without a growth cap a single turn could jump straight to the cap and force
        a large mid-body rewrite that breaks the backend's prefix-cache reuse (the
        v0.7.18 turn-13 regression). The growth ceiling limits expansion to
        ``prev_size + max_context_growth_per_turn`` so the context grows gradually
        and the cached prefix stays valid. On the first turn (no previous size) or
        when the cap is disabled (0), the full dynamic budget applies.
        """
        budget = self.budget_tokens()
        cap = self._config.agentic.max_context_growth_per_turn
        if cap <= 0 or self._last_optimized_token_count is None:
            return budget
        # With the cache-stable rolling summary active, the BATCH FOLD is the
        # size governor: between folds the context grows by pure tail append
        # (cache-safe at any size), and the fold sheds tokens the turn the
        # context crosses its target. The growth ceiling was designed to stop
        # a single turn from forcing a large MID-BODY rewrite (v0.7.18), but
        # here it does the opposite: it chases the previous turn's size with
        # only ``cap`` tokens of slack, so any turn growing more than that
        # (the fixture's turns vary 230-2300 tokens) trips the pressure fold
        # EVERY turn, appending to the summary and sliding the live zone —
        # the per-turn prefix break the fold is supposed to prevent. Let the
        # full (static) budget apply; the fold target derived from it keeps
        # the size bounded with rare, batched invalidations.
        # The optimizer constructs ``hierarchical_summarizer`` non-None exactly
        # when ``_cache_stable_summary`` is set, so the original two-part check
        # reduces to this single config-derived flag (owned by the governor).
        if self._cache_stable_summary:
            return budget
        ceiling = self._last_optimized_token_count + cap
        return min(budget, ceiling)

    def effective_shrink_cap(self) -> int:
        """Return the per-turn SHRINK ceiling (max tokens the context may drop).

        Symmetric to :meth:`effective_budget_tokens`'s growth ceiling (P0.6).
        Bounds the front-eviction rate so the body never collapses in a single
        over-budget turn — the v0.7.21 turn-13 break was an 8.5K->2K tok drop that
        invalidated the backend's cached KV for the whole body. When the next
        turn's optimized size would fall below ``prev_size - shrink_cap``, the
        trimmer only drops down to that floor and leaves the rest for later turns,
        so the cached head stays valid.

        The cap is DYNAMIC (smart default) when ``max_context_shrink_per_turn=0``:
        it is proportional to the CURRENT lean context size
        (``current_size * shrink_context_fraction``), not the model's full window.
        The target is a lean context, so a 12K-tok context may shrink ~1.8K/turn
        while a 2K-tok context only ~300/turn. The cap is floored by the growth
        ceiling (a session that grows fast must be allowed to shrink at least as
        fast) and by ``shrink_min_tokens`` (an absolute floor so tiny contexts
        still have a bounded, non-trivial shrink rate).
        """
        cfg = self._config.agentic
        if cfg.max_context_shrink_per_turn > 0:
            return cfg.max_context_shrink_per_turn
        # Auto: proportional to the current lean context size, floored by the
        # growth rate and an absolute minimum.
        growth = cfg.max_context_growth_per_turn
        current = self._last_optimized_token_count
        if current is None or current <= 0:
            # No baseline yet: fall back to the growth ceiling so shrink is at
            # least as fast as growth.
            return max(growth, cfg.shrink_min_tokens)
        derived = int(current * cfg.shrink_context_fraction)
        return max(derived, growth, cfg.shrink_min_tokens)

    def effective_shrink_floor(self) -> int | None:
        """Return the minimum optimized size allowed this turn, or None if N/A.

        ``prev_size - shrink_cap``. Returns None on the first turn (no previous
        size) or when the shrink cap is disabled (<= 0 and no window), so the
        full budget applies.
        """
        cap = self.effective_shrink_cap()
        if cap <= 0 or self._last_optimized_token_count is None:
            return None
        return max(0, self._last_optimized_token_count - cap)

    def dynamic_cap(self, fraction: float, floor: int) -> int:
        """Return ``max(fraction * dynamic_budget, floor)`` in tokens.

        Used to derive the various sub-caps (tool-output compression threshold,
        code-chunk size, state-step cap, anchor constraints) from the live
        backend window so they scale with the device instead of being fixed.
        Falls back to ``floor`` when the window is unknown or dynamic budgeting
        is disabled (the floor is the configured static value).
        """
        if not self._config.agentic.dynamic_budget_enabled:
            return floor
        return max(floor, int(self.budget_tokens() * fraction))

    def calibrated_token_count(self, raw_tokens: int) -> int:
        """Return the token count scaled by the learned backend ratio (#6).

        ``raw_tokens`` is the caller's raw estimate (e.g. from the token counter);
        the governor does not own a token counter, so the count is passed in.
        """
        return round(raw_tokens * self._token_calibration)
