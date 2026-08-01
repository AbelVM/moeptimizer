"""Diagnostics / session-state / degradation-tracking methods extracted from AgentContextOptimizer (E1 god-object decomposition)."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore

if TYPE_CHECKING:
    from moeptimizer.optimizer import AgentContextOptimizer

logger = logging.getLogger(__name__)


class DiagnosticsOpsMixin:
    """Diagnostics / session-state / degradation-tracking methods (see AgentContextOptimizer)."""

    def _record_degradation(
        self: AgentContextOptimizer,
        stage: str,
        error: Exception | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        """Record a swallowed pipeline-stage failure for the degradation header (P4b).

        Called from the ``except`` guards of each optimization stage. Kept cheap:
        a single list append with a short, header-safe string. The list is reset
        at the start of every ``optimize_messages`` call, so it only ever reflects
        the current turn. Never raises.

        Two message shapes:
        - ``reason`` given → ``stage:reason`` — a short, structured reason code for
          a *fail-open* path that raised no exception (e.g.
          ``code_slicing:parser_unavailable:lang=diff:size=1234:failed_open``). This
          is the REVIEW_luna P1 ask: a reason code plus input size / language /
          failed-open-vs-changed, without a synthesized exception type.
        - ``error`` given → ``stage:ErrorType:message`` — the swallowed-exception shape.
        """
        try:
            if reason is not None:
                msg = f"{stage}:{reason[:200]}"
            elif error is not None:
                msg = f"{stage}:{type(error).__name__}"
                if str(error):
                    msg = f"{msg}:{str(error)[:200]}"
            else:
                msg = stage
            self._last_degradation.append(msg)
            self._last_degradation_counts[stage] = self._last_degradation_counts.get(stage, 0) + 1
            # Cumulative per-stage count (never resets) so a consistently-failing
            # stage is visible in get_debug_info, not just the per-turn header.
            self._degradation_counts[stage] = self._degradation_counts.get(stage, 0) + 1
        except Exception:  # pragma: no cover - defensive
            pass

    @property
    def degradation_counts(self: AgentContextOptimizer) -> dict[str, int]:
        """Cumulative per-stage swallowed-failure counts since session start
        (review §4.8.5 / D4). A stage with a high count is failing consistently
        and silently degrading quality — the signal that was previously invisible."""
        return dict(self._degradation_counts)

    @property
    def last_degradation_counts(self: AgentContextOptimizer) -> dict[str, int]:
        """Return stage failure counts from the most recent optimization turn."""
        return dict(self._last_degradation_counts)

    def _diag_sys(self: AgentContextOptimizer, tag: str, msgs: list[dict[str, Any]]) -> None:
        import hashlib
        import os
        import sys

        if not os.environ.get("MOEPT_DIAG_STAGE"):
            return
        fps = []
        for m in msgs:
            c = m.get("content") or ""
            h = hashlib.md5(c.encode("utf-8", "ignore")).hexdigest()[:6]
            fps.append(f"{m.get('role')[:4]}:{h}")
        sys.stderr.write(f"[DIAG] {tag}: n={len(msgs)} " + " ".join(fps[:22]) + "\n")
        sys.stderr.flush()
        if os.environ.get("MOEPT_DIAG_DUMP"):
            import json

            with open(f"/tmp/diag_{tag.replace(' ', '_')}.json", "w") as _f:
                json.dump(msgs, _f)

    @property
    def last_degradation(self: AgentContextOptimizer) -> list[str]:
        """Per-turn degradation vector (review §11 / P4b).

        Each entry is ``stage:ErrorType:message`` for a pipeline stage that
        swallowed a failure this turn and fell back to a safe default. Empty when
        the turn ran clean. Surfaced to clients via the X-MOEPT-Optimization-Degraded
        response header and in ``get_debug_info``.
        """
        return list(self._last_degradation)

    @property
    def last_evicted_turns(self: AgentContextOptimizer) -> int:
        """Count of complete user-assistant turns dropped by front-eviction on the
        most recent ``optimize_messages`` call (review §11.4 / C8).

        Surfaced to streaming clients as an SSE comment so the client knows history
        was compacted. Reset to 0 at the start of each ``optimize_messages`` call.
        """
        return self._last_evicted_turns

    def get_session_state(self: AgentContextOptimizer) -> str:
        """Get serialized state for persistence across requests."""
        # Route through the optimizer lock (review §4.8.7 / E3): serializes the
        # store/progress/goal, which optimize_messages mutates; the RLock keeps the
        # snapshot consistent (reentrant, so safe if already held).
        with self._lock:
            progress = self.progress_tracker.get_progress()
            goal = self.store.get_goal()
            return json.dumps({
                "store": self.store.serialize(),
                "progress": progress.to_dict(),
                "goal_subtasks": self.goal_decomposer.decompose(
                    goal.original_prompt if goal else ""
                ),
            })

    def _zone_token_breakdown(self: AgentContextOptimizer) -> dict[str, int]:
        """Token sizes of the frozen prefix / rolling summary / live zone for the
        last optimized context (review §4.11.2 / Forward plan D1 debug breakdown).

        Lets operators see how the context is composed — e.g. whether a growing
        summary or a large live zone is driving prefill. Read-only and defensive
        (never affects the optimization path); boundaries are re-derived fresh from
        ``_last_optimized`` so they cannot go stale relative to it.
        """
        msgs = self._last_optimized
        empty = {"frozen_prefix": 0, "rolling_summary": 0, "live_zone": 0, "total": 0}
        if not msgs:
            return empty
        try:
            frozen_end = self._frozen_prefix_end(msgs)
            stable_end = self._stable_prefix_end(msgs)
            frozen = self.token_counter.count_messages(msgs[:frozen_end])
            stable = self.token_counter.count_messages(msgs[:stable_end])
            total = self.token_counter.count_messages(msgs)
            return {
                "frozen_prefix": frozen,
                "rolling_summary": max(0, stable - frozen),
                "live_zone": max(0, total - stable),
                "total": total,
            }
        except Exception:  # pragma: no cover - defensive (debug-only)
            return empty

    def get_debug_info(self: AgentContextOptimizer) -> dict[str, Any]:
        """Return per-session debug snapshot for the operator dashboard (P4).

        Aggregates the live-zone boundary (stable prefix vs. live zone), the
        real prefix-cache outcome, token savings, and the embedding circuit
        breaker state. All fields are read-only and cheap; this is purely
        observational and never affects the optimization path.
        """
        goal = self.store.get_goal()
        cache_stats = {}
        with suppress(Exception):
            cache_stats = self.cache_registry.get_cache_stats()
        breaker_stats: dict[str, Any] = {}
        with suppress(Exception):
            breaker_stats = self.embedding_service.breaker_stats()
        return {
            "session_id": getattr(self, "_session_id", None),
            "live_zone": {
                "live_zone_start": self._live_zone_start,
                "stable_prefix_len": len(self._last_stable_prefix),
                "live_zone_compression_enabled": self._config.agentic.live_zone_compression_enabled,
                "zone_tokens": self._zone_token_breakdown(),
            },
            "cache": {
                "last_static_prefix_hit": self._last_static_prefix_hit,
                "last_optimized_token_count": self.budget.last_optimized_token_count,
                "last_original_token_count": self._last_original_token_count,
                "last_saved_token_count": self.last_saved_token_count,
                "registry": cache_stats,
            },
            "embedding_breaker": breaker_stats,
            "degradation": self.last_degradation,
            "degradation_counts": self.degradation_counts,
            "tool_savings": {
                name: {"chars_in": io[0], "chars_out": io[1], "saved": io[0] - io[1]}
                for name, io in self._tool_savings.items()
            },
            "evicted_turns": self._last_evicted_turns,
            "goal": goal.original_prompt if goal is not None else None,
            "step_count": len(self.store.steps),
        }

    def load_session_state(self: AgentContextOptimizer, state_json: str) -> None:
        """Load state from a previous session."""
        data = json.loads(state_json)
        self.store = AgentStateStore.deserialize(data.get("store", "{}"))
        self.state_rag = StateBasedRAG(self.store)

        if "progress" in data:
            pdata = data["progress"]
            self.progress_tracker._step_count = pdata.get("total_steps", 0)
            self.progress_tracker._tools_used = set(pdata.get("tools_used", []))
            for st in pdata.get("completed_subtasks", []):
                self.progress_tracker._tracked_subtasks[st] = "completed"
            for st in pdata.get("active_subtasks", []):
                self.progress_tracker._tracked_subtasks[st] = "active"

        if "goal_subtasks" in data:
            self.progress_tracker.set_subtasks(data["goal_subtasks"])

        # Load cache registry for cross-session persistence
        self.cache_registry.load_from_disk()
