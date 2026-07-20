"""AgentContextOptimizer — Full pipeline orchestrator.

Pipeline:
  1. Parse message history into AgentStateStore steps
  2. Run ScratchpadCompactor on archived steps
  3. Run ThinkingPreserver on assistant messages (preserves <thinking>)
  4. Optimize code blocks with Tree-Sitter + NPU ranking
  5. Enforce hard token cap for MoE context budget
  6. Apply static layer block alignment for cache optimization

MOE context integrity:
  - RAG context injected as SEPARATE user message (never into assistant content)
  - Loop warnings injected as SEPARATE user message (never into assistant content)
  - Progress tracking is internal only (not injected into context)
  - This preserves the model's expected chat template:
    Ăssistant
<think>
{reasoning}

{response}
    which the model was trained on. Injecting foreign patterns triggers
    KV-cache refills (super slow with MOE prefill).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np

from moeptimizer.async_io_stage import get_async_io_stage
from moeptimizer.attention_sink import apply_attention_sinks
from moeptimizer.cache import (
    get_block_aligned_cache_key,
    get_block_size,
)
from moeptimizer.cache_aware_chunker import get_cache_aware_chunker
from moeptimizer.cache_registry import get_cache_registry
from moeptimizer.chunk_fingerprint import get_chunk_fingerprint_cache
from moeptimizer.code_block_optimizer import (
    extract_code_blocks,
    optimize_code_in_text,
)
from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    deduplicate_chunks,
    detect_language_and_id,
)
from moeptimizer.compactor import ScratchpadCompactor
from moeptimizer.config import AppConfig
from moeptimizer.context_aligner import get_context_aligner
from moeptimizer.context_canonicalizer import get_context_canonicalizer
from moeptimizer.context_compressor import get_context_compressor
from moeptimizer.context_template_matcher import get_context_template_matcher
from moeptimizer.delta_encoder import get_delta_encoder
from moeptimizer.dependency_orderer import get_dependency_orderer
from moeptimizer.embedding import EmbeddingService
from moeptimizer.goal_decomposer import GoalDecomposer
from moeptimizer.hierarchical_index import get_hierarchical_index
from moeptimizer.hierarchical_summarizer import (
    ROLLING_SUMMARY_MARKER,
    get_hierarchical_summarizer,
)
from moeptimizer.hit_prediction_model import get_hit_prediction_model
from moeptimizer.incremental_updater import get_incremental_updater
from moeptimizer.kv_slot_tracker import get_kv_slot_tracker
from moeptimizer.loop_detector import LoopDetector
from moeptimizer.models import AgentStep, LoopWarning
from moeptimizer.mtp_state import get_mtp_state_manager
from moeptimizer.pattern_injector import get_pattern_injector
from moeptimizer.progress_tracker import ProgressTracker
from moeptimizer.prompt_templates import classify_and_template
from moeptimizer.selective_truncator import get_selective_truncator
from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore
from moeptimizer.static_prefix_kv import get_static_prefix_kv_cache
from moeptimizer.symbol_index import SymbolIndex
from moeptimizer.thinking_preserver import ThinkingPreserver
from moeptimizer.token_aware_truncator import TokenAwareTruncator
from moeptimizer.token_counter import TokenCounter
from moeptimizer.tool_output_compressor import ToolOutputCompressor, compress_tool_messages
from moeptimizer.tool_output_filter import ToolOutputFilter
from moeptimizer.tool_streamer import get_tool_streamer

logger = logging.getLogger(__name__)

_UNSUPPORTED_BACKEND_EXTRA_BODY_KEYS = {
    "speculative_decoding",
    "mtp_heads",
    "head_temperatures",
    "expert_hints",
    "cache_control_hints",
}
_OPENAI_MESSAGE_KEYS = {
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "reasoning_content",
    "refusal",
}


class AgentContextOptimizer:
    """Main orchestrator for agentic context optimization."""

    def __init__(
        self,
        config: AppConfig | None = None,
        capability_probe: Any = None,
    ) -> None:
        self._config = config or AppConfig()
        self._lock = threading.RLock()
        self._capability_probe = capability_probe
        # Token-count calibration (review §1/§9, priority fix #6). Initialized
        # early so _budget_tokens() can be called during __init__ (e.g. to seed
        # the adaptive summary-cap ceiling) before the later detailed setup.
        self._token_calibration: float = 1.0
        self.store = AgentStateStore()
        self.context_aligner = get_context_aligner()

        # v0.5.0 components (must be initialized before ScratchpadCompactor
        # because the compactor may reference self.hierarchical_summarizer).
        v050 = self._config.v050
        self._cache_stable_summary = v050.cache_stable_summary_enabled or v050.hierarchical_summary_enabled
        self.hierarchical_summarizer = (
            get_hierarchical_summarizer(max_full_turns=v050.hierarchical_summary_max_full_turns)
            if self._cache_stable_summary
            else None
        )
        self.static_prefix_kv = get_static_prefix_kv_cache() if v050.static_prefix_kv_enabled else None
        self.token_aware_truncator = (
            TokenAwareTruncator(
                cache_stable_mode=v050.cache_stable_mode,
                frozen_prefix_turns=v050.frozen_prefix_turns,
                context_aligner=self.context_aligner,
                token_calibration=1.0,
                keep_full_steps=self._config.agentic.keep_full_steps,
            )
            if v050.token_aware_truncation_enabled
            else None
        )
        self.chunk_fingerprint = get_chunk_fingerprint_cache(max_entries=v050.chunk_fingerprint_max_entries) if v050.chunk_fingerprint_enabled else None
        self.hit_prediction = get_hit_prediction_model(retrain_threshold=v050.hit_prediction_retrain_threshold) if v050.hit_prediction_enabled else None
        self.delta_encoder = get_delta_encoder() if v050.delta_encoding_enabled else None
        self.async_io = get_async_io_stage(max_thread_workers=v050.async_io_max_thread_workers, max_async_concurrency=v050.async_io_max_concurrency) if v050.async_io_enabled else None

        self.thinking_preserver = ThinkingPreserver()
        self.state_rag = StateBasedRAG(self.store)
        self.loop_detector = LoopDetector(threshold=3)
        self.progress_tracker = ProgressTracker()
        self.token_counter = TokenCounter(
            tokenizer=self._config.server.tokenizer,
            capability_probe=capability_probe,
        )
        self.compactor = ScratchpadCompactor(
            keep_full=self._config.agentic.keep_full_steps,
            cache_stable_mode=self._config.v050.cache_stable_mode,
            frozen_prefix_turns=self._config.v050.frozen_prefix_turns,
            context_aligner=self.context_aligner,
            hierarchical_summarizer=self.hierarchical_summarizer,
            token_counter=self.token_counter,
        )
        if self.hierarchical_summarizer is not None:
            # Attach the token counter so the rolling-summary cap is enforced in
            # tokens (not chars) and can preferentially keep code blocks.
            self.hierarchical_summarizer.set_token_counter(self.token_counter)
            # Seed the adaptive summary-cap ceiling from the dynamic context budget
            # (re-derived each turn in _optimize_messages_locked so it tracks the
            # live backend window).
            self.hierarchical_summarizer.set_rolling_summary_ceiling(
                int(self._budget_tokens() * self._config.agentic.rolling_summary_budget_fraction)
            )
        self.goal_decomposer = GoalDecomposer()
        self.embedding_service = EmbeddingService()
        self.symbol_index = SymbolIndex()
        self.cache_registry = get_cache_registry()
        self.cache_registry.load_from_disk()
        self.context_canonicalizer = get_context_canonicalizer()
        self.context_compressor = get_context_compressor()
        self.context_template_matcher = get_context_template_matcher()
        self.dependency_orderer = get_dependency_orderer()  # no-op: instantiated but never called in the pipeline; import ordering not applied
        self.incremental_updater = get_incremental_updater()
        self.pattern_injector = get_pattern_injector()
        self.selective_truncator = get_selective_truncator()  # limited: only remove_duplicates is called (deduplicate code blocks in newest user message)
        self.cache_aware_chunker = get_cache_aware_chunker(block_size=get_block_size())
        self.hierarchical_index = get_hierarchical_index()
        self.mtp_state_manager = get_mtp_state_manager()  # NON-FUNCTIONAL placeholder: cannot capture real MTP state (review03.md §2.1/§10)
        self.tool_streamer = get_tool_streamer()
        self.tool_output_compressor = ToolOutputCompressor(
            max_chars=self._dynamic_tool_output_max_chars()
        )
        self.tool_output_filter = ToolOutputFilter()
        self._task_type: str = "default"
        self._last_mtp_state_key: str | None = None
        self._last_backend_extra_body: dict[str, Any] = {}
        # Throttled cache_registry disk-write counter (review §10).
        self._register_save_counter: int = 0
        # Real cache-outcome signal used to train hit_prediction (replaces the
        # old constant hit=True label). Set when our own static-prefix KV cache
        # is reused; the app layer may override it with the backend's actual
        # cached_tokens from usage.prompt_tokens_details.
        self._last_static_prefix_hit: bool = False
        # Real backend prefix-cache reuse ratio (review P1.3). Rolling mean of
        # the authoritative `cached_tokens > 0` signal reported by the app layer.
        # When reuse collapses we assume our mutations shifted the prefix and
        # *reduce* mutation (skip volatile append + gentler eviction) until reuse
        # recovers — never the reverse, so this cannot make cache stability worse.
        self._real_cache_hit_ratio: float = 1.0
        self._real_cache_samples: int = 0
        self._prefix_drift: bool = False
        self._last_optimized: list[dict[str, Any]] = []
        self._last_optimized_token_count: int | None = None
        self._last_original_token_count: int | None = None
        # Degradation vector (review §11 / P4b). Each pipeline stage swallows
        # failures with a `logger.warning` and falls back to a safe default. To
        # make quality risk visible (and to unblock enabling P3 safely), we
        # accumulate the stage name + error of every swallowed failure into this
        # list. The app layer surfaces it via the X-MOEPT-Optimization-Degraded
        # response header. Reset at the start of every optimize_messages call so
        # it reflects only the current turn. Cheap: no allocation on the hot path
        # when no stage fails (the list stays empty and the header is omitted).
        self._last_degradation: list[str] = []
        # Token-count calibration (review §1/§9, priority fix #6) is initialized
        # earlier in __init__ (see top) so _budget_tokens() is safe during setup.
        # It is later updated by set_token_calibration() from the backend's real
        # prompt_tokens; clamped to [0.5, 2.0].
        # Set once the calibration ratio has been anchored to the backend's exact
        # tokenizer (native /tokenize) so the seed is not re-fetched every turn.
        self._calibration_seeded: bool = False
        # Live-zone compression state (P3). Tracks the boundary between the
        # stable prefix (already optimized, must stay byte-identical for cache
        # reuse) and the live zone (new messages that can be optimized). When
        # the stable prefix is unchanged across turns, only the live zone is
        # re-processed by expensive stages (tree-sitter, RAG, tool compression).
        self._last_stable_prefix: list[dict[str, Any]] = []
        # Raw (incoming, un-optimized) stable prefix from the previous turn. The
        # live-zone boundary is computed by comparing the *incoming raw* prefix of
        # the new turn against this, NOT against the optimized prefix — the client
        # sends raw messages every turn, so comparing raw-to-optimized would never
        # match and the whole conversation would be re-optimized (and re-mutated)
        # each turn, breaking prefix-cache reuse.
        self._last_raw_prefix: list[dict[str, Any]] = []
        self._live_zone_start: int = 0
        # Incremental-optimization memo (review §4). When
        # ``incremental_optimization_enabled`` is on, the fully-optimized stable
        # prefix from the previous turn is cached here keyed by its content hash,
        # so a turn whose leading prefix is unchanged reuses it verbatim and only
        # re-runs the pipeline on the new live zone. ``_stable_prefix_hash`` is the
        # hash of the *incoming* stable prefix that produced ``_stable_prefix_optimized``;
        # a mismatch (new session, reset, edited history) invalidates the memo.
        self._stable_prefix_optimized: list[dict[str, Any]] | None = None
        self._stable_prefix_hash: str | None = None
        # Guard so a live-zone sub-run (incremental path) does not recurse into
        # incremental mode or clobber the session-level memo.
        self._incremental_subrun: bool = False
        # Content-hash cache for tool-output filtering/compression (P3). Avoids
        # re-running regex filters and boundary compression on identical tool
        # outputs that appear across turns.
        self._tool_output_cache: dict[str, dict[str, Any]] = {}
        self._tool_output_cache_max: int = 1024
        # Count of complete user-assistant turns dropped by front-eviction on the
        # most recent optimize_messages call (review §11.4 / C8). Surfaced to the
        # client as an SSE comment so it knows history was compacted.
        self._last_evicted_turns: int = 0
        # C5 (review §5): monotonic quality-anchor state. The anchor is appended as
        # the trailing volatile user turn, so if its content churns turn-to-turn the
        # backend's prefix cache for that turn is invalidated. We accumulate
        # constraints append-only and only ever drop from the FRONT (oldest), keeping
        # the most-recent tail byte-stable across turns.
        self._anchor_first_request: str = ""
        self._anchor_constraints: list[str] = []
        # Active-file tracking (review P0.5): the most-recent file contents the
        # agent read/edited are kept VERBATIM — never skeletonized or evicted —
        # because they are the task-critical state the model must edit/extend.
        # Maps a normalized file path -> exact content string. Bounded to the
        # last few files so memory stays small.
        self._active_files: dict[str, str] = {}
        self._active_file_order: list[str] = []
        # Thinking-block reconstruction store (review P1.1 / cache guide DO #2).
        # The backend caches the KV for the prompt it received, which included
        # the assistant's <think>/reasoning_content block. If the client strips
        # reasoning_content on the next turn (common — many clients only persist
        # `content`), the proxy would re-send an assistant message WITHOUT the
        # thinking block, so the backend's cached prefix no longer matches and
        # it must re-prefill. We capture the thinking we observed on the
        # response stream and re-inject it into the matching assistant message
        # before sending to the backend, keyed by a hash of the assistant
        # `content` (the stable part the client always echoes). Bounded LRU.
        self._thinking_store: dict[str, str] = {}
        self._thinking_order: list[str] = []
        # Tools-schema pinning (review P1.2 / cache guide DO #5). The backend
        # caches the prompt prefix that includes the serialized `tools` array; if
        # the client re-sends tools in a different order or with a different dict
        # layout turn-to-turn, the prefix shifts and the cache is invalidated. We
        # pin the first-seen schema and re-emit it verbatim every turn.
        self._pinned_tools: list[dict[str, Any]] | None = None

    def _budget_tokens(self) -> int:
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
        char_budget = max(1, cfg.max_optimized_chars // 4)
        static = char_budget if cfg.max_optimized_tokens <= 0 else min(char_budget, cfg.max_optimized_tokens)

        if not cfg.dynamic_budget_enabled:
            return static

        window = self._backend_context_window()
        if window is None or window <= 0:
            return static

        derived = int(window * cfg.budget_window_fraction)
        # Never go below the configured floor, and never below the char budget.
        budget = max(derived, cfg.max_optimized_tokens, char_budget)
        # Scale by the learned backend-tokenizer ratio so the cap is enforced
        # against true backend tokens (clamped to [0.5, 2.0] upstream).
        return max(1, round(budget * self._token_calibration))

    def _effective_budget_tokens(self) -> int:
        """Return the budget actually enforced this turn, with the growth ceiling.

        Wraps :meth:`_budget_tokens` with the per-turn growth cap
        (``max_context_growth_per_turn``). The dynamic budget can be much larger
        than the previous turn's optimized context (e.g. ~6.5K on a 262K window);
        without a growth cap a single turn could jump straight to the cap and force
        a large mid-body rewrite that breaks the backend's prefix-cache reuse (the
        v0.7.18 turn-13 regression). The growth ceiling limits expansion to
        ``prev_size + max_context_growth_per_turn`` so the context grows gradually
        and the cached prefix stays valid. On the first turn (no previous size) or
        when the cap is disabled (0), the full dynamic budget applies.
        """
        budget = self._budget_tokens()
        cap = self._config.agentic.max_context_growth_per_turn
        if cap <= 0 or self._last_optimized_token_count is None:
            return budget
        ceiling = self._last_optimized_token_count + cap
        return min(budget, ceiling)

    def _effective_shrink_cap(self) -> int:
        """Return the per-turn SHRINK ceiling (max tokens the context may drop).

        Symmetric to :meth:`_effective_budget_tokens`'s growth ceiling (P0.6).
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

    def _effective_shrink_floor(self) -> int | None:
        """Return the minimum optimized size allowed this turn, or None if N/A.

        ``prev_size - shrink_cap``. Returns None on the first turn (no previous
        size) or when the shrink cap is disabled (<= 0 and no window), so the
        full budget applies.
        """
        cap = self._effective_shrink_cap()
        if cap <= 0 or self._last_optimized_token_count is None:
            return None
        return max(0, self._last_optimized_token_count - cap)

    def _backend_context_window(self) -> int | None:
        """Return the live backend context window in tokens, or None if unknown.

        Reads the cached capability probe so the budget tracks the active device
        (GPU/NPU) without a network call on the hot path. Returns None when no
        probe is wired or the window has not been detected yet.
        """
        probe = self._capability_probe
        if probe is None:
            return None
        try:
            caps = probe.cached()
        except Exception:
            return None
        if caps is None:
            return None
        return caps.max_context_window

    def _finalize_optimized(
        self,
        optimized: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record the final optimized prompt and its token counts.

        Centralizes the bookkeeping that the main pipeline path does at its end
        (Step 14.7 / 14.11) so that EVERY return path — including the fast path
        and the early-exit branches — keeps ``_last_optimized_token_count`` fresh.
        Without this, lean-context turns that take the fast path never update the
        count, so the next over-budget turn sees ``_last_optimized_token_count is
        None`` and computes a ``None`` shrink floor (P0.6 per-turn shrink cap),
        which lets the context collapse in a single turn (the v0.7.22 turn-11
        cliff).
        """
        self._last_optimized = optimized
        try:
            self._last_optimized_token_count = self.token_counter.count_messages(optimized)
        except Exception:  # pragma: no cover - defensive
            self._last_optimized_token_count = None
        try:
            self._last_original_token_count = self.token_counter.count_messages(messages)
        except Exception:  # pragma: no cover - defensive
            self._last_original_token_count = None
        return optimized

    def _dynamic_cap(self, fraction: float, floor: int) -> int:
        """Return ``max(fraction * dynamic_budget, floor)`` in tokens.

        Used to derive the various sub-caps (tool-output compression threshold,
        code-chunk size, state-step cap, anchor constraints) from the live
        backend window so they scale with the device instead of being fixed.
        Falls back to ``floor`` when the window is unknown or dynamic budgeting
        is disabled (the floor is the configured static value).
        """
        if not self._config.agentic.dynamic_budget_enabled:
            return floor
        return max(floor, int(self._budget_tokens() * fraction))

    def _dynamic_tool_output_max_chars(self) -> int:
        """Boundary-compression threshold (chars) for tool/assistant output.

        Derived from ``tool_output_compression_budget_fraction * dynamic budget``
        (tokens), converted to a char floor via ~4 chars/token, then clamped to
        the configured ``tool_output_compression_max_chars`` floor.
        """
        cfg = self._config.agentic
        derived = self._dynamic_cap(cfg.tool_output_compression_budget_fraction, 0) * 4
        return max(cfg.tool_output_compression_max_chars, derived)

    def _dynamic_user_paste_max_chars(self) -> int:
        """Boundary-compression threshold (chars) for large user code pastes."""
        cfg = self._config.agentic
        derived = self._dynamic_cap(cfg.user_paste_compression_budget_fraction, 0) * 4
        return max(cfg.user_paste_compression_max_chars, derived)

    def _dynamic_chunk_max_chars(self) -> int:
        """Max size (chars) of a single tree-sitter code chunk for RAG retrieval."""
        cfg = self._config.code_chunking
        derived = self._dynamic_cap(cfg.chunk_budget_fraction, 0) * 4
        return max(cfg.chunk_max_chars, derived)

    def _dynamic_max_state_steps(self) -> int:
        """Per-session AgentStateStore step cap derived from the dynamic budget."""
        cfg = self._config.agentic
        return self._dynamic_cap(cfg.state_steps_budget_fraction, cfg.max_state_steps)

    def _dynamic_max_anchor_constraints(self) -> int:
        """Cap on accumulated quality-anchor constraints, scaled by the budget."""
        cfg = self._config.agentic
        if not cfg.dynamic_budget_enabled:
            return 5
        derived = int(self._budget_tokens() * cfg.anchor_max_constraints_budget_factor)
        return max(5, derived)

    def set_token_calibration(self, ratio: float | None) -> None:
        """Learn the backend tokenizer ratio from the previous turn (review §1/§9, #6).

        ``ratio`` is ``backend_prompt_tokens / proxy_estimated_tokens`` for the
        optimized prompt we sent. Storing it lets the proxy enforce its budget
        against the backend's true token count instead of the tiktoken estimate.
        Clamped to [0.5, 2.0] so a single noisy measurement cannot swing the
        budget wildly. ``None``/non-positive values are ignored (keep last ratio).
        """
        if ratio is None or ratio <= 0:
            return
        clamped = max(0.5, min(2.0, float(ratio)))
        self._token_calibration = clamped
        if self.token_aware_truncator is not None:
            self.token_aware_truncator._token_calibration = clamped

    def calibrated_token_count(self, messages: list[dict[str, Any]]) -> int:
        """Return the token count scaled by the learned backend ratio (#6)."""
        raw = self.token_counter.count_messages(messages)
        return round(raw * self._token_calibration)

    def _record_degradation(self, stage: str, error: Exception) -> None:
        """Record a swallowed pipeline-stage failure for the degradation header (P4b).

        Called from the ``except`` guards of each optimization stage. Kept cheap:
        a single list append with a short, header-safe string. The list is reset
        at the start of every ``optimize_messages`` call, so it only ever reflects
        the current turn. Never raises.
        """
        try:
            msg = f"{stage}:{type(error).__name__}"
            if str(error):
                msg = f"{msg}:{str(error)[:200]}"
            self._last_degradation.append(msg)
        except Exception:  # pragma: no cover - defensive
            pass

    def calibrate_remote_overhead(
        self, backend_prompt_tokens: int, messages: list[dict[str, Any]]
    ) -> None:
        """Calibrate the remote-path per-message overhead (review B0.6).

        The remote ``/tokenize`` join format is not ChatML, so the fixed
        ``+5 * len(messages)`` overhead in ``TokenCounter.count_messages`` is
        systematically wrong. Feed the measured delta
        ``backend_prompt_tokens - remote_joined_count`` to the counter so it
        applies a single additive correction instead of guessing. Only takes
        effect when the remote path is actually in use (otherwise the delta is
        harmless but unused).
        """
        try:
            raw = self.token_counter.count_messages_remote_raw(messages)
            if raw > 0 and backend_prompt_tokens > raw:
                self.token_counter.set_token_calibration_delta(
                    backend_prompt_tokens - raw
                )
        except Exception:
            logger.debug("Remote overhead calibration failed", exc_info=True)

    def seed_token_calibration(self, sample_text: str, exact_tokens: int) -> None:
        """Seed the calibration ratio from an EXACT reference count (#6).

        Unlike ``set_token_calibration`` (learned from the backend's per-turn
        ``prompt_tokens`` after a response), this anchors the ratio *before* the
        first turn using the backend's own exact tokenizer (native ``/tokenize``)
        on a representative ``sample_text``. This removes turn-1 budget error even
        when the local tokenizer is only the tiktoken fallback. Best-effort and
        idempotent: ignored for empty samples or non-positive counts.
        """
        if not sample_text or exact_tokens <= 0:
            return
        local = self.token_counter.count(sample_text)
        if local > 0:
            self.set_token_calibration(exact_tokens / local)
            self._calibration_seeded = True

    @staticmethod
    def _is_summary_block(msg: dict[str, Any]) -> bool:
        """Return True if ``msg`` is the append-only rolling-summary block.

        Detected by its internal ``_summary_id`` / ``_rolling_summary`` markers OR
        by its content marker (``ROLLING_SUMMARY_MARKER``). The content check is
        required because ``_strip_internal_flags`` removes the ``_summary_id`` key
        before the prompt is sent to the backend, and the stable-prefix detection
        runs on the stripped list on the *next* turn. Without content detection the
        summary would fall into the live zone and be re-optimized every turn,
        breaking the backend's prefix-cache reuse (the turn-11 cliff: cached 3192
        -> 882). The block is part of the STABLE PREFIX, so it must be recognized
        whether or not the internal marker survived stripping.
        """
        if msg.get("_summary_id") or msg.get("_rolling_summary"):
            return True
        content = msg.get("content")
        return isinstance(content, str) and content.startswith(ROLLING_SUMMARY_MARKER)

    def _stable_prefix_end(self, messages: list[dict[str, Any]]) -> int:
        """Return the index just past the byte-stable prefix (frozen + summary).

        The stable prefix is the frozen prefix (system + first user +
        ``frozen_prefix_turns``) PLUS the append-only rolling-summary block, when
        present. The summary block is recognized by its ``_summary_id`` marker OR
        its content marker (see :meth:`_is_summary_block`), not by byte-equality,
        because it grows by appending each turn — its LEADING bytes stay
        byte-identical, which is exactly what the backend prefix cache reuses.
        Everything at/after this index is the live zone and may be re-optimized
        without breaking cache reuse (REVIEW.md P0.5).
        """
        frozen_end = self.context_aligner.frozen_prefix_end(
            messages, self._config.v050.frozen_prefix_turns
        )
        # The summary block sits immediately after the frozen prefix (P0.5 fix).
        # Include it in the stable prefix so optimizer stages never mutate it.
        if frozen_end < len(messages) and self._is_summary_block(messages[frozen_end]):
            return frozen_end + 1
        return frozen_end

    def _compute_live_zone_start(self, messages: list[dict[str, Any]]) -> int:
        """Return the index where the live zone begins.

        The stable prefix is the leading block of messages that is byte-identical
        to the previous turn's optimized prefix. When it matches, everything after
        it is the live zone and can be re-optimized without breaking cache reuse.
        When it does not match (e.g. context reset, first turn), the entire list
        is treated as live.

        The comparison is against the *raw* (incoming) stable prefix stored from the
        previous turn, because the client sends raw messages every turn and the
        stored optimized prefix would never equal the raw incoming prefix.
        """
        if not self._last_raw_prefix:
            return 0

        # Compare role+content of the current raw prefix against the stored raw one.
        def _norm(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"role": m.get("role"), "content": m.get("content")} for m in msgs]

        current_prefix = _norm(messages[: len(self._last_raw_prefix)])
        if current_prefix == _norm(self._last_raw_prefix):
            return self._live_zone_start

        # Prefix changed (new session, reset, or different history). Treat all
        # as live and reset the stored prefix.
        self._last_raw_prefix = []
        self._last_stable_prefix = []
        self._live_zone_start = 0
        return 0

    def _update_stable_prefix(
        self,
        optimized: list[dict[str, Any]],
        live_zone_start: int = 0,
        raw_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record the stable prefix boundary after a successful optimization.

        ``live_zone_start`` is the incoming stable-prefix boundary computed at the
        top of ``_optimize_messages_locked`` (0 on the first turn / after a reset).
        The incremental memo is only populated when that incoming boundary is > 0,
        i.e. a stable prefix already existed from the previous turn and can be
        reused. On the first turn the memo stays empty so the next turn establishes
        it.
        """
        # P0.5: the stable prefix includes the append-only rolling-summary block
        # (when present) so optimizer stages never mutate it and the backend reuses
        # its KV. Use _stable_prefix_end, not just frozen_end.
        self._live_zone_start = self._stable_prefix_end(optimized)
        self._last_stable_prefix = [
            {k: v for k, v in m.items() if k in ("role", "content")}
            for m in optimized[: self._live_zone_start]
        ]
        # Store the RAW (incoming) stable prefix so the next turn's
        # _compute_live_zone_start can compare like-for-like. The client sends raw
        # messages every turn; comparing the incoming raw prefix against the stored
        # *optimized* prefix would never match, forcing a full re-optimization (and
        # re-mutation) of the frozen prefix + summary every turn and breaking
        # prefix-cache reuse. We key on the *computed* stable boundary
        # (self._live_zone_start), not the incoming live_zone_start, so the raw
        # prefix is established on the very first turn that has a stable prefix
        # (turn 1 here) and the next turn can reuse it.
        if raw_messages is not None and self._live_zone_start > 0:
            self._last_raw_prefix = [
                {k: v for k, v in m.items() if k in ("role", "content")}
                for m in raw_messages[: self._live_zone_start]
            ]
        else:
            self._last_raw_prefix = []
        # Incremental-optimization memo (review §4). Cache the fully-optimized
        # stable prefix keyed by the content hash of the *incoming* stable prefix,
        # so the next turn can reuse it verbatim when the prefix is unchanged.
        # Maintained on every path that establishes a stable prefix (fast path,
        # early exits, and the full path) so the memo is always fresh once a
        # stable prefix exists. Only active when the flag is on, not a sub-run,
        # and an incoming stable prefix actually existed (live_zone_start > 0).
        if (
            self._config.agentic.incremental_optimization_enabled
            and not self._incremental_subrun
            and live_zone_start > 0
        ):
            try:
                # The incremental path (see _optimize_messages_locked) compares the
                # *incoming raw* prefix against this memo's hash. The client sends
                # raw messages every turn, so the memo hash MUST be derived from the
                # RAW stable prefix (self._last_raw_prefix), NOT the optimized one —
                # otherwise raw never equals optimized, the hash never matches, and
                # the incremental path is never taken (the full pipeline re-runs
                # every turn, re-applying the tool-output filter inconsistently and
                # breaking prefix-cache reuse). This is the same raw-vs-optimized
                # mismatch class as _compute_live_zone_start (root cause #3).
                incoming_prefix_hash = self._content_hash(self._last_raw_prefix)
                # Cache the FULL optimized stable prefix (frozen prefix + append-only
                # rolling-summary block), bounded by self._live_zone_start — NOT just
                # frozen_end. The summary block is part of the stable prefix and must
                # be reused verbatim; otherwise the next turn regenerates it and the
                # leading bytes change.
                self._stable_prefix_optimized = [
                    dict(m) for m in optimized[: self._live_zone_start]
                ]
                self._stable_prefix_hash = incoming_prefix_hash
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Incremental memo update failed: %s", e)

    @staticmethod
    def _content_hash(messages: list[dict[str, Any]]) -> str:
        """Return a stable hash for a message list based on role+content."""
        h = hashlib.sha256()
        for m in messages:
            role = str(m.get("role", ""))
            content = str(m.get("content") or "")
            h.update(role.encode("utf-8", errors="replace"))
            h.update(b"\x00")
            h.update(content.encode("utf-8", errors="replace"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    @staticmethod
    def _split_live_zone(
        messages: list[dict[str, Any]], live_zone_start: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split messages into (stable_prefix, live_zone)."""
        start = max(0, live_zone_start)
        return messages[:start], messages[start:]

    @staticmethod
    def _merge_live_zone(
        stable_prefix: list[dict[str, Any]],
        live_zone: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Recombine stable prefix and optimized live zone."""
        return [dict(m) for m in stable_prefix] + [dict(m) for m in live_zone]

    @staticmethod
    def _strip_volatile_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return ``messages`` with any trailing volatile (derived) turn removed.

        The volatile context (quality anchor / RAG / loop warnings) is appended as
        a single trailing user turn tagged ``_volatile_turn`` (review §8/§1). When
        reusing a cached optimized prefix for incremental optimization, the cached
        prefix already carries its volatile turn from the previous turn; stripping
        it lets the live-zone sub-run append exactly one fresh volatile turn to the
        merged list, keeping the result byte-identical to the full path.
        """
        result = [dict(m) for m in messages]
        while result and result[-1].get("_volatile_turn"):
            result.pop()
        return result

    def _register_context(self, messages: list[dict[str, Any]]) -> None:
        """Register the optimized context and persist only periodically.

        `cache_registry.save_to_disk` rewrites the whole registry pickle. Doing
        it every turn is unnecessary disk I/O (review §10). The registry already
        skips the write when nothing changed, but the exact-context key changes
        every turn, so we throttle to one write every
        `cache_registry_save_every` turns. A final forced save happens on
        process exit via the registry's own persistence if needed.
        """
        self.cache_registry.register_context(messages)
        self._register_save_counter += 1
        if self._register_save_counter % self._config.v050.cache_registry_save_every == 0:
            self.cache_registry.save_to_disk()

    def _has_nonstandard_message_fields(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when messages contain internal/proxy-only fields."""
        for msg in messages:
            for key in msg:
                if key.startswith("_") or key not in _OPENAI_MESSAGE_KEYS:
                    return True
        return False

    def _has_large_tool_output(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when a tool output is large enough to need special handling."""
        for msg in messages:
            if msg.get("role") == "tool" and len(str(msg.get("content") or "")) > 1000:
                return True
        return False

    def _maybe_fast_path(
        self,
        messages: list[dict[str, Any]],
        total_tokens: int,
        proactive_threshold_tokens: int,
    ) -> list[dict[str, Any]] | None:
        """Return a lean fast path for contexts that already fit the quality budget."""
        if not self._config.agentic.fast_path_enabled:
            return None
        if total_tokens > proactive_threshold_tokens:
            return None
        if self._has_nonstandard_message_fields(messages):
            return None
        if self._has_large_tool_output(messages):
            return None

        logger.info(
            "[AgentOptimizer] Lean fast path: tokens=%d <= threshold=%d",
            total_tokens,
            proactive_threshold_tokens,
        )
        self._register_context(messages)
        return self._strip_internal_flags(messages)

    def optimize_messages(
        self,
        messages: list[dict[str, Any]],
        original_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full optimization pipeline with per-session locking."""
        with self._lock:
            return self._optimize_messages_locked(messages, original_prompt)

    def _optimize_messages_locked(
        self,
        messages: list[dict[str, Any]],
        original_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Full optimization pipeline for a message list in an agentic loop.

        Returns optimized messages ready to send to the MoE model.
        """
        start_time = time.time()

        # Reset the per-turn degradation vector (review §11 / P4b) so it only
        # reflects failures from THIS turn's pipeline run.
        self._last_degradation = []
        # Reset the per-turn eviction counter (review §11.4 / C8).
        self._last_evicted_turns = 0

        # P1.1 (cache guide DO #2): re-inject any thinking block we
        # previously observed for an assistant message whose client stripped it.
        # This makes the message we send byte-match the backend's cached prefix
        # and avoids a forced re-prefill. Must run before the pipeline so the
        # restored thinking is part of the optimized prompt.
        with suppress(Exception):
            messages = self._restore_thinking(messages)

        # Live-zone compression (P3): compute the boundary between the stable
        # prefix (byte-identical to the previous turn) and the live zone (new
        # messages that can be re-optimized). When the stable prefix is unchanged,
        # expensive stages below only touch the live zone, keeping the prefix
        # byte-stable for backend prefix-cache reuse.
        live_zone_start = 0
        if self._config.agentic.live_zone_compression_enabled:
            live_zone_start = self._compute_live_zone_start(messages)

        # Incremental optimization (review §4). When enabled and this turn's
        # leading stable prefix is byte-identical to the previous turn, reuse the
        # previously computed optimized prefix verbatim and only re-run the
        # pipeline on the new live zone. This is a pure latency optimization: the
        # merged result is byte-identical to the full-path output (guaranteed by
        # the test in tests/test_optimizer.py). Skipped during a live-zone sub-run
        # and when the memo is stale (prefix changed, first turn, reset).
        if (
            self._config.agentic.incremental_optimization_enabled
            and not self._incremental_subrun
            and live_zone_start > 0
            and self._stable_prefix_optimized is not None
            and self._stable_prefix_hash is not None
        ):
            incoming_prefix_hash = self._content_hash(messages[:live_zone_start])
            if incoming_prefix_hash == self._stable_prefix_hash:
                live_zone = messages[live_zone_start:]
                # The cached optimized prefix already carries its trailing volatile
                # turn from the previous turn; strip it so the live-zone sub-run
                # appends exactly one fresh volatile turn to the merged list.
                stable_base = self._strip_volatile_turn(self._stable_prefix_optimized)
                self._incremental_subrun = True
                try:
                    optimized_live_zone = self._optimize_messages_locked(
                        live_zone, original_prompt
                    )
                finally:
                    self._incremental_subrun = False
                merged = [dict(m) for m in stable_base] + [
                    dict(m) for m in optimized_live_zone
                ]
                # Update the memo for the next turn: the stable prefix is unchanged,
                # so its optimized form is unchanged; only the live zone grew.
                self._stable_prefix_optimized = stable_base
                self._stable_prefix_hash = incoming_prefix_hash
                return self._finalize_optimized(merged, messages)

        # Step 1: Populate the state store from messages
        self._ingest_messages(messages)
        # Keep the per-session step cap in sync with the live (lean) budget so a
        # long session on a big window retains more goal context without ever
        # growing the optimized context toward the window size.
        self.store.set_max_steps(self._dynamic_max_state_steps())

        # Step 2: Set the root goal if not already set
        if not self.store.get_goal() and original_prompt:
            self.store.set_goal(original_prompt)
        elif not self.store.get_goal() and messages:
            for msg in messages:
                if msg.get("role") == "user":
                    goal_text = (msg.get("content") or "")[:500]
                    self.store.set_goal(goal_text)
                    subtasks = self.goal_decomposer.decompose(goal_text)
                    self.progress_tracker.set_subtasks(subtasks)
                    break

        # REVIEW §6: pin the original request's anchor facts into the rolling
        # summary so they survive front-eviction of Turn 1 (fact_recall would
        # otherwise collapse to 0 by turn 30). Seed once per session from the
        # first user request; the summarizer ignores later calls to keep the
        # summary head byte-stable for prefix-cache reuse.
        if self.hierarchical_summarizer is not None and self._cache_stable_summary:
            first_user = next(
                (m.get("content") or "" for m in messages if m.get("role") == "user"),
                "",
            )
            if first_user:
                self.hierarchical_summarizer.seed_original_request(first_user)

        # Step 3: Run loop detection on each step
        loop_warnings: list[LoopWarning] = []
        for step in self.store.steps:
            warning = self.loop_detector.analyze_step(step)
            if warning:
                loop_warnings.append(warning)

        # Step 4: Update progress tracking
        for step in self.store.steps:
            self.progress_tracker.record_step(step)
        progress = self.progress_tracker.get_progress()

        # Step 5: Apply thinking preservation (pass-through)
        optimized = self.thinking_preserver.process_messages(list(messages))

        # Calculate token count ONCE (P2.1). count_messages is memoized by a
        # content fingerprint, but re-counting the same unchanged list at every
        # gate still costs a full fingerprint pass. We keep a single local
        # ``current_tokens`` and only recompute it right after a stage mutates
        # ``optimized``. Every gate below reads this variable instead of calling
        # count_messages again.
        max_tokens = self._effective_budget_tokens()
        proactive_threshold_tokens = int(max_tokens * self._config.agentic.proactive_trim_ratio)
        compaction_threshold_tokens = int(max_tokens * self._config.agentic.compaction_trigger_ratio)
        current_tokens = self.token_counter.count_messages(optimized)
        total_tokens = current_tokens

        # Single lean-context signal (review §2 / C14). The fast path is the
        # primary early-return gate, but it can be bypassed for lean contexts that
        # carry a large tool output or nonstandard fields. To keep RAG/summary from
        # firing on those small contexts, every heavy stage below consults this one
        # boolean instead of re-deriving its own token-threshold check.
        is_lean_context = total_tokens <= proactive_threshold_tokens

        fast_path = self._maybe_fast_path(optimized, total_tokens, proactive_threshold_tokens)
        if fast_path is not None:
            # Cache-stable boundary transforms must still run on the fast path so the
            # form sent to (and cached by) the backend is the filtered/compressed one.
            fast_path = self._apply_boundary_transforms(fast_path, live_zone_start)
            self._update_stable_prefix(fast_path, live_zone_start, raw_messages=messages)
            return self._finalize_optimized(fast_path, messages)

        # Step 5.0: Check static prefix KV-cache for early exit.
        # Only skip the rest of the pipeline when the context is already lean.
        # If the context is over budget, cache hits must not bypass compaction.
        if self.static_prefix_kv is not None:
            kv_data = self.static_prefix_kv.get(optimized)
            if kv_data is not None:
                self._last_static_prefix_hit = True
                if current_tokens <= proactive_threshold_tokens:
                    logger.info("[AgentOptimizer] Static prefix KV-cache hit, skipping optimization")
                    optimized = self._strip_internal_flags(optimized)
                    optimized = self._apply_boundary_transforms(optimized, live_zone_start)
                    self._register_context(optimized)
                    self._update_stable_prefix(optimized, live_zone_start, raw_messages=messages)
                    return self._finalize_optimized(optimized, messages)
                logger.info(
                    "[AgentOptimizer] Static prefix KV-cache hit, but context is over budget "
                    "(tokens=%d, threshold=%d); continuing compaction",
                    total_tokens,
                    proactive_threshold_tokens,
                )

        # Step 5.1: Check cache hit rate - skip heavy optimization if high and within budget.
        cache_hit_rate = self.cache_registry.predict_hit_rate(optimized)
        if cache_hit_rate > 0.9:
            if total_tokens <= proactive_threshold_tokens:
                logger.info(
                    "[AgentOptimizer] High cache hit rate (%.2f), skipping heavy optimization",
                    cache_hit_rate,
                )
                optimized = self._strip_internal_flags(optimized)
                optimized = self._apply_boundary_transforms(optimized, live_zone_start)
                self._register_context(optimized)
                self._update_stable_prefix(optimized, live_zone_start, raw_messages=messages)
                return self._finalize_optimized(optimized, messages)
            logger.info(
                "[AgentOptimizer] High cache hit rate (%.2f), but context is over budget "
                "(tokens=%d, threshold=%d); continuing compaction",
                cache_hit_rate,
                total_tokens,
                proactive_threshold_tokens,
            )

        # Step 5.1.5: Check hit prediction model for early exit
        # Only allow early exit if context is within budget (not when we need trimming)
        if (self.hit_prediction is not None
            and total_tokens <= proactive_threshold_tokens
            and self.hit_prediction.should_early_exit(optimized)):
            logger.info(
                "[AgentOptimizer] Hit prediction model suggests early exit "
                "(tokens=%d, budget=%d)",
                total_tokens, max_tokens,
            )
            optimized = self._strip_internal_flags(optimized)
            optimized = self._apply_boundary_transforms(optimized, live_zone_start)
            self._register_context(optimized)
            self._update_stable_prefix(optimized, live_zone_start, raw_messages=messages)
            return self._finalize_optimized(optimized, messages)

        # Static layer end is recomputed after compaction/RAG because those stages
        # can change the message list.
        total_chars = sum(len(m.get("content") or "") for m in optimized)
        # NOTE: the quality anchor is no longer injected here. Volatile context
        # (anchor + RAG + loop warnings) is appended as a single trailing user
        # turn at the very end of the pipeline (Step 14.12) so that no historical
        # turn is mutated and the leading prefix stays byte-stable for the
        # backend's prefix cache (review §1/§9, priority fix #1).

        # Step 5.2: Build KV slot map for cache control. This output is only
        # useful when the backend actually receives cache-control hints, so skip
        # the work unless experimental backend hints are enabled (the hints are
        # otherwise stripped before the request is sent).
        if self._config.v050.enable_experimental_backend_hints:
            slot_tracker = get_kv_slot_tracker()
            slot_tracker.build_slot_map(optimized)

        # Step 5.5: Apply context canonicalization only when context pressure
        # justifies normalization. Lean contexts preserve exact user/system text.
        try:
            if current_tokens > proactive_threshold_tokens:
                optimized = self.context_canonicalizer.canonicalize(optimized)
                current_tokens = self.token_counter.count_messages(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Context canonicalization skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Context canonicalization failed: %s", e)
            self._record_degradation("context_canonicalization", e)

        # Step 5.7: Apply code-aware compression when proactive pressure starts.
        # Large code blocks become skeletons to keep the context lean while
        # preserving signatures, imports, comments, and structure. Small code
        # snippets remain exact because they often contain the task semantics.
        try:
            if current_tokens > proactive_threshold_tokens and (
                self._config.agentic.code_skeleton_enabled
                or current_tokens > compaction_threshold_tokens
            ):
                if self.async_io is not None:
                    optimized = self.async_io.run_sync_stage(
                        self.context_compressor.compress,
                        optimized,
                        skip_predicate=self._is_active_file_content,
                        stage_name="compress",
                    )
                else:
                    optimized = self.context_compressor.compress(
                        optimized, skip_predicate=self._is_active_file_content
                    )
                current_tokens = self.token_counter.count_messages(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Context compression skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Context compression failed: %s", e)
            self._record_degradation("context_compression", e)

        # Step 5.9: MoE expert routing is decided internally by the backend; the
        # client-side expert-cache placeholder was removed (review03.md §2.1 / B0.3)
        # because its heuristics provided no real cache locality and only burned CPU.

        # Step 6: Apply prompt template versioning only when context pressure
        # justifies template specialization AND it is enabled. Default OFF for
        # agentic scenarios (review §4.7): template rewrites can change the user's
        # exact wording, which hurts coding tasks and risks cache-guide DONT #4.
        try:
            if (
                self._config.agentic.prompt_template_enabled
                and current_tokens > proactive_threshold_tokens
            ):
                optimized, self._task_type = classify_and_template(optimized)
                current_tokens = self.token_counter.count_messages(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Prompt template specialization skipped: enabled=%s tokens=%d threshold=%d",
                    self._config.agentic.prompt_template_enabled,
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Prompt template versioning failed: %s", e)

        # Step 6.5: Apply context template matching only for over-budget
        # contexts where a task template can preserve quality with fewer tokens.
        # Also gated by prompt_template_enabled (review §4.7).
        if (
            self._config.agentic.prompt_template_enabled
            and current_tokens > proactive_threshold_tokens
        ):
            try:
                if not (optimized and optimized[0].get("role") == "system"):
                    template_name = self.context_template_matcher.match_template(optimized)
                    if template_name:
                        optimized = self.context_template_matcher.apply_template(optimized)
                        current_tokens = self.token_counter.count_messages(optimized)
            except Exception as e:
                logger.warning("Context template matching failed: %s", e)

        # Step 7 (pre-compaction): Cache-stable tiered rolling-summary compaction
        # (review §1/§3/§5, #7). MUST run BEFORE the scratchpad compactor (Step 7
        # below): the compactor drops the entire evictable middle of the
        # conversation in one shot, so if the summary runs after it the evicted
        # turns are already gone and the summary has nothing to fold — that was
        # the turn-11 quality cliff (proxy collapsed to a flat ~1.1K-token stub
        # and never recovered). Running the summary first folds the older dynamic
        # turns into the append-only rolling-summary block (placed right after the
        # frozen prefix, protected from later front-eviction by its _summary_id
        # marker) so the model keeps the task state the compactor would otherwise
        # discard. The compactor's _partition_for_budget already preserves
        # _summary_id messages, so the block survives the later trim.
        if (
            self.hierarchical_summarizer is not None
            and self._cache_stable_summary
            and self._config.v050.cache_stable_mode
            and not is_lean_context
            and current_tokens > proactive_threshold_tokens
        ):
            try:
                # Re-derive the adaptive summary-cap ceiling from the dynamic
                # context budget so it tracks the live backend window (the cap
                # then grows with folded turns up to this ceiling inside the
                # summarizer). Cheap: a single int multiply + attribute set.
                self.hierarchical_summarizer.set_rolling_summary_ceiling(
                    int(max_tokens * self._config.agentic.rolling_summary_budget_fraction)
                )
                frozen_end = self.context_aligner.frozen_prefix_end(
                    optimized, self._config.v050.frozen_prefix_turns
                )
                optimized = self.hierarchical_summarizer.summarize_turns_cache_stable(
                    optimized, frozen_end
                )
                current_tokens = self.token_counter.count_messages(optimized)
            except Exception as e:
                logger.warning("Rolling summary compaction failed: %s", e)

        # Step 7: Apply scratchpad compaction only when the context is already
        # above the proactive threshold. This keeps compaction budget-driven
        # instead of letting it bypass proactive trim on short contexts. The
        # rolling summary (Step 7 pre-compaction) has already folded the evicted
        # turns into the protected _summary_id block, so this only drops what the
        # summary has already captured.
        try:
            if current_tokens > compaction_threshold_tokens:
                # P0.6: bound the per-turn shrink so the compactor never collapses
                # the body below prev_size - shrink_cap in one call (which would
                # invalidate the backend's cached KV for the whole body). The
                # remaining over-budget tokens are shed gradually on later turns.
                shrink_floor = self._effective_shrink_floor()
                optimized = self.compactor.compact_messages(
                    optimized, min_keep_tokens=shrink_floor
                )
                current_tokens = self.token_counter.count_messages(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Scratchpad compaction skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    compaction_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Scratchpad compaction failed: %s", e)
            self._record_degradation("scratchpad_compaction", e)

        # Recalculate after compaction so later stages use the actual context size.
        try:
            total_tokens = current_tokens
            total_chars = sum(len(m.get("content") or "") for m in optimized)
        except Exception as e:
            logger.warning("Post-compaction recount failed: %s", e)

        # Step 7.25: Apply attention sink management only when explicitly
        # enabled and the context is long enough to benefit from it.
        if self._config.agentic.attention_sinks_enabled and total_chars > 4000:
            try:
                optimized = apply_attention_sinks(optimized, self._find_static_layer_end(optimized))
            except Exception as e:
                logger.warning("Attention sink management failed: %s", e)

        # Step 7.5: Apply selective truncation (remove duplicate code blocks)
        try:
            optimized = self.selective_truncator.remove_duplicates(optimized)
        except Exception as e:
            logger.warning("Selective truncation failed: %s", e)

        # Step 7.6: Semantic deduplication is disabled for cache-stable mode.
        # Removing messages from the middle of history changes the serialized
        # prompt prefix even when the remaining text is semantically similar.

        # Step 7.7: Dependency ordering is skipped — dependency_orderer is a
        # no-op (returns messages unchanged). Retained as inert scaffolding.

        # Step 7.8: Apply incremental update for cache preservation
        try:
            optimized = self.incremental_updater.update_context(optimized, "")
            current_tokens = self.token_counter.count_messages(optimized)
        except Exception as e:
            logger.warning("Incremental update failed: %s", e)

        try:
            total_tokens = current_tokens
            total_chars = sum(len(m.get("content") or "") for m in optimized)
        except Exception as e:
            logger.warning("Pre-RAG recount failed: %s", e)

        # Step 8: Compute RAG context and loop warnings. Injection is DEFERRED
        # to the final volatile-context step (Step 14.12, review §1/§9, priority
        # fix #1). These are derived/volatile and must NOT mutate any historical
        # turn or be inserted into the middle of history, or the backend's prefix
        # cache is invalidated and must re-prefill on every turn.
        warning_lines: list[str] = []
        rag_context = ""
        try:
            for w in loop_warnings:
                warning_message = self.loop_detector.get_warning_message(w)
                warning_lines.append(warning_message.replace("[LOOP DETECTED: ", "Loop detected: "))

            rag_tokens = current_tokens
            if (
                self._config.agentic.rag_enabled
                and not is_lean_context
                and rag_tokens > proactive_threshold_tokens
            ):
                last_assistant = None
                for msg in reversed(optimized):
                    if msg.get("role") == "assistant" and not msg.get("_archived"):
                        last_assistant = msg
                        break

                if last_assistant:
                    current_step = AgentStep(
                        role=last_assistant.get("role", "assistant"),
                        content=last_assistant.get("content") or "",
                        tool_name=None,
                        metadata=last_assistant.get("metadata", {}),
                    )
                    rag_context = self.state_rag.get_context_for_step(current_step) or ""
        except Exception as e:
            logger.warning("RAG/loop warning computation failed: %s", e)
            self._record_degradation("rag_loop_warning", e)

        try:
            total_tokens = current_tokens
            total_chars = sum(len(m.get("content") or "") for m in optimized)
        except Exception as e:
            logger.warning("Post-RAG recount failed: %s", e)

        # Step 9: Pre-seed reasoning prefix only when explicitly enabled.
        # Disabled by default because direct-request semantics are the quality target.
        max_tokens = self._budget_tokens()
        if self._config.agentic.reasoning_preseed_enabled and total_tokens < max_tokens * 0.5:
            try:
                optimized = self._preseed_reasoning(optimized)
            except Exception as e:
                logger.warning("Reasoning pre-seeding failed: %s", e)

        # Step 10: Optimize code blocks only when explicitly enabled AND the
        # context is under real pressure. Exact code is preferred for quality,
        # so we skip the tree-sitter parse/dedup on lean contexts (no latency
        # cost, no risk of altering exact code when there is room). The same
        # proactive threshold gates the skeleton compressor (step 5.7).
        #
        # CRITICAL (review P0.2): code-block optimization only ever touches the
        # LIVE ZONE (messages at/after ``live_zone_start``). The stable prefix is
        # frozen byte-for-byte for backend prefix-cache reuse; re-skeletonizing it
        # every turn both destroys quality AND breaks the cache-stability
        # guarantee. When ``live_zone_start == 0`` (first turn / reset) the whole
        # list is the live zone, but on later turns we must never walk into the
        # prefix.
        if self._config.agentic.optimize_code_blocks:
            try:
                if current_tokens > proactive_threshold_tokens:
                    live_start = max(0, live_zone_start)
                    for msg in optimized[live_start:]:
                        # Never skeletonize the append-only rolling-summary block:
                        # it is part of the STABLE PREFIX, so rewriting its code
                        # changes the leading bytes and breaks prefix-cache reuse.
                        if self._is_summary_block(msg):
                            continue
                        content = msg.get("content") or ""
                        if isinstance(content, str) and self._has_code_blocks(content):
                            # P0.5: never skeletonize the active file body — it is
                            # the task-critical state the model must edit/extend.
                            if self._is_active_file_content(content):
                                continue
                            msg["content"] = self._optimize_code_block_content(content)
                    current_tokens = self.token_counter.count_messages(optimized)
                else:
                    logger.debug(
                        "[AgentOptimizer] Code block optimization skipped: tokens=%d <= threshold=%d",
                        current_tokens,
                        proactive_threshold_tokens,
                    )
            except Exception as e:
                logger.warning("Code block optimization failed: %s", e)
                self._record_degradation("code_block_optimization", e)

        try:
            total_tokens = current_tokens
        except Exception as e:
            logger.warning("Post-code recount failed: %s", e)

        # Step 10.5: Apply cache-aware chunking only for contexts that remain
        # over the compaction threshold after cheaper quality-preserving passes.
        if current_tokens > compaction_threshold_tokens:
            try:
                optimized = self.cache_aware_chunker.chunk_context(optimized)
                current_tokens = self.token_counter.count_messages(optimized)
            except Exception as e:
                logger.warning("Cache-aware chunking failed: %s", e)

        try:
            total_tokens = current_tokens
        except Exception as e:
            logger.warning("Post-chunking recount failed: %s", e)

        # Step 11: Proactive context trimming for MoE KV-cache efficiency.
        # Keep this aligned with the configured proactive threshold so early gates
        # and final trimming use the same quality/leanness policy.
        proactive_threshold_tokens = int(max_tokens * self._config.agentic.proactive_trim_ratio)
        # Drift-safe mode (review P1.3): when real prefix-cache reuse has
        # collapsed, skip the aggressive proactive trim — it is the mutation most
        # likely to have shifted the prefix. The hard budget cap (Step 12) still
        # runs, so we never exceed the model window; we only stop *voluntarily*
        # shrinking the context further until reuse recovers.
        if total_tokens > proactive_threshold_tokens and not self._prefix_drift:
            try:
                optimized = self._proactive_trim(
                    optimized, proactive_threshold_tokens, use_tokens=True,
                    shrink_floor=self._effective_shrink_floor(),
                )
            except Exception as e:
                logger.warning("Proactive trimming failed: %s", e)

        # Step 11.5/11.6/11.7: Boundary-compress large tool/assistant outputs and
        # user code pastes (review §3/§5.1/§5/C13). These are content-rewrite stages
        # that replace verbose tool output / pasted files with concise markers. They
        # are applied before the content enters the stable prefix so the compressed
        # form is frozen and the backend's prefix cache stays valid. Idempotent on
        # already-compressed content, so re-running is safe.
        #
        # P0.6 (per-turn shrink cap): these rewrites can collapse the WHOLE body in
        # a single turn when many messages match (the v0.7.22 turn-11 cliff was
        # ``filter_tool_messages`` dropping 19K->2.5K tok at once, invalidating the
        # backend's cached KV for the entire body). We therefore bound them by the
        # same per-turn shrink floor used for eviction: transform messages
        # front-to-back (oldest first) and stop once the next transform would drop
        # the total below ``prev_size - shrink_cap``. Recent messages are left
        # verbatim for later turns to compress gradually, so the cached body head
        # stays valid.
        optimized = self._apply_boundary_transforms(optimized, live_zone_start)
        # Step 11.7 mutations above may have changed the token count; refresh the
        # local so the Step 11.8 gate below reads the post-filter size.
        current_tokens = self.token_counter.count_messages(optimized)

        # Recalculate total_tokens after entropy trim (calibrated to the backend
        # tokenizer so the budget is enforced against true token counts, #6).
        try:
            total_tokens = self.calibrated_token_count(optimized)
        except Exception as e:
            logger.warning("Token recount failed: %s", e)

        # Step 11.7: MTP state bookkeeping is skipped — mtp_state is a
        # NON-FUNCTIONAL placeholder (review03.md §2.1/§10). MTP hidden states
        # cannot be captured/restored by an OpenAI client proxy.

        # Step 11.8: Apply sliding window for long contexts
        # This is the preferred method for context management with MTP state preservation
        # Skipped in drift-safe mode (review P1.3) for the same reason as Step 11:
        # it is a prefix-shifting mutation we avoid while reuse is collapsed.
        if total_tokens > int(max_tokens * 0.8) and not self._prefix_drift:
            try:
                optimized = self._sliding_window_trim(
                    optimized, use_tokens=True, shrink_floor=self._effective_shrink_floor()
                )
            except Exception as e:
                logger.warning("Sliding window trim failed: %s", e)

        # Step 12: Enforce hard token cap (calibrated token counts, #6)
        try:
            total_tokens = self.calibrated_token_count(optimized)
            if total_tokens > max_tokens:
                optimized = self._trim_to_budget(optimized, use_tokens=True)
                total_tokens = self.calibrated_token_count(optimized)
                if total_tokens > max_tokens and self.token_aware_truncator is not None:
                    optimized = self.token_aware_truncator.trim_messages_to_budget(
                        optimized,
                        max_tokens,
                    )
        except Exception as e:
            logger.warning("Token budget enforcement failed: %s", e)
            self._record_degradation("token_budget_enforcement", e)

        # Step 13: Strip internal metadata before sending to model
        try:
            optimized = self._strip_internal_flags(optimized)
        except Exception as e:
            logger.warning("Internal flag stripping failed: %s", e)

        # Step 14: Register context in cache registry for hit prediction.
        # _register_context persists only when cache state changed.
        try:
            self._register_context(optimized)
        except Exception as e:
            logger.warning("Cache registry registration failed: %s", e)

        # Step 14.6: Store static prefix KV-cache for reuse.
        # Store the *stable* prefix content (not a timestamped blob) so that
        # repeated turns with the same system+first-user prefix produce identical
        # bytes. The put() change-detection then skips the per-turn disk write,
        # which is the whole point of the _last_context_changed optimization.
        if self.static_prefix_kv is not None:
            try:
                prefix = self.static_prefix_kv.get_static_prefix(optimized)
                if prefix:
                    self.static_prefix_kv.put(optimized, prefix.encode("utf-8"))
            except Exception as e:
                logger.warning("Static prefix KV-cache storage failed: %s", e)

        # Step 14.7: Record the real cache outcome for hit-prediction learning.
        # The authoritative signal (backend cached_tokens) is reported by the app
        # layer via record_cache_outcome(); if it has not arrived yet we fall back
        # to whether our own static-prefix KV cache was reused this turn. The old
        # constant hit=True label is gone because it trained the model on noise.
        self._last_optimized = optimized
        # Cache the token count of the optimized prompt so the app layer can set
        # the X-Optimized-Prompt-Tokens header without re-tokenizing the whole
        # prompt on the event loop (expensive with the real HF Qwen tokenizer).
        try:
            self._last_optimized_token_count = self.token_counter.count_messages(optimized)
        except Exception:  # pragma: no cover - defensive
            self._last_optimized_token_count = None
        if self.hit_prediction is not None:
            try:
                self.hit_prediction.record_outcome(
                    optimized,
                    hit=self._last_static_prefix_hit,
                )
            except Exception as e:
                logger.warning("Hit prediction recording failed: %s", e)

        # Step 14.8: Store code snapshots for delta encoding
        if self.delta_encoder is not None:
            try:
                # C4 (review §4): only scan the live zone (messages at/after
                # live_zone_start) for code blocks. The stable prefix is
                # byte-identical to the previous turn, so re-scanning it every
                # turn is pure wasted CPU. Scanning the live zone only covers
                # new/mutated turns.
                scan_from = max(0, live_zone_start)
                for msg in optimized[scan_from:]:
                    # Skip the append-only rolling-summary block: its verbatim code
                    # is a folded historical snapshot, not a live file. Storing it as
                    # a delta-encoder snapshot would let a later turn inject a diff
                    # into the summary (mutating its leading bytes and breaking the
                    # backend's prefix-cache reuse — the turn-11 cliff).
                    if self._is_summary_block(msg):
                        continue
                    content = msg.get("content") or ""
                    if isinstance(content, str) and "```" in content:
                        for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
                            lang = match.group(1)
                            code = match.group(2)
                            # Stable per-language id so re-reads of the same file
                            # map to one encoder entry and yield a real prior delta.
                            file_path = f"inline:{lang}"
                            self.delta_encoder.store_snapshot(file_path, code)
                # P2.2 (review §3.4): when a file is re-read after an edit, inject
                # a compact unified diff instead of the full re-read file body. The
                # flag is ON by default, but injection is decided dynamically per
                # re-read: it only fires when the prior snapshot is already present
                # in the optimized context (verified by substring), so the model can
                # apply the diff to a file it already has. On a first read, or when
                # the prior version was evicted/summarized out, the full current code
                # is kept verbatim — edits stay correct and the model never loses
                # context it needs.
                if self._config.agentic.delta_encode_inject:
                    self._inject_code_deltas(optimized, scan_from)
            except Exception as e:
                logger.warning("Delta encoding snapshot storage failed: %s", e)

        # Step 14.9: Save hierarchical summaries to disk. Gated behind
        # persist_state_to_disk (review §8): per-turn pickle writes add latency to
        # the request path for no benefit on a single-process proxy. State is still
        # kept in memory every turn regardless.
        if self._config.v050.persist_state_to_disk and self.hierarchical_summarizer is not None:
            try:
                self.hierarchical_summarizer.save_to_disk()
            except Exception as e:
                logger.warning("Hierarchical summarizer save failed: %s", e)

        # Step 14.10: Save static prefix KV-cache only when it changed.
        if self._config.v050.persist_state_to_disk and self.static_prefix_kv is not None:
            try:
                self.static_prefix_kv.save_to_disk()
            except Exception as e:
                logger.warning("Static prefix KV-cache disk save failed: %s", e)

        # Log metrics
        original_chars = sum(len(m.get("content") or "") for m in messages)
        optimized_chars = sum(len(m.get("content") or "") for m in optimized)
        original_tokens = self.token_counter.count_messages(messages)
        optimized_tokens = self.token_counter.count_messages(optimized)
        duration = time.time() - start_time
        saved_tokens = max(0, original_tokens - optimized_tokens)

        # Cache the original count so the app layer can report token savings
        # (review §11.1 / fix #10) without re-tokenizing the raw messages.
        self._last_original_token_count = original_tokens

        logger.info(
            "[AgentOptimizer] %d -> %d chars (%d -> %d tokens, %d saved, %.3fs, %d -> %d msgs, progress: %.0f%%, loops: %d detected)",
            original_chars,
            optimized_chars,
            original_tokens,
            optimized_tokens,
            saved_tokens,
            duration,
            len(messages),
            len(optimized),
            progress.estimated_completion * 100,
            len(loop_warnings),
        )

        self._last_optimized = optimized
        # Cache the token count of the optimized prompt so the app layer can set
        # prefix cache can reuse it across turns. This is the core KV-cache
        # preservation guarantee (review §1/§3/§7). When cache-stable mode is on,
        # the frozen block is system + first user + the next `frozen_prefix_turns`
        # turns (all taken verbatim from the incoming `messages`, which the client
        # sends identically every turn). The incoming `messages` are the source of
        # truth for the immutable prefix.
        if self._config.agentic.immutable_prefix_enabled:
            try:
                frozen_turns = (
                    self._config.v050.frozen_prefix_turns
                    if self._config.v050.cache_stable_mode
                    else 0
                )
                optimized = self.context_aligner.freeze_static_prefix(
                    optimized, optimized, frozen_prefix_turns=frozen_turns
                )
            except Exception as e:
                logger.warning("Static prefix freeze failed: %s", e)

        # Step 14.12: Append volatile (derived) context as a SINGLE trailing user
        # turn (review §1/§9, priority fix #1). The quality anchor, RAG context,
        # and loop warnings are volatile: they differ every turn. By appending
        # them as one trailing user turn — instead of mutating the active user
        # turn or inserting into the middle of history — every historical turn
        # stays byte-identical across turns. The backend's native prefix cache
        # then reuses its KV for the whole stable leading prefix and avoids an
        # expensive MoE re-prefill. This is the core KV-cache preservation change.
        try:
            anchor = self._build_quality_anchor(optimized)
            # Drift-safe mode (review P1.3): when real prefix-cache reuse has
            # collapsed we stop appending the volatile trailing turn, because
            # even that append shifts the prefix the backend would otherwise
            # reuse. Skipping it reduces mutation and lets reuse recover.
            if not self._prefix_drift:
                optimized = self._append_volatile_context(
                    optimized, anchor, rag_context, warning_lines, proactive_threshold_tokens
                )
        except Exception as e:
            logger.warning("Volatile context append failed: %s", e)
            self._record_degradation("volatile_context_append", e)

        # Record the final optimized prompt (with the trailing volatile turn) so
        # the app layer's cache-outcome and token-calibration signals match what
        # was actually sent to the backend (#6, review §1/§9).
        self._last_optimized = optimized

        # Live-zone compression (P3): update the stable prefix boundary so the
        # next turn can skip re-optimizing unchanged content.
        if self._config.agentic.live_zone_compression_enabled:
            try:
                self._update_stable_prefix(optimized, live_zone_start, raw_messages=messages)
            except Exception as e:
                logger.debug("Live-zone prefix update failed: %s", e)

        return optimized

    def _strip_internal_flags(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove internal metadata keys and section markers that shouldn't reach the model.

        Preserves the message structure and content while stripping:
        - _archived: compactor marker
        - Section markers (<!-- STATIC/CONTEXT/DYNAMIC_LAYER -->)
        - Proxy-only fields such as chunk_index
        - Any other _prefixed keys (future-proof)
        """
        result: list[dict[str, Any]] = []
        internal_prefix = "_"
        # Pre-compiled pattern for performance
        marker_pattern = re.compile(
            r"<!-- (STATIC|CONTEXT|DYNAMIC)_LAYER -->\n?",
        )

        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                content = marker_pattern.sub("", content)

            cleaned = {
                key: value
                for key, value in msg.items()
                if key in _OPENAI_MESSAGE_KEYS and not key.startswith(internal_prefix)
            }
            if "content" in msg or msg.get("role") != "assistant":
                cleaned["content"] = content
            result.append(cleaned)

        return result

    def _append_volatile_context(
        self,
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

    def _build_quality_anchor(self, messages: list[dict[str, Any]]) -> str:
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

    def _placeholder_code_blocks(self, text: str) -> str:
        """Replace large code fences with a compact placeholder for anchors."""
        return re.sub(r"```[\s\S]*?```", "[code block]", text)

    def _compact_anchor_text(self, text: str, max_chars: int) -> str:
        """Compact whitespace and truncate text for quality anchors."""
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3].rstrip()}..."

    def _should_run_hierarchical_summary(
        self,
        messages: list[dict[str, Any]],
        proactive_threshold_tokens: int,
    ) -> bool:
        """Gate hierarchical summarization behind length and budget thresholds."""
        if not messages:
            return False
        max_full_turns = max(self._config.v050.hierarchical_summary_max_full_turns, 1)
        min_messages = max_full_turns + 3
        return len(messages) >= min_messages and self.token_counter.count_messages(messages) > proactive_threshold_tokens

    # Tool names whose result content is a file body the agent is actively
    # editing. Those contents are tracked verbatim and protected from
    # skeletonization/eviction (review P0.5).
    _FILE_TOOL_NAMES = frozenset({"read_file", "edit_file", "write_file", "open_file", "view_file"})

    def _ingest_messages(self, messages: list[dict[str, Any]]) -> None:
        """Convert message list into AgentStateStore steps."""
        for i, msg in enumerate(messages):
            role = msg.get("role", "assistant")
            content = msg.get("content") or ""

            step = AgentStep(
                role=role,
                content=content if isinstance(content, str) else json.dumps(content),
                step_index=i,
                metadata=msg.get("metadata", {}),
            )

            if role == "assistant":
                tool_calls = msg.get("tool_calls") or msg.get("metadata", {}).get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    step.tool_name = (
                        tool_calls[0].get("function", {}).get("name", "") if tool_calls else ""
                    )
                    step.tool_call_id = tool_calls[0].get("id", "") if tool_calls else ""
            elif role == "tool":
                step.tool_call_id = msg.get("tool_call_id", "")
                # Active-file tracking: capture the file path + body from a
                # read/edit/write tool result so the optimizer can keep it
                # verbatim later in the pipeline.
                name = msg.get("name", "")
                if name in self._FILE_TOOL_NAMES and isinstance(content, str) and content.strip():
                    path = self._extract_file_path(msg, tool_calls=None, content=content)
                    if path:
                        self._track_active_file(path, content)

            fingerprint = self.store.step_fingerprint(step)
            if self.store.has_step_fingerprint(fingerprint):
                continue

            self.store.add_step(step)
            self.progress_tracker.record_step(step)

    @staticmethod
    def _extract_file_path(msg: dict[str, Any], tool_calls: Any, content: str) -> str | None:
        """Best-effort extraction of the file path a file-tool acted on."""
        # Prefer the tool_call arguments on the matching assistant message; we
        # only have the tool message here, so fall back to scanning the content
        # for a path-like token and to the tool message's own metadata.
        meta = msg.get("metadata", {}) or {}
        for key in ("path", "file_path", "filename"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Scan the first lines of the content for a path-like token.
        for line in content.splitlines()[:5]:
            m = re.search(r"(?:[/\\][\w.\-/\\]+){2,}\.\w{1,6}", line)
            if m:
                return m.group(0)
        return None

    def _track_active_file(self, path: str, content: str) -> None:
        """Record an active file's verbatim content (bounded LRU)."""
        norm = path.strip()
        if norm in self._active_files:
            self._active_file_order.remove(norm)
        self._active_files[norm] = content
        self._active_file_order.append(norm)
        # Keep only the most recent few files.
        while len(self._active_file_order) > 5:
            old = self._active_file_order.pop(0)
            self._active_files.pop(old, None)

    def _is_active_file_content(self, content: str) -> bool:
        """True if ``content`` is (a prefix of) a tracked active file body."""
        if not self._active_files:
            return False
        for body in self._active_files.values():
            if body and content and (content == body or content in body or body in content):
                return True
        return False

    def _inject_code_deltas(
        self, optimized: list[dict[str, Any]], scan_from: int
    ) -> None:
        """Replace re-read full file bodies with a diff against the prior snapshot.

        P2.2 (review §3.4): when a file is re-read after an edit, the full new
        file body is redundant if the model already has the prior version in
        context. For each code block we ask the delta encoder for the diff vs the
        previously stored version of the same block. The diff is only injected
        when the prior content is already present somewhere in ``optimized``
        (verified by substring), so the model can apply the diff to a file it
        already sees — never on a first read, and never when the prior version is
        absent (which would make the diff unapplicable).
        """
        if self.delta_encoder is None:
            return
        # Serialize the whole context once to test prior-content presence.
        ctx_blob = "\n".join(
            (m.get("content") or "") if isinstance(m.get("content"), str) else ""
            for m in optimized
        )
        for msg in optimized[scan_from:]:
            # Never mutate the append-only rolling-summary block: it is part of the
            # STABLE PREFIX, so rewriting its (verbatim-preserved) code block with a
            # delta diff changes its leading bytes and invalidates the backend's
            # cached KV for the whole body (the turn-11 cliff: cached 3192 -> 882).
            # The summary's code is a folded historical snapshot, not a live file
            # the model is actively editing, so delta injection does not apply.
            if self._is_summary_block(msg):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or "```" not in content:
                continue
            new_parts: list[str] = []
            last = 0
            changed = False
            for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
                lang = match.group(1)
                code = match.group(2)
                # Stable per-language block id so re-reads of the same file map to
                # the same encoder entry and produce a real prior-version delta.
                file_path = f"inline:{lang}"
                delta = self.delta_encoder.get_delta_vs_previous(file_path, code)
                if delta:
                    # The diff is only applicable if the PRIOR version is already
                    # present in the context (so the model can apply the diff to a
                    # file it already has). If the prior version is absent, keep the
                    # full current code so the model can still see it.
                    prev = self.delta_encoder.get_previous_content(file_path)
                    if prev and prev in ctx_blob:
                        new_parts.append(content[last : match.start()])
                        new_parts.append(
                            "```diff\n# file changed since last read; "
                            "apply this diff to the version you already have:\n"
                            f"{delta}\n```"
                        )
                        changed = True
                        last = match.end()
            if changed:
                new_parts.append(content[last:])
                msg["content"] = "".join(new_parts)

    # --- Thinking-block reconstruction (review P1.1 / cache guide DO #2) ---

    @staticmethod
    def _thinking_key(content: str) -> str:
        """Stable key for an assistant message: hash of its ``content``."""
        return hashlib.md5((content or "").encode("utf-8", "replace")).hexdigest()[:16]

    def pin_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
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

    def capture_thinking(self, content: str, reasoning: str | None) -> None:
        """Store the thinking block observed for an assistant ``content``.

        Called by the app layer after a streaming response completes, so the
        proxy remembers the reasoning block the backend cached alongside this
        assistant message. Bounded LRU.
        """
        if not content or not reasoning:
            return
        key = self._thinking_key(content)
        if key in self._thinking_store:
            self._thinking_order.remove(key)
        self._thinking_store[key] = reasoning
        self._thinking_order.append(key)
        while len(self._thinking_order) > 32:
            old = self._thinking_order.pop(0)
            self._thinking_store.pop(old, None)

    def _restore_thinking(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def _has_code_blocks(self, text: str) -> bool:
        """Check if text contains fenced code blocks."""
        return bool(re.search(r"```[\s\S]*?```", text))

    def _optimize_code_block_content(self, text: str) -> str:
        """Optimize code blocks while reusing identical chunk fingerprints."""
        if self.chunk_fingerprint is None:
            return optimize_code_in_text(
                text,
                self._config,
                self.embedding_service,
            )

        cached = self.chunk_fingerprint.get(text)
        if cached is not None:
            cached_text = cached.get("optimized_text")
            if isinstance(cached_text, str):
                return cached_text

        optimized = optimize_code_in_text(
            text,
            self._config,
            self.embedding_service,
        )
        self.chunk_fingerprint.put(text, {"optimized_text": optimized})
        return optimized

    def _optimize_code_in_text(self, text: str) -> str:
        """Optimize code blocks within a text string using Tree-Sitter + NPU.

        Returns the original text if optimization would reduce code block count.
        """
        regex_pattern = r"(```[\s\S]*?```)"
        blocks = re.findall(regex_pattern, text)
        base_text = re.sub(regex_pattern, "", text).strip()

        if not blocks:
            return text

        detected_langs: set[str] = set()
        all_chunks: list[str] = []
        block_langs: list[str] = []  # Track language per block

        for block in blocks:
            clean = block.replace("```", "").strip()
            lines = clean.split("\n")
            first_line = lines[0].strip().lower() if lines else ""
            lang_id = None
            code = clean

            if first_line in LANG_MAP:
                lang_id = LANG_MAP[first_line]
                code = "\n".join(lines[1:])
            else:
                lang_id = detect_language_and_id(clean)

            detected_langs.add(lang_id if lang_id != "generic" else "unknown-text")
            block_langs.append(first_line if first_line in LANG_MAP else (lang_id if lang_id != "generic" else ""))

            chunks = chunk_code_with_treesitter(code, lang_id or "generic", self._dynamic_chunk_max_chars())
            all_chunks.extend(chunks)

        if not all_chunks:
            return text

        all_chunks = deduplicate_chunks(all_chunks)

        # If we have fewer chunks than original blocks, we'd lose code
        # Return original text to preserve all code blocks
        if len(all_chunks) < len(blocks):
            return text

        if len(all_chunks) >= 2 and len(base_text) > 100:
            try:
                ranked = self._sync_embed_and_rank(base_text, all_chunks)
                all_chunks = ranked
            except Exception:
                pass

        # Reassemble text with optimized code blocks
        placeholder = "__CODE_BLOCK_{}__"
        for i, block in enumerate(blocks):
            text = text.replace(block, placeholder.format(i))

        for i, chunk in enumerate(all_chunks):
            placeholder_str = placeholder.format(i) if i < len(blocks) else ""
            if i < len(blocks):
                # Preserve original language from the block
                original_lang = block_langs[i] if i < len(block_langs) else ""
                replacement = f"```{original_lang}\n{chunk}\n```"
                text = text.replace(placeholder_str, replacement)

        return text

    def _sync_embed_and_rank(self, base_text: str, chunks: list[str]) -> list[str]:
        """Synchronous embedding and ranking (optionally offloaded to a thread pool)."""
        if self.async_io is not None:
            return self.async_io.run_sync_stage(
                self._embed_and_rank_impl, base_text, chunks, stage_name="embed_rank"
            )
        return self._embed_and_rank_impl(base_text, chunks)

    def _embed_and_rank_impl(self, base_text: str, chunks: list[str]) -> list[str]:
        """Core embedding + cosine ranking, run on the request or worker thread."""
        query_vec = self.embedding_service._sync_get_embedding(base_text)
        vecs = self.embedding_service.embed_batch_sync(chunks)
        return self._rank_chunks(query_vec, vecs, chunks)

    def _rank_chunks(
        self,
        query_vec: np.ndarray[Any, Any],
        chunk_vecs: list[np.ndarray[Any, Any]],
        chunks: list[str],
    ) -> list[str]:
        """Rank chunks by cosine similarity, return top-K."""
        if not chunk_vecs:
            return chunks

        matrix = np.vstack(chunk_vecs)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return chunks[: self._config.code_chunking.top_k_chunks]

        norms = np.linalg.norm(matrix, axis=1)
        dots = np.dot(matrix, query_vec)
        scores = np.where(norms != 0, dots / (norm_q * norms), -1.0)

        valid = scores >= self._config.code_chunking.min_chunk_score
        if np.any(valid):
            indices = np.where(valid)[0]
            local_top = np.argsort(scores[indices])[::-1][
                : self._config.code_chunking.top_k_chunks
            ]
            return [chunks[i] for i in indices[local_top]]
        return [
            chunks[i]
            for i in np.argsort(scores)[::-1][: self._config.code_chunking.top_k_chunks]
        ]

    def _trim_to_budget(
        self,
        messages: list[dict[str, Any]],
        use_tokens: bool = False,
    ) -> list[dict[str, Any]]:
        """Trim messages to stay within the budget.

        Uses front-loading eviction: drops complete user-assistant pairs from
        the front of the evictable body. No content modification or truncation.

        Three immutable zones:
          1. System Anchor: system + first user (never modified)
          2. Evictable Body: historical turns (dropped from front)
          3. Protected Tail: last N turns (never modified)

        This preserves token offsets and sequence patterns for MTP heads.

        P0.6 (per-turn shrink cap): the evictable body is never dropped below
        ``_effective_shrink_floor()`` (``prev_size - shrink_cap``) in a single
        turn, even if that leaves the context slightly over the hard budget. The
        remaining over-budget tokens are left for later turns to shed gradually,
        so the backend's cached KV for the body head stays valid (the v0.7.21
        turn-13 break was an 8.5K->2K tok one-shot collapse that invalidated the
        whole cached body).

        Args:
            messages: The message list to trim
            use_tokens: If True, use token-based budget; if False, use character-based
        """
        if use_tokens:
            max_tokens = self._budget_tokens()
        else:
            max_chars = self._config.agentic.max_optimized_chars

        # Partition into zones
        system_anchor, evictable_body, protected_tail = self._partition_for_budget(messages)

        # Reserve space for non-evictable zones; remaining budget is what's available
        if use_tokens:
            reserved = (self.calibrated_token_count(system_anchor)
                        + self.calibrated_token_count(protected_tail))
            evictable_budget = max(0, max_tokens - reserved)
        else:
            reserved = (sum(len(m.get("content") or "") for m in system_anchor)
                        + sum(len(m.get("content") or "") for m in protected_tail))
            evictable_budget = max(0, max_chars - reserved)

        # P0.6: bound the per-turn shrink. The evictable body may not drop below
        # the shrink floor (prev_size - shrink_cap). Compute the floor in the same
        # unit as evictable_budget (tokens or chars).
        shrink_floor: int | None = None
        if use_tokens:
            shrink_floor = self._effective_shrink_floor()
            if shrink_floor is not None:
                # Floor applies to the WHOLE optimized context; convert to an
                # evictable-body floor by subtracting the non-evictable reserved
                # tokens (never below 0).
                shrink_floor = max(0, shrink_floor - reserved)
        else:
            # Char mode: derive a floor from the token shrink cap if available.
            cap = self._effective_shrink_cap()
            if cap > 0 and self._last_optimized_token_count is not None:
                # Approximate char floor: prev_chars - cap*4 (4 chars/token).
                prev_chars = (self._last_optimized_token_count - cap) * 4
                shrink_floor = max(0, prev_chars - reserved)

        # Evict from front of evictable body until under remaining budget AND
        # not below the per-turn shrink floor.
        evictable_body = self._evict_for_budget(
            evictable_body, evictable_budget, use_tokens, shrink_floor=shrink_floor
        )

        return system_anchor + evictable_body + protected_tail

    def _partition_for_budget(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Partition messages into three immutable zones for budget trimming."""
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
        # turns as part of the immutable anchor so front-eviction never shifts
        # the stable prefix. The frozen block is system + first user + the next
        # `frozen_prefix_turns` turns; eviction below only touches what follows.
        if self._config.v050.cache_stable_mode:
            frozen_end = self.context_aligner.frozen_prefix_end(
                messages, self._config.v050.frozen_prefix_turns
            )
            if frozen_end > i:
                system_anchor.extend(messages[i:frozen_end])
                i = frozen_end
            # Keep a rolling summary block (placed right after the frozen prefix)
            # in the immutable anchor so front-eviction never drops or moves it
            # (review §1/§3/§5, #7). Its _summary_id marker identifies it.
            while i < len(messages) and messages[i].get("_summary_id"):
                system_anchor.append(messages[i])
                i += 1

        # Group remaining messages into turns.
        # Each turn starts with a user message and includes following assistant/tool messages.
        # Leading assistant/tool messages are attached to the first following user turn.
        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []
        orphaned_leading: list[dict[str, Any]] = []

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            if role == "user":
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
                current_turn = [msg]
            elif current_turn:
                current_turn.append(msg)
            else:
                orphaned_leading.append(msg)
            i += 1

        if current_turn:
            turns.append(current_turn)

        if orphaned_leading:
            if turns:
                turns[0] = orphaned_leading + turns[0]
            else:
                turns.append(orphaned_leading)

        # Separate complete and pending turns; evict from complete only.
        complete_turns = [t for t in turns if any(m.get("role") == "assistant" for m in t)]
        pending_turns = [t for t in turns if not any(m.get("role") == "assistant" for m in t)]

        keep = max(self._config.agentic.keep_full_steps, 1)
        if len(complete_turns) > keep:
            evictable = [m for t in complete_turns[:-keep] for m in t]
            protected = [m for t in complete_turns[-keep:] for m in t]
        else:
            evictable = []
            protected = [m for t in complete_turns for m in t]

        # Always preserve pending (unpaired user-only) turns and hierarchical summaries.
        for turn in pending_turns:
            protected.extend(turn)
        protected.extend(summary_messages)

        return system_anchor, evictable, protected

    def _evict_for_budget(
        self,
        evictable_body: list[dict[str, Any]],
        budget: int,
        use_tokens: bool = False,
        shrink_floor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Drop pairs from front of evictable body until under budget.

        Args:
            evictable_body: Messages to potentially evict
            budget: Target budget (tokens or chars depending on use_tokens)
            use_tokens: If True, use token-based budget; if False, use character-based
            shrink_floor: If set, the evictable body may not be reduced below this
                size in a single call (P0.6 per-turn shrink cap). When evicting
                would drop below the floor, stop early and leave the context
                slightly over budget — the remaining over-budget tokens are shed
                gradually on later turns so the backend's cached KV stays valid.
        """
        if not evictable_body:
            return evictable_body

        # Group into pairs: (user, [assistant, tools...])
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

        # Drop from front until under budget.
        if use_tokens:
            total = sum(self.token_counter.count_messages(pair) for pair in pairs)
        else:
            total = sum(len(m.get("content") or "") for p in pairs for m in p)

        # High/low watermark (review03.md §6/§9): only start evicting once the
        # body exceeds the budget (high water), but then trim in one batch down
        # to budget * low_water_ratio (low water). This keeps the oldest kept
        # turn byte-stable across many subsequent turns, so the backend's native
        # prefix cache is reused instead of being invalidated every over-budget
        # turn. When already under budget this is a no-op.
        if total <= budget:
            return [m for pair in pairs for m in pair]

        initial_pairs = len(pairs)
        ratio = self._config.agentic.eviction_low_water_ratio
        ratio = max(0.1, min(1.0, ratio))
        low_water = int(budget * ratio)

        # P0.6: the per-turn shrink floor bounds how far we may drop the body in
        # one call. Evict down to at most the low water, BUT never below the
        # shrink floor (prev_size - shrink_cap). So the target is the LARGER of
        # the two: we shed down to low_water normally, but if low_water would
        # drop the body below the floor we stop at the floor and leave the
        # context slightly over budget, shedding the rest gradually on later
        # turns. This prevents the one-shot collapse that invalidated the
        # backend's cached KV for the whole body (the v0.7.21 turn-13 break).
        target = low_water
        if shrink_floor is not None:
            target = max(low_water, shrink_floor)

        # Carry-forward ledger: when a front pair is evicted, extract any code
        # signatures it contained so the model retains awareness of code that
        # existed in dropped turns (fixes has_code_proxy=0 / code_block_loss).
        # The ledger is appended to the protected tail, never the frozen prefix,
        # so backend prefix-cache reuse is preserved.
        ledger_sigs: list[str] = []
        while total > target:
            if not pairs:
                break
            if use_tokens:
                total -= self.token_counter.count_messages(pairs[0])
            else:
                total -= sum(len(m.get("content") or "") for m in pairs[0])
            ledger_sigs.extend(self._extract_code_signatures(pairs[0]))
            pairs = pairs[1:]

        # Record how many complete turns were dropped (review §11.4 / C8).
        dropped = initial_pairs - len(pairs)
        if dropped > 0:
            self._last_evicted_turns += dropped

        result = [m for pair in pairs for m in pair]
        if ledger_sigs:
            result.extend(self._build_code_ledger(ledger_sigs))
        return result

    def _transform_stable_then_live(
        self,
        messages: list[dict[str, Any]],
        live_zone_start: int,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        shrink_floor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply a content ``transform`` with cache-stable live-zone bounding.

        The stable prefix (messages before ``live_zone_start``) is transformed
        WITHOUT the shrink floor: it arrives raw from the client every turn but
        was sent (and cached by the backend) in its transformed form on previous
        turns, so re-applying the (idempotent) transform reproduces that exact
        byte form and the backend reuses its cached KV. The live zone is
        transformed WITH the floor (gradual, bounded compression) so new content
        is never collapsed more than the per-turn cap in a single turn.

        When ``live_zone_start <= 0`` (first turn / reset) the whole list is the
        live zone and the floor applies throughout.
        """
        if live_zone_start <= 0 or live_zone_start >= len(messages):
            return self._apply_transform_with_floor(
                messages, transform, shrink_floor=shrink_floor
            )
        stable_prefix = messages[:live_zone_start]
        live_zone = messages[live_zone_start:]
        # Stable prefix: no floor — idempotent transform reproduces the cached form.
        transformed_stable = [transform(m) for m in stable_prefix]
        # Live zone: floor-bounded gradual compression.
        transformed_live = self._apply_transform_with_floor(
            live_zone, transform, shrink_floor=shrink_floor
        )
        return [*transformed_stable, *transformed_live]

    def _apply_transform_with_floor(
        self,
        messages: list[dict[str, Any]],
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        shrink_floor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply a per-message content ``transform`` while honoring the shrink floor.

        P0.6 (per-turn shrink cap) for *content-rewrite* stages (tool-output
        filtering/compression, user-paste compression). Unlike front-eviction,
        these stages rewrite message content in place, so they can collapse the
        whole body in a single turn when many messages match (the v0.7.22 turn-11
        cliff was ``filter_tool_messages`` dropping 19K->2.5K tok at once, which
        invalidated the backend's cached KV for the entire body).

        To keep the per-turn shrink rate bounded, we transform messages
        **front-to-back** (oldest first) and stop as soon as transforming the next
        message would bring the total below ``shrink_floor`` (``prev_size -
        shrink_cap``). The remaining (recent) messages are left verbatim for later
        turns to compress gradually, so the cached body head stays valid.

        Args:
            messages: Message list to transform.
            transform: Maps a message to its (possibly rewritten) form. Must return
                the SAME message object when no change is made, so unchanged
                messages are not double-counted.
            shrink_floor: Minimum total size (tokens) allowed this turn, or None to
                skip the bound (full transform).
        """
        if shrink_floor is None:
            return [transform(m) for m in messages]

        total = self.token_counter.count_messages(messages)
        if total <= shrink_floor:
            # Already at/under the floor: do not shrink further this turn.
            return list(messages)

        result: list[dict[str, Any]] = []
        for msg in messages:
            transformed = transform(msg)
            candidate = [*result, transformed]
            candidate_tokens = self.token_counter.count_messages(candidate)
            if candidate_tokens < shrink_floor and result:
                # Transforming this message would breach the floor; keep original.
                result.append(msg)
            else:
                result.append(transformed)
                total = candidate_tokens
        return result

    def _apply_boundary_transforms(
        self,
        messages: list[dict[str, Any]],
        live_zone_start: int,
    ) -> list[dict[str, Any]]:
        """Apply the cache-stable boundary transforms (tool-output filter,
        tool-output compression, user-paste compression) to ``messages``.

        These idempotent, lossless-ish transforms must run on EVERY turn, including
        turns that take an early-exit gate (fast path / static-KV hit / cache-hit /
        hit-prediction). Running them before an early-exit return guarantees the form
        sent to (and cached by) the backend is the filtered/compressed one, so the
        next turn's re-filter is a no-op and the cached KV is reused. Skipping them on
        an early exit left the raw (unfiltered) form cached, then collapsed it on the
        following turn — the v0.7.22 turn-11 cliff class (P0.6 regression).

        Uses the same cache-stability split as the main pipeline: the stable prefix is
        transformed WITHOUT the shrink floor (idempotent, reproduces the cached form),
        the live zone WITH the floor (gradual compression).
        """
        optimized = messages
        shrink_floor = self._effective_shrink_floor()
        if self._config.agentic.tool_output_compression_enabled:
            try:
                optimized = self._transform_stable_then_live(
                    optimized,
                    live_zone_start,
                    lambda m: self._filter_tool_message(m),
                    shrink_floor=shrink_floor,
                )
            except Exception as e:
                logger.warning("Tool output filtering failed: %s", e)
                self._record_degradation("tool_output_filtering", e)

        if self._config.agentic.tool_output_compression_enabled:
            try:
                optimized = self._transform_stable_then_live(
                    optimized,
                    live_zone_start,
                    lambda m: self._compress_tool_output_message(m),
                    shrink_floor=shrink_floor,
                )
            except Exception as e:
                logger.warning("Tool output compression failed: %s", e)
                self._record_degradation("tool_output_compression", e)

        if self._config.agentic.user_paste_compression_enabled:
            try:
                optimized = self._transform_stable_then_live(
                    optimized,
                    live_zone_start,
                    lambda m: self._compress_user_paste_message(m),
                    shrink_floor=shrink_floor,
                )
            except Exception as e:
                logger.warning("User paste compression failed: %s", e)
                self._record_degradation("user_paste_compression", e)
        return optimized

    def _extract_code_signatures(self, pair: list[dict[str, Any]]) -> list[str]:
        """Extract compact code signatures from an evicted turn pair.

        Returns a list of short signature strings (function/class defs) so the
        model keeps awareness of code that lived in a now-evicted turn.
        """
        sigs: list[str] = []
        for msg in pair:
            content = msg.get("content") or ""
            if not isinstance(content, str) or "```" not in content:
                continue
            for _lang, code, _s, _e in extract_code_blocks(content):
                for line in code.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("def ", "class ", "function ", "async def ")):
                        sig = stripped.split(":", 1)[0] if ":" in stripped else stripped
                        if sig and sig not in sigs:
                            sigs.append(sig)
        return sigs

    def _build_code_ledger(self, sigs: list[str]) -> list[dict[str, Any]]:
        """Build a compact code-ledger message from accumulated signatures."""
        # Cap the ledger so it cannot itself blow the budget.
        capped = sigs[: self._config.agentic.code_ledger_max_sigs]
        if not capped:
            return []
        body = "[Evicted-turn code index]\n" + "\n".join(capped)
        return [{"role": "system", "content": body, "_code_ledger": True}]

    @property
    def last_optimized_token_count(self) -> int | None:
        """Token count of the most recent optimized prompt (cached from optimize_messages)."""
        return self._last_optimized_token_count

    @property
    def last_original_token_count(self) -> int | None:
        """Token count of the most recent raw (pre-optimization) prompt."""
        return self._last_original_token_count

    @property
    def last_saved_token_count(self) -> int | None:
        """Tokens saved by the last optimization (original - optimized), if known."""
        if self._last_original_token_count is None or self._last_optimized_token_count is None:
            return None
        return max(0, self._last_original_token_count - self._last_optimized_token_count)

    @property
    def last_degradation(self) -> list[str]:
        """Per-turn degradation vector (review §11 / P4b).

        Each entry is ``stage:ErrorType:message`` for a pipeline stage that
        swallowed a failure this turn and fell back to a safe default. Empty when
        the turn ran clean. Surfaced to clients via the X-MOEPT-Optimization-Degraded
        response header and in ``get_debug_info``.
        """
        return list(self._last_degradation)

    @property
    def last_evicted_turns(self) -> int:
        """Count of complete user-assistant turns dropped by front-eviction on the
        most recent ``optimize_messages`` call (review §11.4 / C8).

        Surfaced to streaming clients as an SSE comment so the client knows history
        was compacted. Reset to 0 at the start of each ``optimize_messages`` call.
        """
        return self._last_evicted_turns

    def get_cache_key(self, messages: list[dict[str, Any]]) -> str:
        """Generate cache key with canonicalization for static layer."""
        return get_block_aligned_cache_key(messages)

    def get_backend_extra_body(
        self,
        messages: list[dict[str, Any]],
        existing_extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return only standard OpenAI-compatible extra_body fields.

        Lemonade exposes a standard OpenAI API. Do not inject proxy-internal
        MTP, KV-cache, or expert-routing fields into the backend request; most
        Lemonade/llama.cpp builds will ignore them, and some may reject them.
        """
        del messages
        extra_body = {
            key: value
            for key, value in (existing_extra_body or {}).items()
            if key not in _UNSUPPORTED_BACKEND_EXTRA_BODY_KEYS
        }
        self._last_backend_extra_body = extra_body
        return extra_body

    def record_cache_outcome(self, cached_tokens: int | None = None) -> None:
        """Record the real backend prefix-cache outcome for hit-prediction learning.

        Called by the app layer after the backend responds. Only trains on the
        authoritative backend signal (``cached_tokens`` from
        ``usage.prompt_tokens_details``). When that is unavailable (e.g. the
        backend does not report it, or streaming without usage) the turn is
        **skipped** rather than labeled with the proxy's weak local
        ``_last_static_prefix_hit`` signal — training on a guessed label biases
        the predictor (review §2). This replaces the previous constant
        ``hit=True`` label that trained the model on noise.
        """
        if cached_tokens is None:
            return
        hit = cached_tokens > 0
        # Rolling real-reuse ratio (review P1.3). Exponential moving average so a
        # single transient miss does not flip the drift flag, but a sustained
        # collapse (prefix shifted) does.
        self._real_cache_samples += 1
        alpha = 0.3
        self._real_cache_hit_ratio = (
            self._real_cache_hit_ratio * (1 - alpha) + (1.0 if hit else 0.0) * alpha
        )
        # Declare drift only after we have a few real samples and reuse has
        # genuinely collapsed. This is a *reduction* of mutation, so it is
        # cache-safe: it never increases eviction or prefix mutation.
        if self._real_cache_samples >= 3 and self._real_cache_hit_ratio < 0.34:
            if not self._prefix_drift:
                logger.info(
                    "[AgentOptimizer] Prefix-cache reuse collapsed (ratio=%.2f); "
                    "entering drift-safe mode (reduce mutation).",
                    self._real_cache_hit_ratio,
                )
            self._prefix_drift = True
        elif self._real_cache_hit_ratio > 0.6:
            self._prefix_drift = False
        if self.hit_prediction is not None:
            try:
                self.hit_prediction.record_outcome(self._last_optimized, hit=hit)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Hit prediction outcome recording failed: %s", e)

    def _find_static_layer_end(self, messages: list[dict[str, Any]]) -> int:
        """Find the end index of the static layer (system + first user)."""
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif (msg.get("role") == "user" and static_end > 0) or (msg.get("role") == "user" and static_end == 0):
                static_end = i + 1
                break
        return static_end

    def _get_static_layer_content(self, messages: list[dict[str, Any]]) -> str:
        """Extract the static layer content for expert cache warming."""
        static_end = self._find_static_layer_end(messages)
        if static_end == 0:
            return ""
        return "\n".join(m.get("content") or "" for m in messages[:static_end])

    def _preseed_reasoning(
        self,
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

    def _proactive_trim(
        self,
        messages: list[dict[str, Any]],
        target: int,
        use_tokens: bool = False,
        shrink_floor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Proactively trim context while preserving complete recent turns.
        it becomes a problem. This method evicts complete user-assistant turns
        from the front of the evictable body instead of dropping the newest
        dynamic message when it cannot fit. That prevents the optimizer from
        collapsing long conversations down to only the static system/first-user
        prefix.

        P0.6 (per-turn shrink cap): when ``shrink_floor`` is set, the evictable
        body is never dropped below it in a single call, even if that leaves the
        context slightly over the proactive target. The remaining over-budget
        tokens are shed gradually on later turns so the backend's cached KV for
        the body head stays valid.

        Args:
            messages: The message list to trim
            target: Target size (chars or tokens depending on use_tokens)
            use_tokens: If True, target is in tokens; if False, in characters
            shrink_floor: If set, the evictable body may not be reduced below this
                size in tokens (P0.6 per-turn shrink cap).
        """
        if use_tokens:
            total_tokens = self.token_counter.count_messages(messages)
            if total_tokens <= target:
                return messages
        else:
            total_chars = sum(len(m.get("content") or "") for m in messages)
            if total_chars <= target:
                return messages

        system_anchor, evictable_body, protected_tail = self._partition_for_budget(messages)

        if use_tokens:
            reserved = (self.token_counter.count_messages(system_anchor)
                        + self.token_counter.count_messages(protected_tail))
            evictable_budget = max(0, target - reserved)
            # P0.6: convert the whole-context floor to an evictable-body floor.
            if shrink_floor is not None:
                shrink_floor = max(0, shrink_floor - reserved)
        else:
            reserved = (sum(len(m.get("content") or "") for m in system_anchor)
                        + sum(len(m.get("content") or "") for m in protected_tail))
            evictable_budget = max(0, target - reserved)

        evictable_body = self._evict_for_budget(
            evictable_body, evictable_budget, use_tokens, shrink_floor=shrink_floor
        )
        return system_anchor + evictable_body + protected_tail

    def _sliding_window_trim(
        self,
        messages: list[dict[str, Any]],
        window_size: int | None = None,
        overlap_size: int = 256,
        use_tokens: bool = False,
        shrink_floor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Drop whole old turns from the top while preserving the static prefix.

        The model-visible chat history must remain a contiguous prefix. This
        method never truncates message content and never inserts a middle summary;
        it only removes complete user/assistant turns from the front of the
        dynamic layer after the immutable system message.

        P0.6 (per-turn shrink cap): when ``shrink_floor`` is set, eviction stops
        once the kept context reaches the floor, even if that leaves the context
        over ``window_size``. The remaining over-budget tokens are shed gradually
        on later turns so the backend's cached KV for the body head stays valid.
        """
        del overlap_size

        if window_size is None:
            window_size = self._budget_tokens() if use_tokens else self._config.agentic.max_optimized_chars

        if not messages:
            return messages

        static_end = self._find_static_layer_end(messages)
        # Cache-stable mode (review §1/§3/§7): never evict from inside the frozen
        # prefix block, so the stable prefix stays byte-identical across turns and
        # the backend reuses its KV cache. Eviction below only touches the dynamic
        # layer after the frozen block.
        if self._config.v050.cache_stable_mode:
            frozen_end = self.context_aligner.frozen_prefix_end(
                messages, self._config.v050.frozen_prefix_turns
            )
            if frozen_end > static_end:
                static_end = frozen_end
        # Keep a rolling summary block (placed right after the frozen prefix) in
        # the static region so sliding-window eviction never drops it
        # (review §1/§3/§5, #7). Its _summary_id marker identifies it.
        while static_end < len(messages) and messages[static_end].get("_summary_id"):
            static_end += 1
        static_messages = messages[:static_end]
        dynamic_messages = messages[static_end:]

        def msg_size(msg: dict[str, Any]) -> int:
            if use_tokens:
                return self.token_counter.count_messages([msg])
            return len(msg.get("content") or "")

        static_size = sum(msg_size(msg) for msg in static_messages)
        if static_size >= window_size:
            return messages

        if not dynamic_messages:
            return messages

        # Keep the newest user turn as the active request. Evict only complete
        # turns from the front of the dynamic layer. A dynamic assistant/tool
        # message before the next user belongs to that next user's turn.
        active_turn: list[dict[str, Any]] = []
        evictable_turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []

        for msg in dynamic_messages:
            role = msg.get("role", "")
            if role == "user":
                if current_turn:
                    if current_turn[0].get("role") == "user":
                        evictable_turns.append(current_turn)
                    else:
                        current_turn.append(dict(msg))
                        evictable_turns.append(current_turn)
                    current_turn = []
                else:
                    current_turn = [dict(msg)]
            else:
                current_turn.append(dict(msg))

        if current_turn:
            if current_turn[0].get("role") == "user":
                active_turn = current_turn
            else:
                evictable_turns.append(current_turn)

        result_turns = list(evictable_turns)
        while result_turns and static_size + sum(
            sum(msg_size(msg) for msg in turn) for turn in [*result_turns, active_turn]
        ) > window_size:
            # P0.6: stop evicting once the kept context would drop below the
            # per-turn shrink floor. Leave the rest for later turns.
            if shrink_floor is not None:
                kept = static_size + sum(
                    sum(msg_size(msg) for msg in turn) for turn in [*result_turns[1:], active_turn]
                )
                if kept <= shrink_floor:
                    break
            result_turns = result_turns[1:]

        result = list(static_messages)
        for turn in result_turns:
            result.extend(turn)
        result.extend(active_turn)
        return result

    def _calculate_message_entropy(self, content: str) -> float:
        """Calculate entropy of a message.

        High entropy = unpredictable content (logs, errors)
        Low entropy = predictable patterns (code, structure)
        """
        if not content:
            return 0.0

        # Count unique symbols vs total tokens
        symbols = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content))
        tokens = content.split()

        if not tokens:
            return 0.0

        # Symbol diversity ratio
        ratio = len(symbols) / len(tokens)
        return min(1.0, ratio)

    def _compress_tool_outputs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Boundary-compress large tool/assistant outputs (review §3/§5.1).

        Cheap, lossless-ish transforms (truncate, collapse repeated lines/frames,
        strip ANSI) applied before the output enters the stable prefix. Idempotent
        on small/already-compressed content, so it is safe to run every turn.

        Uses a content-hash cache to avoid re-compressing identical tool outputs
        that appear across turns (P3 live-zone compression).
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    cache_key = self._content_hash([msg])
                    if cache_key in self._tool_output_cache:
                        result.append(self._tool_output_cache[cache_key])
                        continue
                    compressed = compress_tool_messages(
                        [msg], ToolOutputCompressor(max_chars=self._dynamic_tool_output_max_chars())
                    )
                    compressed_msg = compressed[0] if compressed else msg
                    self._tool_output_cache[cache_key] = compressed_msg
                    if len(self._tool_output_cache) > self._tool_output_cache_max:
                        self._tool_output_cache.pop(next(iter(self._tool_output_cache)))
                    result.append(compressed_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    def _compress_user_pastes(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Boundary-compress large user code pastes (review §5 / C13).

        Same cheap, lossless-ish transforms as ``_compress_tool_outputs`` but scoped
        to ``user`` turns whose content exceeds ``user_paste_compression_max_chars``.
        Applied before the paste enters the stable prefix, so the compressed form is
        frozen into the prefix and the backend's prefix cache stays valid. Idempotent
        on small/already-compressed content, so it is safe to run every turn.

        Uses a content-hash cache to avoid re-compressing identical pastes that appear
        across turns (P3 live-zone compression).
        """
        max_chars = self._dynamic_user_paste_max_chars()
        compressor = ToolOutputCompressor(max_chars=max_chars)
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str) and compressor.should_compress(content):
                    cache_key = self._content_hash([msg])
                    if cache_key in self._tool_output_cache:
                        result.append(self._tool_output_cache[cache_key])
                        continue
                    compressed = compress_tool_messages(
                        [msg], compressor, roles=("user",)
                    )
                    compressed_msg = compressed[0] if compressed else msg
                    self._tool_output_cache[cache_key] = compressed_msg
                    if len(self._tool_output_cache) > self._tool_output_cache_max:
                        self._tool_output_cache.pop(next(iter(self._tool_output_cache)))
                    result.append(compressed_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    def _filter_tool_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Per-message wrapper around :func:`filter_tool_messages` for the floor-bound transform.

        Returns the message unchanged when it is not a filterable tool/assistant
        output or no rule matches, so the floor logic can detect "no change" and
        avoid double-counting.
        """
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("tool", "assistant") or not isinstance(content, str):
            return msg
        # Never rewrite the append-only rolling-summary block: it is part of the
        # STABLE PREFIX, so re-compressing/re-filtering it changes its leading
        # bytes and breaks backend prefix-cache reuse (the turn-11 cliff class).
        if self._is_summary_block(msg):
            return msg
        if not self.tool_output_filter.should_filter(content):
            return msg
        filtered = self.tool_output_filter.filter(content)
        if filtered is content:
            return msg
        new_msg = dict(msg)
        new_msg["content"] = filtered
        return new_msg

    def _compress_tool_output_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Per-message wrapper around tool-output compression for the floor-bound transform."""
        if msg.get("role") != "tool":
            return msg
        # Never rewrite the append-only rolling-summary block (see _filter_tool_message).
        if self._is_summary_block(msg):
            return msg
        content = msg.get("content") or ""
        if not isinstance(content, str):
            return msg
        compressor = ToolOutputCompressor(max_chars=self._dynamic_tool_output_max_chars())
        if not compressor.should_compress(content):
            return msg
        cache_key = self._content_hash([msg])
        if cache_key in self._tool_output_cache:
            return self._tool_output_cache[cache_key]
        compressed = compress_tool_messages([msg], compressor)
        compressed_msg = compressed[0] if compressed else msg
        self._tool_output_cache[cache_key] = compressed_msg
        if len(self._tool_output_cache) > self._tool_output_cache_max:
            self._tool_output_cache.pop(next(iter(self._tool_output_cache)))
        return compressed_msg

    def _compress_user_paste_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Per-message wrapper around user-paste compression for the floor-bound transform."""
        if msg.get("role") != "user":
            return msg
        # Never rewrite the append-only rolling-summary block (see _filter_tool_message).
        if self._is_summary_block(msg):
            return msg
        content = msg.get("content") or ""
        if not isinstance(content, str):
            return msg
        compressor = ToolOutputCompressor(max_chars=self._dynamic_user_paste_max_chars())
        if not compressor.should_compress(content):
            return msg
        cache_key = self._content_hash([msg])
        if cache_key in self._tool_output_cache:
            return self._tool_output_cache[cache_key]
        compressed = compress_tool_messages([msg], compressor, roles=("user",))
        compressed_msg = compressed[0] if compressed else msg
        self._tool_output_cache[cache_key] = compressed_msg
        if len(self._tool_output_cache) > self._tool_output_cache_max:
            self._tool_output_cache.pop(next(iter(self._tool_output_cache)))
        return compressed_msg

    def get_optimal_temperature(
        self,
        messages: list[dict[str, Any]],
    ) -> float:
        """Get optimal temperature based on context characteristics.

        For MoE models, lower temperature = more predictable patterns =
        better MTP predictions and cache hits.

        For precise coding tasks, recommended temperature is ~0.6.
        This optimizer adjusts based on context entropy:
        - Low entropy (code): 0.5-0.6 for precision
        - Medium entropy: 0.3-0.5 for balance
        - High entropy (explanations): 0.1-0.3 for exploration
        """
        # Calculate context entropy
        total_entropy = 0.0
        msg_count = 0

        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str) and len(content) > 100:
                entropy = self._calculate_message_entropy(content)
                total_entropy += entropy
                msg_count += 1

        avg_entropy = total_entropy / max(1, msg_count)

        # High entropy = unpredictable = need higher temperature
        # Low entropy = predictable = can use lower temperature
        # For coding tasks, target ~0.6 for precision
        if avg_entropy > 0.6:
            return 0.3  # High entropy, allow exploration
        elif avg_entropy > 0.3:
            return 0.5  # Medium entropy, balanced for coding
        else:
            return 0.6  # Low entropy, deterministic coding

    def get_session_state(self) -> str:
        """Get serialized state for persistence across requests."""
        progress = self.progress_tracker.get_progress()
        goal = self.store.get_goal()
        return json.dumps({
            "store": self.store.serialize(),
            "progress": progress.to_dict(),
            "goal_subtasks": self.goal_decomposer.decompose(
                goal.original_prompt if goal else ""
            ),
            "mtp_state_key": self._last_mtp_state_key,
        })

    def get_debug_info(self) -> dict[str, Any]:
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
            },
            "cache": {
                "last_static_prefix_hit": self._last_static_prefix_hit,
                "last_optimized_token_count": self._last_optimized_token_count,
                "last_original_token_count": self._last_original_token_count,
                "last_saved_token_count": self.last_saved_token_count,
                "registry": cache_stats,
            },
            "embedding_breaker": breaker_stats,
            "degradation": self.last_degradation,
            "evicted_turns": self._last_evicted_turns,
            "goal": goal.original_prompt if goal is not None else None,
            "step_count": len(self.store.steps),
        }

    def load_session_state(self, state_json: str) -> None:
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

        # Restore MTP state key for potential state restoration
        self._last_mtp_state_key = data.get("mtp_state_key")

        if "goal_subtasks" in data:
            self.progress_tracker.set_subtasks(data["goal_subtasks"])

        # Load cache registry for cross-session persistence
        self.cache_registry.load_from_disk()
