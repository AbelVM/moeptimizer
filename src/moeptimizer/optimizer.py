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
from moeptimizer.expert_cache import get_expert_cache
from moeptimizer.goal_decomposer import GoalDecomposer
from moeptimizer.goal_relevance_scorer import GoalRelevanceScorer
from moeptimizer.hierarchical_index import get_hierarchical_index
from moeptimizer.hierarchical_summarizer import get_hierarchical_summarizer
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
from moeptimizer.tool_output_filter import ToolOutputFilter, filter_tool_messages
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
            )
            if v050.token_aware_truncation_enabled
            else None
        )
        self.chunk_fingerprint = get_chunk_fingerprint_cache(max_entries=v050.chunk_fingerprint_max_entries) if v050.chunk_fingerprint_enabled else None
        self.hit_prediction = get_hit_prediction_model(retrain_threshold=v050.hit_prediction_retrain_threshold) if v050.hit_prediction_enabled else None
        self.delta_encoder = get_delta_encoder() if v050.delta_encoding_enabled else None
        self.async_io = get_async_io_stage(max_thread_workers=v050.async_io_max_thread_workers, max_async_concurrency=v050.async_io_max_concurrency) if v050.async_io_enabled else None

        self.compactor = ScratchpadCompactor(
            keep_full=self._config.agentic.keep_full_steps,
            cache_stable_mode=self._config.v050.cache_stable_mode,
            frozen_prefix_turns=self._config.v050.frozen_prefix_turns,
            context_aligner=self.context_aligner,
            hierarchical_summarizer=self.hierarchical_summarizer,
        )
        self.thinking_preserver = ThinkingPreserver()
        self.state_rag = StateBasedRAG(self.store)
        self.loop_detector = LoopDetector(threshold=3)
        self.progress_tracker = ProgressTracker()
        self.token_counter = TokenCounter(
            tokenizer=self._config.server.tokenizer,
            capability_probe=capability_probe,
        )
        self.goal_decomposer = GoalDecomposer()
        self.goal_relevance_scorer = GoalRelevanceScorer(self._config.agentic)
        self.embedding_service = EmbeddingService()
        self.expert_cache = get_expert_cache()  # NON-FUNCTIONAL placeholder: fabricated expert masks, no real MoE routing (review03.md §2.1)
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
            max_chars=self._config.agentic.tool_output_compression_max_chars
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
        self._last_optimized: list[dict[str, Any]] = []
        self._last_optimized_token_count: int | None = None
        self._last_original_token_count: int | None = None
        # Token-count calibration (review §1/§9, priority fix #6). Scales the
        # proxy's tiktoken estimate toward the backend's real tokenizer so the
        # budget is enforced against true token counts. Learned from the backend's
        # actual `prompt_tokens` on the previous turn; clamped to [0.5, 2.0].
        self._token_calibration: float = 1.0
        # Set once the calibration ratio has been anchored to the backend's exact
        # tokenizer (native /tokenize) so the seed is not re-fetched every turn.
        self._calibration_seeded: bool = False
        # Live-zone compression state (P3). Tracks the boundary between the
        # stable prefix (already optimized, must stay byte-identical for cache
        # reuse) and the live zone (new messages that can be optimized). When
        # the stable prefix is unchanged across turns, only the live zone is
        # re-processed by expensive stages (tree-sitter, RAG, tool compression).
        self._last_stable_prefix: list[dict[str, Any]] = []
        self._live_zone_start: int = 0
        # Content-hash cache for tool-output filtering/compression (P3). Avoids
        # re-running regex filters and boundary compression on identical tool
        # outputs that appear across turns.
        self._tool_output_cache: dict[str, dict[str, Any]] = {}
        self._tool_output_cache_max: int = 1024

    def _budget_tokens(self) -> int:
        """Return the configured token budget without letting defaults override chars."""
        cfg = self._config.agentic
        char_budget = max(1, cfg.max_optimized_chars // 4)

        if cfg.max_optimized_tokens <= 0:
            return char_budget

        return min(char_budget, cfg.max_optimized_tokens)

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

    def _compute_live_zone_start(self, messages: list[dict[str, Any]]) -> int:
        """Return the index where the live zone begins.

        The stable prefix is the leading block of messages that is byte-identical
        to the previous turn's optimized prefix. When it matches, everything after
        it is the live zone and can be re-optimized without breaking cache reuse.
        When it does not match (e.g. context reset, first turn), the entire list
        is treated as live.
        """
        if not self._last_stable_prefix:
            return 0

        # Compare role+content of the current prefix against the stored one.
        def _norm(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"role": m.get("role"), "content": m.get("content")} for m in msgs]

        current_prefix = _norm(messages[: len(self._last_stable_prefix)])
        if current_prefix == _norm(self._last_stable_prefix):
            return self._live_zone_start

        # Prefix changed (new session, reset, or different history). Treat all
        # as live and reset the stored prefix.
        self._last_stable_prefix = []
        self._live_zone_start = 0
        return 0

    def _update_stable_prefix(self, optimized: list[dict[str, Any]]) -> None:
        """Record the stable prefix boundary after a successful optimization."""
        frozen_end = self.context_aligner.frozen_prefix_end(
            optimized, self._config.v050.frozen_prefix_turns
        )
        self._live_zone_start = frozen_end
        self._last_stable_prefix = [
            {k: v for k, v in m.items() if k in ("role", "content")}
            for m in optimized[:frozen_end]
        ]

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

        # Live-zone compression (P3): compute the boundary between the stable
        # prefix (byte-identical to the previous turn) and the live zone (new
        # messages that can be re-optimized). When the stable prefix is unchanged,
        # expensive stages below only touch the live zone, keeping the prefix
        # byte-stable for backend prefix-cache reuse.
        live_zone_start = 0
        if self._config.agentic.live_zone_compression_enabled:
            live_zone_start = self._compute_live_zone_start(messages)

        # Step 1: Populate the state store from messages
        self._ingest_messages(messages)

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

        # Step 2.5: Task-aware goal-relevance pruning (P3, review §10).
        # Evict low-relevance steps from the evictable body (oldest archived
        # steps) before heavy optimization. The recent/protected tail and the
        # frozen prefix are never mutated, so backend prefix-cache reuse holds.
        try:
            threshold = self._config.agentic.goal_relevance_threshold
            if threshold > 0 and len(self.store.steps) > self._config.agentic.keep_full_steps:
                self.store.prune_by_relevance(
                    threshold=threshold,
                    goal=self.store.get_goal(),
                    keep_recent=self._config.agentic.keep_full_steps,
                )
        except Exception as e:
            logger.warning("Goal-relevance pruning failed: %s", e)

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

        # Calculate token count early (needed for cache early-exit decisions).
        max_tokens = self._budget_tokens()
        proactive_threshold_tokens = int(max_tokens * self._config.agentic.proactive_trim_ratio)
        compaction_threshold_tokens = int(max_tokens * self._config.agentic.compaction_trigger_ratio)
        total_tokens = self.token_counter.count_messages(optimized)

        fast_path = self._maybe_fast_path(optimized, total_tokens, proactive_threshold_tokens)
        if fast_path is not None:
            self._update_stable_prefix(fast_path)
            return fast_path

        # Step 5.0: Check static prefix KV-cache for early exit.
        # Only skip the rest of the pipeline when the context is already lean.
        # If the context is over budget, cache hits must not bypass compaction.
        if self.static_prefix_kv is not None:
            kv_data = self.static_prefix_kv.get(optimized)
            if kv_data is not None:
                self._last_static_prefix_hit = True
                if total_tokens <= proactive_threshold_tokens:
                    logger.info("[AgentOptimizer] Static prefix KV-cache hit, skipping optimization")
                    optimized = self._strip_internal_flags(optimized)
                    self._register_context(optimized)
                    self._update_stable_prefix(optimized)
                    return optimized
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
                self._register_context(optimized)
                self._update_stable_prefix(optimized)
                return optimized
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
            self._register_context(optimized)
            self._update_stable_prefix(optimized)
            return optimized

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
            current_tokens = self.token_counter.count_messages(optimized)
            if current_tokens > proactive_threshold_tokens:
                optimized = self.context_canonicalizer.canonicalize(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Context canonicalization skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Context canonicalization failed: %s", e)

        # Step 5.7: Apply code-aware compression when proactive pressure starts.
        # Large code blocks become skeletons to keep the context lean while
        # preserving signatures, imports, comments, and structure. Small code
        # snippets remain exact because they often contain the task semantics.
        try:
            current_tokens = self.token_counter.count_messages(optimized)
            if current_tokens > proactive_threshold_tokens and (
                self._config.agentic.code_skeleton_enabled
                or current_tokens > compaction_threshold_tokens
            ):
                if self.async_io is not None:
                    optimized = self.async_io.run_sync_stage(
                        self.context_compressor.compress, optimized, stage_name="compress"
                    )
                else:
                    optimized = self.context_compressor.compress(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Context compression skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Context compression failed: %s", e)

        # Step 5.9: Expert cache warming is skipped — expert_cache is a
        # NON-FUNCTIONAL placeholder (review03.md §2.1). The backend decides
        # MoE expert routing internally; client-side heuristics provide no
        # real cache locality.

        # Step 5.10: Prefetch dependencies only when the context is over budget.
        # This keeps RAG/cache enrichment quality-focused instead of always-on.
        if total_tokens > proactive_threshold_tokens:
            try:
                self._prefetch_dependencies(optimized)
            except Exception as e:
                logger.warning("Dependency prefetch failed: %s", e)

        # Step 6: Apply prompt template versioning only when context pressure
        # justifies template specialization.
        try:
            current_tokens = self.token_counter.count_messages(optimized)
            if current_tokens > proactive_threshold_tokens:
                optimized, self._task_type = classify_and_template(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Prompt template specialization skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    proactive_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Prompt template versioning failed: %s", e)

        # Step 6.5: Apply context template matching only for over-budget
        # contexts where a task template can preserve quality with fewer tokens.
        if self.token_counter.count_messages(optimized) > proactive_threshold_tokens:
            try:
                if not (optimized and optimized[0].get("role") == "system"):
                    template_name = self.context_template_matcher.match_template(optimized)
                    if template_name:
                        optimized = self.context_template_matcher.apply_template(optimized)
            except Exception as e:
                logger.warning("Context template matching failed: %s", e)

        # Step 7: Apply scratchpad compaction only when the context is already
        # above the proactive threshold. This keeps compaction budget-driven
        # instead of letting it bypass proactive trim on short contexts.
        try:
            current_tokens = self.token_counter.count_messages(optimized)
            if current_tokens > compaction_threshold_tokens:
                optimized = self.compactor.compact_messages(optimized)
            else:
                logger.debug(
                    "[AgentOptimizer] Scratchpad compaction skipped: tokens=%d <= threshold=%d",
                    current_tokens,
                    compaction_threshold_tokens,
                )
        except Exception as e:
            logger.warning("Scratchpad compaction failed: %s", e)

        # Recalculate after compaction so later stages use the actual context size.
        try:
            total_tokens = self.token_counter.count_messages(optimized)
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
        except Exception as e:
            logger.warning("Incremental update failed: %s", e)

        try:
            total_tokens = self.token_counter.count_messages(optimized)
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

            rag_tokens = self.token_counter.count_messages(optimized)
            if self._config.agentic.rag_enabled and rag_tokens > proactive_threshold_tokens:
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

        try:
            total_tokens = self.token_counter.count_messages(optimized)
            total_chars = sum(len(m.get("content") or "") for m in optimized)
        except Exception as e:
            logger.warning("Post-RAG recount failed: %s", e)

        # Step 8.5: Cache-stable tiered rolling-summary compaction (review §1/§3/§5, #7).
        # When enabled, older dynamic turns are folded into a single append-only
        # rolling summary block placed right after the frozen prefix (never in the
        # middle of history). The block retains constraints / the task's "don'ts"
        # so the model does not re-derive them verbosely (the 2.17x verbosity
        # regression). The block is protected from later front-eviction by
        # _partition_for_budget / _sliding_window_trim via its _summary_id marker,
        # so the leading prefix stays byte-stable and the backend reuses its cache.
        if (
            self.hierarchical_summarizer is not None
            and self._cache_stable_summary
            and self._config.v050.cache_stable_mode
            and self.token_counter.count_messages(optimized) > proactive_threshold_tokens
        ):
            try:
                frozen_end = self.context_aligner.frozen_prefix_end(
                    optimized, self._config.v050.frozen_prefix_turns
                )
                optimized = self.hierarchical_summarizer.summarize_turns_cache_stable(
                    optimized, frozen_end
                )
            except Exception as e:
                logger.warning("Rolling summary compaction failed: %s", e)

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
        if self._config.agentic.optimize_code_blocks:
            try:
                current_tokens = self.token_counter.count_messages(optimized)
                if current_tokens > proactive_threshold_tokens:
                    if live_zone_start > 0 and live_zone_start < len(optimized):
                        stable_prefix, live_zone = self._split_live_zone(optimized, live_zone_start)
                        for msg in live_zone:
                            content = msg.get("content") or ""
                            if isinstance(content, str) and self._has_code_blocks(content):
                                msg["content"] = self._optimize_code_block_content(content)
                        optimized = self._merge_live_zone(stable_prefix, live_zone)
                    else:
                        for msg in optimized:
                            content = msg.get("content") or ""
                            if isinstance(content, str) and self._has_code_blocks(content):
                                msg["content"] = self._optimize_code_block_content(content)
                else:
                    logger.debug(
                        "[AgentOptimizer] Code block optimization skipped: tokens=%d <= threshold=%d",
                        current_tokens,
                        proactive_threshold_tokens,
                    )
            except Exception as e:
                logger.warning("Code block optimization failed: %s", e)

        try:
            total_tokens = self.token_counter.count_messages(optimized)
        except Exception as e:
            logger.warning("Post-code recount failed: %s", e)

        # Step 10.5: Apply cache-aware chunking only for contexts that remain
        # over the compaction threshold after cheaper quality-preserving passes.
        total_tokens = self.token_counter.count_messages(optimized)
        if total_tokens > compaction_threshold_tokens:
            try:
                optimized = self.cache_aware_chunker.chunk_context(optimized)
            except Exception as e:
                logger.warning("Cache-aware chunking failed: %s", e)

        try:
            total_tokens = self.token_counter.count_messages(optimized)
        except Exception as e:
            logger.warning("Post-chunking recount failed: %s", e)

        # Step 11: Proactive context trimming for MoE KV-cache efficiency.
        # Keep this aligned with the configured proactive threshold so early gates
        # and final trimming use the same quality/leanness policy.
        proactive_threshold_tokens = int(max_tokens * self._config.agentic.proactive_trim_ratio)
        if total_tokens > proactive_threshold_tokens:
            try:
                optimized = self._proactive_trim(optimized, proactive_threshold_tokens, use_tokens=True)
            except Exception as e:
                logger.warning("Proactive trimming failed: %s", e)

        # Step 11.5: Filter large tool/assistant outputs (rtk/snip pattern, review §3/§5.1).
        # Applied before compression so the filter's concise markers enter the
        # stable prefix instead of the verbose original. Idempotent on already-
        # filtered content.
        if self._config.agentic.tool_output_compression_enabled:
            try:
                if live_zone_start > 0 and live_zone_start < len(optimized):
                    stable_prefix, live_zone = self._split_live_zone(optimized, live_zone_start)
                    live_zone = filter_tool_messages(live_zone, self.tool_output_filter)
                    optimized = self._merge_live_zone(stable_prefix, live_zone)
                else:
                    optimized = filter_tool_messages(optimized, self.tool_output_filter)
            except Exception as e:
                logger.warning("Tool output filtering failed: %s", e)

        # Step 11.6: Boundary-compress large tool/assistant outputs (headroom/
        # snip-style, review §3/§5.1). Applied once when a tool message first
        # appears, so the compressed form is frozen into the stable leading
        # prefix and the backend's prefix cache stays valid. Idempotent on
        # already-compressed (small) outputs, so re-running is safe.
        if self._config.agentic.tool_output_compression_enabled:
            try:
                if live_zone_start > 0 and live_zone_start < len(optimized):
                    stable_prefix, live_zone = self._split_live_zone(optimized, live_zone_start)
                    live_zone = self._compress_tool_outputs(live_zone)
                    optimized = self._merge_live_zone(stable_prefix, live_zone)
                else:
                    optimized = self._compress_tool_outputs(optimized)
            except Exception as e:
                logger.warning("Tool output compression failed: %s", e)

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
        if total_tokens > int(max_tokens * 0.8):
            try:
                optimized = self._sliding_window_trim(optimized, use_tokens=True)
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
                for msg in optimized:
                    content = msg.get("content") or ""
                    if isinstance(content, str) and "```" in content:
                        import re
                        for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
                            lang = match.group(1)
                            code = match.group(2)
                            file_path = f"inline:{lang}:{hashlib.md5(code.encode()).hexdigest()[:8]}"
                            self.delta_encoder.store_snapshot(file_path, code)
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

        # Step 14.11: Freeze the stable prefix verbatim so the backend's automatic
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
            optimized = self._append_volatile_context(
                optimized, anchor, rag_context, warning_lines, proactive_threshold_tokens
            )
        except Exception as e:
            logger.warning("Volatile context append failed: %s", e)

        # Record the final optimized prompt (with the trailing volatile turn) so
        # the app layer's cache-outcome and token-calibration signals match what
        # was actually sent to the backend (#6, review §1/§9).
        self._last_optimized = optimized

        # Live-zone compression (P3): update the stable prefix boundary so the
        # next turn can skip re-optimizing unchanged content.
        if self._config.agentic.live_zone_compression_enabled:
            try:
                self._update_stable_prefix(optimized)
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
        # Avoid duplicating an identical trailing volatile turn from a prior pass.
        if messages[-1].get("role") == "user" and messages[-1].get("content") == content:
            return messages

        result = [dict(msg) for msg in messages]
        result.append({"role": "user", "content": content})
        return result

    def _build_quality_anchor(self, messages: list[dict[str, Any]]) -> str:
        """Build a compact anchor from the original request and short constraints."""
        marker = "\n\n# Conversation Quality Anchor\n"
        user_messages = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content") or "", str):
                content = msg.get("content") or ""
                if marker in content:
                    content = content.split(marker, 1)[0]
                user_messages.append(content)
        if not user_messages:
            return ""

        first_request = self._compact_anchor_text(
            self._placeholder_code_blocks(user_messages[0]),
            max_chars=700,
        )
        constraints: list[str] = []
        seen: set[str] = set()
        for content in user_messages[1:]:
            compact = self._compact_anchor_text(self._placeholder_code_blocks(content), max_chars=160)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            constraints.append(f"- {compact}")

        lines = [f"Original request:\n{first_request}"]
        if constraints:
            lines.append("Accumulated constraints:")
            lines.extend(constraints[-5:])

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

            fingerprint = self.store.step_fingerprint(step)
            if self.store.has_step_fingerprint(fingerprint):
                continue

            self.store.add_step(step)
            self.progress_tracker.record_step(step)

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

            chunks = chunk_code_with_treesitter(code, lang_id or "generic", self._config.code_chunking.chunk_max_chars)
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

        # Evict from front of evictable body until under remaining budget
        evictable_body = self._evict_for_budget(evictable_body, evictable_budget, use_tokens)

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
    ) -> list[dict[str, Any]]:
        """Drop pairs from front of evictable body until under budget.

        Args:
            evictable_body: Messages to potentially evict
            budget: Target budget (tokens or chars depending on use_tokens)
            use_tokens: If True, use token-based budget; if False, use character-based
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

        ratio = self._config.agentic.eviction_low_water_ratio
        ratio = max(0.1, min(1.0, ratio))
        low_water = int(budget * ratio)

        while total > low_water:
            if not pairs:
                break
            if use_tokens:
                total -= self.token_counter.count_messages(pairs[0])
            else:
                total -= sum(len(m.get("content") or "") for m in pairs[0])
            pairs = pairs[1:]

        return [m for pair in pairs for m in pair]

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

        Called by the app layer after the backend responds. Prefers the
        authoritative backend signal (``cached_tokens`` from
        ``usage.prompt_tokens_details``); when that is unavailable (e.g. the
        backend does not report it, or streaming without usage) it falls back to
        whether the proxy's own static-prefix KV cache was reused this turn, which
        is a genuine local signal. This replaces the previous constant ``hit=True``
        label that trained the model on noise.
        """
        if self.hit_prediction is None:
            return
        hit = cached_tokens > 0 if cached_tokens is not None else self._last_static_prefix_hit
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

    def _prefetch_dependencies(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        """Prefetch dependencies for files in context.

        Skipped — expert_cache is a NON-FUNCTIONAL placeholder (review03.md §2.1).
        """
        # No-op: expert cache warming is inert scaffolding.

    def _extract_file_references(
        self,
        messages: list[dict[str, Any]],
    ) -> list[str]:
        """Extract file references from messages."""
        file_refs: list[str] = []
        file_pattern = re.compile(r"[\w/]+\.(py|js|ts|go|rs|cpp|h|java)")

        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                matches = file_pattern.findall(content)
                file_refs.extend(matches)

        return file_refs

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
    ) -> list[dict[str, Any]]:
        """Proactively trim context while preserving complete recent turns.

        For MoE models, KV-cache fill is expensive, so context is trimmed before
        it becomes a problem. This method evicts complete user-assistant turns
        from the front of the evictable body instead of dropping the newest
        dynamic message when it cannot fit. That prevents the optimizer from
        collapsing long conversations down to only the static system/first-user
        prefix.

        Args:
            messages: The message list to trim
            target: Target size (chars or tokens depending on use_tokens)
            use_tokens: If True, target is in tokens; if False, in characters
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
        else:
            reserved = (sum(len(m.get("content") or "") for m in system_anchor)
                        + sum(len(m.get("content") or "") for m in protected_tail))
            evictable_budget = max(0, target - reserved)

        evictable_body = self._evict_for_budget(evictable_body, evictable_budget, use_tokens)
        return system_anchor + evictable_body + protected_tail

    def _sliding_window_trim(
        self,
        messages: list[dict[str, Any]],
        window_size: int | None = None,
        overlap_size: int = 256,
        use_tokens: bool = False,
    ) -> list[dict[str, Any]]:
        """Drop whole old turns from the top while preserving the static prefix.

        The model-visible chat history must remain a contiguous prefix. This
        method never truncates message content and never inserts a middle summary;
        it only removes complete user/assistant turns from the front of the
        dynamic layer after the immutable system message.
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
                    compressed = compress_tool_messages([msg], self.tool_output_compressor)
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
