"""AgentContextOptimizer — Full pipeline orchestrator.

Pipeline:
  1. Parse message history into AgentStateStore steps
  2. Run ScratchpadCompactor on archived steps
  3. Run ThinkingPreserver on assistant messages (preserves <thinking>)
  4. Optimize code blocks with Tree-Sitter + NPU ranking
  5. Enforce hard token cap for MoE context budget
  6. Apply static layer block alignment for cache optimization
  7. Apply syntax-stable MTP prompt engineering

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

import json
import logging
import re
import time
from typing import Any

import numpy as np

from moeptimizer.attention_sink import apply_attention_sinks
from moeptimizer.cache import (
    align_to_block_boundary,
    canonicalize_code_for_cache,
    get_block_aligned_cache_key,
    set_block_size,
)
from moeptimizer.cache_aware_chunker import get_cache_aware_chunker
from moeptimizer.cache_registry import get_cache_registry
from moeptimizer.code_block_optimizer import (
    extract_code_blocks,
    has_code_blocks,
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
from moeptimizer.dependency_orderer import get_dependency_orderer
from moeptimizer.embedding import EmbeddingService
from moeptimizer.expert_cache import get_expert_cache
from moeptimizer.goal_decomposer import GoalDecomposer
from moeptimizer.incremental_updater import get_incremental_updater
from moeptimizer.kv_slot_tracker import get_kv_slot_tracker
from moeptimizer.loop_detector import LoopDetector
from moeptimizer.models import AgentStep, LoopWarning
from moeptimizer.pattern_injector import get_pattern_injector
from moeptimizer.progress_tracker import ProgressTracker
from moeptimizer.prompt_templates import classify_and_template
from moeptimizer.selective_truncator import get_selective_truncator
from moeptimizer.semantic_dedup import get_semantic_deduplicator
from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore
from moeptimizer.symbol_index import SymbolIndex
from moeptimizer.thinking_preserver import ThinkingPreserver
from moeptimizer.hierarchical_index import get_hierarchical_index
from moeptimizer.mtp_state import get_mtp_state_manager
from moeptimizer.token_counter import TokenCounter
from moeptimizer.tool_streamer import get_tool_streamer

logger = logging.getLogger(__name__)


class AgentContextOptimizer:
    """Main orchestrator for agentic context optimization."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()
        self.store = AgentStateStore()
        self.compactor = ScratchpadCompactor()
        self.thinking_preserver = ThinkingPreserver()
        self.state_rag = StateBasedRAG(self.store)
        self.loop_detector = LoopDetector(threshold=3)
        self.progress_tracker = ProgressTracker()
        self.token_counter = TokenCounter()
        self.goal_decomposer = GoalDecomposer()
        self.embedding_service = EmbeddingService()
        self.expert_cache = get_expert_cache()
        self.symbol_index = SymbolIndex()
        self.cache_registry = get_cache_registry()
        self.context_aligner = get_context_aligner()
        self.context_canonicalizer = get_context_canonicalizer()
        self.context_compressor = get_context_compressor()
        self.context_template_matcher = get_context_template_matcher()
        self.dependency_orderer = get_dependency_orderer()
        self.incremental_updater = get_incremental_updater()
        self.pattern_injector = get_pattern_injector()
        self.selective_truncator = get_selective_truncator()
        self.cache_aware_chunker = get_cache_aware_chunker()
        self.hierarchical_index = get_hierarchical_index()
        self.mtp_state_manager = get_mtp_state_manager()
        self.tool_streamer = get_tool_streamer()
        self.semantic_deduplicator = get_semantic_deduplicator()
        self._task_type: str = "default"
        self._last_mtp_state_key: str | None = None

    def optimize_messages(
        self,
        messages: list[dict[str, Any]],
        original_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Full optimization pipeline for a message list in an agentic loop.

        Returns optimized messages ready to send to the MoE model.
        """
        start_time = time.time()

        # Step 1: Populate the state store from messages
        self._ingest_messages(messages)

        # Step 2: Set the root goal if not already set
        if not self.store.get_goal() and original_prompt:
            self.store.set_goal(original_prompt)
        elif not self.store.get_goal() and messages:
            for msg in messages:
                if msg.get("role") == "user":
                    goal_text = msg.get("content", "")[:500]
                    self.store.set_goal(goal_text)
                    subtasks = self.goal_decomposer.decompose(goal_text)
                    self.progress_tracker.set_subtasks(subtasks)
                    break

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

        # Step 5.1: Check cache hit rate - skip heavy optimization if high
        cache_hit_rate = self.cache_registry.predict_hit_rate(optimized)
        if cache_hit_rate > 0.9:
            logger.info(
                "[AgentOptimizer] High cache hit rate (%.2f), skipping heavy optimization",
                cache_hit_rate,
            )
            # Still need to strip internal flags and register context
            optimized = self._strip_internal_flags(optimized)
            self.cache_registry.register_context(optimized)
            self.cache_registry.save_to_disk()
            return optimized

        # Calculate static layer end once (used by multiple steps)
        static_end = self._find_static_layer_end(optimized)
        total_chars = sum(len(m.get("content", "")) for m in optimized)
        total_tokens = self.token_counter.count_messages(optimized)

        # Step 5.2: Build KV slot map for cache control
        slot_tracker = get_kv_slot_tracker()
        slot_map = slot_tracker.build_slot_map(optimized)

        # Step 5.5: Apply context canonicalization for cache-friendly formatting
        optimized = self.context_canonicalizer.canonicalize(optimized)

        # Step 5.7: Apply context compression to reduce token usage
        optimized = self.context_compressor.compress(optimized)

        # Step 5.8: Apply attention sink management for long-context stability
        if total_chars > 4000:
            optimized = apply_attention_sinks(optimized, static_end)

        # Step 5.9: Warm expert cache for static layer patterns
        if total_chars > 1000:
            static_content = self._get_static_layer_content(optimized)
            if static_content:
                self.expert_cache.warm_cache_for_static_layer(static_content)

        # Step 5.10: Prefetch dependencies for files in context
        self._prefetch_dependencies(optimized)

        # Step 6: Apply prompt template versioning
        optimized, self._task_type = classify_and_template(optimized)

        # Step 6.5: Apply context template matching
        if not (optimized and optimized[0].get("role") == "system"):
            template_name = self.context_template_matcher.match_template(optimized)
            if template_name:
                optimized = self.context_template_matcher.apply_template(optimized)

        # Step 7: Apply scratchpad compaction
        optimized = self.compactor.compact_messages(optimized)

        # Step 7.5: Apply selective truncation (remove duplicate code blocks)
        optimized = self.selective_truncator.remove_duplicates(optimized)

        # Step 7.6: Apply semantic deduplication for near-duplicate context
        if total_tokens > 1000:  # Only for larger contexts
            optimized = self.semantic_deduplicator.deduplicate(
                optimized,
                self.embedding_service,
            )

        # Step 7.7: Apply dependency ordering for cache locality
        optimized = self.dependency_orderer.order_by_dependencies(optimized)

        # Step 7.8: Apply incremental update for cache preservation
        optimized = self.incremental_updater.update_context(optimized, "")

        # Step 8: Inject RAG context and loop warnings as SEPARATE user messages
        if loop_warnings:
            warning_lines: list[str] = []
            for w in loop_warnings:
                warning_lines.append(self.loop_detector.get_warning_message(w))
            if warning_lines:
                optimized.append({
                    "role": "user",
                    "content": "\n".join(warning_lines),
                })

        # RAG context: append as a separate user message
        last_assistant = None
        for msg in reversed(optimized):
            if msg.get("role") == "assistant" and not msg.get("_archived"):
                last_assistant = msg
                break

        if last_assistant:
            current_step = AgentStep(
                role=last_assistant.get("role", "assistant"),
                content=last_assistant.get("content", ""),
                tool_name=None,
                metadata=last_assistant.get("metadata", {}),
            )
            rag_context = self.state_rag.get_context_for_step(current_step)
            if rag_context:
                optimized.append({
                    "role": "user",
                    "content": rag_context,
                })

        # Step 8: Apply static layer block alignment for cache optimization
        if total_tokens > 500:  # ~2000 chars
            optimized = self._align_static_layer(optimized)

        # Step 9: Pre-seed reasoning prefix for MTP (only if budget allows)
        max_tokens = self._config.agentic.max_optimized_chars // 4
        if total_tokens < max_tokens * 0.5:
            optimized = self._preseed_reasoning(optimized)

        # Step 10: Optimize code blocks in all messages
        for msg in optimized:
            content = msg.get("content", "")
            if isinstance(content, str) and has_code_blocks(content):
                msg["content"] = optimize_code_in_text(
                    content,
                    self._config,
                    self.embedding_service,
                )

        # Step 10.5: Apply cache-aware chunking for large contexts
        if total_tokens > 750:  # ~3000 chars
            optimized = self.cache_aware_chunker.chunk_context(optimized)

        # Step 11: Proactive context trimming for MoE KV-cache efficiency
        # Use token-based threshold for more accurate budget enforcement
        proactive_threshold_tokens = int(max_tokens * 0.7)
        if total_tokens > proactive_threshold_tokens:
            optimized = self._proactive_trim(optimized, proactive_threshold_tokens, use_tokens=True)

        # Step 11.5: Entropy-guided trimming for MTP-friendly content
        optimized = self._entropy_guided_trim(optimized)

        # Step 11.6: Stream large tool outputs for better context management
        optimized = self._stream_large_tool_outputs(optimized)

        # Recalculate total_tokens after entropy trim
        total_tokens = self.token_counter.count_messages(optimized)

        # Step 11.7: Save MTP state before trimming for context switching
        # This preserves prediction quality across evictions
        state_key = self.mtp_state_manager.get_state_key(optimized)
        # Store the state key in the optimizer for potential restoration
        self._last_mtp_state_key = state_key

        # Step 11.8: Apply sliding window for long contexts
        # This is the preferred method for context management with MTP state preservation
        if total_tokens > int(max_tokens * 0.8):
            optimized = self._sliding_window_trim(optimized, use_tokens=True)

        # Step 11.9: Align to MTP prediction boundary for better MTP accuracy
        # This ensures context length is a multiple of 128 tokens
        optimized = self.mtp_state_manager.align_prediction_boundary(optimized)

        # Step 12: Enforce hard token cap
        total_tokens = self.token_counter.count_messages(optimized)
        if total_tokens > max_tokens:
            optimized = self._trim_to_budget(optimized, use_tokens=True)

        # Step 13: Strip internal metadata before sending to model
        optimized = self._strip_internal_flags(optimized)

        # Step 14: Register context in cache registry for hit prediction
        self.cache_registry.register_context(optimized)

        # Step 14.5: Persist cache registry for cross-session reuse
        self.cache_registry.save_to_disk()

        # Log metrics
        original_chars = sum(len(m.get("content", "")) for m in messages)
        optimized_chars = sum(len(m.get("content", "")) for m in optimized)
        original_tokens = self.token_counter.count_messages(messages)
        optimized_tokens = self.token_counter.count_messages(optimized)
        duration = time.time() - start_time
        saved_tokens = max(0, original_tokens - optimized_tokens)

        logger.info(
            "[AgentOptimizer] %d -> %d chars (%d -> %d tokens, %d saved, %.3fs, %d -> %d msgs, progress: %.0%%, loops: %d detected)",
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

        return optimized

    def _strip_internal_flags(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove internal metadata keys and section markers that shouldn't reach the model.

        Preserves the message structure and content while stripping:
        - _archived: compactor marker
        - Section markers (<!-- STATIC/CONTEXT/DYNAMIC_LAYER -->)
        - Attention sink markers (# CONTEXT_ANCHOR, STATIC_LAYER_END)
        - Any other _prefixed keys (future-proof)
        """
        result: list[dict[str, Any]] = []
        internal_prefix = "_"
        # Pre-compiled pattern for performance
        _MARKER_PATTERN = re.compile(
            r"<!-- (STATIC|CONTEXT|DYNAMIC)_LAYER -->\n?|# (CONTEXT_ANCHOR|STATIC_LAYER_END).*\n?",
        )

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = _MARKER_PATTERN.sub("", content)

            cleaned = {
                k: v
                for k, v in msg.items()
                if not k.startswith(internal_prefix)
            }
            cleaned["content"] = content
            result.append(cleaned)

        return result

    def _ingest_messages(self, messages: list[dict[str, Any]]) -> None:
        """Convert message list into AgentStateStore steps."""
        for i, msg in enumerate(messages):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

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

            self.store.add_step(step)
            self.progress_tracker.record_step(step)

    def _has_code_blocks(self, text: str) -> bool:
        """Check if text contains fenced code blocks."""
        return bool(re.search(r"```[\s\S]*?```", text))

    def _optimize_code_in_text(self, text: str) -> str:
        """Optimize code blocks within a text string using Tree-Sitter + NPU.

        Returns the original text if optimization would reduce code block count.
        """
        regex_pattern = r"(```[\s\S]*?```)"
        blocks = re.findall(regex_pattern, text)
        base_text = re.sub(regex_pattern, "", text).strip()

        if not blocks:
            return text

        # Store original blocks for fallback
        original_blocks = list(blocks)

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
        """Synchronous embedding and ranking."""
        query_vec = self.embedding_service._sync_get_embedding(base_text)
        vecs = self.embedding_service.embed_batch_sync(chunks)
        return self._rank_chunks(query_vec, vecs, chunks)

    def _rank_chunks(
        self,
        query_vec: np.ndarray,
        chunk_vecs: list[np.ndarray],
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
            max_tokens = self._config.agentic.max_optimized_chars // 4
        else:
            max_chars = self._config.agentic.max_optimized_chars

        # Partition into zones
        system_anchor, evictable_body, protected_tail = self._partition_for_budget(messages)

        # Reserve space for non-evictable zones; remaining budget is what's available
        if use_tokens:
            reserved = (self.token_counter.count_messages(system_anchor)
                        + self.token_counter.count_messages(protected_tail))
            evictable_budget = max(0, max_tokens - reserved)
        else:
            reserved = (sum(len(m.get("content", "")) for m in system_anchor)
                        + sum(len(m.get("content", "")) for m in protected_tail))
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

        if i < len(messages) and messages[i].get("role") == "user":
            system_anchor.append(messages[i])
            i += 1

        # Group remaining into user-assistant pairs.
        # Each turn starts with a user message and includes following assistant/tool messages.
        # Leading assistants (orphaned from system anchor's first user) are attached to the next user.
        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            if role == "user":
                # Save previous turn (complete or pending) before starting new one
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                # Assistant/tool messages without a preceding user belong to the next turn.
                # Collect them until we hit a user message, then attach them there.
                if not current_turn:
                    # No pending user yet — save these as an orphaned group
                    # that will be attached to the next user's turn
                    current_turn = [{"_orphan": True}]  # marker for attachment
                    current_turn.append(msg)
                else:
                    current_turn.append(msg)
            i += 1

        if current_turn and not any(m.get("_orphan", False) for m in current_turn):
            turns.append(current_turn)
        elif current_turn:
            # This turn starts with orphans — merge into previous turn's end
            if turns:
                turns[-1].extend([m for m in current_turn if not m.get("_orphan")])
            else:
                # No previous turn to attach to — keep as standalone pending
                turns.append(current_turn)

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

        # Always preserve pending (unpaired user-only) turns.
        for t in pending_turns:
            protected.extend(t)

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
            total_tokens = sum(
                self.token_counter.count_messages(pair) for pair in pairs
            )
        else:
            total_chars = sum(len(m.get("content", "")) for p in pairs for m in p)

        while (use_tokens and total_tokens > budget) or (not use_tokens and total_chars > budget):
            if not pairs:
                break
            if use_tokens:
                pair_tokens = self.token_counter.count_messages(pairs[0])
                total_tokens -= pair_tokens
            else:
                pair_size = sum(len(m.get("content", "")) for m in pairs[0])
                total_chars -= pair_size
            pairs = pairs[1:]

        return [m for pair in pairs for m in pair]

    def _align_static_layer(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Align static layer to block boundary for cache optimization.

        Pads the static layer (system + first user) to align to
        CONTEXT_BLOCK_SIZE boundary, improving prefix cache hit rates.
        Only applies padding if it doesn't exceed budget.
        """
        if not messages:
            return messages

        # Find static layer end (system + first user)
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif msg.get("role") == "user" and static_end > 0:
                static_end = i + 1
                break
            elif msg.get("role") == "user" and static_end == 0:
                static_end = i + 1
                break

        if static_end == 0:
            return messages

        # Calculate current static layer size
        static_content = "\n".join(
            m.get("content", "") for m in messages[:static_end]
        )
        aligned_content = align_to_block_boundary(static_content)

        # Only add padding if it's small and won't break the budget
        # (padding is for cache alignment, not content)
        padding_needed = len(aligned_content) - len(static_content)
        # Only pad if less than 100 newlines (small alignment)
        if padding_needed > 0 and padding_needed < 100:
            # Create a copy of messages to avoid mutation
            result = [dict(m) for m in messages]
            # Add padding to the last static message
            result[static_end - 1] = {
                **result[static_end - 1],
                "content": result[static_end - 1].get("content", "") + "\n" * padding_needed,
            }
            return result

        return messages

    def _apply_syntax_stable_mtp(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply syntax-stable MTP prompt engineering.

        Pre-seeds code-specific patterns to improve MTP prediction accuracy:
        - Indentation level markers
        - Type signature anchors
        - Section markers for code structure
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and self._has_code_blocks(content):
                content = self._inject_syntax_markers(content)
            result.append({**msg, "content": content})
        return result

    def _inject_syntax_markers(self, text: str) -> str:
        """Inject syntax markers for MTP stability.

        Adds predictable patterns that help MTP heads converge faster.
        """
        # Add section markers for code blocks
        lines = text.split("\n")
        result_lines = []
        in_code_block = False
        code_block_lang = ""

        for i, line in enumerate(lines):
            if line.strip().startswith("```") and not in_code_block:
                in_code_block = True
                code_block_lang = line.strip().replace("```", "").strip()
                result_lines.append(line)
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                code_block_lang = ""
                result_lines.append(line)
            elif in_code_block and code_block_lang:
                # Inject section markers for common patterns
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("function "):
                    result_lines.append(f"# SECTION: function {stripped[:40]}")
                elif stripped.startswith("class "):
                    result_lines.append(f"# SECTION: class {stripped[:40]}")
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    result_lines.append(f"# SECTION: import")
                result_lines.append(line)
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def get_cache_key(self, messages: list[dict[str, Any]]) -> str:
        """Generate cache key with canonicalization for static layer."""
        return get_block_aligned_cache_key(messages)

    def _find_static_layer_end(self, messages: list[dict[str, Any]]) -> int:
        """Find the end index of the static layer (system + first user)."""
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif msg.get("role") == "user" and static_end > 0:
                static_end = i + 1
                break
            elif msg.get("role") == "user" and static_end == 0:
                static_end = i + 1
                break
        return static_end

    def _get_static_layer_content(self, messages: list[dict[str, Any]]) -> str:
        """Extract the static layer content for expert cache warming."""
        static_end = self._find_static_layer_end(messages)
        if static_end == 0:
            return ""
        return "\n".join(m.get("content", "") for m in messages[:static_end])

    def _prefetch_dependencies(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        """Prefetch dependencies for files in context.

        Warms the expert cache for related code to avoid cold starts.
        """
        # Extract file references from context
        file_refs = self._extract_file_references(messages)

        for file_path in file_refs:
            # Get dependency context
            dep_context = self.state_rag.get_dependency_context(file_path)
            if dep_context:
                # Warm expert cache for dependency patterns
                self.expert_cache.warm_cache_for_static_layer(dep_context)

    def _extract_file_references(
        self,
        messages: list[dict[str, Any]],
    ) -> list[str]:
        """Extract file references from messages."""
        file_refs: list[str] = []
        file_pattern = re.compile(r"[\w/]+\.(py|js|ts|go|rs|cpp|h|java)")

        for msg in messages:
            content = msg.get("content", "")
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
        max_tokens = self._config.agentic.max_optimized_chars // 4
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
        content = user_msg.get("content", "")

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
        """Proactively trim context to prevent KV-cache performance degradation.

        For MoE models, KV-cache fill is extremely slow. This method trims
        context before it becomes a problem, preserving the most recent turns.

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
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars <= target:
                return messages

        # Find the static layer (system + first user)
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif msg.get("role") == "user" and static_end > 0:
                static_end = i + 1
                break
            elif msg.get("role") == "user" and static_end == 0:
                static_end = i + 1
                break

        # Calculate how much we can keep from dynamic layer
        if use_tokens:
            static_tokens = self.token_counter.count_messages(messages[:static_end])
            available_for_dynamic = target - static_tokens
        else:
            static_chars = sum(len(m.get("content", "")) for m in messages[:static_end])
            available_for_dynamic = target - static_chars

        if available_for_dynamic <= 0:
            return messages[:static_end]  # Only keep static layer

        # Keep only the most recent dynamic content
        result = [dict(m) for m in messages[:static_end]]
        dynamic_messages = messages[static_end:]

        # Add messages from the end until we hit the limit
        for msg in reversed(dynamic_messages):
            if use_tokens:
                msg_tokens = self.token_counter.count_messages([msg])
                if available_for_dynamic >= msg_tokens:
                    result.insert(static_end, dict(msg))
                    available_for_dynamic -= msg_tokens
                else:
                    break
            else:
                msg_chars = len(msg.get("content", ""))
                if available_for_dynamic >= msg_chars:
                    result.insert(static_end, dict(msg))
                    available_for_dynamic -= msg_chars
                else:
                    break

        return result

    def _sliding_window_trim(
        self,
        messages: list[dict[str, Any]],
        window_size: int | None = None,
        overlap_size: int = 256,
        use_tokens: bool = False,
    ) -> list[dict[str, Any]]:
        """Apply sliding window with MTP state preservation.

        Uses a sliding window approach that preserves MTP state in the overlap region.
        This maintains prediction quality across context switches.

        Args:
            messages: The message list to trim
            window_size: Target size (chars or tokens depending on use_tokens)
            overlap_size: Size of overlap region for state continuity
            use_tokens: If True, window_size is in tokens; if False, in characters
        """
        if window_size is None:
            window_size = self._config.agentic.max_optimized_chars

        if use_tokens:
            total_tokens = self.token_counter.count_messages(messages)
            if total_tokens <= window_size:
                return messages
        else:
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars <= window_size:
                return messages

        # Find static layer
        static_end = self._find_static_layer_end(messages)
        if use_tokens:
            static_tokens = self.token_counter.count_messages(messages[:static_end])
        else:
            static_chars = sum(len(m.get("content", "")) for m in messages[:static_end])
            static_tokens = static_chars // 4  # Convert to tokens for comparison

        # If static layer alone exceeds window, only keep static
        if use_tokens:
            if static_tokens >= window_size:
                return messages[:static_end]
        else:
            if static_chars >= window_size:
                return messages[:static_end]

        # Calculate available space for dynamic content
        if use_tokens:
            available = window_size - static_tokens
        else:
            available = window_size - static_chars

        # Keep overlap at the end for MTP state preservation
        # The overlap region maintains hidden state continuity
        result = [dict(m) for m in messages[:static_end]]
        dynamic_messages = messages[static_end:]

        # Add messages from end, keeping overlap
        current_size = 0
        kept_for_overlap: list[dict[str, Any]] = []

        for msg in reversed(dynamic_messages):
            if use_tokens:
                msg_tokens = self.token_counter.count_messages([msg])
                if current_size + msg_tokens <= available:
                    result.insert(static_end, dict(msg))
                    current_size += msg_tokens
                else:
                    # This message would exceed the window
                    # Keep it for overlap (will be added at the end)
                    kept_for_overlap.insert(0, dict(msg))
            else:
                msg_chars = len(msg.get("content", ""))
                if current_size + msg_chars <= available:
                    result.insert(static_end, dict(msg))
                    current_size += msg_chars
                else:
                    # This message would exceed the window
                    # Keep it for overlap (will be added at the end)
                    kept_for_overlap.insert(0, dict(msg))

        # Add overlap messages at the end (after the kept content)
        # This preserves MTP state continuity
        # Always add at least one overlap message for state continuity
        if kept_for_overlap:
            # Add the most recent overlap message (the one closest to the current turn)
            result.append(kept_for_overlap[0])

        return result

    def _entropy_guided_trim(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Trim high-entropy content while preserving low-entropy code structures.

        MTP heads perform better with low-entropy contexts. This method:
        - Identifies high-entropy "noise" messages (tool logs, errors)
        - Trims them first while preserving code structures
        - Never modifies assistant content to avoid KV-cache refill
        - Never removes assistant messages (they're part of the chat template)
        """
        if not messages:
            return messages

        # Find static layer (system + first user)
        static_end = self._find_static_layer_end(messages)

        # Separate static and dynamic
        static_messages = messages[:static_end]
        dynamic_messages = messages[static_end:]

        # Only trim tool messages with high entropy
        # Never trim assistant messages (they're part of the chat template)
        trimmed_dynamic = []
        for msg in dynamic_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Always keep assistant messages (chat template integrity)
            if role == "assistant":
                trimmed_dynamic.append(msg)
                continue

            # For tool messages, check entropy
            if role == "tool" and isinstance(content, str):
                entropy = self._calculate_message_entropy(content)
                # High entropy tool output - can be trimmed
                if entropy > 0.7 and len(content) > 500:
                    # Replace with summary instead of removing
                    # This preserves the turn structure
                    trimmed_dynamic.append({
                        **msg,
                        "content": f"[Tool output truncated - {len(content)} chars]",
                    })
                else:
                    trimmed_dynamic.append(msg)
            else:
                trimmed_dynamic.append(msg)

        return static_messages + trimmed_dynamic

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

    def _stream_large_tool_outputs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Stream large tool outputs as separate user messages.

        Large tool outputs are split into chunks to avoid context bloat
        while maintaining MTP prediction patterns.

        CRITICAL: Tool messages are kept as tool role to preserve turn structure.
        The model expects user→assistant→tool turn patterns.
        """
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool" and isinstance(content, str):
                if self.tool_streamer.should_stream(content):
                    # Stream as separate tool messages (not user!)
                    # This preserves the turn structure for the model
                    tool_name = msg.get("metadata", {}).get("name", "unknown")
                    streamed = self.tool_streamer.stream_output(content, tool_name)
                    for i, chunk in enumerate(streamed):
                        # Keep as tool role, add chunk index for tracking
                        result.append({
                            **msg,
                            "content": chunk,
                            "chunk_index": i,
                        })
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
            content = msg.get("content", "")
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
