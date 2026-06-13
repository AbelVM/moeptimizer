"""AgentContextOptimizer — Full pipeline orchestrator.

Pipeline:
  1. Parse message history into AgentStateStore steps
  2. Run ScratchpadCompactor on archived steps
  3. Run ThinkingPreserver on assistant messages (preserves <think>/</think>)
  4. Optimize code blocks with Tree-Sitter + NPU ranking
  5. Enforce hard character cap for MoE context budget

MOE context integrity:
  - RAG context injected as SEPARATE user message (never into assistant content)
  - Loop warnings injected as SEPARATE user message (never into assistant content)
  - Progress tracking is internal only (not injected into context)
  - This preserves the model's expected chat template:
    ăssistant\n<think>\n{reasoning}\n</think>\n\n{response}
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

from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    deduplicate_chunks,
    detect_language_and_id,
)
from moeptimizer.compactor import ScratchpadCompactor
from moeptimizer.config import AppConfig
from moeptimizer.embedding import EmbeddingService
from moeptimizer.goal_decomposer import GoalDecomposer
from moeptimizer.loop_detector import LoopDetector
from moeptimizer.models import AgentStep, LoopWarning
from moeptimizer.progress_tracker import ProgressTracker
from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore
from moeptimizer.thinking_preserver import ThinkingPreserver
from moeptimizer.token_counter import TokenCounter

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

        # Step 5: Apply thinking preservation
        optimized = self.thinking_preserver.process_messages(list(messages))

        # Step 6: Apply scratchpad compaction
        optimized = self.compactor.compact_messages(optimized)

        # Step 7: Inject RAG context and loop warnings as SEPARATE user messages
        # CRITICAL: Never inject into assistant messages — this breaks the model's
        # expected chat template (ăssistant\n<think>\n...\n</think>\n\n...) and
        # triggers KV-cache refills during expensive MOE prefill.
        #
        # Instead, append user messages AFTER the last assistant turn. The model
        # was trained on user→assistant turn pairs, so this pattern is safe.
        if loop_warnings:
            # Build a compact loop warning for the model
            warning_lines: list[str] = []
            for w in loop_warnings:
                warning_lines.append(self.loop_detector.get_warning_message(w))
            if warning_lines:
                optimized.append({
                    "role": "user",
                    "content": "\n".join(warning_lines),
                })

        # RAG context: append as a separate user message (not injected into assistant)
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

        # Step 8: Optimize code blocks in all messages
        for msg in optimized:
            content = msg.get("content", "")
            if isinstance(content, str) and self._has_code_blocks(content):
                msg["content"] = self._optimize_code_in_text(content)

        # Step 9: Enforce hard character cap
        # Use character counts consistently — _trim_to_budget uses char budget
        total_chars = sum(len(m.get("content", "")) for m in optimized)
        if total_chars > self._config.agentic.max_optimized_chars:
            optimized = self._trim_to_budget(optimized)

        # Step 10: Strip internal metadata before sending to model
        optimized = self._strip_internal_flags(optimized)

        # Log metrics
        original_chars = sum(len(m.get("content", "")) for m in messages)
        optimized_chars = sum(len(m.get("content", "")) for m in optimized)
        original_tokens = self.token_counter.count_messages(messages)
        optimized_tokens = self.token_counter.count_messages(optimized)
        duration = time.time() - start_time
        saved_tokens = max(0, original_tokens - optimized_tokens)

        logger.info(
            "[AgentOptimizer] %d -> %d chars (%d tokens saved, %.3fs, %d -> %d msgs, progress: %.0%%, loops: %d detected)",
            original_chars,
            optimized_chars,
            saved_tokens,
            duration,
            len(messages),
            len(optimized),
            progress.estimated_completion * 100,
            len(loop_warnings),
        )

        return optimized

    def _strip_internal_flags(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove internal metadata keys that shouldn't reach the model.

        Preserves the message structure and content while stripping:
        - _archived: compactor marker
        - Any other _prefixed keys (future-proof)
        """
        result: list[dict[str, Any]] = []
        internal_prefix = "_"

        for msg in messages:
            cleaned = {
                k: v
                for k, v in msg.items()
                if not k.startswith(internal_prefix)
            }
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
        """Optimize code blocks within a text string using Tree-Sitter + NPU."""
        regex_pattern = r"(```[\s\S]*?```)"
        blocks = re.findall(regex_pattern, text)
        base_text = re.sub(regex_pattern, "", text).strip()

        if not blocks:
            return text

        detected_langs: set[str] = set()
        all_chunks: list[str] = []

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

            chunks = chunk_code_with_treesitter(code, lang_id or "generic", self._config.code_chunking.chunk_max_chars)
            all_chunks.extend(chunks)

        if not all_chunks:
            return text

        all_chunks = deduplicate_chunks(all_chunks)

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
                replacement = f"```{next(iter(detected_langs)) if detected_langs else ''}\n{chunk}\n```"
                text = text.replace(placeholder_str, replacement)

        if len(all_chunks) > len(blocks):
            extra = "\n\n".join(
                f"```{next(iter(detected_langs)) if detected_langs else ''}\n{c}\n```"
                for c in all_chunks[len(blocks):]
            )
            text = text.rstrip() + "\n\n" + extra

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

    def _trim_to_budget(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim messages to stay within the character budget.

        Uses front-loading eviction: drops complete user-assistant pairs from
        the front of the evictable body. No content modification or truncation.

        Three immutable zones:
          1. System Anchor: system + first user (never modified)
          2. Evictable Body: historical turns (dropped from front)
          3. Protected Tail: last N turns (never modified)

        This preserves token offsets and sequence patterns for MTP heads.
        """
        max_chars = self._config.agentic.max_optimized_chars

        # Partition into zones
        system_anchor, evictable_body, protected_tail = self._partition_for_budget(messages)

        # Reserve space for non-evictable zones; remaining budget is what's available
        reserved = (sum(len(m.get("content", "")) for m in system_anchor)
                    + sum(len(m.get("content", "")) for m in protected_tail))
        evictable_budget = max(0, max_chars - reserved)

        # Evict from front of evictable body until under remaining budget
        evictable_body = self._evict_for_budget(evictable_body, evictable_budget)

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
    ) -> list[dict[str, Any]]:
        """Drop pairs from front of evictable body until under budget."""
        if not evictable_body:
            return evictable_body

        # Group into user-assistant pairs
        pairs: list[list[dict[str, Any]]] = []
        current_pair: list[dict[str, Any]] = []

        for msg in evictable_body:
            role = msg.get("role", "")
            if role == "user":
                if current_pair:
                    pairs.append(current_pair)
                current_pair = [msg]
            else:
                current_pair.append(msg)

        if current_pair:
            pairs.append(current_pair)

        # Drop from front until under budget.
        total_chars = sum(len(m.get("content", "")) for p in pairs for m in p)
        while total_chars > budget:
            if not pairs:
                break
            pair_size = sum(len(m.get("content", "")) for m in pairs[0])
            total_chars -= pair_size
            pairs = pairs[1:]

        return [m for pair in pairs for m in pair]

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

        if "goal_subtasks" in data:
            self.progress_tracker.set_subtasks(data["goal_subtasks"])
