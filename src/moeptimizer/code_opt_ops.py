"""Code-optimization / embedding-ranking methods extracted from AgentContextOptimizer (E1 god-object decomposition)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from moeptimizer.code_block_optimizer import (
    extract_code_blocks,
    optimize_code_in_text,
    slice_code_to_query,
)
from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    deduplicate_chunks,
    detect_language_and_id,
)

if TYPE_CHECKING:
    from moeptimizer.optimizer import AgentContextOptimizer

logger = logging.getLogger(__name__)


class CodeOptOpsMixin:
    """Code-optimization / embedding-ranking methods (see AgentContextOptimizer)."""

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

    def _track_active_file(self: AgentContextOptimizer, path: str, content: str) -> None:
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

    def _is_active_file_content(self: AgentContextOptimizer, content: str) -> bool:
        """True if ``content`` is (a prefix of) a tracked active file body."""
        if not self._active_files:
            return False
        for body in self._active_files.values():
            if body and content and (content == body or content in body or body in content):
                return True
        return False

    def _inject_code_deltas(
        self: AgentContextOptimizer, optimized: list[dict[str, Any]], scan_from: int
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

    def _has_code_blocks(self: AgentContextOptimizer, text: str) -> bool:
        """Check if text contains fenced code blocks."""
        return bool(re.search(r"```[\s\S]*?```", text))

    def _optimize_code_block_content(self: AgentContextOptimizer, text: str) -> str:
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

    def _optimize_code_in_text(self: AgentContextOptimizer, text: str) -> str:
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

    def _sync_embed_and_rank(self: AgentContextOptimizer, base_text: str, chunks: list[str]) -> list[str]:
        """Synchronous embedding and ranking (optionally offloaded to a thread pool)."""
        if self.async_io is not None:
            return self.async_io.run_sync_stage(
                self._embed_and_rank_impl, base_text, chunks, stage_name="embed_rank"
            )
        return self._embed_and_rank_impl(base_text, chunks)

    def _embed_and_rank_impl(self: AgentContextOptimizer, base_text: str, chunks: list[str]) -> list[str]:
        """Core embedding + cosine ranking, run on the request or worker thread."""
        query_vec = self.embedding_service._sync_get_embedding(base_text)
        vecs = self.embedding_service.embed_batch_sync(chunks)
        return self._rank_chunks(query_vec, vecs, chunks)

    def _rank_chunks(
        self: AgentContextOptimizer,
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
            # Embedding unavailable (server down / breaker open / uninitialized
            # service): ranking is a no-op. Surface it so operators can tell RAG is
            # degraded instead of failing silently (review §4.8.1 / §7.2).
            self._record_degradation(
                "embed_rank",
                RuntimeError("zero query embedding; chunks returned unranked"),
            )
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

    def _extract_code_signatures(self: AgentContextOptimizer, pair: list[dict[str, Any]]) -> list[str]:
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

    def _build_code_ledger(self: AgentContextOptimizer, sigs: list[str]) -> list[dict[str, Any]]:
        """Build a compact code-ledger message from accumulated signatures."""
        # Cap the ledger so it cannot itself blow the budget.
        capped = sigs[: self._config.agentic.code_ledger_max_sigs]
        if not capped:
            return []
        body = "[Evicted-turn code index]\n" + "\n".join(capped)
        return [{"role": "system", "content": body, "_code_ledger": True}]

    def _slice_code_in_text(self: AgentContextOptimizer, text: str, query: str) -> str:
        """Slice each fenced code block in ``text`` to the query-referenced defs (B4).

        Replaces each block with ``slice_code_to_query(code, lang, query)`` — strictly
        fail-open (unknown language / no match returns the block unchanged) and never
        expands. Blocks are processed back-to-front so earlier offsets stay valid.
        Idempotent: re-slicing a collapsed block sees only the kept definitions (all
        matching the query) and fails open, returning it unchanged — so the transform
        is cache-stable when applied on every turn.

        A block whose grammar is unavailable fails open (full file retained) and is
        surfaced as a ``code_slicing``/``parser_unavailable`` degradation carrying the
        language, retained input size, and ``failed_open`` outcome, so the silent
        no-delta path is visible in ``/v1/metrics`` (REVIEW_luna P1 reason codes).
        """
        blocks = extract_code_blocks(text)
        if not blocks:
            return text
        result = text
        unavailable: set[str] = set()
        retained_chars: dict[str, int] = {}
        for lang, code, start, end in reversed(blocks):
            lang_id = LANG_MAP.get(lang, lang) if lang else detect_language_and_id(code)
            resolved = lang_id or "generic"
            sliced = slice_code_to_query(code, resolved, query, unavailable)
            if resolved in unavailable:
                retained_chars[resolved] = retained_chars.get(resolved, 0) + len(code)
            if sliced != code:
                result = result[:start] + f"```{lang}\n{sliced}\n```" + result[end:]
        for lang_id in sorted(unavailable):
            self._record_degradation(
                "code_slicing",
                reason=(
                    f"parser_unavailable:lang={lang_id}"
                    f":size={retained_chars.get(lang_id, 0)}:failed_open"
                ),
            )
        return result

    def _slice_message_code_to_query(self: AgentContextOptimizer, msg: dict[str, Any], query: str) -> dict[str, Any]:
        """Per-message wrapper: slice code blocks in the content to ``query`` (B4)."""
        if msg.get("role") not in ("tool", "assistant", "user"):
            return msg
        # Never rewrite the append-only rolling-summary block.
        if self._is_summary_block(msg):
            return msg
        content = msg.get("content") or ""
        if not isinstance(content, str) or "```" not in content:
            return msg
        sliced = self._slice_code_in_text(content, query)
        if sliced == content:
            return msg
        new_msg = dict(msg)
        new_msg["content"] = sliced
        return new_msg
