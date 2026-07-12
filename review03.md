# MOEptimizer Architecture Review (review03)

**Scope:** Transparent OpenAI proxy fronting Lemonade/Qwen3.6-35B-A3B-MTP, for multi-turn agentic coding (OpenCode client).
**Date:** 2026-07-12
**Inputs:** README.md, notes.md, cache_preservation_guide.md, review01.md, review02.md, full `src/moeptimizer/*` read, `scripts/benchmark.py`, latest benchmark `benchmark_refactor_long_30_13.json`, web refs (headroom, snip, rtk, opencode).
**Benchmark under review:** refactor_long, 30 turns, max_tokens=256 → token_savings 85.79%, latency +23.6% (proxy slower), semantic_similarity 0.7063 (**Grade D**), per_turn_cached median 31 tok (~5.5% reuse), length_ratio ~2.17 (verbosity regression).

---

## Implementation progress (updated 2026-07-12, 15:40)

This review is being actioned in-place. Status is tracked against the prioritized
fix list at the bottom. Legend: ✅ done · 🔄 in progress · ⬜ pending.

| Fix | Status | Notes |
|-----|--------|-------|
| #1 Re-anchor prefix (volatile → trailing turn) | ✅ done | `optimizer._append_volatile_context` (replaces `_inject_quality_anchor`); RAG/loop injection deferred to Step 14.12; `hierarchical_summarizer` rolling summary now a trailing turn; `_last_optimized` recorded; calibration in `app.py` uses the actual optimized prompt. All 305 tests green. |
| #2 Real Qwen tokenizer + wire calibration | ✅ done | `server.tokenizer` config (default `auto`); `TokenCounter` loads the real Qwen HF tokenizer (module-cached, local-only for `auto` so no surprise downloads), falls back to tiktoken with an honest warning. Confirmed `auto` resolves to `hf:Qwen/Qwen2.5-Coder-32B-Instruct` on this box. Runtime calibration (fix #1) remains the safety net. |
| #3 Quarantine/relabel phantom subsystems | ✅ done | Module docstrings for `static_prefix_kv`/`mtp_state`/`expert_cache` now state plainly they are text memo / inert / placeholder (not real KV/MTP/expert state). Config toggles `mtp_boundary_alignment_enabled`, `SpeculativeConfig.enabled`, `static_prefix_kv_enabled` relabeled honestly. `expert_cache.warm_cache_for_static_layer` now gated behind `enable_experimental_backend_hints` (was unconditional). `mtp_state.get_state_key` char→token bug (#6) fixed and reuses the optimizer's loaded tokenizer. |
| #4 Async / streaming off critical path | ✅ done | Streaming responses already implemented (default `stream=True`, `StreamingResponse` + backend `chat_completions_stream`); the CPU-bound optimizer already runs in a worker thread via `run_in_executor` so the event loop stays free (review §2/§4/§5). Heavy optional passes (tree-sitter AST, embedding ranking) are gated behind `async_io_stage` and disabled by default. Removed the one remaining event-loop hotspot: the `X-Optimized-Prompt-Tokens` header no longer re-tokenizes the whole prompt on the loop — it reuses the count cached during `optimize_messages` (`last_optimized_token_count`), avoiding a second full HF-Qwen tokenization per request. |
| #5 Boundary compression for tool I/O | ✅ done | New `tool_output_compressor.py` (`ToolOutputCompressor` + `compress_tool_messages`): strips ANSI/CR, collapses 3+ repeated lines, dedupes repeated stack-frame blocks, truncates oversized outputs (head+tail, idempotent). Config `agentic.tool_output_compression_enabled` (default True) + `_max_chars` (default 4000). No-op `_stream_large_tool_outputs` replaced by `_compress_tool_outputs` (runs every turn, idempotent so the frozen prefix stays byte-stable). 7 new tests; suite 312 passed / 2 skipped. |
| #6 Eviction maximizes backend `cached_tokens` | ✅ done | `_evict_for_budget` now uses a high/low watermark (`agentic.eviction_low_water_ratio`, default 0.8): eviction only triggers above budget but then batches down to `budget * ratio`, so the oldest kept turn stays byte-stable across many subsequent turns instead of evicting one pair every over-budget turn (which invalidated the backend prefix cache every turn). `record_cache_outcome` already feeds the authoritative `cached_tokens` signal into hit-prediction. `ratio=1.0` restores exact-budget trimming. 2 new tests; suite 314 passed / 2 skipped. |
| #7 Memoize optimization per fingerprint | ✅ done | `TokenCounter.count_messages` now memoizes by a cheap content fingerprint (role+content SHA-1) in a bounded per-instance LRU (`max_cache`, default 256). The stable prefix is re-counted every turn and per-pair inside `_evict_for_budget`, so identical content is no longer re-tokenized. The optimizer also caches the last optimized count (`last_optimized_token_count`) for the response header. 2 new tests; suite 316 passed / 2 skipped. |
| #8 Remove per-turn pickle disk writes | ✅ done | The two per-turn pickle writes on the request path (`hierarchical_summarizer.save_to_disk`, `static_prefix_kv.save_to_disk`) are now gated behind a new `v050.persist_state_to_disk` flag (default **False**). State is still kept in memory every turn; disk persistence only happens when explicitly enabled (crash recovery / cross-process). `delta_encoder.store_snapshot` was already in-memory. Removes per-turn disk I/O latency (review §8). |
| #9 Benchmark methodology fix | ✅ done | `benchmark.py --max-tokens` default raised from 256 → 1024 (256 understated proxy token-savings because it capped the baseline's own output). `_looks_like_cached_response` false-hit fixed: it now requires a positive `cached_tokens` signal to label a turn cached; the old `prompt_tokens==0 and completion_tokens>0` rule mislabeled normal responses as cache hits. Zero-prompt is now treated as "usage omitted", not "cached". |
| #10 UX: metrics endpoint, dry-run/explain, quality profile, config check, regression gate | ✅ done | **All sub-items shipped.** (1) **Metrics:** `_ProxyMetrics` aggregate (process-wide, lock-protected) fed from both streaming (L~401) and non-streaming (L~517) `record_cache_outcome` sites; `optimizer` exposes `last_original_token_count` / `last_saved_token_count` (cached during `optimize_messages`, no re-tokenization); `chat_completions_proxy` passes `_turn_start` into both paths so `record_turn` captures `cached_tokens`/`prompt_tokens`/`saved_tokens`/`latency_ms`; `GET /v1/metrics` + `POST /v1/metrics/reset`. (2) **Quality profiles:** `config.py` `agentic.quality_profile` (quality/balanced/aggressive) + `QUALITY_PROFILES` + `apply_quality_profile()` (routes each override to the owning sub-config — `agentic` or `v050`; unknown→balanced with warning); applied at app-build time so explicit env/field overrides still win; `SessionManager(config=...)` passes it to the optimizer. (3) **Dry-run/explain:** `agentic.explain_mode_enabled` + per-request `X-MOEPT-Explain` header/`_explain` body opt-in; proxy attaches base64-JSON `X-MOEPT-Optimized-Messages` + `X-MOEPT-Explain: true` response headers (set before backend call, survive 500s). (4) **Config check CLI:** new `config_check.py` (`check_config`→`ConfigIssue` ERROR/WARN/INFO, exits non-zero on ERROR); `moeptimizer-config-check` script + `python -m moeptimizer --check-config`. (5) **Regression gate:** `benchmark.py` `--min-similarity` + `_check_similarity_gate` (exit 2 on fail) in both single and all-scenario paths; `--profile` expanded to quality/balanced/aggressive. Suite **331 passed / 2 skipped**. |

**Note on the "core flaw" (§0/§2.1):** confirmed in code. The three phantom
subsystems remain non-functional by construction (an OpenAI client-side proxy
cannot access backend KV/MTP/expert state). Fix #1 addresses the *only* real
lever — byte-stable leading prefix so the backend's native prefix cache is
reused — and is now implemented.

---

## 0. TL;DR verdict

The proxy delivers large token savings (good) but is **net-negative on the two metrics that matter for this workload**: it is **slower** than direct and its responses are only **Grade D** similar to the un-proxified baseline. The root cause is not tuning — it is a **category error in architecture**: three headline "MoE/MTP optimization" subsystems (`static_prefix_kv`, `mtp_state`, `expert_cache`) are built to manipulate KV-cache tensors and expert routing that **the OpenAI API cannot expose to a client-side proxy**. They store text/keys/placeholder masks, not real model state, so they cannot do what their names and README claims promise. Meanwhile the *only* real cache lever — keeping the backend's native prefix cache byte-stable — is being actively undermined by the proxy's own mutations (compaction, RAG/loop injection, rolling summaries).

**Recommendation:** Re-scope the proxy from "KV/MTP optimizer" to "byte-stable prefix anchor + boundary compressor" (the review02 strategic framing, now confirmed by code), delete or quarantine the three phantom subsystems, fix the tokenizer, and re-benchmark against headroom/snip/rtk-class approaches.

---

## 1. Missing optimizations

1. **Real prefix-cache anchoring is not actually achieved.** The backend (llama.cpp/llama-server) already does prefix caching (`cache_prompt`, `--cache-reuse` KV shifting). The proxy's job is to guarantee a byte-identical leading prefix every turn. But `optimizer.py` mutates the prefix region: it appends a "Conversation Quality Anchor" to the last *user* turn (`_inject_quality_anchor`, L886), injects RAG/loop text into the last user turn (L540-590), and (in cache-stable mode) inserts a rolling summary block "right after the frozen prefix" (L598-619). Any change to the last user turn or insertion after the frozen prefix **shifts the token boundary the backend hashes**, defeating reuse. The benchmark's `per_turn_cached` median = 31 tok confirms the backend is re-prefilling almost everything.
   - *Fix:* Append **all** volatile context (anchor, RAG, loop warnings, rolling summary) as a **single trailing user/tool turn**, never into the active prompt and never inside the frozen prefix. Keep system + first N turns verbatim and untouched. This is the single highest-leverage change.

2. **No streaming / incremental response handling for latency.** `app.py` collects the full completion before returning (non-streaming path dominates the benchmark). For a coding agent, first-token latency matters more than total. The proxy adds a full synchronous optimization pass per request (L276-852) on the critical path.
   - *Fix:* Optimize on the request path but stream the response immediately; move heavy passes (compaction, embedding-based ranking) to the async stage that already exists (`async_io_stage`) and is currently only used for compression.

3. **No token-budget enforcement against the real tokenizer** (see §6 bug). The budget is enforced against `cl100k_base` counts, so the "hard token cap" is wrong by the Qwen ratio on every turn.

4. **No differential/structured tool-output compression at the boundary.** headroom/snip/rtk all win by compressing *tool outputs, logs, file reads, RAG blobs* before they enter context. MOEptimizer compresses chat history (semantic dedup, compaction) but the biggest agentic bloat is tool I/O, which is only lightly handled (`_stream_large_tool_outputs`, L685-690, gated behind `proactive_threshold_tokens`).

---

## 2. Design weaknesses

1. **Phantom subsystems (the core design flaw).** Three modules are architected around capabilities the OpenAI API does not grant a client:
   - `static_prefix_kv.py` — `put()` stores `prefix.encode("utf-8")` (L762), i.e. **the prompt text as bytes, not KV tensors**. A "KV-cache" that holds text cannot be loaded into the model; it is just a string memo. The `get()` early-exit (L331-339) skips optimization when the *text* matches, which is at best a no-op fast path and at worst returns an un-compacted context when over budget (guarded, but still semantically wrong — it is not caching KV).
   - `mtp_state.py` — `save_state()` exists (L36) but is **never called** from `optimizer.py`; only `get_state_key()` (L704) and the no-op `align_prediction_boundary()` (L90-101, explicitly a no-op) are used. No real MTP hidden state is ever captured or restored.
   - `expert_cache.py` — `warm_cache_for_static_layer()` stores **placeholder** expert masks derived from language heuristics (`_predict_experts_for_pattern`, L226-245: "Python → experts 0-15", etc.). These are fabricated, not observed. `update_from_model_feedback` (L247) is never called, so the cache never learns real routing. The hints are then **stripped** before sending (`_UNSUPPORTED_BACKEND_EXTRA_BODY_KEYS`, L91-97; `backend_client.py` L40-54) unless `enable_experimental_backend_hints` is on — and even then they are guesses.
   - *Consequence:* README claims of "KV-cache reuse", "MTP state preservation", and "expert routing cache" are not implemented. This is the most important thing to correct before any further investment.

2. **Over-engineered pipeline with ~25 stages** (`optimizer.py` L1-24 documents 14+ numbered steps; `notes.md` lists 20+ modules). Many stages are gated behind the same `proactive_threshold_tokens` check and silently `except`-swallow failures (e.g. L405, L430, L467, L495, L516, L526, L590, L619, L646, L661, L675, L683, L690, L716, L724, L738, L744, L750, L764, L779, L793, L800, L808, L850). The pipeline is hard to reason about, hard to benchmark per-stage, and most stages only fire on over-budget contexts — so on lean contexts the proxy is pure overhead.

3. **Quality anchor is self-referential and grows.** `_inject_quality_anchor` appends an anchor to the last user turn every turn (L886-937) and re-parses/strips prior anchors (L928-932). This both breaks prefix stability (§1.1) and adds tokens the direct run never has — contributing to the verbosity regression.

4. **`hierarchical_summarizer` drops constraints** (review02 finding, confirmed by code path L598-619 + the 2.17x length_ratio). Folding older turns into a summary without retaining the task's "don'ts" makes the model re-derive them verbosely. The fix attempted in code (retain constraints) is partial; the benchmark still shows inflation.

---

## 3. Better alternatives

The web references show the winning pattern for this exact problem is **boundary compression of tool/I/O blobs**, not history rewriting:
- **headroom** (headroomlabs-ai/headroom): a library/proxy/MCP-server that compresses tool outputs, logs, files, and RAG context *before* they reach the LLM (60-95% fewer tokens). Drop-in for agents.
- **snip** (edouard-claude/snip): CLI proxy, YAML filters, 60-90% reduction, Go.
- **rtk** (rtk-ai/rtk): CLI proxy, 60-90% reduction, Rust, zero-dep.
- **opencode** (anomalyco/opencode): the typical client agent this proxy sits in front of.

**Strategic recommendation:** MOEptimizer should become the *OpenAI-protocol* equivalent of headroom — a transparent proxy that (a) anchors the prefix byte-stably, (b) compresses tool/assistant outputs at the boundary using cheap, lossless-ish transforms (truncate long logs, collapse repeated stack frames, keep signatures of code), and (c) gets out of the way of the backend's native prefix cache. This is simpler, faster, and actually improves quality (less hallucinated summarization) than the current 25-stage rewriter.

---

## 4. Throughput improvements

1. **Move optimization off the critical path.** Today every request pays the full `optimize_messages` cost before the first token. Use the existing `async_io_stage` (ThreadPoolExecutor) for all non-essential passes; stream the response as soon as the backend starts emitting.
2. **Cache the optimization result per exact incoming context hash.** `chunk_fingerprint` (L1046) and `cache_registry` exist but the per-turn optimization is recomputed even when the incoming `messages` are identical to last turn (common in agentic loops where only the new user turn differs). Memoize on the incoming-message fingerprint and reuse the optimized form.
3. **Drop the per-turn disk writes.** `static_prefix_kv.save_to_disk()` (L806) and `hierarchical_summarizer.save_to_disk()` (L799) run every turn; `cache_registry` is throttled (L225) but the others are not. These pickle writes on the request path add latency for no per-request benefit.
4. **Avoid redundant token recounts.** `token_counter.count_messages` is called ~15 times per `optimize_messages` (L322, 396, 413, 458, 472, 485, 499, 535, 550, 593, 650, 656, 664, 695, 728, 813-814). Count once, track deltas.

---

## 5. Context-efficiency improvements

1. **Compress tool I/O at the boundary** (§3) — highest ROI, lowest quality risk.
2. **Stop injecting the quality anchor into the prompt** (§2.3); if kept, put it in a trailing tool/metadata turn, not the active user turn.
3. **Make rolling summary append-only at the very end** (after the last user turn), never inside the frozen prefix (§1.1).
4. **Preserve constraints in summaries** (§2.4) — verify with the benchmark's `length_ratio` returning to ~1.0.
5. **Use the backend's real `cached_tokens` as the optimization objective**, not token count alone. The proxy already receives it (`app.py` L217-326, `record_cache_outcome` L437). Train/decide eviction to *maximize* `cached_tokens`, since that is what actually saves prefill.

---

## 6. Bugs (confirmed in code)

1. **Wrong tokenizer everywhere (critical).** `TokenCounter` uses `tiktoken` `cl100k_base` (GPT-4), explicitly "a reasonable approximation" for Qwen (L45-50). Qwen uses a different BPE; the ratio is off by a meaningful factor, so:
   - the token budget (`_budget_tokens`, L182-190) is enforced against the wrong count;
   - `X-Optimized-Prompt-Tokens` (app.py L678) reports GPT-4 tokens, not Qwen tokens;
   - the benchmark's own `_estimate_prompt_tokens` (benchmark.py L659-679) also uses `cl100k_base`, so **the headline 85.79% savings is computed with the wrong tokenizer** and is not directly comparable to the backend's `prompt_tokens`.
   - *Fix:* Use the actual Qwen tokenizer (e.g. `transformers` `Qwen2Tokenizer` or a local tiktoken-style Qwen encoding) for counting, or at minimum calibrate from the backend's reported `prompt_tokens` (the `set_token_calibration` plumbing at L192-211 already exists but is only a clamped [0.5,2.0] fallback and is not wired to `app.py`'s real `cached_tokens`/`prompt_tokens` signal).

2. **`static_prefix_kv` stores text, not KV** (§2.1) — mislabeled; either rename to "static prefix memo" or delete.

3. **`mtp_state.save_state` never called** (§2.1) — dead code; the "MTP state preservation" feature does not exist.

4. **`expert_cache` placeholder masks** (§2.1) — fabricated routing; misleading and stripped before send. Delete or replace with a real, backend-fed signal (none exists via OpenAI API).

5. **`align_prediction_boundary` is a no-op** (L90-101) but is still invoked (L722) and documented as "MTP prompt engineering" — dead/aspirational.

6. **`get_state_key` hashes the last 128 *characters*** (L86-88: `content[-overlap_tokens:]` with `overlap_tokens=128` as chars, not tokens). A char-slice key collides across very different contexts and is not token-aligned; the name says "tokens" but the code uses chars.

7. **Benchmark methodology can mask quality loss.** `execution_order = direct_full_conversation_then_proxy_full_conversation` (benchmark.py L1759-1762, L1782) runs direct first, then proxy — fine for isolation, but the proxy re-uses the *same* `base_messages` and replays the *same* user tasks, so the proxy sees a different model state (the direct run already warmed the backend's KV cache and the model's RNG/draft state). More importantly, `max_tokens=256` (L1758, L2328) caps responses so semantic similarity is measured on truncated answers — a 256-token ceiling hides verbosity/truncation differences and may inflate similarity. Re-run with realistic `max_tokens` for a fair quality grade.

8. **`_resolve_prompt_tokens` can report 0 cached_tokens as a "hit".** `_looks_like_cached_response` (benchmark.py L692-703) returns `True` when `prompt_tokens==0 and completion_tokens>0`; combined with the `estimated_missing_usage` fallback (L719-725), a missing-usage response is scored as cached. This can overstate cache reuse in the report.

---

## 7. Performance bottlenecks

1. **Synchronous 25-stage pipeline on the request path** (§4.1) — dominant latency source; explains +23.6% slower despite 85% fewer tokens.
2. **Per-turn pickle disk writes** (§4.3).
3. **Redundant token counting** (§4.4).
4. **`embedding.py` NPU calls on the critical path** for code-block ranking (`_sync_embed_and_rank`, L1099-1103) — embedding a coding LLM on NPU per request is expensive; should be async/batched.
5. **`ScratchpadCompactor`, `context_compressor`, `hierarchical_summarizer` all run LLM-free regex/structure passes but still per-turn and synchronously**; acceptable individually, costly in aggregate with no memoization.

---

## 8. Memory leaks

1. **`StaticPrefixKVCache` pickles to `~/.moeptimizer/kv_cache.pkl` every turn** (L806) and grows unbounded in entries (`max_entries=64`, L31 — bounded, OK) but the *pickle file* is rewritten fully each time; not a leak but wasteful.
2. **`MTPStateManager._states` bounded at 100 (L26) — OK.** `ExpertRoutingCache` bounded (L66) — OK. `cache_registry` loads from disk and is throttled — OK.
3. **No leak found in the caches themselves**, but `AgentStateStore` (`state_store.py`) accumulates `AgentStep` objects per session and is never trimmed except via compaction markers; long sessions grow unbounded in memory. Verify `session_manager.py` evicts old sessions (it isolates per session, L109-125, but check cleanup on session expiry).
4. **`embedding.py` `_sync_loop` global thread** (noted in prior context) — ensure it is daemonized and joined on shutdown to avoid hang; not a leak but a lifecycle risk.

---

## 9. KV-cache preservation techniques (what actually works)

The backend already preserves KV via prefix caching. The proxy's only valid techniques are:
1. **Freeze the leading prefix verbatim** — system + first user + first N turns, byte-identical every turn (`context_aligner.freeze_static_prefix`, L839-848, is the right idea; ensure nothing else mutates those messages).
2. **Append all volatile context as the final turn** (anchor, RAG, loop warnings, rolling summary) so the hashed prefix never moves.
3. **Never reorder, never re-serialize, never re-tokenize the prefix** — the cache_preservation_guide.md DOs are correct; the code violates them via anchor injection and summary insertion.
4. **Use `chat/completions` (not `/completion`)** and keep `tools` array order/schema stable — already followed.
5. **Echo every `<thinking>` token** — `thinking_preserver` pass-through (L316) is correct.
6. **Report the backend's real `cached_tokens`** and optimize to maximize it (§5.5). The plumbing exists (`app.py` L217-326, `record_cache_outcome`); wire it into eviction decisions.

Delete the fake KV techniques (`static_prefix_kv` text cache, `mtp_state` key-only, `expert_cache` placeholders) — they provide no preservation and mislead.

---

## 10. MTP-preservation techniques

MTP (multi-token prediction) is a **model-internal** decoder optimization. A client-side OpenAI proxy **cannot** read or write MTP hidden states or draft tokens — there is no OpenAI field for it. Therefore:
- The only MTP-relevant thing the proxy can do is **keep the prompt prefix stable** so the decoder's draft cache (if the backend maintains one) stays warm. That is the same as KV preservation (§9).
- `mtp_state.py` and `mtp_speculative.py` (extra_body hints) are **ineffective by construction** unless the backend exposes a native MTP endpoint — `config.py` L254/L282 correctly default them off, but the modules should be removed or clearly marked experimental/non-functional to avoid implying MTP optimization exists.
- If the backend is llama.cpp with `--speculative`, the *correct* integration is a native endpoint call, not OpenAI extra_body guessing. Out of scope for a transparent proxy; document as unsupported.

---

## 11. New UX features

1. **Live cache-reuse dashboard.** ✅ Shipped — `X-Prefix-Cache-Hit-Tokens` is yielded on the streaming path (app.py L424-425) and set as a response header on the non-streaming path (L542), and `GET /v1/metrics` now reports process-wide `cached_tokens`, token savings, cache-hit rate, prefix-cache reuse ratio, and average latency. Operators can now see whether the proxy is helping.
2. **`--explain` / dry-run mode.** Return the optimized prompt alongside the response (or via a header) so users can inspect what the proxy changed — critical for trust given Grade D similarity.
3. **Per-profile presets** beyond `balanced`/`aggressive` (benchmark.py L2342-2346): e.g. `quality` (no summarization, anchor off, only boundary compression) for when similarity must stay Grade A/B.
4. **Config validation feedback.** `config.py` has many toggles; a `moept --check-config` that warns about contradictory settings (e.g. `static_prefix_kv_enabled` while the stored bytes are not KV) would prevent the phantom-feature trap.
5. **Benchmark regression gate.** Add a CI check that fails if `semantic_similarity < 0.82` (Grade B) or `latency_delta > 0` — currently the proxy ships while being both slower and Grade D.
6. **Tokenizer selector.** Let the user point the proxy at the real model tokenizer (path or HF id) instead of hard-coding `cl100k_base` (§6.1).

---

## Prioritized fix list

| # | Priority | Fix | File(s) |
|---|----------|-----|---------|
| 1 | P0 | Re-anchor prefix: move all volatile context (anchor/RAG/loop/summary) to a trailing turn; freeze system+first N turns verbatim | `optimizer.py` L540-619, L839-848, L886-937 |
| 2 | P0 | Replace `cl100k_base` with real Qwen tokenizer; wire `app.py` `prompt_tokens` into `set_token_calibration` | `token_counter.py`, `app.py` L678, `optimizer.py` L192-211 |
| 3 | P0 | Delete/quarantine phantom subsystems: `static_prefix_kv` (text≠KV), `mtp_state` (key-only), `expert_cache` (placeholders) — or relabel honestly | `static_prefix_kv.py`, `mtp_state.py`, `expert_cache.py`, `optimizer.py` L130/143/165/331/443/704/758 |
| 4 | P1 | Move heavy/optional passes to `async_io_stage`; stream responses | `optimizer.py`, `app.py` |
| 5 | P1 | Boundary compression for tool I/O (headroom-style) | new module / `optimizer.py` L685-690 |
| 6 | P1 | Optimize eviction to maximize backend `cached_tokens` | `optimizer.py` L437, `app.py` L326 |
| 7 | P2 | Memoize optimization per incoming-context fingerprint; count tokens once | `optimizer.py`, `chunk_fingerprint.py` |
| 8 | P2 | Remove per-turn pickle disk writes on request path | `optimizer.py` L799/806 |
| 9 | P2 | Re-run benchmark with realistic `max_tokens`; fix `_looks_like_cached_response` false-hit | `benchmark.py` L692-703, L1758 |
| 10 | P3 | UX: metrics endpoint, dry-run/explain, quality profile, config check, regression gate (✅) | `app.py`, `benchmark.py`, `config.py`, `config_check.py` |

---

## What is genuinely good (keep)

- `thinking_preserver` pass-through (L316) — correct, preserves `<thinking>`.
- `cache_preservation_guide.md` DOs — sound principles; the code just doesn't follow them.
- `backend_client.py` stripping of unsupported extra_body (L40-54) — correct defensive design.
- `context_aligner.freeze_static_prefix` concept (L839-848) — right idea, undermined by other mutations.
- Per-session isolation in `session_manager.py` — sound.
- The benchmark harness is thorough (many quality metrics); only the tokenizer and `max_tokens` methodology need fixing.
