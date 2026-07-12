I've verified the key claims against the actual code. Here is the senior-architect review.

---

# MOEptimizer — Senior Architect Review

**Scope:** `src/moeptimizer/` (transparent OpenAI proxy in front of Lemonade/Qwen3.6-35B-A3B-MTP).
**Evidence base:** source read of `optimizer.py`, `app.py`, `backend_client.py`, `async_io_stage.py`, `hit_prediction_model.py`, `template_selector.py`, `config.py`, `prompt_templates.py`; benchmark `scripts/benchmark_refactor_long_30_12.json` (30-turn `refactor_long`).

**Headline numbers (30-turn refactor_long):**
- Token savings: **84.79%** (direct 568,121 → proxy 86,439 prompt tokens)
- `total_cached_tokens`: **2,998** of ~654k prompt tokens → **~0.46%** reused
- `per_turn_cached` **median 0.0** → the proxy achieves **~0% real prefix-cache reuse**, despite "KV-cache preservation" being its stated mission
- Latency: proxy **44,839 ms** mean vs direct **26,559 ms** → **+18.3 s / ~68% slower**
- Semantic similarity: **0.7589** (Grade C)

The proxy is excellent at *compression* but fails at its *second* job (cache preservation) and is a net latency loss. Below is the breakdown.

---

## 1. Missing optimizations

- **Byte-stable prefix anchor / cache breakpoints.** There is no concept of a frozen, never-mutated prefix. Every turn the whole list is re-serialized and re-transformed, so the prefix before the new user turn is almost never byte-identical to the previous turn. OpenAI-style automatic prefix caching needs byte-identical prefixes from the start — the proxy never guarantees that.
- **Token-accurate budget.** `_budget_tokens()` (optimizer.py:170) derives the budget from `max_optimized_chars // 4` (a 4-char/token heuristic). `token_counter.count_messages` is called repeatedly but the *hard cap* is enforced against a char-based estimate, not real tokens. On a MoE where prefill cost is the bottleneck, a 4-char heuristic is too loose.
- **Tiered eviction (summarize, don't drop).** When over budget, the proxy *drops* whole turns (`_evict_for_budget`, optimizer.py:1186). Middle history is discarded, not summarized. `hierarchical_summary_enabled` exists (config.py:247) but is **default False** and explicitly disabled "because middle-history summaries break contiguous KV-cache prefixes" — so the only lever is deletion.
- **Streaming proxy path is not actually optimized.** `app.py:546` forces `stream=True` by default, but the optimizer runs fully synchronously before the first token, so TTFT includes the entire pipeline. There is no incremental/streaming optimization.
- **Provider prefix-cache accounting.** The proxy never reads `usage.prompt_tokens_details.cached_tokens` to *adapt*; it only logs a warning when `prompt_tokens == 0` (backend_client.py:291). It cannot tell whether its own transformations are preserving the cache.
- **MoE/MTP actually wired to the backend.** All MoE/MTP machinery is computed and then stripped before send (see §2/§8). The backend never sees a single hint.

## 2. Design weaknesses

- **The MoE/MTP layer is pure overhead, never delivered.** `expert_cache`, `mtp_head_checkpoint`, `mtp_state`, `kv_cache_warmup`, `kv_slot_tracker`, `segment_wise_speculative` are all instantiated (optimizer.py:133–168) and several *run* during the pipeline, but their output never reaches the model:
  - `get_backend_extra_body` (optimizer.py:1306) strips every MoE key via `_UNSUPPORTED_BACKEND_EXTRA_BODY_KEYS`.
  - `enable_experimental_backend_hints` is **default False** (config.py:276).
  - `backend_client._strip_unsupported_extra_body` (backend_client.py:39) strips the same keys a second time.
  - `SpeculativeDecoder` is never enabled — `enable_speculative_decoding` (backend_client.py:176) is never called, and even if it were, its `mtp_body` is stripped at line 279.
  → These modules burn CPU every request to produce data that is thrown away.
- **`async_io` is dead code.** `async_io_enabled` is **default True** (config.py:282), so a `ThreadPoolExecutor` is created (async_io_stage.py:48), but `self.async_io` is **never invoked anywhere in the pipeline** (grep: only import + init). Tree-sitter compression and embedding run synchronously on the request thread. The async stage adds construction cost and zero benefit.
- **`hit_prediction` is trained on a constant label.** `record_outcome(optimized, hit=True)` (optimizer.py:693) *always* records `hit=True` ("We got here, so request succeeded"). The XGBoost model (hit_prediction_model.py:153) therefore trains on 100% positive labels and learns nothing about real cache hits. Its `should_early_exit` gate is saved only because it is also gated by `total_tokens <= proactive_threshold_tokens`.
- **`template_selector` is a no-op for output.** `select_template` is called (optimizer.py:405) but its result only sets `self._task_type` (a label). The actual templating is re-done independently by `classify_and_template` (optimizer.py:403 → prompt_templates.py:176), which re-classifies and re-applies. The selector's "learning" is recorded against a **fake** semantic-similarity metric: `1.0 if optimized_tokens <= original_tokens else original_tokens/ax(...)` (optimizer.py:740) — a token ratio, not similarity. So it optimizes toward token savings, not quality.
- **Front-eviction conflicts with prefix stability.** `_evict_for_budget` drops complete turns from the *front* of the evictable body (optimizer.py:1218). This keeps `system + first user` stable but changes the set of middle turns every time eviction triggers, which invalidates the cached prefix for everything after the first user. The design oscillates between "keep lean" (evict) and "keep prefix stable" (don't evict) and satisfies neither well.
- **RAG/anchor injected into the last user turn is correct but volatile.** `_inject_quality_anchor` (optimizer.py:801) and RAG (optimizer.py:487) append to the *last* user turn. That's the right place for stability, but the anchor is **rebuilt every turn** from all user messages (optimizer.py:854), so the last user turn is never byte-identical across turns — and it's the turn that grows, guaranteeing the tail is always a cache miss.

## 3. Better alternatives

- **Headroom-style CacheAligner.** Freeze a byte-stable prefix (system + first user + a fixed early-context block) and move *all* volatile content (RAG, anchor, new prompt) to the **tail**. Never mutate the middle. This is the opposite of front-eviction: keep the prefix immutable, compress/summarize only the tail. The proxy's own `cache_preservation_guide.md` already states this DO ("inject volatile data only in last user turn", "never mutate mid-sequence") but the pipeline doesn't enforce an immutable prefix block.
- **Explicit cache breakpoints (vLLM/llama.cpp `cache_control` / `prefix_cache`).** If the backend supports it, mark the stable prefix as a cache breakpoint instead of relying on byte-identity. Currently disabled by default and stripped anyway.
- **Tiered compaction instead of deletion.** Summarize evicted turns into a rolling summary block appended *after* the immutable prefix (not in the middle), preserving both leanness and prefix stability. This is the standard fix the proxy explicitly declined.
- **Don't re-tokenize embeddings per turn.** Embeddings for unchanged code chunks should be cached across turns (a `chunk_fingerprint` cache exists for code, optimizer.py:947, but RAG/state embeddings are recomputed).

## 4. Throughput (make the proxy not slower than direct)

- **Wire `async_io` for real, or delete it.** Offload tree-sitter compression (`context_compressor.compress`, optimizer.py:370) and embedding (`_sync_embed_and_rank`, optimizer.py:1035) to the thread pool / async tasks. Today they block the request thread, which is the main reason proxy TTFT (44.8 s) ≫ direct (26.6 s).
- **Batch embeddings.** `run_batch_embeddings` exists (async_io_stage.py:203) but is unused. Embed all chunks in one batch instead of per-call.
- **Cache embeddings across turns** so unchanged context isn't re-embedded every request.
- **Drop the unused MoE modules** (§2) — they run for nothing and add measurable CPU before the first token.
- **Parallelize CPU-bound stages** that are currently sequential (canonicalize → compress → template → compact → dedupe → order → incremental-update → RAG → align → trim → entropy → sliding-window). Several are independent and could run concurrently.

## 5. Context-efficiency

- **Relevance-ranked RAG, not whole-graph injection.** `state_rag.get_context_for_step` (optimizer.py:512) returns a context blob appended wholesale. Rank by relevance and cap size; inject only the top-K.
- **Delta-only tool outputs.** Large tool outputs are streamed (optimizer.py:614) but not diffed; `delta_encoder` exists (optimizer.py:701) but only snapshots code, never sends deltas to the model.
- **Tighter budget.** Replace the 4-char/token heuristic with real token counts for the hard cap, and lower `proactive_trim_ratio` once prefix stability is fixed (today aggressive trimming fights cache preservation).
- **Keep code exact; summarize prose.** The proxy already prefers exact code (optimizer.py:568, 1009). Extend that philosophy to summaries of old turns rather than dropping them.

## 6. MTP-preservation techniques

- **Freeze the first N turns byte-stable** and never re-serialize them. MTP heads predict on contiguous token sequences; any byte change in the prefix forces a full MTP re-warm.
- **Stable tool-schema ordering.** The guide (DO) requires stable tool schemas; verify the optimizer never reorders `tools` or `tool_calls` between turns.
- **Avoid per-turn message-count changes in the prefix.** Front-eviction changes message count in the middle; prefer tail-only mutation so the prefix length is constant.
- **Don't pre-seed reasoning** (`reasoning_preseed_enabled`, optimizer.py:560) — it injects synthetic tokens into the prefix that differ from the direct request, breaking MTP alignment with the reference distribution.

## 7. KV-cache preservation techniques

- **Immutable prefix anchor** (system + first user + frozen early block) serialized once and reused verbatim.
- **Inject volatile data only in the last user turn** (guide DO) — already done, but the anchor must be *stable per turn* (append-only, not rebuilt from scratch).
- **Echo `reasoning_content`** from assistant turns verbatim (guide DO) so the model's own reasoning is preserved as part of the stable prefix rather than dropped.
- **Measure, don't assume.** Read `usage.prompt_tokens_details.cached_tokens` and feed it back into `hit_prediction` as the *real* label (replacing the constant `hit=True`).

## 8. Bugs

- **MTP speculative body is built then stripped.** `SpeculativeDecoder._mtp_speculative_generate` (backend_client.py:128) builds `mtp_body` and merges it, but `_send_chat_completions_request` (backend_client.py:279) calls `_strip_unsupported_extra_body`, deleting `speculative_decoding`/`mtp_heads`/`head_temperatures`. Net effect: no speculative decoding, plus wasted compute. (Also never enabled — see §2.)
- **`hit_prediction` constant-label bug** (optimizer.py:693) — trains on `hit=True` always.
- **`template_selector` selection discarded** (optimizer.py:404–407) — `select_template` result only sets a label; `classify_and_template` re-does it.
- **`static_ratio` feature bug** (hit_prediction_model.py:116–122): counts `system` + first `user` as static then `break`, but the `elif` condition `static_end == 0` means for a no-system prompt it also breaks after the first user — fine — yet for multi-user prompts it double-counts oddly. Minor, but the feature is meaningless given the constant label anyway.
- **`get_backend_extra_body` ignores `messages`** (optimizer.py:1317 `del messages`) — it can't make prefix-stable decisions because it has no message context; it's a pure pass-through filter.

## 9. Bottlenecks

- **Synchronous tree-sitter + embedding on the request thread** (optimizer.py:370, 1035) — primary TTFT cost; this is why proxy is 68% slower.
- **Repeated `token_counter.count_messages` calls** — counted ~10+ times per pipeline (optimizer.py:279, 348, 365, 401, 419, 446, 482, 540, 577, 584, 592, 622, 653, 732, 733). Each is a full walk; cache it once per stage boundary.
- **Unused MoE modules running** (§2) add CPU before any token is produced.
- **`async_io` executor construction** on every optimizer init (config default True) with no usage.

## 10. Memory leaks

- **`AgentStateStore` grows unbounded per session.** `_ingest_messages` (optimizer.py:911) appends a step per message and de-dupes by fingerprint, but archived/evicted turns are never pruned from the store; the store retains the full original history even after the *optimized* list drops them. Over a long agentic session this is a steady leak.
- **`hit_prediction._history`** is a `deque(maxlen=200)` — bounded, OK. But `_load_model` (hit_prediction_model.py:341) marks `_trained=True` from disk without restoring weights, so on restart it believes it's trained but has no model → falls back to heuristic; harmless but misleading.
- **`cache_registry` persisted every turn** (`_register_context` → `save_to_disk`, optimizer.py:182) — disk I/O every request, and the on-disk registry grows; verify it's bounded.
- **`static_prefix_kv` stores `optimized[:3]` JSON every turn** (optimizer.py:682) and `save_to_disk` every turn (optimizer.py:723) — unbounded growth + per-turn disk write.
- **`delta_encoder` snapshots** bounded by config (100) — OK. `mtp_head_checkpoint` bounded (256) — OK.

## 11. UX features

- **Expose real cache-hit rate in response headers.** Today only `X-Optimized-Prompt-Tokens` (app.py:526) is returned. Add `X-Prefix-Cache-Hit-Tokens` / `X-Cache-Reuse-Pct` from `usage.prompt_tokens_details` so users can see whether the proxy is actually preserving the cache (currently it isn't).
- **A "passthrough mode" toggle** to disable optimization per-request (e.g. `extra_body={"moeptimizer": {"passthrough": true}}`) for debugging quality regressions — useful given Grade C similarity.
- **Per-session optimization stats endpoint** (`/v1/debug/{session_id}`) showing token savings, cache reuse, pipeline stage timings — the pipeline logs timings only at INFO, not per-stage.
- **Surface the latency penalty honestly.** The README claims "proxy is faster" (notes.md: latency -1.9%), but the 30-turn benchmark shows +68%. The UX/docs should state the trade-off (big token savings, slower TTFT) rather than implying a speed win.
- **Graceful degradation already exists** (app.py:489 fallback to recent-turn context) — good; consider exposing `X-Optimization-Error` already returned (app.py:541) in a more visible way.

---

## Priority fixes (highest ROI)

1. **Make the proxy faster than direct, or stop claiming it is:** offload tree-sitter + embedding to the existing (but unused) `async_io` stage, or delete it. *(§4, §9)*
2. **Actually preserve the prefix cache:** implement an immutable prefix anchor + tail-only mutation (headroom/CacheAligner), and feed real `cached_tokens` back as the `hit_prediction` label. *(§1, §2, §7)*
3. **Stop computing MoE/MTP data that is thrown away:** gate those modules behind a real backend-capability check, or remove them until the backend supports the hints. *(§2, §8)*
4. **Fix the two no-op "learning" components** (`hit_prediction` constant label, `template_selector` discarded result) or remove them — they add latency and give a false sense of adaptation. *(§2, §8)*
5. **Bound `AgentStateStore` and per-turn disk writes** (`cache_registry`, `static_prefix_kv` save every turn). *(§10)*

Net: the compression engine is strong (84.8% token savings), but the cache-preservation mission is unmet (0% real reuse) and the proxy is a latency regression. The fastest wins are (a) using or deleting `async_io`, and (b) an immutable-prefix design that lets the backend's own prefix cache do the heavy lifting.

---

## Implementation Status (2026-07-12)

The following priority fixes from §11 / "Priority fixes" have been implemented across `v0.5.2` and `v0.5.3` (verified against source; full test suite: **356 passed, 3 skipped**).

| # | Priority fix | Status | What changed |
|---|---|---|---|
| 1 | Use or delete `async_io` | **Done (wired for real)** | `async_io_enabled` default `True`→`False` in v0.5.2, then reverted to `True` in v0.5.3 once the pipeline actually invokes it. `optimizer.py` now offloads tree-sitter compression (`run_sync_stage(self.context_compressor.compress, ...)`, Step 5.7) and embedding ranking (`run_sync_stage(self._embed_and_rank_impl, ...)`) to the thread pool. |
| 2 | Preserve prefix cache + real `cached_tokens` label | **Done** | `app.py` reads `usage.prompt_tokens_details.cached_tokens` and (a) calls `optimizer.record_cache_outcome(cached_tokens)` and (b) emits the `X-Prefix-Cache-Hit-Tokens` header in both paths. `hit_prediction` trains on this real signal. `ContextAligner.freeze_static_prefix` now freezes **only the system prompt** verbatim at the end of the pipeline (Step 14.11) so the backend's automatic prefix cache can reuse it; the first user message is not frozen (it is deterministically compressed and stays stable on its own). Gated by `immutable_prefix_enabled` (default `true`). |
| 3 | Stop computing thrown-away MoE/MTP data | **Done (gated)** | `expert_cache.warm_cache_for_static_layer`, `kv_slot_tracker.build_slot_map`, and `mtp_state_manager.get_state_key` run only when `enable_experimental_backend_hints` is `True` (default `False`). |
| 4 | Fix no-op "learning" components | **Done (disabled by default)** | `template_selector_enabled` default `True`→`False` (config.py:237); dead `select_template` call and fake `record_quality` block removed. `hit_prediction` constant-label bug fixed via `record_cache_outcome`. |
| 5 | Bound `AgentStateStore` + per-turn disk writes | **Done** | `StaticPrefixKVCache.put` stores the stable prefix content (not a timestamped blob), so repeated identical prefixes skip the per-turn pickle rewrite via `_last_context_changed`. `AgentStateStore` now prunes oldest archived steps beyond `max_state_steps` (default 200) via `_prune_if_needed`/`_rebuild_indices` (state_store.py). `cache_registry` already gates writes on new keys only. |

### Config defaults changed
- `MOEPT_V050__TEMPLATE_SELECTOR_ENABLED`: `true` → `false`
- `MOEPT_V050__ASYNC_IO_ENABLED`: `true` → `false` (v0.5.2), then `false` → `true` (v0.5.3, now used)
- `MOEPT_V050__ENABLE_EXPERIMENTAL_BACKEND_HINTS`: remains `false` (MoE/MTP hints still stripped before send)
- `MOEPT_AGENTIC__IMMUTABLE_PREFIX_ENABLED`: new, default `true`
- `MOEPT_AGENTIC__MAX_STATE_STEPS`: new, default `200`

### Honest latency trade-off (documented)
The proxy is a *token-reduction* proxy, not a speed win. On the 30-turn `refactor_long` benchmark the proxy is **~68% slower** (44,839 ms vs 26,559 ms direct) despite **84.8% token savings** (0.7589 similarity, Grade C). README "Priority Fixes" now states this trade-off explicitly.

### Remaining work (not started)
- None of the original five priority fixes remain open. Future hardening: verify real prefix-cache reuse end-to-end against a live backend (the 30-turn benchmark still showed ~0% real reuse — the immutable-prefix freeze is necessary but not sufficient; the backend must actually honor the stable system prefix).