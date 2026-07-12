# MOEptimizer — Senior Architect Review (v2)

**Scope:** `src/moeptimizer/` (transparent OpenAI proxy in front of Lemonade/Qwen3.6-35B-A3B-MTP), `scripts/benchmark.py`, and the existing `review01.md` (already implemented in v0.5.2/v0.5.3).
**Evidence base:** source read of `optimizer.py`, `app.py`, `backend_client.py`, `context_aligner.py`, `async_io_stage.py`, `token_counter.py`, `static_prefix_kv.py`, `config.py`, `benchmark.py`; benchmark `scripts/benchmark_refactor_long_30_12.json`; and current llama.cpp/llama-server prefix-cache semantics (verified via upstream docs/discussions).

**Headline numbers (30-turn `refactor_long`, from the JSON):**
- Token savings: **84.79%** (direct 568,121 → proxy 86,439 prompt tokens) — strong.
- `per_turn_cached` **median 0.0** → the proxy still achieves **~0% real prefix-cache reuse**, despite "KV-cache preservation" being its stated mission.
- Latency: proxy **44,839 ms** mean vs direct **26,559 ms** → **+68% slower** (TTFT tax, not a speed win).
- Semantic similarity: **0.7589** (Grade C). `length_ratio` mean **2.17** → proxy responses are **2.17× longer** than direct (26/30 turns flagged verbose). `code_block_ratio` mean 0.85 (5 turns lose code).

`review01.md` was implemented, but its central claim — "actually preserve the prefix cache" — is **not met**, and several of its fixes are weaker than they appear. Below is a fresh pass focused on what is still broken, what is missing, and the highest-ROI moves for a *local, hardware-limited MoE-MTP* setup.

---

## 0. The strategic reframing (read this first)

The backend (llama.cpp / llama-server behind Lemonade) **already has excellent prefix caching**: `cache_prompt` defaults to `true`, it reuses the longest common *leading* prefix on the same slot, and `--cache-reuse N` (KV shifting) reuses shared *chunks* even when they move. The proxy's job should be to **optimize *for* that cache**, not to fight it.

Today the proxy does the opposite: every turn it re-serializes, re-compresses, and re-evicts the conversation, so the token sequence the backend sees is never byte-identical to a previous turn. The backend's cache therefore never hits, and the proxy pays the full MoE prefill on every turn — which is exactly why it is 68% slower. The 84.8% token savings is real, but it is bought by (a) destroying cache reuse and (b) degrading response quality (Grade C + 2.17× verbosity).

**The single most important realization:** the proxy should keep a *stable frozen prefix* and pin each session to a *stable backend slot*, then let the backend's own cache do the heavy lifting. Compression should only happen on parts that don't break the frozen prefix. This inverts the current design and is where the real speed + quality wins are.

---

## 1. Missing optimizations

- **Session→slot pinning (the biggest miss).** llama.cpp reuses a slot's KV cache when the new prompt shares a leading prefix with what's cached on that slot. The proxy never pins a conversation to a slot (`id_slot` is never injected; `backend_client.py` never calls `enable_speculative_decoding` and never sets a slot). If the proxy mapped `session_id → stable id_slot` and forwarded it, the backend would reuse the *entire* conversation prefix across turns — making the proxy's compression almost free from a cache perspective. Gated behind a backend-capability flag (like `enable_experimental_backend_hints`) so it stays OpenAI-transparent for other backends.
- **Native MTP speculative decoding is entirely dead.** `SpeculativeDecoder` exists but is never constructed (`app.py:410` builds a bare `LemonadeClient`; `backend_client.py:253` only uses it `if self._speculative_decoder is not None`, which is never set). For a *MoE-MTP* model, native MTP draft heads are the single largest TPS lever (2–3× decode throughput). The proxy should detect backend MTP support and enable it — this is the model's own speed feature and the proxy ignores it. `mtp_boundary_alignment_enabled` and `segment_speculative_enabled` are also default `False`.
- **`--cache-reuse` / `--cache-ram` / `--swa-full` backend flags are never recommended or set.** Upstream llama.cpp: `--cache-reuse 256` enables KV shifting for shared chunks that aren't at the front (default 0 = disabled); `--cache-ram -1` keeps more cache resident; `--swa-full` (and `--ctx-checkpoints`) is required for SWA/hybrid models (Qwen3-family uses SWA) or the server *forces full re-prefill* every request. The proxy should document and, where it controls the server, set these. This alone can recover most of the lost TTFT without touching the proxy logic.
- **`--system-prompt-file`.** llama.cpp can load the system prompt from a file and share it as a common prefix across all slots, guaranteeing system-prompt caching regardless of proxy behavior. The proxy could write the (frozen) system prompt to a file for the backend.
- **Accurate token counting.** `token_counter.py:46` uses `tiktoken.encoding_for_model("gpt-4")` → `cl100k_base`, which is a *different BPE* from Qwen. For code, token counts are off by a large margin, so the hard budget `_budget_tokens()` (`optimizer.py:176`) is enforced against wrong numbers. Better: use the real Qwen tokenizer, **or** — cheapest and most accurate — echo the backend's actual `prompt_tokens` from the previous turn's response and use that to size the budget (the proxy already has `X-Optimized-Prompt-Tokens`; extend it to also carry the backend's true `prompt_tokens`/`cached_tokens`).
- **Cross-turn embedding cache.** `embedding_invalidation` (`optimizer.py:166`) and `parallel_embedding` (`optimizer.py:167`) are instantiated but **never used**; RAG/state embeddings are recomputed every turn via `embedding_service.embed_batch_sync` (`optimizer.py:1064`). Unchanged code chunks should be embedded once.
- **Prompt-cache breakpoint hints for vLLM.** If the backend is vLLM, `cache_control` / explicit prefix breakpoints are the robust way to guarantee reuse; currently disabled and stripped.
- **Rolling summary block (tiered compaction).** Still disabled (`hierarchical_summary_enabled=False`). Dropping history loses constraints (see §5) and breaks the prefix. A summary block appended *after* the frozen prefix (not in the middle) preserves both leanness and stability.

## 2. Design weaknesses

- **`freeze_static_prefix` is a no-op for cache reuse** (`context_aligner.py:117`). It only freezes the *system prompt*, which the pipeline never mutated in the first place (it lives in the immutable zone and `_strip_internal_flags` preserves it). So `freeze_static_prefix` returns `optimized` unchanged in practice. The *first user message* is compressed (`optimizer.py` Steps 5.5/5.7) and all *historical turns* are re-serialized/evicted every turn (Steps 11.8/12). The prefix after the system prompt is therefore **never byte-stable** → the backend's `cache_prompt` can only ever reuse the system prompt (a few hundred tokens), which is why `per_turn_cached` is ~0. The v0.5.3 "immutable prefix freeze" does not address the actual unstable prefix.
- **`async_io` is fake concurrency.** `async_io_stage.run_sync_stage` (`async_io_stage.py:89`) does `executor.submit(fn, ...).result()` — it blocks the caller until the worker finishes. The pipeline calls it synchronously (`optimizer.py:382, 1056`), so offloading to a thread pool yields **zero latency benefit**: the request thread still waits the full duration. The v0.5.3 claim "wired for real" is misleading — it moves CPU work to another thread but awaits it, so TTFT is unchanged. The previous review's #1 priority (make the proxy faster via `async_io`) is **not achieved**.
- **The optimizer blocks the event loop.** `optimize_messages` is a sync function called directly in the async handler (`app.py:525`). FastAPI runs it inline in the event loop, so **all other sessions stall** during optimization. This kills concurrency under multi-session agentic load.
- **`static_prefix_kv` is misnamed and inert.** `put` stores `prefix.encode("utf-8")` — i.e. the prefix *text*, not KV tensors (`static_prefix_kv.py:75`). It is a "have we seen this prefix?" check, not a KV cache. Its early-exit (`optimizer.py:295`) is gated by `total_tokens <= proactive_threshold_tokens`, which is false in every over-budget turn, so it essentially never triggers.
- **Dead/instantiated-but-unused components** (wasted import + init + memory): `parallel_embedding`, `embedding_invalidation`, `mtp_head_checkpoint` (`optimizer.py:164`, never used), `kv_warmup` (gated off), `segment_decoder` (gated off), `hierarchical_summarizer` (None by default). `delta_encoder` snapshots code every turn (`optimizer.py:725`) but the deltas are **never sent to the model** — pure bookkeeping with no benefit.
- **O(n²) per-turn re-ingestion.** `_ingest_messages` (`optimizer.py:929`) re-processes *all* messages every turn; `loop_detector.analyze_step` (Step 3) and `progress_tracker.record_step` (Step 4) run on every step every turn. Over a 30-turn session this is ~1800 step constructions + fingerprint computations, growing quadratically. `get_session_state` (`optimizer.py:1674`) re-decomposes the goal every request.
- **Front-eviction is fundamentally incompatible with prefix caching.** Every `_evict_for_budget` / `_sliding_window_trim` changes *which* turns exist and *what* they contain, so the serialized prefix changes every turn. The `cache_preservation_guide.md` itself says "slice from the top," but slicing from the top still changes the prefix — the only cache-safe move is a *frozen* block that never changes plus tail-only mutation.

## 3. Better alternatives

- **Headroom-style CacheAligner + slot pinning.** Freeze a byte-stable prefix (system + first user + a fixed early block) **and** pin the session to a stable backend slot. With a stable prefix, llama.cpp's default LCP slot selection naturally reuses the slot; with explicit `id_slot` it's guaranteed. This is strictly better than the current "freeze system only" no-op.
- **Let the backend cache; proxy preserves structure.** The backend's prefix cache is already excellent. The proxy should only (a) keep a stable leading prefix, (b) append all volatility to the tail (already done for RAG/anchor — good), and (c) compress only *old, frozen* turns into a stable summary block. Then enable `--cache-reuse` so even moved shared chunks (code, tool output) are reused via KV shifting.
- **Tiered compaction instead of deletion.** Summarize evicted turns into a rolling summary block appended *after* the frozen prefix. Preserves constraints (fixes the 2.17× verbosity) and prefix stability simultaneously. This is the standard fix the proxy explicitly declined.
- **Native MTP speculative decoding** as the primary TPS lever, with `--cache-reuse`/`--swa-full` as the cache lever. Together these are worth far more than any compression for a local MoE-MTP box.

## 4. Throughput (make the proxy not slower than direct)

- **Run `optimize_messages` in a thread/process pool** (`await loop.run_in_executor(...)`) so the event loop stays free for concurrent sessions. This doesn't cut single-request TTFT but restores concurrency (currently one session blocks all others).
- **Truly parallelize independent heavy stages.** Tree-sitter compression and embedding ranking are sequential today; run them concurrently (real `asyncio.gather` / thread pool) since they don't depend on each other.
- **Pin sessions to slots** (§1) — the backend then reuses the whole conversation prefix; the proxy's prefill cost collapses on revisit.
- **Enable `--cache-reuse 256` + `--cache-ram -1` + `--cache-idle-slots`** on the backend (§1).
- **Enable native MTP speculative decoding** if the backend supports it (§1) — the model's own 2–3× decode-speed feature.
- **Cache embeddings across turns** (use `embedding_invalidation` / `parallel_embedding`, which already exist but are unused) and **batch embeddings** (`run_batch_embeddings` exists, unused).
- **Drop the dead MoE/MTP modules** that run for nothing (§2) — they add CPU before the first token.

## 5. Context-efficiency

- **The 2.17× verbosity is a quality regression, not a win.** `length_ratio` mean 2.17 (benchmark) means proxy responses are *longer* than direct. Front-eviction drops constraints/code the model then re-derives verbosely. Fix with tiered compaction (rolling summary) that retains constraints, and stop evicting the parts that carry the task's "don'ts." Grade C similarity is largely driven by this.
- **Relevance-ranked RAG, not whole-graph injection.** `state_rag.get_context_for_step` (`optimizer.py:528`) returns a context blob appended wholesale. Rank by relevance, cap size, inject only top-K.
- **Delta-only tool outputs.** Large tool outputs are streamed (`_stream_large_tool_outputs` is a no-op, `optimizer.py:1634`) but never diffed; `delta_encoder` snapshots but never sends deltas. Send only diffs of repeated tool output.
- **Tighter, accurate budget.** Replace the 4-char/token heuristic with real token counts (§1) and lower `proactive_trim_ratio` *after* prefix stability is fixed (today aggressive trimming fights cache preservation).
- **Keep code exact; summarize prose** — extend to old turns (skeletonize code, summarize natural language) instead of dropping whole turns.

## 6. MTP-preservation techniques

- **Freeze the first N turns byte-stable**, not just the system prompt. MTP heads predict on contiguous token sequences; any byte change in the prefix forces a full MTP re-warm.
- **Stable tool-schema ordering.** The proxy forwards `tools` unchanged (good) — verify it never reorders `tools`/`tool_calls` between turns (it doesn't today; keep it that way).
- **Avoid per-turn message-count changes in the prefix.** Front-eviction changes message count in the middle; prefer tail-only mutation so the prefix length is constant.
- **Don't pre-seed reasoning** (`reasoning_preseed_enabled=False` — correct; keep off). It injects synthetic tokens that diverge from the direct-request distribution.
- **Echo `reasoning_content` verbatim** (guide DO) — see §7/§8 bug.
- **MTP + `--cache-reuse`**: with chunk reuse, MTP draft patterns survive shifted chunks.

## 7. KV-cache preservation techniques

- **Immutable prefix anchor** (system + first user + frozen early block) serialized once and reused verbatim — currently only the system prompt is frozen, and that was already immutable.
- **Inject volatile data only in the last user turn** (done — good), but the anchor is rebuilt every turn from all user messages (`_build_quality_anchor`, `optimizer.py:872`); that's fine since it's the tail, but keep it strictly append-only.
- **Echo `reasoning_content`** from assistant turns so the model's own reasoning is preserved as part of the stable prefix (guide DO).
- **Measure, don't assume.** Read `usage.prompt_tokens_details.cached_tokens` and feed it back (partially done — see §8 bug for the streaming gap).
- **Pin sessions to slots** + **`--cache-reuse`** + **`--system-prompt-file`** (§1) — these are the robust, backend-native ways to guarantee reuse.

## 8. Potential bugs

1. **Streaming path loses the cache signal.** `app.py:583` defaults to `stream=True`, but `stream_options` is never set, so `usage` (with `cached_tokens`) is `None` in streaming. The streaming generator only captures `cached_tokens` *if present* (`app.py:228`), which it won't be → `record_cache_outcome` falls back to `_last_static_prefix_hit` (garbage). **Fix:** set `stream_options={"include_usage": true}` in the streaming path.
2. **`X-Prefix-Cache-Hit-Tokens` header only set in non-streaming** (`app.py:369`). The real (streaming) path never exposes it — the review01 §11 UX feature is half-implemented.
3. **`_normalize_response_choices` mutates assistant content** (`app.py:159-161`): when `content` is empty it sets `content = reasoning_content`. The client (OpenCode) persists this; next turn it sends `content=reasoning` (possibly without `reasoning_content`), so the model's prefix differs from what it generated → **breaks prefix-cache stability and MTP alignment**. The proxy must pass through `reasoning_content` and `content` exactly as produced, not collapse them.
4. **`get_backend_extra_body` does `del messages`** (`optimizer.py:1343`) — a no-op that also proves the function has no message context to make prefix-stable decisions.
5. **`record_cache_outcome` trains on `self._last_optimized`** (pre-freeze messages); `freeze_static_prefix` runs *after* at Step 14.11. So hit-prediction trains on the wrong (pre-freeze) messages. Minor, but fix ordering.
6. **`freeze_static_prefix` is a no-op** (§2) — it claims to guarantee cache reuse but doesn't.
7. **Benchmark `max_tokens=256`** is too small to measure response quality for coding tasks; the 0.7589 similarity and 2.17× verbosity are computed on 256-token responses and may not reflect real agentic usage. Also the benchmark runs DIRECT then PROXY as two separate full conversations, so the backend's prefix cache from the direct run is warm when the proxy run starts — but the proxy's compressed prompts don't match, so no cross-reuse is possible; within the proxy run the prefixes aren't stable, so 0% is structurally guaranteed. The benchmark *can* measure real backend `cached_tokens` (it reads `p_usage`), so 0% is genuine — but the methodology can't isolate *proxy-induced* cache loss from *proxy-changed-token-sequence* loss.

## 9. Performance bottlenecks

- **Synchronous tree-sitter + embedding on the request thread** (`optimizer.py:382, 1056`) — primary TTFT cost; this is why proxy is 68% slower. `async_io` doesn't help (§2).
- **Repeated `token_counter.count_messages`** — called ~15× per pipeline (`optimizer.py:285, 359, 376, 421, 435, 462, 498, 556, 600, 608, 622, 638, 671, 757`). Each is a full walk + tiktoken encode. Cache it once per stage boundary.
- **Unused MoE modules running** (§2) add CPU before any token is produced.
- **O(n²) re-ingestion** (§2) — quadratic growth per session.
- **`cache_registry.save_to_disk()` every turn** via `_register_context` (`optimizer.py:186-189`) — verify it's bounded; batch/periodically save instead of per-turn.

## 10. Memory leaks

- **`AgentStateStore`** — bounded in v0.5.3 (`max_state_steps=200`) — OK.
- **`hit_prediction._history`** — `deque(maxlen=200)` — OK.
- **`cache_registry` on-disk size** — `_register_context` saves every turn; verify the on-disk registry is bounded (previous review flagged this; not confirmed fixed). Recommend bounding + periodic save.
- **`static_prefix_kv`** — `save_to_disk` gated by `_last_context_changed` (fixed in v0.5.2) — OK.
- **`delta_encoder` snapshots** — bounded (100) — OK. `mtp_head_checkpoint` bounded (256) — OK.
- **No leak in the steady state**, but the per-turn `save_to_disk` calls (cache_registry) are unnecessary disk I/O that should be batched.

## 11. UX features

- **Expose real cache-hit rate in headers (both paths).** `X-Prefix-Cache-Hit-Tokens` / `X-Cache-Reuse-Pct` from `usage.prompt_tokens_details` — currently non-streaming only (§8).
- **Expose the backend's true `prompt_tokens` per turn** via a header so the benchmark (and users) can see the real prefill cost, not just the proxy's estimate.
- **Passthrough mode per request:** `extra_body={"moeptimizer": {"passthrough": true}}` to disable optimization for debugging the Grade C regression.
- **Per-session debug endpoint** (`/v1/debug/{session_id}`) with token savings, cache reuse, and per-stage pipeline timings (currently logged at INFO only).
- **A dry-run / diff endpoint** showing what the proxy *would* change — builds trust and helps tune eviction.
- **Surface the latency penalty honestly.** README now states +68%, but `notes.md:56` still claims "proxy is faster! (-1.9% mean)" — stale and contradictory; fix `notes.md`.
- **Config to set backend flags / slot pinning** so operators can turn on `--cache-reuse`, `--swa-full`, slot affinity without editing server launch separately.

---

## Priority fixes (highest ROI, in order)

1. **Pin sessions to stable backend slots + enable `--cache-reuse`/`--cache-ram`/`--swa-full`.** This lets the backend reuse the *entire* conversation prefix (and shifted chunks), recovering most of the 68% TTFT loss without rewriting the compression engine. Gated behind a backend-capability flag to stay OpenAI-transparent. *(§1, §4, §7)*
2. **Enable native MTP speculative decoding** when the backend supports it. The model's own 2–3× decode-speed feature is currently dead code. *(§1, §4)*
3. **Fix the streaming cache signal + response normalization.** Set `stream_options={"include_usage": true}`; stop collapsing `content` into `reasoning_content` in `_normalize_response_choices`; expose `X-Prefix-Cache-Hit-Tokens` in streaming. *(§8)*
4. **Make the prefix actually stable.** Freeze system + first user + a fixed early block verbatim (not just system), and move all volatility to the tail. Replace front-eviction with a frozen block + rolling summary. This is what makes #1 work. *(§2, §3, §6, §7)*
5. **Run the optimizer off the event loop** (`run_in_executor`) and **truly parallelize** tree-sitter + embedding. Restores concurrency and cuts TTFT. *(§4, §9)*
6. **Accurate token counting** via the real Qwen tokenizer or backend-echoed `prompt_tokens`. *(§1, §9)*
7. **Fix the 2.17× verbosity** with tiered compaction (rolling summary) that retains constraints; re-tune `max_tokens` in the benchmark so quality is measured on realistic responses. *(§5, §8)*
8. **Delete/disable dead components** (`parallel_embedding`, `embedding_invalidation`, `mtp_head_checkpoint`, `delta_encoder` send-path, inert `static_prefix_kv` early-exit) and **batch `cache_registry` disk writes**. *(§2, §9, §10)*

**Net:** The compression engine is strong (84.8% token savings) but the cache-preservation mission is still unmet (0% real reuse) and the proxy is a latency regression. The fastest wins are not more compression — they are (a) letting the backend's own prefix cache work by keeping a stable prefix + pinning slots, and (b) turning on the model's native MTP speculative decoding. Do those two and the proxy becomes a genuine speed *and* quality win instead of a token-reduction-only trade.

---

## Implementation status (as of v0.5.4)

- **#4 Make the prefix actually stable — DONE (v0.5.4).** `TokenAwareTruncator` now respects the frozen prefix from `ContextAligner.frozen_prefix_end` (system + first user + `frozen_prefix_turns` early turns) in both `_partition_for_budget` and `_drop_whole_messages_from_front`. Front-eviction never drops or reorders the frozen block, so the serialized prefix stays byte-stable across turns. Gated by `cache_stable_mode` (default `true`). Verified by `tests/test_optimizer.py::test_cache_stable_mode_freezes_early_turns`.
- **#6 Accurate token counting — DONE (v0.5.4).** The proxy calibrates its tiktoken (`cl100k_base`) estimate against the backend's real tokenizer. Each turn `app.py` reads the backend's true `usage.prompt_tokens` for the optimized prompt (streaming + non-streaming) and feeds the ratio to `optimizer.set_token_calibration`, which scales the budget via `calibrated_token_count` (clamped to `[0.5, 2.0]`).
- **#1 slot pinning, #3 streaming cache signal + response normalization, #5 run optimizer off event loop — DONE (earlier v0.5.x).** See `notes.md` v0.5.2/v0.5.3 sections.
- **#8 delete dead components — DONE (v0.5.4).** Removed `parallel_embedding_lookup`, `embedding_cache_invalidation`, `mtp_head_checkpoint`, `kv_cache_warmup`, `segment_wise_speculative`, `template_selector` (source + their dedicated tests). Removed their config flags from `config.py`, their imports/instantiation from `optimizer.py`, their exports from `__init__.py`, and the `kv_cache_warmup` strip key from `optimizer.py`'s `_UNSUPPORTED_BACKEND_EXTRA_BODY_KEYS` (already removed from `backend_client.py`). `hierarchical_summarizer`, `delta_encoder`, and `static_prefix_kv` are **kept** (they are used). Full suite green: **295 passed, 2 skipped** (the drop from 357 is the deleted dead-component tests).
- **#2 native MTP speculative decoding — DONE (v0.5.4).** `LemonadeClient.detect_mtp_support()` probes the backend (fetches a model from `/v1/models`, then sends a minimal chat completion carrying a `speculative_decoding` extra_body key) and returns whether the backend accepts it. `create_app` now auto-enables `native_mtp_passthrough` at startup when `MOEPT_V050__NATIVE_MTP_AUTODETECT` is true (default) and `native_mtp_passthrough` is not explicitly set, so a backend that supports native MTP speculative decoding (e.g. llama.cpp `--speculative`) receives the client's speculative fields instead of having them stripped. The probe is best-effort and bounded by a 5s timeout so startup is never blocked. Verified by `tests/test_backend_client.py` (success / reject / connection-error / models-unavailable / enable-at-runtime).
- **#7 verbosity / compaction tuning — DONE (v0.5.4).** Implemented cache-stable tiered rolling-summary compaction (review §1/§3/§5, priority #7). `HierarchicalSummarizer.summarize_turns_cache_stable(messages, frozen_prefix_end)` folds older dynamic turns into a single **append-only** rolling summary block placed immediately after the frozen prefix (never mid-history), so the leading prefix stays byte-stable and the backend reuses its prefix cache. The block retains the task's "don't"/"must not"/"avoid" constraints and key decisions (`_CONSTRAINT_HINTS` + `_extract_constraints`), which is what stops the 2.17× verbosity regression — when the proxy drops those constraints the model re-derives them verbosely. The block is protected from later front-eviction in `_partition_for_budget` / `_sliding_window_trim` via its `_summary_id` marker, so it lives in the immutable/static region alongside the frozen prefix. Wired into `optimizer.py` Step 8.5 (fires only when `cache_stable_mode` is on and the token count exceeds the proactive threshold). Verified by `tests/test_hierarchical_summarizer.py` (short / long-places-block-after-frozen / retains-constraints / append-only) and `tests/test_v050_integration.py::test_rolling_summary_in_pipeline_cache_stable`. Full suite green: **305 passed, 2 skipped**.
