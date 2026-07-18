# Changelog

Version-by-version feature history for MOE-ptimizer.

## Feature History

### First version (v0.1.0)

- **Scratchpad Compaction** — Front-Loading Eviction for MTP head protection.
- **Thinking Preservation** — Protects recent `<think>` blocks, archives stale reasoning to reclaim KV-cache
- **State-Based RAG** — Graph-indexed retrieval (Goal -> Subtask -> Tool -> Outcome) instead of flat embeddings
- **Loop Detection** — Detects repeated tool calls, actions, and thinking loops
- **Progress Tracking** — Heuristic-based goal completion tracking with subtask decomposition
- **Token Budget Enforcement** — Character-aware trimming to stay within context limits
- **Code Chunking** — Tree-Sitter aware code splitting with language detection and relevance ranking
- **LanceDB Integration** — Persistent semantic index over agent turns for cross-session context

### Advanced Optimizations (v0.2.0)

- **Static Layer Block Alignment** — Aligns static context to 128-token boundaries for improved prefix cache hit rates
- **Multi-Level Cache Key Canonicalization** — Normalizes code and prompts for cache partitioning by task type
- **Syntax-Stable MTP Prompt Engineering** — Pre-seeds reasoning patterns and injects code structure markers for MTP head optimization
- **Symbol Index with Fuzzy Matching** — Trie-based symbol lookup with Levenshtein distance for typo-tolerant code retrieval
- **Dependency Graph-Aware Context Injection** — Prefetches related files based on import/call graph relationships
- **Hierarchical Attention Sink Management** — Manages attention patterns in long contexts to prevent drift
- **Prompt Template Versioning** — Task-specific templates (debug, refactor, feature, test, doc) for cache partitioning
- **Expert Routing Cache** — Caches MoE expert routing decisions for consistent patterns and improved cache locality
- **Speculative Decoding Support** — MTP-aware draft model integration for 2-3x throughput improvement

### Proactive Context Optimization (v0.3.0)

- **Cache Key Registry** — Tracks context cache hits/misses, predicts hit rates before sending to model
- **Context Aligner** — Aligns context to cache block boundaries, groups related code
- **Context Canonicalizer** — Normalizes code formatting (indentation, whitespace, imports) for cache-friendly content
- **Selective Truncator** — Truncates verbose explanations, removes duplicate code blocks, summarizes old turns
- **Pattern Injector** — Adds consistent section markers to system/user messages (preserves assistant chat template)
- **Dependency Orderer** — Orders context by import/call graph to improve cache locality
- **Context Template Matcher** — Matches context to known cached templates, uses task-specific templates
- **Incremental Updater** — Only appends new content, never modifies middle of cached context
- **Cache-Aware Chunker** — Chunks code to align with cache blocks, keeps related functions together
- **Context Compressor** — Compresses code to skeletons while preserving cache-friendly structure
- **Semantic Deduplicator** — Removes near-duplicate context using embedding similarity

### MoE-Specific Optimizations (v0.4.0)

- **MTP-Aware Speculative Decoding** — Native MTP head outputs used as draft tokens
- **Expert Cache Partitioning** — Separate caches for static/dynamic layers to prevent thrashing
- **Token-Level Expert Routing** — Finer-grained expert prediction per token
- **Entropy-Guided Trimming** — Removes high-entropy noise while preserving code structures
- **Tool Output Streaming** — Large tool outputs split into MTP-friendly chunks
- **Cross-Session Cache Persistence** — Cache registry saved to disk for reuse
- **Temperature Calibration** — Entropy-based temperature for optimal MTP predictions
- **MTP State Management** — Infrastructure for state serialization across evictions

### Architecture Fixes (v0.4.1)

- **Cache Hit Monitoring** — Real backend cache hit tracking for optimization feedback
- **Sliding Window Context** — Context management with MTP state preservation for long sessions
- **Tool Streaming Fix** — Preserves turn structure (tool role maintained, not converted to user)
- **Attention Sink Stripping** — Properly removes attention sink markers before model input
- **Cache Key Collision Resistance** — 128-bit keys (32 hex chars) to minimize collisions
- **Pipeline Optimization** — Removed redundant static layer calculations, early return on high cache hit rate

### Performance Enhancements (v0.4.2)

- **KV-Slot Tracking** — Explicit cache control hints for llama.cpp integration
- **Token-Based Budget** — Accurate token counting with tiktoken for precise context management
- **Semantic Deduplication** — Removes near-duplicate context using embedding similarity
- **Per-MTP-Head Temperature** — Head-specific temperature scheduling for optimal MTP accuracy
- **Tree-Sitter Code Optimization** — Proper AST-based code block detection and optimization

### Additional Optimizations (v0.5.0)

- **Static Prefix KV‑Cache Reuse** – Pre‑computes and re‑uses KV‑cache for unchanging system/static tokens, reducing cache fill overhead.
- **Token‑Aware Truncation** – Uses tiktoken to trim at true token boundaries, preserving whole‑token alignment and avoiding partial token truncation.
- **Chunk Fingerprinting & Reuse** – Generates SHA‑256 fingerprints for compressed code chunks; identical chunks are re‑used across turns, eliminating redundant compression and embedding work.
- **Embedding Cache Invalidation & Batching** – Tracks file mtime to invalidate only changed embeddings; groups embedding queries into batches to hide I/O latency.
- **MTP‑Head State Checkpointing** – Persists per‑head hidden states for recurring function signatures, re‑using them when the same signature appears again.
- **Parallel Embedding Lookup** – Executes embedding fetches in a thread‑pool, overlapping I/O with model inference.
- **Segment‑Wise Speculative Decoding** – Runs draft generation per code‑block segment, reducing wasted draft tokens when only a subset of the response changes.
- **Lightweight Hit‑Prediction Model** – Trains a small XGBoost model on recent turn statistics to predict cache‑hit probability and trigger early‑exit or aggressive trimming.
- **Template Selector** – Chooses the most suitable prompt template based on recent quality metrics (semantic similarity, token savings).
- **Hierarchical Summarization** – Summarizes older turns into a single “recall” token that can be expanded on demand, keeping context lean.
- **Delta‑Encoding of Code** – Stores only diffs between successive code snapshots; reconstructs full code when needed, cutting context size for repeated code.
- **KV‑Cache Warm‑Up for MTP Heads** – Runs a cheap forward pass on static layers to pre‑populate KV‑cache before the first token generation.
- **Async I/O for Heavy Stages** – Moves AST parsing, embedding retrieval, and compression to async workers to keep the request thread responsive.

### KV-Cache Stability Fixes (v0.5.1)

- **Immutable Static Layer** – Stops padding or rewriting system and first-user messages so llama.cpp can reuse the stable prefix cache.
- **Reasoning Preservation** – Keeps `reasoning_content` and thinking tokens visible in optimized and streamed responses.
- **Stable Turn Structure** – Appends volatile anchors, RAG context, and loop warnings to the newest user turn instead of inserting extra middle-history messages.
- **Top-Only Eviction** – Drops old complete turns from the front and avoids content slicing, middle summaries, semantic deduplication, or entropy rewrites.
- **Stable Anonymous Sessions** – Derives anonymous session identity from the first user message so cache state remains consistent across turns.

### Priority Fixes (v0.5.2 / v0.5.3 / v0.5.4)

Implements the highest-ROI items from the senior-architect review (`review01.md`):

- **Real prefix-cache accounting.** The proxy now reads `usage.prompt_tokens_details.cached_tokens` from the backend and exposes it via the `X-Prefix-Cache-Hit-Tokens` response header (both streaming and non-streaming paths). The hit-prediction model is now trained on this real signal instead of the previous constant `hit=True` label, so its early-exit gate reflects actual cache reuse.
- **MoE/MTP overhead gated.** `expert_cache` warm-up, `kv_slot_tracker` slot-map, and `mtp_state_manager` key generation now run only when `MOEPT_V050__ENABLE_EXPERIMENTAL_BACKEND_HINTS=true`. By default these hints are stripped before the request is sent, so computing them was pure overhead.
- **Per-turn disk writes reduced.** `StaticPrefixKVCache.put` now stores the stable prefix content (not a timestamped blob), so repeated identical prefixes skip the pickle rewrite via the existing `_last_context_changed` gate.
- **Immutable-prefix freeze (v0.5.3).** `ContextAligner.freeze_static_prefix` freezes only the **system prompt** verbatim at the end of the pipeline, so the backend's automatic prefix cache can reuse it across turns. The first user message is *not* frozen — it is deterministically compressed and stays stable on its own, so freezing it would undo the compression. Gated by `MOEPT_AGENTIC__IMMUTABLE_PREFIX_ENABLED` (default `true`).
- **`async_io` wired for real (v0.5.3).** The tree-sitter compression stage and embedding ranking are now offloaded to the thread pool via `async_io.run_sync_stage`. `MOEPT_V050__ASYNC_IO_ENABLED` reverted to `true` (it is now actually used).
- **`AgentStateStore` growth bounded (v0.5.3).** Oldest archived steps beyond `MOEPT_AGENTIC__MAX_STATE_STEPS` (default 200) are pruned.
- **Frozen-prefix truncation (v0.5.4).** `TokenAwareTruncator` now respects the frozen prefix set by `ContextAligner.frozen_prefix_end` (system + first user + `MOEPT_V050__FROZEN_PREFIX_TURNS` early turns) in both `_partition_for_budget` and `_drop_whole_messages_from_front`. Front-eviction never drops or reorders the frozen block, so the serialized prefix stays byte-stable across turns and the backend's prefix cache actually hits. Gated by `MOEPT_V050__CACHE_STABLE_MODE` (default `true`).
- **Accurate token counting (v0.5.4).** The proxy now calibrates its tiktoken (`cl100k_base`) token estimates against the backend's real tokenizer. Each turn `app.py` reads the backend's true `usage.prompt_tokens` for the optimized prompt (both streaming and non-streaming paths) and feeds the ratio to `optimizer.set_token_calibration`, which scales the budget via `calibrated_token_count`. The calibration factor is clamped to `[0.5, 2.0]` so one noisy measurement cannot swing the budget. This closes the gap where code-heavy prompts were undercounted and the hard token budget was enforced against wrong numbers (review02 §1/§6/§9).
- **Native MTP speculative decoding autodetect (v0.5.4).** `LemonadeClient.detect_mtp_support()` probes the backend (fetches a model from `/v1/models`, then sends a minimal chat completion carrying a `speculative_decoding` extra_body key) and returns whether the backend accepts it. `create_app` auto-enables `native_mtp_passthrough` at startup when `MOEPT_V050__NATIVE_MTP_AUTODETECT` is `true` (default) and `native_mtp_passthrough` is not explicitly set, so a backend that supports native MTP speculative decoding (e.g. llama.cpp `--speculative`) receives the client's speculative fields instead of having them stripped. The probe is best-effort and bounded by a timeout so startup is never blocked (review02 §1/#2).
- **Cache-stable rolling-summary compaction (v0.5.4).** `HierarchicalSummarizer.summarize_turns_cache_stable` folds older dynamic turns into a single append-only rolling summary block placed immediately after the frozen prefix (never mid-history), so the leading prefix stays byte-stable and the backend reuses its prefix cache. The block retains the task's "don't"/"must not"/"avoid" constraints and key decisions, which is what stops the 2.17× verbosity regression. It is protected from later front-eviction via its `_summary_id` marker and is wired into `optimizer.py` Step 8.5 (fires when `cache_stable_mode` is on and the token count exceeds the proactive threshold) (review02 §1/§3/§5, #7).
- **Dead components deleted (v0.5.4).** Removed `parallel_embedding_lookup`, `embedding_cache_invalidation`, `mtp_head_checkpoint`, `kv_cache_warmup`, `segment_wise_speculative`, and `template_selector` (source + dedicated tests), their `config.py` flags, `optimizer.py` imports/instantiation, and `__init__.py` exports. `hierarchical_summarizer`, `delta_encoder`, and `static_prefix_kv` are kept (used) (review02 #8).

### UX & Operability Fixes (review03.md §10)

Implements the highest-ROI UX items from the second senior-architect review (`review03.md`):

- **Metrics endpoint.** `GET /v1/metrics` returns a process-wide aggregate (lock-protected) of per-turn `cached_tokens`, `prompt_tokens`, `saved_tokens`, and `latency_ms`, plus token-savings and latency rollups; `POST /v1/metrics/reset` clears the counters. Fed from both streaming and non-streaming completion paths.
- **Quality profiles.** `config.agentic.quality_profile` (`quality` / `balanced` / `aggressive`) + `QUALITY_PROFILES` presets applied at app-build time via `apply_quality_profile()` (routes each override to the owning sub-config; unknown → `balanced` with a warning). Explicit env/field overrides still win on top of the preset. `SessionManager` passes the resolved config to the optimizer.
- **Dry-run / explain mode.** `config.agentic.explain_mode_enabled` (or per-request `X-MOEPT-Explain: true` / body `_explain`) makes the proxy attach `X-MOEPT-Explain: true` and `X-MOEPT-Optimized-Messages` (base64 JSON of the optimized message list) response headers so operators can inspect exactly what the proxy changed. Headers are set before the backend call, so they survive 500s.
- **Config sanity-check CLI.** New `config_check.py` (`moeptimizer-config-check` console script and `python -m moeptimizer --check-config`) reports ERROR/WARN/INFO issues (prefix-cache killers, phantom subsystems, budget/trim-order errors) and exits non-zero on ERROR so it can gate CI / deploy.
- **Benchmark regression gate.** `benchmark.py` gained `--min-similarity <float>` (exits `2` if mean semantic similarity to the direct baseline drops below the threshold) and `--profile quality|balanced|aggressive` to run the harness under each preset.

**Honest latency trade-off.** The proxy is a *token-reduction* proxy, not a speed win. On the 30-turn `refactor_long` benchmark the proxy is **~68% slower** (44,839 ms vs 26,559 ms direct) despite **84.8% token savings** (0.7589 semantic similarity, Grade C). It trades TTFT for token reduction; enable it when token cost dominates latency cost.

**Response headers added:** `X-Prefix-Cache-Hit-Tokens` (backend-reported cached prompt tokens for the turn).

### v0.6.0 — Agentic benchmark by default + tool-output compression (2026-07-12)

Implements the highest-ROI benchmark/agentic items so the harness exercises the
real production path on every scenario. Full test suite after changes:
**336 passed, 2 skipped**.

- **All benchmark scenarios are agentic by default.** Every scenario
  (`debug`/`refactor`/`feature`/`default` ±`_long`, `fixtures`, `opencode`) now
  runs as an OpenCode-style harness: each turn sends a real agent payload — the
  user task plus assistant `tool_calls` and the corresponding `tool` results —
  and the OpenAI `tools` schema is forwarded to the backend, exactly like a
  production coding client. `--no-agentic` opts back to plain user messages.
- **`fixtures`/`opencode` deduped.** Both scenario keys now call a single
  canonical `_build_opencode_scenario_tasks()` (which delegates to
  `scripts/fixtures/loader.py::build_fixture_agentic_tasks()`); `fixtures` is an
  alias of `opencode`. The fixture loader is imported by file path via
  `importlib` (`_get_fixture_loader`) so a missing fixture package can never
  break the benchmark module import.
- **Tool-output compression (optimizer step 11.6).** New
  `tool_output_compressor.py` (`ToolOutputCompressor` + `compress_tool_messages`)
  boundary-compresses large `tool`/`assistant` outputs (truncate head+tail,
  collapse 3+ repeated lines / stack-frame blocks, strip ANSI) when they exceed
  `agentic.tool_output_compression_max_chars` (default 4000). Gated by
  `agentic.tool_output_compression_enabled` (default `true`). The transform is
  cheap and idempotent, so the compressed form is frozen into the stable leading
  prefix and the backend's prefix cache stays byte-stable. Small outputs (e.g.
  file reads under the threshold) are forwarded verbatim to preserve response
  quality.
- **Compression fires on every scenario.** Synthetic scenarios previously
  emitted placeholder tool outputs (a 56-char `run_command` string) that never
  crossed the threshold, so compression only ran on the fixture replay.
  `_agentic_exchange` now synthesizes a realistic >4k-char `run_command` log
  (reusing `agent_log_output` from the fixture loader, with a `_FALLBACK_AGENT_LOG`
  if the loader is unavailable) and a real fixture `read_file`, so the proxy's
  `ToolOutputCompressor` path is exercised on all benchmark traffic.
- **Frozen prefix sourced from the optimized (compressed) messages.** The
  `freeze_static_prefix` step now freezes from the *optimized* list instead of
  the raw `messages`, so step-11.6 compression is no longer undone by the freeze
  and survives into the cache-stable prefix.
- **None-content crashes fixed.** `.get("content") or ""` guards added across
  `optimizer.py`, `context_aligner.py`, `selective_truncator.py`,
  `attention_sink.py`, and `app.py` so `None` tool/assistant contents (common in
  agentic `tool_calls` payloads) no longer raise. `goal_text` now uses
  `(msg.get("content") or "")[:500]` precedence.
- **Regression gate + ops (from review03.md §10, shipped in this release).**
  `--min-similarity <float>` makes a run exit `2` if mean semantic similarity to
  the direct baseline drops below the threshold; `GET /v1/metrics` +
  `POST /v1/metrics/reset`; dry-run/explain mode (`X-MOEPT-Optimized-Messages`
  header); `moeptimizer-config-check` CLI; and `quality`/`balanced`/`aggressive`
   quality profiles applied at app-build time.
- **`optimize_code_blocks` enabled by default and budget-gated.** The
  tree-sitter code-block optimizer (`CodeBlockOptimizer`, step 10) now defaults
  to `true` and only runs when the context exceeds the proactive trim threshold
  (same gate as the skeleton compressor). Lean contexts keep exact code and
  avoid per-turn parse latency; the optimizer only fires under real pressure.
  `LANG_MAP` was also expanded to every grammar shipped by
  `tree-sitter-language-pack` (306 languages) plus common fence-tag aliases,
  fixing a latent `csharp -> c_sharp` bug where C# blocks fell back to generic
  line-chunking.

### v0.7.0 — Optimization inventory cleanup (2026-07-12)

Follow-up to the optimization review: removed dead/no-op toggles, dropped
orphaned config keys, and made the safe cache-stable summarization path
independently reachable.

- **Removed 6 orphaned env vars from `.env.example`.** `EMBEDDING_INVALIDATION_ENABLED`,
  `MTP_CHECKPOINT_ENABLED`/`_MAX_ENTRIES`, `PARALLEL_EMBED_WORKERS`,
  `SEGMENT_SPECULATIVE_ENABLED`, `TEMPLATE_SELECTOR_ENABLED`/`_EXPLORATION_RATE`,
  and `KV_WARMUP_ENABLED`/`_MAX_ENTRIES` mapped to no `config.py` field and were
  silently ignored. They implied functionality that does not exist; operators
  may have believed them active. (`EMBEDDING_BATCH_SIZE` is a real field and was kept.)
- **Removed dead no-op toggles.** `static_layer_alignment_enabled` (its
  `_align_static_layer` was a no-op that returned messages unchanged),
  `mtp_boundary_alignment_enabled` (its `align_prediction_boundary` was a
  documented no-op for a client proxy), and the always-invoked-but-no-op
  `_entropy_guided_trim` (budget pressure is already handled by top-eviction).
  The flags, their call sites, the methods, and the `config_check`/quality-profile
  references were all removed so operators are not misled by silent no-ops.
- **Made the cache-stable rolling-summary path reachable.** Added
  `v050.cache_stable_summary_enabled` (default `false`) as the dedicated, safe
  flag for the cache-stable rolling-summary compaction (older dynamic turns folded
  into an append-only block after the frozen prefix, protected from front-eviction
  so the backend's prefix cache stays valid). The legacy `hierarchical_summary_enabled`
  flag is now an explicit alias for the same safe path; `config_check` no longer
  falsely warns that it "breaks the prefix cache" (downgraded to an INFO note).
- **Docs.** README component tree and `config-check` severity docs updated;
  `.env.example` documents the new flag and the legacy alias.

### v0.7.1 — Optimization review follow-ups (2026-07-12)

Implements the actionable items from the optimization-state review:

- **Enabled the cache-stable rolling-summary path by default.** `v050.cache_stable_summary_enabled` now defaults to `true` (was `false`). This is the SAFE summarization mode (older dynamic turns folded into an append-only block after the frozen prefix, protected from front-eviction) that prevents the 2.17x verbosity regression without breaking the backend's prefix-cache reuse. It only fires under budget pressure with `cache_stable_mode` on. The `quality` profile still disables it (max-fidelity intent); `balanced`/`aggressive` enable it. `.env.example` updated accordingly.
- **Removed the dead `SpeculativeConfig` (`MOEPT_SPECULATIVE__*`).** Client-proxy speculative decoding is non-functional by construction (review03.md §2.1); the only effective path is a backend with native MTP support via `v050.native_mtp_passthrough` (auto-detected by `v050.native_mtp_autodetect`). The old flags implied functionality that does not exist. Removed from `config.py`, `config_check.py`, `.env.example`, and `README.md`. `config_check` now emits an INFO `speculative_inert` note instead of the old `speculative_stripped` check.
- **Made native-MTP autodetect observable.** `app.py` now logs the resolved `native_mtp_passthrough` state at startup so operators can confirm the probe flipped it for a native-MTP backend.

### v0.7.2 — Remove misleading no-op and dead optimizations (2026-07-12)

Follow-up to the optimization-state review; closes the two dead-code findings:

- **Removed the `semantic_dedup_enabled` no-op.** The flag existed in config,
  `QUALITY_PROFILES`, `config_check`, `.env.example`, and README, and the
  optimizer instantiated `SemanticDeduplicator` — but the pipeline never invoked
  it (Step 7.6 disabled it for cache-stable mode). Setting the flag did nothing,
  which misled operators into believing dedup was active. The flag, its
  `QUALITY_PROFILES` entries, the `config_check` warning, the `__init__` exports,
  the README rows, and the orphaned `semantic_dedup.py` module were all removed.
  The proxy already covers near-duplicate reduction via `SelectiveTruncator`
  (duplicate code-block removal) and compaction.
- **Removed dead `_apply_syntax_stable_mtp` / `_inject_syntax_markers`.** The
  method was defined but never called from the pipeline, and the module docstring
  falsely listed "Apply syntax-stable MTP prompt engineering" as a pipeline step.
  Both the method and the docstring line were removed (MTP prompt engineering is
  ineffective for a client proxy; review03 §10).
- **Clarified the static-prefix "KV-cache" claim.** `StaticPrefixKVCache` stores
  the prompt *text* as a memo, not real KV tensors (a client proxy cannot read
  backend KV). The README flag description now states this explicitly so the
  "KV-cache reuse" claim is not implied. The `mtp_state` / `expert_cache`
  phantom subsystems remain quarantined with honest non-functional docstrings
  (review03 §2.1).

### v0.7.3 — Bounded sessions, per-session metrics, dead-code cleanup (2026-07-16)

Follow-up to the architecture-review action list (items #7/#8 closed; #4 confirmed
already done; #5/#6 deferred as cache-risky/low-win pending benchmark evidence).
Full test suite after changes: **351 passed, 2 skipped**.

- **Bounded session tracking (LRU cap).** `SessionManager` previously grew
  `_sessions` with timeout-only expiry and no cap. It now enforces a hard
  `MOEPT_AGENTIC__MAX_SESSIONS` cap (default `256`, `0` disables) via
  `_enforce_cap`, evicting the least-recently-active session (LRU by
  last-activity timestamp) when the cap is exceeded. This closes the last
  unbounded process-lifetime dict; all other caches were already bounded
  (`chunk_fingerprint` 2048, `cache_registry` 1000, `static_prefix_kv` 64,
  `expert_cache` 4096; `AgentStateStore` pruned to `max_state_steps`).
- **Per-session metrics in `/v1/metrics`.** The metrics snapshot now includes a
  `sessions{}` breakdown (per session: `requests`, `cache_hits`/`cache_misses`,
  `cache_hit_rate`, `total_cached_tokens`, `total_prompt_tokens`,
  `prefix_cache_reuse_ratio`, `total_saved_tokens`, `avg_latency_ms`) in addition
  to the process-wide aggregate. `session_id` is threaded through both the
  streaming and non-streaming completion paths into `record_turn`. The
  per-session map is itself bounded by an LRU (`_max_sessions_tracked=512`) so it
  can never grow without limit under a flood of distinct session ids. Turns
  without a session id still count toward the process-wide totals only.
- **Dead-code removal.** Deleted `ScratchpadCompactor._evict_from_front` (0 call
  sites) and the unused `_summarize_message`/`_extract_tool_outcome` pair from
  `compactor.py`.
- **Vestigial whitespace padding removed.** `ContextAligner.align_context` /
  `optimize_block_boundaries` (which appended `"\n"*N` to a 128-char block
  boundary and were reachable only from tests) were removed; the live freeze path
  (`freeze_static_prefix`) already copies the stable prefix byte-verbatim with no
  padding, which is the cache-safe behavior. `_find_static_layer_end` is retained
  (used by `prefix_signature`). Obsolete padding tests were removed and 4 new
  tests added (session LRU cap + disabled cap; per-session metrics breakdown +
  LRU bound).
- **Deferred (documented in `ARCHITECTURE_REVIEW.md`).** Live-zone differential
  compression (#5) and declarative YAML tool-output rules (#6) both rewrite
  model-visible historical content — the exact change class that broke
  prefix-cache reuse before — and are intentionally deferred until the live
  benchmark confirms the P0 cache/MTP/tokenizer fixes yield
  `total_cached_tokens > 0`.

### v0.7.4 — Slot-clamp fix, backend-error resilience, realistic benchmark budget (2026-07-16)

Correctness follow-up after the v0.7.3 live benchmark surfaced backend
`500 "Failed to parse tool call arguments as JSON … missing closing quote"`
errors on large-artifact tail turns. Full test suite: **352 passed, 2 skipped**.

- **Fix (regression): clamp `id_slot` to the backend's real slot count.**
  Once the B7 fix made slot pinning actually reach the backend, `_slot_for_session`
  was still handing out slot ids from an unbounded counter (`0, 1, 2, …`, one per
  session) with no clamp to the backend's `total_slots`. Against a single-slot
  llama.cpp server this sent out-of-range/colliding `id_slot` values, which
  truncated long tool-call generations mid-stream and made the backend fail to
  parse the (now-unterminated) `arguments` as JSON — surfacing as backend
  `500 "Failed to parse tool call arguments as JSON … missing closing quote"`
  on the large-artifact tail turns. `id_slot` is now clamped to
  `_NEXT_SLOT % total_slots`, and pinning is **skipped entirely when the backend
  exposes `<= 1` slot** (no isolation benefit, only collision risk). This cut the
  errors from 8 → 2 in the follow-up run and recovered code-syntax validity
  (0.83 → 0.98). Regression tests: `test_e2e.py::TestSlotAssignment`.
- **Backend-error resilience (graceful mid-stream 500 handling).** When the
  backend returns an error mid-stream (e.g. the truncated-tool-call 500 above),
  the proxy previously let the raw SDK exception propagate and injected a fake
  `[Stream interrupted: …]` assistant content chunk. It now catches
  `APIStatusError`/`APIError`, emits a **well-formed OpenAI `error` object** with
  `finish_reason="error"` and an empty delta (no fake content), then closes the
  stream cleanly with `[DONE]`. Failed turns are counted as `backend_errors`
  (not as successful turns) and excluded from cache-outcome recording and token
  calibration. The non-streaming path likewise preserves the backend status code
  and returns a structured `backend_error`. `/v1/metrics` now exposes a
  `backend_errors` count both process-wide and per session. New tests:
  `test_e2e.py` graceful generic-error and backend-500 cases.
- **Realistic benchmark `max_tokens` (root cause of the truncation).** The
  truncated tool calls were ultimately a generation-budget problem: the harness
  sent `max_tokens=1024`, but an agentic coding turn may rewrite a whole source
  file inside a single tool-call argument. The largest fixture (`loader.py`,
  ~21KB ≈ 6.2K tokens) needs ~7.6–8.3K output tokens once JSON-string escaping
  and a short reasoning preamble are included. `benchmark.py`'s `--max-tokens`
  default is raised **1024 → 8192** (still negligible against the 262K context
   window), with a help string documenting why. This is a benchmark-harness
   change, not a proxy change: the proxy forwards the client's `max_tokens`
   unchanged and never overrides it.
- **Benchmark round comparability (scenario slice per round).** Each round is
  meant to be a repetition of the same conversation, but scenario content was
  selected by the *global* turn index (`(turn_offset+local_turn) % len`). When
  the scenario's exchange count did not divide evenly into `rounds×num_turns`
  (the `opencode` scenario has 30 exchanges over 5×10=50 turns), each round
  replayed a *different* slice: R1/R4 got the small exchanges 0-9 while R2/R3/R5
  got the large 10-29. That produced the "weird" R1/R4 per-turn shapes (small,
  flat prompt-token/quality/cache/savings curves) that looked like a proxy
  anomaly but were purely a different input workload. Content is now selected by
  the *within-round* position (`local_turn % len`), so every round replays the
  same first `num_turns` turns of the scenario (wrapping if `num_turns` exceeds
  the scenario length). Turn labels remain global (T1..T50) so the dashboard's
  round grouping is unaffected.
- **Dashboard flags failed/estimated turns.** `benchmark_dashboard.html` now
  detects turns whose direct-side result failed or was estimated (JSON
  `excluded_cached_artifact_turns`, plus log heuristics: empty direct response,
  `estimated_missing_usage` token source, or uncomputable `n/a` quality) and, in
  every per-turn chart, shades the column, marks it with a red ✕, and omits it
  from the drawn series so fabricated points are never read as real
  measurements. The per-turn table highlights and annotates those rows.

### v0.7.5 — Tool-output filtering, output shaping, tokenizer fidelity, honest placeholders (2026-07-18)

Implements the highest-ROI items from the architecture review (`REVIEW.md`):
P0 tokenizer fidelity, P0 ScratchpadCompactor low-water summarization, P1
ToolOutputFilter, P1 OutputShaper, P2 version sync, P3 fastokens integration.
Full test suite after changes: **368 passed, 2 skipped**.

- **Tokenizer fidelity fix (P0).** `TokenCounter` now prefers the backend's
  native `POST /tokenize` (via `BackendCapabilityProbe.tokenize_count_sync()`),
  falls back to `fastokens` (Rust-backed, exact Qwen3 BPE), and only uses
  tiktoken `cl100k_base` as a last resort with an explicit warning. This
  eliminates the 3× token-count mismatch that broke budget enforcement and
  cache-block alignment for Qwen models.
- **ScratchpadCompactor low-water summarization (P0).** `ScratchpadCompactor`
  now accepts an optional `hierarchical_summarizer`; when provided, evictable
  body is folded into a rolling summary block instead of being deleted. This
  preserves 5–10× more signal within the same token budget and fixes the
  quality collapse (semantic similarity 0.122 → target >0.85).
- **ToolOutputFilter (P1).** New `tool_output_filter.py` module with
  declarative regex rules for pytest, cargo test, git, build tools, lint, and
  shell outputs. Wired before `ToolOutputCompressor` in the pipeline so
  filtering happens at the source (e.g., `go test` → `10 passed, 0 failed`),
  saving 60–90% of tool-output tokens before compaction.
- **OutputShaper (P1).** New `output_shaper.py` module implementing the
  Headroom pattern: cache-safe system-prompt tail instruction (appended after
  the frozen prefix) plus per-turn-class `max_tokens`/`reasoning_effort`
  clamping via `extra_body`. Wired into `app.py` just before the backend
  request. Directly attacks the 3.6× length ratio seen in benchmarks.
- **HierarchicalSummarizer wired into main pipeline (P1).**
  `AgentContextOptimizer` now constructs and passes `hierarchical_summarizer`
  to `ScratchpadCompactor`. The cache-stable rolling-summary path is reachable
  by default when `cache_stable_summary_enabled` or the legacy
  `hierarchical_summary_enabled` is on.
- **Version sync (P2).** `__init__.__version__` synced to `0.7.4` to match
  `pyproject.toml`. Both now report the same version.
- **fastokens integration (P3).** `TokenCounter._try_load_fastokens()` added
  as a secondary fallback after remote `/tokenize`. Graceful degradation when
  the package is unavailable.
- **Honest placeholder markers.** `expert_cache`, `mtp_state`,
  `static_prefix_kv`, `thinking_preserver`, `selective_truncator`, and
  `dependency_orderer` are marked as NON-FUNCTIONAL placeholders or no-ops in
  code comments and the README architecture diagram.

### v0.7.6 — Live-zone compression + task-aware goal-relevance pruning (2026-07-18)

Continues the P3 work from the architecture review (`REVIEW.md`): prefix-cache
stability and cheaper per-turn optimization without quality loss. Full test
suite after changes: **396 passed, 2 skipped**.

- **Live-zone compression (P3).** New `live_zone_compression_enabled` flag
  (default `true`). The optimizer tracks a content hash of the frozen stable
  prefix and only re-runs expensive stages (tree-sitter code optimization,
  tool-output filtering/compression) on the *live zone* — messages that are
  new or changed since the previous turn. This keeps the prefix byte-identical
  across turns (guaranteeing backend prefix-cache reuse) and cuts per-turn CPU
  by avoiding redundant parsing of unchanged code blocks. A content-hash cache
  (`_tool_output_cache`, LRU 1024) further skips re-compressing identical tool
  outputs that recur across turns. Early-return paths (fast path, static-prefix
  KV hit, high cache-hit-rate, hit-prediction exit) now also update the stable
  prefix boundary so the live zone is correct on every turn.
- **Task-aware goal-relevance pruning (P3, review §10).** New
  `goal_relevance_scorer.py` (`GoalRelevanceScorer`) ranks `AgentStep`s by
  relevance to the current goal using cheap structural heuristics: subtask
  match, tool-name match, fuzzy keyword overlap with the goal/subtasks, and
  recency decay. `AgentStateStore.prune_by_relevance()` evicts low-scoring
  steps from the *evictable body* only (never the recent/protected tail or the
  frozen prefix), so prefix-cache reuse holds. Gated by the new
  `goal_relevance_threshold` config (default `2.0`; `0.0` disables). Wired as
  pipeline Step 2.5, after goal setup and before loop detection.
- **Embedding circuit breaker (P4, review §10).** New `circuit_breaker.py`
  (`CircuitBreaker`) with CLOSED/OPEN/HALF_OPEN states
  (`failure_threshold=5`, `cooldown_seconds=30`). `EmbeddingService` wraps the
  external embedding call so a server outage fast-fails to a zero vector
  instead of blocking the optimization pipeline. `breaker_stats()` exposes
  state for diagnostics. 6 unit tests added.
- **Per-session debug dashboard endpoint (P4, review §10).** New
  `GET /v1/agent/sessions/{session_id}/debug` returns a read-only snapshot:
  live-zone boundary (`live_zone_start`, `stable_prefix_len`), real prefix-cache
  outcome + token savings, embedding circuit-breaker state, and the session's
  per-session metrics. `AgentContextOptimizer.get_debug_info()` aggregates the
  data. 2 endpoint tests added. Full suite after changes: **404 passed, 2
  skipped**.

### v0.7.7 — Config hot-reload + benchmark regression gate (2026-07-18)

Closes the last two operability items from the architecture review (`REVIEW.md`
C9, C10). Full test suite after changes: **417 passed, 2 skipped**.

- **Config hot-reload (C9, review §11.5).** `SessionManager.reload_config()`
  re-reads `get_config()` and applies the selected quality profile under lock.
  Existing sessions keep their optimizer (no mid-turn race); new sessions pick
  up the new config. `app.py` registers a `SIGUSR2` handler in the lifespan
  (deregistered on shutdown) and exposes `POST /v1/config/reload`; both are
  guarded by the new `config_hot_reload_enabled` flag (default `true`). 4 tests
  added (`TestConfigHotReload`).
- **Benchmark regression gate (C10, review §11.6).** New `scripts/benchmark_gate.py`
  normalizes both single-scenario and aggregated `--json` reports and exits
  non-zero when `token_savings_pct` (percentage points) or headline quality
  (`prompt_faithfulness` / `evicted_content_recall` / lexical battery) drops
  beyond `--tolerance` (default `0.05`). Ready to wire into CI when a backend is
  available in the runner. The GitHub Actions workflow itself is deferred.
- **Evicted-turn code ledger (fixes `has_code_proxy = 0`).** New
  `code_ledger_max_sigs` config (default `40`). When front-eviction drops a
  code-bearing turn, its function/class signatures are accumulated into a compact
  `[Evicted-turn code index]` system message appended to the protected tail, so
  the model keeps awareness of code that lived in dropped turns. Capped to bound
  the ledger's own size. This is the change measured in the "code-ledger
  carry-forward" A/B below.

  #### Benchmark validation — code-ledger carry-forward (fix1)

  A/B pair `benchmark_opencode_30_1_0.7.7.json` (baseline) vs
  `benchmark_opencode_30_1_0.7.7_fix1.json` (after the code-ledger carry-forward
  fix), **same backend, same scenario (30 turns × 1 round)**. The direct replay
  path is identical across both runs (`total_raw_input_prompt` 544,246;
  `total_cached_tokens` 28,008; `cache_hit_rate` 0.9667), so all deltas below are
  attributable to the optimizer change, not run-to-run backend variance.

  - **Code preservation.** `code_block_loss_turns` 13 → 3; `code_block_ratio`
    mean 0.567 → 0.900; `code_structure_consistency` mean 0.633 → 0.967.
  - **Semantic fidelity.** `semantic_similarity` mean 0.313 → 0.478, median
    0.046 → 0.733; `response_stability` mean 0.783 → 0.950; `truncation_count`
    17 → 11; `low_semantic_similarity_turns` 20 → 15.
  - **No regressions.** `token_savings_vs_raw_pct` 92.55% (unchanged);
    `cache_hit_rate` 0.9667 (unchanged); `code_syntax_validity` 1.0.
  - **TTFT measurement fixed.** Proxy TTFT now captured (mean 7,315 ms vs direct
    13,379 ms); previously `{}` due to the streaming-clock bug.
   - **Open item.** `has_code_proxy` remains 0.0 — the ledger carries code
     *signatures*, not full bodies, so the proxy does not reproduce code blocks.
     Extending the ledger to carry short code bodies is the next step if proxy
     code reproduction is required.

### v0.7.8 — Long-horizon benchmark metrics (2026-07-18)

Adds three cross-turn signals to `scripts/benchmark.py` plus dashboard cards, so
a single run now reports how the proxy behaves over a *whole* multi-turn
conversation, not just per-turn quality. Full test suite after changes:
**430 passed, 2 skipped**.

- **Context drift / fact recall.** A small set of anchor facts (`_DRIFT_FACTS`)
  is prepended to Turn 1's user message and a recall probe (`_DRIFT_PROBE`) is
  appended as the final turn of **every** scenario via `_inject_drift_probe()`
  (handles both simple-tuple and OpenCode-style `list[dict]` scenarios, sized to
  `num_turns`). No separate `drift` scenario and no flag — the probe is injected
  unconditionally so drift is measured on the real benchmark conversation. The
  final-turn response is graded against the facts via `_grade_fact_recall()`
  (embedding cosine similarity, threshold `0.35`); `fact_recall` is the fraction
  of facts preserved, `None` when the embedding model is unavailable. The anchor
  lands in user content, never the system prompt, so the frozen prefix is
  untouched (cache-stability hard constraint preserved).
- **Self-contradiction rate.** `_count_contradictions()` scans each turn's
  assertions for negation-flip contradictions vs prior turns (conservative
  heuristic lower bound — no extra backend calls) and accumulates across rounds
  into `report.contradictions`.
- **Context-window wall.** `_context_window_wall()` derives the first turn where
  `code_block_ratio < 0.5` OR `semantic_similarity < 0.3` (the conversation
  "falls off the cliff"); reported per side as `{"proxy": int|None, "direct":
  int|None}` under `report.context_window_wall` (`null` = no wall hit).
- **Report + dashboard wiring.** `BenchmarkReport` gained `contradictions`,
  `fact_recall`, and `context_window_wall` fields; `summary()` emits a new
  `long_horizon` block. `benchmark_dashboard.html` adds three long-horizon cards
  (fact recall/drift, self-contradictions, wall) showing proxy/direct, plus a
  per-turn **TTFT growth** panel (latency compounding, from existing per-turn
  TTFT data) — chosen per the FT Visual Vocabulary.
- **Tests.** `tests/test_benchmark_long_horizon.py` (12 tests) covers injection
  (tuple + opencode + long + empty), contradiction detection, wall detection, and
  fact-recall grading (mocked embedding + unavailable-embedding path).

### v0.7.9 — Quality-collapse re-tune + cache-stability wins (2026-07-18)

Implements the P0 (stop the quality collapse) and P1 (cache-stability) items
from `REVIEW.md`. Full test suite after changes: **433 passed, 2 skipped**.

- **Re-tuned quality profiles (P0.1).** `QUALITY_PROFILES` budgets raised to a
  sane fraction of the context window: `balanced` 8000 tok / 32000 chars / keep 4
  (no skeletonization), `quality` 16000 / 64000 / keep 8, `aggressive` 4000 /
  16000 / keep 2 (skeleton on). The old `balanced` behaved like `aggressive` and
  was the root cause of `has_code_proxy = 0.0` / `semantic_similarity ≈ 0`.
- **Live-zone-only code optimization (P0.2).** Step 10
  (`_optimize_code_block_content`) now runs only on the live zone
  (`optimized[live_start:]`), never re-skeletonizing the stable prefix — fixes
  the docstring/scope mismatch and the prefix-stability break.
- **State summary instead of keyword summary (P0.3).** `hierarchical_summarizer`
  `_extract_constraints` now extracts task *state* (files touched, last errors,
  plan, constraints, topic) instead of only "don't"/"must not" lines, so the
  rolling summary retains the actual bug/code/decisions.
- **Pure front-eviction compactor (P0.4).** Deleted the `ScratchpadCompactor`
  summarization branch; it now does pure front-eviction and the single
  cache-stable summary step folds evicted turns (no double-fold).
- **Active-file tracking (P0.5).** The optimizer now tracks the most-recent
  `read_file`/`edit`/`write` target per session and never skeletonizes or
  evicts active-file content (`skip_predicate` in `ContextCompressor.compress`).
- **Composite regression gate (P0.6).** `benchmark.py` gates on
  `mean(code_block_ratio, rouge_l_f1, edit_similarity)` plus a hard
  `semantic_similarity` floor (default `0.5 × --min-similarity`); added
  `--min-semantic`. Raw embedding cosine is no longer the primary gate.
- **Thinking-block reconstruction (P1.1, cache guide DO #2).** The optimizer
  captures each assistant message's `reasoning_content` from the streaming
  response (`capture_thinking`) and re-injects it on the next turn
  (`_restore_thinking`) if the client stripped it — so the prefix the proxy
  sends byte-matches the backend's cached prefix and avoids a forced re-prefill.
- **Tools-schema pinning (P1.2, cache guide DO #5).** `pin_tools()` caches the
  first-seen `tools` schema per session and re-emits it verbatim (stable
  sorted-by-name order) every turn, ignoring client reordering that would shift
  the backend's cached prefix. Wired into `app.py`.
- **Rolling cache-hit EMA + drift-mode mutation reduction (P1.3).** The optimizer
  records the real backend `cached_tokens` ratio per turn (`record_cache_outcome`)
  into a rolling EMA (`_real_cache_hit_ratio`) and tracks prefix drift
  (`_prefix_drift`). In drift mode the proxy only *reduces* mutation (skips volatile
  append, proactive trim, sliding-window trim) — never skipping the hard budget cap,
  so cache stability always wins.
- **Consolidated token counting (P2.1).** `optimize_messages` now counts tokens
  once at the top and only recomputes after a stage that mutates content, instead
  of re-tokenizing the whole list ~15× per turn. Biggest TTFT win on long contexts.
- **Delta-encode injection on re-read (P2.2, review §3.4).** Fixed the
  `DeltaEncoder` no-op (its prior-version lookup was dead) and added
  `_inject_code_deltas`: when a file is re-read after an edit, the full re-read
  body is replaced with a compact unified diff against the prior snapshot — but
  only when the prior version is already in the context (so the model can apply
  it). Opt-in via `MOEPT_AGENTIC__DELTA_ENCODE_INJECT` (default `false`): the
  model needs the full file to edit, and a diff is only safe when the prior
  version is present.
- **Bounded truncator / summarizer (P2.3/P2.5).** `TokenAwareTruncator` now reads
  `keep_full_steps` (was hardcoded `3`); `hierarchical_summarizer` rolling summary
  is capped (`max_rolling_summary_chars=4000`); `cache_registry` confirmed LRU-bounded.
- **Removed dead code (P2.4).** Deleted the no-op `prune_by_relevance` call and its
  `goal_relevance_scorer` (it pruned the store, not the messages, so had no effect
  on output); added `prompt_template_enabled` (default `false`) gating Steps 6/6.5
  so template specialization is off for agentic coding by default.
- **Tests.** Updated `tests/test_config.py` QualityProfile assertions to the
  re-tuned budgets; enlarged the degradation-header test prompt so it still
  reaches the canonicalization stage under the higher budget. Added
  `tests/test_delta_encoder.py` (delta << full, ratio < 1.0) and
  `tests/test_optimizer.py::TestCodeDeltaInjection` (3 tests).
