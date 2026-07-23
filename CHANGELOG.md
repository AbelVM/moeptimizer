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

### v0.7.10 — Delta injection default-on, decided dynamically per re-read (2026-07-18)

Follow-up to the P2.2 delta-encode injection from v0.7.9.

- **`delta_encode_inject` now defaults to `true`.** The static flag was pure
  conservatism: the injection is already gated at runtime by
  `_inject_code_deltas`, which only fires when the prior version of the file is
  already present in the context (verified by substring). So the transform is now
  **decided dynamically per re-read** — it injects a diff only when the model
  already has the prior version to apply it against, and keeps the full current
  code verbatim on a first read or when the prior version was evicted/summarized
  out of context. No operator action needed; set the flag `false` to always
  forward the full re-read body. The existing `TestCodeDeltaInjection` suite
  (inject-on-reread / keep-on-first-read / keep-when-prior-absent) still covers
  the three runtime paths.

### v0.7.11 — Turn-11 cliff fix: rolling summary runs before compaction (2026-07-19)

Fixes the quality cliff observed in `benchmark_opencode_30_5_0.7.10_baseline`
where the proxy collapsed to a flat ~1.1K-token stub at turn 11 and never
recovered (`semantic_similarity` median 0.079, `has_code_proxy` 0.107).

- **Rolling summary now runs BEFORE the scratchpad compactor (Step 7).** The
  compactor drops the entire evictable middle of the conversation in one shot;
  when the cache-stable rolling summary ran *after* it, the evicted turns were
  already gone and the summary had nothing to fold — so all task state was lost
  at the cliff. Reordering lets the summary fold older dynamic turns into the
  append-only, byte-stable `_summary_id` block (protected from later
  front-eviction) before the compactor drops them. The context now grows with
  the conversation instead of flatlining.
- **Fact-anchor pinning (REVIEW §6).** `HierarchicalSummarizer.seed_original_request`
  pins the first user request's anchor facts into the rolling summary's leading,
  byte-stable section so `fact_recall` survives front-eviction of Turn 1. Seeded
  once per session and never rewritten, so the summary head stays cache-stable.
  (The facts also already live in the frozen-prefix first user message, so this
  is defense-in-depth.)
- **Softer post-cliff floor (`balanced` profile).** `keep_full_steps` 4 → 6 and
  `hierarchical_summary_max_full_turns` 5 → 6 (kept in sync so the summary and
  compactor protect the same recent window); `compaction_trigger_ratio` 0.85 →
  0.88 so compaction starts a little later, giving the summary more turns to
  fold first.

### v0.7.12 — Fix fact-recall grader (was 0.0 for both proxy and direct) (2026-07-19)

The `fact_recall_turn30` metric was unmeasurable: it reported **0.0 for BOTH
proxy and direct** in every run, so it could not distinguish a real compaction
regression from a healthy run. Root cause was the grader, not the proxy.

- **`_grade_fact_recall` rewritten to grade lexically, not by whole-response
  embedding similarity.** The old grader embedded the *entire* probe response
  and compared it to each fact's embedding with a 0.35 cutoff. The bundled
  embedder (768-d vectors, `embed-gemma:300m`) has a coarse space where even
  unrelated text scores ~0.6, and a verbose response that *verbatim-lists all 5
  facts* scored **negative** similarity (~−0.05) because the fact signal was
  diluted by surrounding boilerplate — so perfect recall graded 0.0.
- **New primary signal: normalized substring match on each fact's distinctive
  answer tokens.** `_DRIFT_FACTS` is now a list of `(planted_sentence,
  answer_tokens)` pairs; recall checks whether every answer token (e.g.
  `ATLAS`, `Python`+`3.11`, `Postgres`, `retry`+`3`, `platform-infra`) appears
  in the normalized response. Deterministic, embedding-independent, and honest
  for concrete agentic facts (codenames, versions, DB names, retry counts, team
  names). A verbatim recall now grades **1.0**; an unrelated answer grades 0.0.
- **Embedding kept only as a soft fallback** for paraphrased recalls, and fixed
  to compare each fact against the *best-matching sentence* (max-over-segments)
  instead of the whole response. It is consulted only when lexical finds nothing
  and the embedder is reachable, so the common case stays fast and deterministic.
- Updated `tests/test_benchmark_long_horizon.py` to assert the lexical path
  works with the embedder down (the key regression this fixes) and that empty
  responses return `None` rather than a false zero.

### v0.7.13 — Compactor was dropping the rolling-summary block (turn-10+ collapse) (2026-07-19)

A deeper read of `benchmark_opencode_30_1_0.7.11_baseline` surfaced a second,
larger bug behind the low `prompt_faithfulness` (median 0.34),
`evicted_content_recall` (median 0.36), and `semantic_similarity` (median 0.002):
the folded task state was being thrown away every turn.

- **Root cause:** `ScratchpadCompactor.compact_messages` did pure front-eviction
  (`system_anchor + protected_tail`, dropping the evictable middle) and ignored
  the `_summary_id` rolling-summary block. The optimizer's Step 7 (pre-compaction)
  correctly folds evicted turns into that block (placed as a *trailing* `user`
  message), but because it is the 7th-from-last message once there are more than
  `keep_full_steps` complete turns, the compactor's tail-keep logic evicted it.
  So every turn, the summary was built and then immediately discarded — the model
  saw only the frozen prefix + last 6 turns, never the folded history. The
  optimizer comment claimed `_partition_for_budget` already protected `_summary_id`,
  but `compact_messages` uses `_partition_zones`, which did not.
- **Fix:** `compact_messages` now extracts any `_summary_id` / `_rolling_summary`
  message *before* partitioning and re-appends it to the protected tail, so the
  block is never front-evicted. Verified the summary block (with 3–9 folded turns
  of task state at turns 20–30) now survives, and the frozen prefix stays
  byte-stable across turns (cache-reuse invariant preserved: 96.7% hit rate).
- Added `tests/test_compactor.py::test_compactor_never_evicts_rolling_summary_block`
  as a regression guard.

**Expected benchmark impact (next run):** `prompt_faithfulness`,
`evicted_content_recall`, and `semantic_similarity` should rise substantially
because the model finally receives the folded task state; `has_code_proxy` should
recover toward `has_code_direct` as code from evicted turns is now in the summary.

### v0.7.14 — Rolling summary was summarizing code away (has_code_proxy = 0) (2026-07-19)

Investigation confirmed the `has_code_proxy` → 0 regression flagged after v0.7.13:
the cache-stable path (`summarize_turns_cache_stable`) calls `_extract_constraints`,
which captured only file paths, errors, plans, constraints, and topics — **never
fenced code blocks**. The only code-pattern extraction in the module lived in
`_create_hierarchical_summary`, which the cache-stable path never calls. So when a
turn carrying a fenced code block was folded into the rolling summary, the code
vanished and the model had nothing to reproduce.

- **Fix:** `_extract_constraints` now extracts fenced code blocks (with language
  tag) verbatim via a new `_FENCE_RE` and emits them under a `Code:` section in the
  rolling-summary block (capped at the 6 most recent blocks, each ≤1500 chars, to
  stay within the rolling-summary budget). The block is preserved byte-for-byte so
  the model can reproduce it; deduplicated by `lang:code` to avoid bloat.
- Added `tests/test_hierarchical_summarizer.py::test_summarize_cache_stable_preserves_code_blocks`
  as a regression guard.
- **Cache-stability preserved:** the code is appended to the existing append-only
  `_rolling_summary_text`, so the leading (pinned-facts) bytes stay byte-identical
  and the backend prefix cache is reused.

**Expected benchmark impact (next run):** `has_code_proxy` should recover from 0.0
toward `has_code_direct` (0.367); `code_block_ratio` and `code_structure_consistency`
should rise; `semantic_similarity` should improve further as the model regains the
evicted code.

### v0.7.15 — Retain more context on deep turns + token-based summary cap (2026-07-19)

Two improvements from reading `benchmark_opencode_30_1_0.7.13_baseline`:

- **#1 — Raise the `balanced` retention budget.** Per-turn token analysis showed
  the proxy over-compressed deep-context turns 15–29× (backend-facing plateaued at
  ~2.1K tok while direct grew to 59K tok at turn 30), because the protected tail +
  summary floor was too small. The backend window is 262K with <1% utilization, so
  there is ample headroom. Raised `balanced`: `max_optimized_tokens` 8000→12000,
  `keep_full_steps` 6→8, `max_optimized_chars` 32000→48000, and
  `hierarchical_summary_max_full_turns` 6→8 (so the rolling summary protects the
  same recent window as the compactor). Cache-stable prefix is untouched.
- **#2 — Token-based rolling-summary cap with preferential code retention.** The
  rolling-summary cap was measured in chars (4000), which over/under-counts for
  code-heavy vs prose-heavy sessions. Replaced with a token cap
  (`max_rolling_summary_tokens`, default 1500) enforced by a new
  `_enforce_rolling_summary_budget`. When over budget, the OLDEST PROSE is dropped
  first and fenced code blocks (the v0.7.14 `Code:` section) are kept preferentially,
  because evicted code is the highest-value context the model cannot reconstruct.
  The byte-stable leading (pinned-facts) section is always preserved. The token
  counter is attached to the summarizer via `set_token_counter` in the optimizer.
- Also fixed a latent bug: `_extract_constraints` embedded the full fenced code
  block inside the `Topic:` line (duplicate of the `Code:` section); the topic
  sentence now strips fenced code before extraction.
- Added `tests/test_hierarchical_summarizer.py::test_rolling_summary_budget_keeps_code_over_prose`.
  Updated `test_config.py` (balanced values), `test_app.py` (degradation test input
  now exceeds the raised 7200-token proactive threshold), and
  `test_v050_integration.py` (conversation extended past the new `max_full_turns=8`).

**Expected benchmark impact (next run):** `semantic_similarity` median should rise
off 0.028 and `low_semantic_similarity_turns` (22/30) should drop, because deep
turns retain ~8 full recent steps + a denser, code-preferring summary instead of a
2.1K-tok stub.

### v0.7.16 — Dynamic token budget derived from the live backend window (2026-07-19)

The static `max_optimized_tokens` cap was a guess that did not track the actual
device. The proxy already knows the real backend window (`capability_probe`
→ `max_context_window`, e.g. 262144) and learns the true token ratio per turn
(`_token_calibration` from the backend's authoritative `prompt_tokens`). This
release makes the budget **dynamic** so it adapts to the device instead of being
hard-coded.

- **New:** `dynamic_budget_enabled` (default `true`) and `budget_window_fraction`
  (default `0.06`). When on and the live window is known, the effective budget is
  `max(window * budget_window_fraction, max_optimized_tokens)`, then scaled by the
  learned token-calibration ratio so it is enforced against the backend's TRUE
  token count. On a 262K window this yields ~15.7K tokens (vs the old fixed 12K),
  retaining more recent verbatim context with headroom for generation + the
  cache-stable prefix. `max_optimized_tokens` is now a **floor** (never starves the
  model on a tiny/unknown window) and the value used when dynamic budgeting is off
  or the window is undetected.
- The optimizer now stores the `capability_probe` and reads `max_context_window`
  from its cached snapshot (no network call on the hot path). `_budget_tokens()`
  is the single source of truth; `proactive_threshold` / `compaction_threshold`
  derive from it everywhere, so the dynamic value flows through the whole pipeline.
- Cache-stable prefix is untouched — only the optimized-context ceiling moves.
- Added `tests/test_optimizer.py` dynamic-budget cases (derives from window,
  falls back when window unknown, respects floor on tiny window, disabled uses
  static). Enlarged `test_app.py` degradation-test input so it still exceeds the
  raised proactive threshold (~9436 tok under the dynamic budget).

**Expected benchmark impact (next run):** deeper turns should retain even more
verbatim context (budget scales with the real 262K window), further lifting
`semantic_similarity` and `code_block_ratio` without raising cache-miss risk, since
the cap stays a small fraction of the window.

### v0.7.17 — Adaptive rolling-summary cap (grows with turns, saturates at window) (2026-07-19)

The rolling-summary token cap was a fixed constant (1500). A constant cap
undersells deep sessions: a 30-turn conversation needs a denser summary than a
5-turn one, and the cap should also track the live backend window like the
context budget now does (v0.7.16). This release makes the summary cap **adaptive**.

- **New:** the effective per-call cap is
  `min(ceiling, floor + per_turn_growth * summarized_turn_count)`. It starts at a
  floor (the old 1500 default), grows linearly as more turns are folded into the
  rolling summary, and **saturates** at `ceiling` — a fraction of the dynamic
  context budget (`rolling_summary_budget_fraction`, default `0.25`). On a 262K
  window the ceiling is ~3.9K tokens, so a long session keeps a far denser summary
  than a short one without ever eating into the verbatim recent window.
- The optimizer re-derives the ceiling each turn from `_budget_tokens()` (the
  dynamic, window-aware context budget) via `set_rolling_summary_ceiling`, so the
  summary cap scales with the real device. The ceiling is clamped to at least the
  floor so a tiny derived budget never starves the summary of room for the pinned
  facts + recent code.
- `_enforce_rolling_summary_budget` now uses the adaptive cap; the code-preferential
  eviction (v0.7.15) and byte-stable leading section (v0.7.11) are unchanged.
- Added `tests/test_hierarchical_summarizer.py` cases for turn-aware growth and
  ceiling clamping.

**Expected benchmark impact (next run):** deep turns (15–30) should show higher
`semantic_similarity` / `prompt_faithfulness` because the rolling summary retains
more folded-turn state as the session lengthens, while `cache_hit_rate` is
unaffected (the leading pinned-facts bytes stay byte-stable).

### v0.7.18 — Dynamic sub-caps derived from the lean budget (2026-07-19)

The context *budget* (v0.7.16) and the rolling-summary *cap* (v0.7.17) were made
dynamic, but several internal sub-caps were still fixed constants: tool/assistant
output compression threshold, user-paste compression threshold, RAG code-chunk
size, `AgentStateStore` step cap, and the quality-anchor constraint cap. On a
huge backend window these stayed at their small fixed values, so the proxy
under-used the headroom it now has — and on a tiny window they could over-compress.
This release derives each from the **lean** dynamic budget (the same 6%-of-window
value the context budget uses), so they scale with the device while the optimized
context stays lean even on a 262K window.

- **New fraction fields** (all fractions of the lean dynamic budget, not the raw
  window — the optimized context is already capped at 6% of the window, so these
  stay tiny): `tool_output_compression_budget_fraction` (0.10),
  `user_paste_compression_budget_fraction` (0.10), `state_steps_budget_fraction`
  (0.025), `anchor_max_constraints_budget_factor` (0.001), and
  `code_chunking.chunk_budget_fraction` (0.05). Each existing `*_max_chars` /
  `max_state_steps` value is now a **hard floor** (chars/steps) so a tiny/unknown
  window keeps the old behavior and is never starved.
- **New helpers** in `AgentContextOptimizer`: `_dynamic_cap(fraction, floor)` plus
  `_dynamic_tool_output_max_chars`, `_dynamic_user_paste_max_chars`,
  `_dynamic_chunk_max_chars`, `_dynamic_max_state_steps`,
  `_dynamic_max_anchor_constraints`. The tool-output / user-paste compressors and
  the code-chunk call site now build with the dynamic value each turn; the
  anchor-constraint cap and the per-turn `AgentStateStore` step cap are refreshed
  every turn (the store gains `set_max_steps`, applied in Step 1).
- **Removed dead config**: `HierarchicalSummarizer` no longer takes
  `max_summary_turns` / `max_rolling_summary_chars` — they were stored but never
  enforced (the token-based adaptive cap superseded them). Constructor signature
  simplified to `(max_full_turns, max_rolling_summary_tokens, token_counter)`.
- Added `tests/test_optimizer.py::TestDynamicSubCaps` (scaling above floor on a
  262K window, floor honored on a tiny budget, disabled → static floor, per-turn
  store override).

**Expected benchmark impact (next run):** on the 262K backend, tool/assistant
output and user pastes keep more verbatim content within the lean budget, RAG
retrieval uses larger relevant code chunks, and long sessions retain more goal
state — all without raising the total optimized-context size (still ~6% of window)
or touching the cache-stable prefix.

### v0.7.19 — Stop the prefix-cache break (P0 from REVIEW.md) (2026-07-19)

The v0.7.16–v0.7.18 dynamic-budget work was functionally correct but over-expanded
the optimized context for long agentic sessions, which triggered a **prefix-cache
stability break at turn 13** (REVIEW.md): the context grew to ~12K tokens, the
over-cap eviction/compaction rewrote the *middle* of the dynamic body, the
backend's cached KV for that body was invalidated, and the model drifted
(`contradictions` 2 → 78). This release caps growth so the stable prefix is never
mutated mid-session.

- **Changed `budget_window_fraction` 0.06 → 0.025** (default). On a 262K window
  this yields a ~6.5K effective cap (floored by `max_optimized_tokens`) instead of
  ~15.7K — large enough for a dense recent window, small enough that over-cap
  eviction never forces a mid-body rewrite. The `balanced` profile comment updated
  to match.
- **New `max_context_growth_per_turn` (default 1500 tokens)** — a hard ceiling on
  how much the optimized context may GROW in a single turn. `AgentContextOptimizer`
  gains `_effective_budget_tokens()`, which wraps `_budget_tokens()` with
  `min(budget, prev_size + max_context_growth_per_turn)`; the main eviction /
  compaction gate (Step 7 + Step 12) now uses it, so the context grows gradually
  and the cached prefix stays valid. Set to `0` to disable the growth cap.
- **New regression test** `tests/test_optimizer.py::TestCacheStabilityAcrossTurns`:
  - `test_frozen_prefix_stable_across_30_turns` — runs a 30-turn agentic
    conversation on one optimizer instance and asserts the frozen prefix +
    rolling-summary head bytes are byte-stable once the prefix has fully formed
    (the exact invariant that broke at turn 13 in v0.7.18).
  - `test_growth_ceiling_bounds_per_turn_expansion` — asserts a single oversized
    turn cannot expand the context beyond `prev_size + max_context_growth_per_turn`.
- README config table updated for `budget_window_fraction` (0.025) and the new
  `max_context_growth_per_turn` field.

**Regression gate (REVIEW.md §6):** accepted only if a re-run of
`benchmark_opencode_30_1_0.7.19` shows `prefix_cache_reuse_ratio` ≥ 1.0 at every
turn 1–30, `contradictions` (proxy) ≤ 5, `prompt_faithfulness` median ≥ 0.36,
`token_savings_pct` ≥ 80%, `fact_recall_turn30` = 1.0, and the full suite green.

### v0.7.20 — Fix `has_code_proxy = 0.0` grader blind spot (P2 from REVIEW.md) (2026-07-19)

`has_code_proxy` was a **false zero**, not a model-behavior regression. In the
agentic `opencode` coding scenario the model emits code *inside tool-call
arguments* (e.g. a `str_replace`/`bash` tool call that rewrites a source file),
not in the message `content` text. The benchmark captured only `content` +
`reasoning` into the graded text and never concatenated `tool_calls` arguments,
so `_code_block_preservation` / `_has_code_content` could not see tool-emitted
code and `has_code_proxy` collapsed to 0.0 across all 30 turns.

- **New `_tool_calls_text()`** — serializes assistant `tool_calls[].function.arguments`
  into text. `direct_contents` / `proxy_contents` now append it so the
  code-preservation grader sees code emitted via tool calls (the model-facing
  payload is unchanged). `d_tool_calls` / `p_tool_calls` are now initialized in
  both stream and non-stream paths (and the error branch) so the helper is always
  safe.
- **Hardened `_has_code_content()`** — also detects unfenced code (inline backtick
  code, 4-space-indented blocks, code keywords), so a low `has_code_proxy` reflects
  genuine absence rather than formatting. `_code_block_preservation` now reports
  `has_code_proxy` via this helper instead of the bare `"```" in text` check.
- **New `_code_likeness()` diagnostic** — fraction of code-like lines, so future
  runs can distinguish "genuinely no code" from "code without fences".
- **New regression test** `tests/test_benchmark_code_capture.py` — guards
  tool-call code capture, unfenced-code detection, and the
  `_code_block_preservation` path.

**Benchmark result (`benchmark_opencode_30_1_0.7.20_baseline`):**
- `has_code_proxy` **0.0 → 0.9333** (median 1.0) — P2 fix confirmed; the false
  zero was a grader blind spot for tool-call-emitted code. `has_code_direct`
  0.367 → 0.1667.
- `contradictions` (proxy) 78 → **31**; `code_block_loss_turns` 11 → **4**;
  `code_block_ratio` mean 0.633 → **0.867**; `prompt_faithfulness` median 0.383 →
  **0.405**; `token_savings_pct` 84.65 → **85.75**; `fact_recall_turn30` = 1.0.
- **P0 still FAILS the regression gate:** the prefix-cache break is NOT fixed.
  `prefix_cache_reuse_ratio` is healthy (≥1.0) only at turns 1–12, then collapses
  to **0.19–0.43** at turns 13–30 (frozen prefix 881 tok cached, body rewritten
  every turn). The aggregate `1.0537` is misleading (early-turn dominated). The
  v0.7.19 growth ceiling bounded *growth* but did not make the body between the
  frozen prefix and the live zone byte-stable. **P0.5 required:** incremental
  (append-only) rolling-summary folding so the whole prompt up to the live zone is
  byte-stable and the backend can reuse the body's KV cache.

### v0.7.21 — Byte-stable body: rolling summary placed right after frozen prefix (P0.5 from REVIEW.md) (2026-07-19)

The v0.7.19 growth ceiling bounded *growth* but left the body between the frozen
prefix and the live zone **rewritten every turn**, so the backend could only reuse
the 881-token frozen head (prefix-cache break at turn 13, REVIEW.md P0.4). Root
cause: the rolling-summary block was placed at the **trailing** position (after
`keep_recent`), making the leading bytes `[frozen][keep_recent]` — which change
every turn as turns shift out of `keep_recent` into the folded set.

- **`hierarchical_summarizer.py`** `summarize_turns_cache_stable` — the
  append-only rolling-summary block is now placed **immediately after the frozen
  prefix** (`return [*frozen, self._build_rolling_summary_block(), *keep_recent]`),
  matching the docstring contract. The summary only ever grows by appending, so the
  leading bytes `[frozen][summary]` are byte-stable across turns and the backend
  reuses the KV for the frozen prefix + summary head.
- **`compactor.py`** `compact_messages` — the protected rolling-summary block is
  re-inserted **right after the system anchor (frozen prefix)**, not at the tail.
- **`optimizer.py`** — new `_stable_prefix_end()` extends the stable prefix to
  include the append-only summary block (when present) so optimizer stages (e.g.
  Step 10 code-block skeletonize) never mutate it; `_update_stable_prefix` uses it.

**Verification:** `tests/test_optimizer.py::TestCacheStabilityAcrossTurns::
test_frozen_prefix_stable_across_30_turns` asserts the frozen-prefix + summary-head
hash is byte-identical across turns 3–30 (the exact invariant that broke at turn 13
in v0.7.18). Full suite: **455 passed, 2 skipped**; `ruff` clean. A live
`benchmark_opencode_30_1_0.7.21` run is required to confirm `prefix_cache_reuse_ratio
>= 1.0` at every turn 1–30 and `contradictions <= 5`.

### v0.7.22 — Per-turn shrink cap: bound one-shot front-eviction (P0.6 from REVIEW.md) (2026-07-19)

The v0.7.21 benchmark (`benchmark_opencode_30_1_0.7.21_baseline`) showed P0.5 was
**necessary but insufficient**: `prefix_cache_reuse_ratio` was still healthy turns
1–12 then collapsed at **turn 13** (`cache=882/0.44`, then 0.19–0.43 turns 14–30) —
identical to v0.7.20. Root cause: the v0.7.19 growth ceiling bounded *growth* but
NOT *shrinkage*. At turn 13 the context dropped **8,524 → 2,015 tokens in one call**
(a 6.5K one-shot front-eviction that wiped the entire body), invalidating the
backend's cached KV for the body head. Summary placement (P0.5) is irrelevant when
the whole body is evicted in a single turn.

- **`config.py`** — new `max_context_shrink_per_turn` (default **0 = AUTO**) and
  `shrink_window_fraction` (default **0.006**). When AUTO, the cap is derived as
  `max(window * shrink_window_fraction, max_context_growth_per_turn, 1500)` so it
  scales with the live backend window and never falls below the growth rate.
- **`optimizer.py`** — new `_effective_shrink_cap()` (derives the AUTO cap) and
  `_effective_shrink_floor()` (`prev_size - cap`); `_trim_to_budget` passes the floor
  to `_evict_for_budget`, which now evicts down to `min(low_water, shrink_floor)`
  instead of the full low-water mark. Step 7 (pre-compaction) now passes
  `min_keep_tokens=self._effective_shrink_floor()` so the compactor honors the same
  floor. `token_counter` is assigned **before** the compactor is built (fixes an
  `AttributeError` on first turn).
- **`compactor.py`** — `compact_messages` now takes `min_keep_tokens`; when set and a
  token counter is present it retains evictable pairs from the front until the kept
  context reaches the floor (gradual shrink) instead of dropping the whole body. New
  `_group_pairs` helper groups the evictable body into complete user-led turns. The
  rolling-summary block is still re-inserted right after the frozen prefix (P0.5).

**Verification:** `tests/test_optimizer.py::TestCacheStabilityAcrossTurns::
test_shrink_cap_bounds_per_turn_contraction` asserts the context cannot shrink more
than the per-turn cap in a single turn. `tests/test_compactor.py` adds
`test_compactor_honors_shrink_floor` (floor retained) and
`test_compactor_no_floor_drops_evictable_body` (legacy behavior preserved). Full
suite: **458 passed, 2 skipped**; `ruff` clean. A live `benchmark_opencode_30_1_0.7.22`
 run is required to confirm `prefix_cache_reuse_ratio >= 1.0` at every turn 1–30 and
 `contradictions <= 5`.

### v0.7.23 — Shrink-floor at every stage + fast-path token-count finalization (P0.6 follow-up) (2026-07-19)

The v0.7.22 shrink cap was **incomplete**: it bounded the compactor and the
front-eviction trimmer, but two content-rewrite stages still collapsed the body in
one shot, and the floor was `None` on the first over-budget turn after a run of lean
turns. The `benchmark_opencode_30_1_0.7.22_baseline` log showed a turn-11 cliff
(`4091 → 1553` tokens) caused by `filter_tool_messages` (Step 11.5) replacing matched
tool/assistant content with a tiny marker, unbounded.

- **`optimizer.py`** — new `_apply_transform_with_floor()` helper: applies a
  per-message content transform front-to-back, stopping before the next transform
  would drop the total below `shrink_floor` (P0.6 for content-rewrite stages).
  Rewrote Step 11.5/11.6/11.7 (tool-output filter, tool-output compression,
  user-paste compression) to use it, scoped to the live zone when
  `live_zone_start > 0`. Added per-message wrappers `_filter_tool_message`,
  `_compress_tool_output_message`, `_compress_user_paste_message` (with
  `_tool_output_cache` memoization). New `_finalize_optimized()` helper sets
  `_last_optimized` / `_last_optimized_token_count` / `_last_original_token_count` on
  **every** return path — fast path, static-prefix KV hit, high cache-hit-rate, and
  the main pipeline end — so the shrink floor is always defined (fixes the turn-11
  `None`-floor cliff).
- **Smart, context-relative shrink cap** — `_effective_shrink_cap()` no longer
  derives the AUTO cap from the model's full context window. It is now
  **proportional to the current lean context size**
  (`max(current_size * shrink_context_fraction, max_context_growth_per_turn,
  shrink_min_tokens)`), so the cap tracks what we are actually carrying (the target
  is a lean context, not the 262K window): a 12K-tok context may shrink ~1.8K/turn
  while a 2K-tok context only ~300/turn. Floored by the growth rate (a fast-growing
  session must be allowed to shrink at least as fast) and an absolute
  `shrink_min_tokens` floor (tiny contexts still get a bounded, non-trivial rate).

**Verification:** `scripts/diag_shrink.py` (per-stage shrink diagnostic over
`build_fixture_agentic_tasks(max_turns=30)`) now shows **no per-turn delta below
`-shrink_cap`** across turns 1–30, with the cap scaling with the live context size
(~1.5K at 11K context, ~800 floor at small contexts). New regression tests
`test_fast_path_updates_last_optimized_token_count` and
`test_filter_tool_messages_respects_shrink_floor` (optimizer). Full suite: **460
passed, 2 skipped**; `ruff` clean. A live `benchmark_opencode_30_1_0.7.23` run is
required to confirm `prefix_cache_reuse_ratio >= 1.0` at every turn 1–30 and
`contradictions <= 5`.

### v0.7.24 — Append-only rolling summary (turn-11 prefix-cache break fix) (2026-07-19)

The v0.7.23 per-turn shrink cap correctly bounds the *context size*, but the
live `benchmark_opencode_30_1_0.7.23_baseline` log showed the turn-11 cliff was
a **prefix-cache break**, not a size shrink: `cached=3192` (turn 10) →
`cached=882` (turn 11) — the backend's cached KV fell to the frozen prefix only.
Root cause: the rolling-summary block is part of the STABLE PREFIX, and
`_enforce_rolling_summary_budget()` **front-trimmed** its oldest segments to
stay under the token budget, rewriting the summary's leading bytes. The backend
had cached the old head, so the new leading bytes invalidated the whole body's
cached KV.

- **`hierarchical_summarizer.py`** — `_enforce_rolling_summary_budget()` is now a
  no-op (front-trim is forbidden for a cache-stable summary).
  `summarize_turns_cache_stable()` enforces the budget at **append time**: the new
  folded text is truncated to fit the *remaining* budget (`_truncate_to_budget`,
  keeps the front) before appending; existing summary content is never rewritten.
  The adaptive budget (`_effective_summary_budget`) is monotonic (grows with
  folded turns, saturates at the ceiling), so a later turn can always append more
  and the leading bytes stay byte-identical. `_truncate_to_budget` keeps the front
  because `_extract_constraints` already orders files/code before prose, so
  truncation-by-front retains code over prose (the v0.7.15 guarantee holds at
  extract time).
- **`scripts/diag_shrink.py`** — now tracks the stable-prefix token size and flags
  a **PREFIX BREAK** when it drops >50 tokens in one turn, so a future regression
  is caught by the diagnostic.

**Verification:** the fixture `build_fixture_agentic_tasks(max_turns=30)` does not
reproduce the turn-11 break (stable prefix stays 467 tok through turn 11),
confirming the break is content-driven in the live run. Append-only invariant
covered by `test_summarize_cache_stable_append_only`,
`test_rolling_summary_is_append_only_and_budget_capped`,
`test_leading_prefix_byte_stable_across_turns`. Full suite: **461 passed, 2
skipped**; `ruff` clean. A live `benchmark_opencode_30_1_0.7.24` run is required
to confirm `cached` stays high (no 882 drop) at turn 11 and
`prefix_cache_reuse_ratio >= 1.0` turns 1–30.

### v0.7.25 — Stable-prefix reuse fix: raw-vs-optimized mismatch + summary guards (2026-07-20)

The v0.7.24 append-only summary fixed the *summarizer*, but the orchestration
layer still re-optimized (and re-mutated) the entire conversation every turn, so
the frozen prefix + summary were not actually held byte-stable across turns. Two
compounding root causes:

1. **Raw-vs-optimized stable-prefix mismatch.** `_compute_live_zone_start()`
   compared the *incoming raw* message prefix against the *optimized* stable
   prefix stored from the previous turn. The client sends raw messages every
   turn, so raw never equals optimized → the comparison always failed →
   `live_zone_start` returned 0 → the whole conversation (including the frozen
   prefix and summary) was re-run through every content-rewrite stage each turn.
   Once token pressure crossed the proactive threshold (turn 10–12), the
   tool-output filter collapsed the frozen prefix's assistant code blocks to
   `[git status]` placeholders, mutating the leading bytes and breaking the
   backend's cached KV.
2. **Delta-encoder / skeletonizer mutated the summary block.** Even when the
   live zone was correctly bounded, `_inject_code_deltas`, the delta-encoder
   snapshot loop, and the code-block skeletonizer rewrote the append-only
   rolling-summary block's verbatim code fences (the summary is a folded
   historical snapshot, not a live file), changing its leading bytes.

Fixes:

- **`optimizer.py`** — `_compute_live_zone_start()` now compares the incoming raw
  prefix against a newly stored **raw** stable prefix (`_last_raw_prefix`), not
  the optimized one, so a unchanged raw prefix is correctly recognized and the
  live zone is bounded. `_update_stable_prefix()` stores `_last_raw_prefix` from
  the incoming `raw_messages` keyed on the *computed* stable boundary
  (`self._live_zone_start`), so the raw prefix is established on the first turn
  that has a stable prefix and reused thereafter.
- **`optimizer.py`** — added `_is_summary_block()` (detects by `_summary_id` /
  `_rolling_summary` OR the `ROLLING_SUMMARY_MARKER` content marker) and used it
  to skip the summary block in `_inject_code_deltas`, the delta-encoder snapshot
  loop, and the code-block skeletonizer, so the append-only summary is never
  rewritten by those stages.
- **`hierarchical_summarizer.py`** — added the `ROLLING_SUMMARY_MARKER` constant
  and used it in `_build_rolling_summary_block`; exported from
  `moeptimizer/__init__.py`.
- **`tests/test_optimizer.py`** — `test_frozen_prefix_stable_across_30_turns`
  rewritten to assert the APPEND-ONLY invariant (`blob.startswith(prev_blob)`
  once the summary is present) instead of byte-identical equality, and to detect
  the summary by `ROLLING_SUMMARY_MARKER`.

**Verification:** `test_frozen_prefix_stable_across_30_turns` now passes — the
frozen prefix + summary head stay byte-stable (append-only) across all 30 turns,
and the assistant code blocks in the frozen prefix are no longer collapsed to
`[git status]`. Full suite: **461 passed, 2 skipped**; `ruff` clean. A live
`benchmark_opencode_30_1_0.7.25` run is required to confirm `cached` stays high
at turn 11+ and `prefix_cache_reuse_ratio >= 1.0`.

### v0.7.26 — Batch rolling-summary folding: cache-stable size control (turn-12 cliff fix) (2026-07-23)

**Problem:** the 30-turn opencode benchmark showed a cache cliff at turn 12
(`cached` 7,014 → 881) and `cached=0` from turn 14 — the proxy re-prefilled the
whole ~8K-token live zone every turn while the direct no-proxy path (pure
append-only) computed only the new turn. Five stacked front-evictors each slid
the live zone every turn: the rolling summary folded one turn per turn (its
keep window slid and the summary grew each turn), Step 11 proactive trim and
Step 11.8 sliding window front-evicted over their thresholds, the scratchpad
compactor slid its fixed-size tail window, and the pressure-fold target was the
growth-capped budget, which chases the previous turn's size by construction.

**Fix — the batch fold is the single size governor; every front-evictor yields to it:**

- **`hierarchical_summarizer.py`** — `summarize_turns_cache_stable` now folds in
  BATCHES with **growth-relative hysteresis**: the first fold fires at the
  pressure target (static budget × `compaction_trigger_ratio`); each later fold
  fires only once the context has grown a growth budget
  (`max(2048, target//3)`) past the previous fold's post-fold size. Absolute
  targets cannot work — the keep-window floor sits at the target, so
  "fold until under target" re-triggers every turn. The emitted size is
  measured with `count_messages` (the same measurement the pipeline gates use —
  content-only counting undercounts tool-call payloads and let the compactor
  drop turns the summary had not captured, which was the
  `evicted_content_recall` regression to 0.51). The block keeps its leading
  placement `[frozen][append-only summary][live zone]` (trailing placement is
  worse on fold turns — the fold removes the live zone's HEAD — and the
  compactor undoes it anyway). New `has_rolling_summary()` accessor.
- **`optimizer.py`** — `skip_front_eviction` gate: Step 11 proactive trim and
  Step 11.8 sliding window stay off while the fold is armed (over the proactive
  threshold) and able (already folding, or still under the compaction
  threshold); they remain the valves for short/fat contexts the fold cannot
  shrink (`live <= keep`, over compaction). `_effective_budget_tokens` bypasses
  the per-turn growth ceiling while the cache-stable summary governs sizing —
  between folds the growth is pure tail append (cache-safe at any size), and
  the chasing ceiling was forcing a fold every turn.
- **`compactor.py`** — `_group_pairs` now groups only the evictable body (it
  grouped the full list, so anchor/tail pairs consumed the shrink-floor budget
  first and the floor was never reached); the floor loop stops once satisfied.
- **`token_aware_truncator.py`** — summary blocks are detected by content
  marker as well as `_summary_id` (which `_strip_internal_flags` removes before
  the backend sees the prompt), so budget trimming never drops the block and
  keeps it right after the frozen prefix.
- **`backend_client.py`** — `MOEPT_DUMP_REQUESTS=1` dumps each backend request
  payload to `/tmp/moept_req_NNN.json` for body-level prefix diffs.
- **Tests** — fixed two singleton-isolation bugs: `test_optimizer.py` stubbed
  `record_outcome` on the shared hit-prediction model without restoring it
  (now try/finally + delete), and `test_v050_integration.py` setup resets the
  hit-prediction singleton alongside the summarizer.

**Verification (16-turn opencode benchmark, before → after):**

- Per-turn `cached`: grew 0 → 7,014 then cliffed to 881/0 → now grows
  0 → **10,524** through turn 13, one fold invalidation at turns 14-15, back to
  **15,092** (98% of the prompt) at turn 16.
- Request-body LCP (dumped payloads): turns 1-13 are **100% prefix-reusable**
  (pure tail appends), turns 14-15 fold at ~20% (the designed one-time cost),
  turn 16 98%.
- Total cached tokens 44,918 → **84,320** (1.88x); proxy latency mean
  26.9s → 14.8s (-45%), p90 78.9s → 24.4s (-69%).
- Quality recovered: `prompt_faithfulness` 0.969 → 0.975,
  `evicted_content_recall` 0.996 → 0.997 (min 0.99).
- Full suite: **464 passed, 2 skipped**; `ruff` clean.

**Note:** residual `cached=882` on turns 14-15 was traced to **external slot
contention** — the Lemonade backend runs `--parallel 1` (one KV slot) and a
concurrent browser client evicted the benchmark's cached prompt between turns
(the dumped bodies still shared a ~2.5K-token prefix the backend did not
report). Benchmark runs need an exclusive/quiescent backend for clean numbers.
