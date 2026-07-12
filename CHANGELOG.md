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
