# MOE-ptimizer

Transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turns agentic tasks.

![img](moe2.jpg)

## Features

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

**Honest latency trade-off.** The proxy is a *token-reduction* proxy, not a speed win. On the 30-turn `refactor_long` benchmark the proxy is **~68% slower** (44,839 ms vs 26,559 ms direct) despite **84.8% token savings** (0.7589 semantic similarity, Grade C). It trades TTFT for token reduction; enable it when token cost dominates latency cost.

**Response headers added:** `X-Prefix-Cache-Hit-Tokens` (backend-reported cached prompt tokens for the turn).

## Architecture

```
Client (OpenAI SDK) → moeptimizer:8080 → Lemonade Server:13305
                                │
                                ├── SessionManager (per-session isolation)
                                │   └── Stable Anonymous Session Resolver
                                ├── AgentContextOptimizer (cache-stability policy)
                                │   ├── Immutable Static Layer Guard
                                │   ├── Reasoning Content Preserver
                                │   ├── Stable Turn Structure Normalizer
                                │   └── Top-Only Eviction Policy
                                ├── AgentStateStore (KV graph)
                                ├── ScratchpadCompactor
                                ├── ThinkingPreserver
                                ├── StateBasedRAG
                                │   └── SymbolIndex (fuzzy symbol lookup)
                                ├── LoopDetector
                                ├── ProgressTracker
                                ├── PromptTemplateManager (task classification)
                                │   └── ContextTemplateMatcher (template matching)
                                ├── AttentionSinkManager (internal cache hint only; no model-visible markers)
                                ├── ExpertRoutingCache (MoE routing cache)
                                ├── CacheKeyRegistry (hit prediction)
                                │   └── HitPredictionModel (XGBoost early-exit)
                                ├── KVSlotTracker (explicit cache control)
                                ├── StaticPrefixKVCache (internal cache-key reuse only)
                                ├── ContextAligner (internal alignment; no prompt padding)
                                ├── ContextCanonicalizer (newest-user-turn only)
                                ├── SelectiveTruncator (newest-user-turn only)
                                ├── SemanticDeduplicator (disabled in stable pipeline)
                                ├── PatternInjector (section markers; stripped before model input)
                                ├── DependencyOrderer (import ordering)
                                ├── IncrementalUpdater (cache preservation)
                                ├── CacheAwareChunker (aligned chunking)
                                ├── ContextCompressor (newest-user-turn only)
                                ├── CodeBlockOptimizer (tree-sitter code optimization)
                                ├── ChunkFingerprintCache (SHA-256 chunk reuse)
                                ├── DeltaEncoder (code delta compression)
                                ├── HierarchicalSummarizer (cache-stable rolling-summary compaction; enabled in the stable pipeline when hierarchical_summary_enabled)
                                ├── TokenAwareTruncator (whole-message top-only fallback)
                                ├── AsyncIOStage (async heavy stage offloading)
                                └── EmbeddingService (LanceDB + embeddings model)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Environment variables use the `MOEPT_` prefix with `__` for nested config.
For example, `server.url` maps to `MOEPT_SERVER__URL`.

### Proxy Server

| Variable | Default | Description |
|---|---|---|
| `MOEPT_PORT` | `8080` | Proxy listen port |

### Lemonade Server

| Variable | Default | Description |
|---|---|---|
| `MOEPT_SERVER__URL` | `http://localhost:13305/api/v1` | Base URL of the Lemonade chat-completions server API |
| `MOEPT_SERVER__EMBED_URL` | `http://localhost:13305/api/v1` | Base URL of the OpenAI-compatible embedding server; set separately when embeddings are hosted elsewhere |
| `MOEPT_SERVER__LLM_MODEL` | `Qwen3.6-35B-A3B-MTP-GGUF` | LLM model identifier for chat completions |
| `MOEPT_SERVER__EMBED_MODEL` | `embed-gemma-300m-FLM` | Embedding model identifier |
| `MOEPT_SERVER__TIMEOUT` | `300.0` | Request timeout in seconds for long context conversations |

### Agentic Loop Tuning

| Variable | Default | Description |
|---|---|---|
| `MOEPT_AGENTIC__KEEP_FULL_STEPS` | `3` | Last N user-assistant pairs kept in full detail; older complete turns are evicted from the top |
| `MOEPT_AGENTIC__ARCHIVE_THRESHOLD` | `3` | Steps before this index are archived/compressed |
| `MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS` | `12000` | Character fallback cap for optimized context |
| `MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS` | `3000` | Token budget cap; takes precedence over character cap |
| `MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO` | `0.45` | Ratio of max tokens where proactive top-only trimming starts |
| `MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO` | `0.75` | Ratio of max tokens where compaction/compression starts |
| `MOEPT_AGENTIC__THINKING_PROTECT_RECENT` | `2` | Keep full thinking for last N steps |
| `MOEPT_AGENTIC__SESSION_TIMEOUT` | `3600` | Session inactivity timeout in seconds |
| `MOEPT_AGENTIC__USE_TOKEN_BUDGET` | `true` | Use token-based budget enforcement |
| `MOEPT_AGENTIC__FAST_PATH_ENABLED` | `true` | Bypass expensive transformations for contexts already under budget |
| `MOEPT_AGENTIC__RAG_ENABLED` | `true` | Enable state-based RAG injection for long/over-budget sessions |
| `MOEPT_AGENTIC__OPTIMIZE_CODE_BLOCKS` | `false` | Run tree-sitter code-block optimization |
| `MOEPT_AGENTIC__CODE_SKELETON_ENABLED` | `true` | Compress large code blocks to skeletons under context pressure |
| `MOEPT_AGENTIC__SEMANTIC_DEDUP_ENABLED` | `false` | Enable embedding-based semantic deduplication |
| `MOEPT_AGENTIC__ATTENTION_SINKS_ENABLED` | `false` | Inject model-visible attention-sink markers |
| `MOEPT_AGENTIC__STATIC_LAYER_ALIGNMENT_ENABLED` | `false` | Pad static layer to cache-block boundaries |
| `MOEPT_AGENTIC__REASONING_PRESEED_ENABLED` | `false` | Inject reasoning scaffolding into user messages |
| `MOEPT_AGENTIC__MTP_BOUNDARY_ALIGNMENT_ENABLED` | `false` | Pad final context to an MTP prediction boundary |
| `MOEPT_AGENTIC__IMMUTABLE_PREFIX_ENABLED` | `true` | Freeze the system prompt verbatim across turns so the backend's automatic prefix cache can reuse it. The first user message is not frozen (it is deterministically compressed and stays stable on its own). |
| `MOEPT_AGENTIC__MAX_STATE_STEPS` | `200` | Maximum steps retained in AgentStateStore per session; oldest archived steps are pruned beyond this cap |

### Code Chunking

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CODE_CHUNKING__CHUNK_MAX_CHARS` | `1500` | Maximum characters per code chunk |
| `MOEPT_CODE_CHUNKING__TOP_K_CHUNKS` | `5` | Number of top relevant chunks to retrieve |
| `MOEPT_CODE_CHUNKING__MIN_CHUNK_SCORE` | `0.05` | Minimum embedding similarity score |
| `MOEPT_CODE_CHUNKING__EMBEDDING_DIM` | `384` | Embedding vector dimension |

### Cache

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CACHE__EMBED_CACHE_MAX` | `512` | Maximum embeddings in memory cache |
| `MOEPT_CACHE__LANCEDB_PATH` | `~/.moeptimizer/lancedb` | LanceDB vector database data directory |

### Speculative Decoding

| Variable | Default | Description |
|---|---|---|
| `MOEPT_SPECULATIVE__ENABLED` | `false` | Enable MTP-aware speculative decoding |
| `MOEPT_SPECULATIVE__MTP_LOOKAHEAD` | `4` | Number of tokens to predict ahead with MTP heads |
| `MOEPT_SPECULATIVE__CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for accepting speculative tokens |

### v0.5.x Optimizations

| Variable | Default | Description |
|---|---|---|
| `MOEPT_V050__STATIC_PREFIX_KV_ENABLED` | `true` | Enable static prefix KV-cache reuse |
| `MOEPT_V050__STATIC_PREFIX_KV_MAX_ENTRIES` | `64` | Max entries in static prefix KV-cache |
| `MOEPT_V050__TOKEN_AWARE_TRUNCATION_ENABLED` | `true` | Enable token-aware truncation with tiktoken |
| `MOEPT_V050__CACHE_STABLE_MODE` | `true` | Freeze a stable prefix block (system + first user + early turns) verbatim and never front-evict from it, so the backend reuses the KV cache across turns. Disable for backends without prefix caching to maximize token savings. |
| `MOEPT_V050__FROZEN_PREFIX_TURNS` | `2` | Number of early complete user-assistant turns (after the first user message) to freeze verbatim as part of the stable prefix when cache-stable mode is enabled. |
| `MOEPT_V050__CHUNK_FINGERPRINT_ENABLED` | `true` | Enable chunk fingerprinting and reuse |
| `MOEPT_V050__CHUNK_FINGERPRINT_MAX_ENTRIES` | `2048` | Max entries in chunk fingerprint cache |
| `MOEPT_V050__EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding queries |
| `MOEPT_V050__HIT_PREDICTION_ENABLED` | `true` | Enable lightweight hit-prediction model |
| `MOEPT_V050__HIT_PREDICTION_RETRAIN_THRESHOLD` | `50` | New samples before retraining |
| `MOEPT_V050__HIERARCHICAL_SUMMARY_ENABLED` | `false` | Enable hierarchical summarization of old turns |
| `MOEPT_V050__HIERARCHICAL_SUMMARY_MAX_FULL_TURNS` | `5` | Max recent turns to keep in full |
| `MOEPT_V050__DELTA_ENCODING_ENABLED` | `true` | Enable delta-encoding of code snapshots |
| `MOEPT_V050__DELTA_ENCODING_MAX_SNAPSHOTS` | `100` | Max code snapshots to keep |
| `MOEPT_V050__ENABLE_EXPERIMENTAL_BACKEND_HINTS` | `false` | Send optional llama.cpp/MTP cache-control hints |
| `MOEPT_V050__NATIVE_MTP_PASSTHROUGH` | `false` | Forward MTP/speculative `extra_body` keys to the backend instead of stripping them, so a backend that supports native MTP speculative decoding (e.g. llama.cpp `--speculative`) uses the model's own 2–3× decode-speed feature. Disabled by default because most backends reject unknown fields. |
| `MOEPT_V050__NATIVE_MTP_AUTODETECT` | `true` | At startup, probe the backend for native MTP speculative decoding support and automatically enable `native_mtp_passthrough` when detected. Best-effort and bounded by a timeout, so startup is never blocked. |
| `MOEPT_V050__ASYNC_IO_ENABLED` | `true` | Enable async I/O for heavy pipeline stages. Offloads tree-sitter compression and embedding ranking to a thread pool (primary TTFT cost). |
| `MOEPT_V050__ASYNC_IO_MAX_THREAD_WORKERS` | `4` | Max thread workers for CPU-bound stages |
| `MOEPT_V050__ASYNC_IO_MAX_CONCURRENCY` | `16` | Max concurrent async tasks |

## Usage

```bash
# Start the middleware
python -m moeptimizer
# or
./scripts/run.sh

# Test with curl
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-MTP-GGUF",
    "messages": [
      {"role": "user", "content": "Build a REST API with auth"}
    ],
    "stream": true
  }'
```

Conversation continuity is OpenAI-compatible by default. Clients can set the standard `user` field as a conversation/user key, and moeptimizer combines it with the first user message to reuse optimizer state across turns. If `user` is absent, the proxy fingerprints the `messages` history. Legacy `_session_id` and `_session_state` fields are still accepted for existing integrations, but they are stripped before forwarding requests to Lemonade.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (proxy + optimize) |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings (proxy to embeddings) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/agent/state` | Get agent session state |
| `POST` | `/v1/agent/state/reset` | Reset agent session |
| `GET` | `/v1/agent/sessions` | List active sessions |
| `DELETE` | `/v1/agent/session/{id}` | Delete a session |
| `POST` | `/v1/cache/clear` | Clear caches |

## Benchmarking

The benchmark script compares direct Lemonade vs moeptimizer proxy performance:

```bash
# Run with defaults (proxy on 8080, lemonade on localhost:13305)
python scripts/benchmark.py

# Real-life coding scenarios
python scripts/benchmark.py --scenario debug --turns 15 --live
python scripts/benchmark.py --scenario debug_long --turns 30 --live
python scripts/benchmark.py --scenario refactor_long --turns 30 --live
python scripts/benchmark.py --scenario feature_long --turns 30 --live
python scripts/benchmark.py --scenario default_long --turns 30 --live
python scripts/benchmark.py --scenario feature --turns 20 --live

# Stress test with context eviction
python scripts/benchmark.py --turns 50 --budget 8000 --live

# Aggressive token-savings profile (top-only eviction, 3000-token cap)
python scripts/benchmark.py --scenario refactor_long --turns 30 --profile aggressive --json > report.json

# JSON output for analysis
python scripts/benchmark.py --turns 20 --json > report.json 2> benchmark.log

# Dump full response pairs
python scripts/benchmark.py --turns 10 --dump-responses

# Complete example
python benchmark.py --scenario refactor_long --turns 30 --json > benchmark_refactor_long_30_12.json 2> benchmark_refactor_long_30_12.log
```

You might need to run the benchmark as background task to avoid hitting command timeouts. Progress and per-turn information is dumped to stderr.


### Benchmark Scenarios

| Scenario | Description |
|----------|-------------|
| `debug` | Debugging session with error analysis and fix suggestions |
| `debug_long` | 30-turn real-life debug conversation with evolving code blocks and context growth beyond 32k tokens |
| `refactor` | Code refactoring with performance optimization and type hints |
| `refactor_long` | 30-turn real-life refactor conversation with evolving code blocks and context growth beyond 32k tokens |
| `feature` | Feature implementation with API design and testing |
| `feature_long` | 30-turn real-life feature conversation with evolving code blocks and context growth beyond 32k tokens |
| `default` | General coding conversation (Fibonacci example) |
| `default_long` | 30-turn general coding conversation with evolving code blocks and context growth beyond 32k tokens |

### Metrics Collected

- **Latency**: Direct vs proxy response times (mean, median, p95)
- **Token Usage**: Prompt tokens, cached tokens, token savings percentage
- **Context Window**: Final utilization percentage
- **Response Quality**: Compares proxy-optimized responses against the direct Lemonade baseline. Similarity metrics are reported on a 0–1 scale, where `1.0` is the perfect score; they are not percentages, so `0.99` means near-perfect alignment against the 1.0 top score. Exceptions are called out below.
  - **Semantic similarity**: Embedding cosine similarity. Higher is better; it means the proxy response preserves the same meaning even if wording differs.
  - **Token Jaccard**: Word-set overlap between proxy and baseline responses. Higher is better; it indicates shared vocabulary/content coverage.
  - **ROUGE-L F1**: Longest common subsequence overlap between proxy and baseline text. Higher is better; it rewards shared wording and ordering.
  - **Trigram overlap**: Shared three-character sequence overlap. Higher is better; it indicates surface-level similarity in wording and local structure.
  - **Edit similarity**: Normalized longest-common-subsequence edit similarity. Higher is better; it means fewer insertions, deletions, or rewrites are needed to transform one response into the other.
  - **Code block ratio**: Fraction of baseline code blocks preserved by the proxy response. Higher is better; 1.0 means all baseline code blocks were preserved or no baseline code blocks existed.
  - **Markdown structure similarity**: Jaccard similarity of markdown structural elements such as headings, lists, code fences, and blockquotes. Higher is better.
  - **Length ratio**: Proxy response length divided by baseline response length. Closer to 1.0 is better; `<0.5` flags severe truncation and `>2.0` flags verbosity inflation.
  - **Vocabulary richness delta**: Absolute difference in type-token ratio between proxy and baseline. Lower is better; 0 means identical vocabulary diversity.
  - **MTP stability**: Proxy preservation of baseline code-block and thinking-tag structure. Higher is better; it indicates stable MTP-style response structure.
  - **Syntax consistency**: Similarity of code-structure keywords preserved in code blocks. Higher is better; it indicates the proxy kept similar code constructs when code was present.
- **MTP Stability**: Code block preservation, syntax consistency
- **Eviction**: Turns triggering context eviction, chars before optimization

## Development

```bash
./scripts/dev.sh    # Install deps, run tests, lint, type-check
pytest tests/ -v    # Run tests only
ruff check src/ tests/  # Lint
mypy src/moeptimizer/  # Type-check
```

## License

MIT
