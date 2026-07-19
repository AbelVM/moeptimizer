# MOE-ptimizer

Transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turns agentic tasks.

![img](moe2.jpg)

## Features

MOE-ptimizer is a transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turn agentic tasks — large token savings with byte-stable prefixes so the backend's native prefix cache is reused.

The full version-by-version feature history (v0.1.0 → v0.5.4, plus the review03.md §10 UX/operability fixes) lives in [CHANGELOG.md](CHANGELOG.md).


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
                                 │   ├── Top-Only Eviction Policy
                                 │   ├── ToolOutputFilter (declarative regex filter for tool outputs)
                                 │   ├── ToolOutputCompressor (boundary-compress large tool outputs)
                                 │   └── OutputShaper (cache-safe terse instruction + turn-class max_tokens/reasoning_effort clamping)
                                 ├── AgentStateStore (KV graph)
                                 ├── ScratchpadCompactor
                                 │   └── HierarchicalSummarizer (cache-stable rolling-summary compaction)
                                 ├── ThinkingPreserver (no-op; preserves <think> blocks verbatim)
                                 ├── StateBasedRAG
                                 │   └── SymbolIndex (fuzzy symbol lookup)
                                 ├── LoopDetector
                                 ├── ProgressTracker
                                 ├── PromptTemplateManager (task classification)
                                 │   └── ContextTemplateMatcher (template matching)
                                 ├── AttentionSinkManager (internal cache hint only; no model-visible markers)
                                 ├── ExpertRoutingCache (placeholder; fabricated expert masks)
                                 ├── CacheKeyRegistry (hit prediction)
                                 │   └── HitPredictionModel (XGBoost early-exit)
                                 ├── KVSlotTracker (explicit cache control)
                                 ├── StaticPrefixKVCache (text memo; not real KV tensors)
                                 ├── ContextAligner (internal alignment; no prompt padding)
                                 ├── ContextCanonicalizer (newest-user-turn only)
                                 ├── SelectiveTruncator (no-op; returns unchanged)
                                 ├── PatternInjector (section markers; stripped before model input)
                                 ├── DependencyOrderer (no-op; import ordering not applied)
                                 ├── IncrementalUpdater (cache preservation)
                                 ├── CacheAwareChunker (aligned chunking)
                                 ├── ContextCompressor (newest-user-turn only)
                                 ├── CodeBlockOptimizer (tree-sitter code optimization)
                                 ├── ChunkFingerprintCache (SHA-256 chunk reuse)
                                 ├── DeltaEncoder (code delta compression)
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
| `MOEPT_SERVER__TOKENIZER` | `auto` | Tokenizer for budget/counting. `auto` tries a local Qwen HF tokenizer, else falls back to tiktoken `cl100k_base` (GPT-4 BPE, approximate for Qwen). Set to a HuggingFace repo id or a local `tokenizer.json` path for exact Qwen counts. Runtime calibration from the backend's real `prompt_tokens` corrects the residual ratio each turn regardless. |
| `MOEPT_SERVER__TIMEOUT` | `300.0` | Request timeout in seconds for long context conversations |

### Agentic Loop Tuning

| Variable | Default | Description |
|---|---|---|
| `MOEPT_AGENTIC__KEEP_FULL_STEPS` | `3` | Last N user-assistant pairs kept in full detail; older complete turns are evicted from the top |
| `MOEPT_AGENTIC__ARCHIVE_THRESHOLD` | `3` | Steps before this index are archived/compressed |
| `MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS` | `32000` | Character fallback cap for optimized context. Should be a meaningful fraction of the backend context window, not ~1% of it: evicting/skeletonizing below this starves the model of task context and collapses quality. |
| `MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS` | `8000` | Floor / fallback token budget cap; takes precedence over the character cap. When `dynamic_budget_enabled` is true and the live backend window is known, the effective budget is derived from the real window (`budget_window_fraction`) and this value acts as a hard floor so a tiny/unknown window never starves the model. |
| `MOEPT_AGENTIC__DYNAMIC_BUDGET_ENABLED` | `true` | Derive the optimized-context token budget from the LIVE backend context window (`capability_probe.max_context_window`) instead of the static `max_optimized_tokens` guess. Budget = `max(window * budget_window_fraction, max_optimized_tokens)`, scaled by the learned token-calibration ratio so it is enforced against the backend's true token count. Falls back to the static `max_optimized_tokens` when the window is unknown. |
| `MOEPT_AGENTIC__BUDGET_WINDOW_FRACTION` | `0.025` | Fraction of the live backend context window used as the dynamic token budget (only when `dynamic_budget_enabled`). Kept deliberately small (2.5%) for cache-stable workloads: on a 262K window this yields ~6.5K effective cap (floored by `max_optimized_tokens`), large enough for a dense recent window but small enough that over-cap eviction never forces a mid-body rewrite that breaks backend prefix-cache reuse. The per-turn growth ceiling (`max_context_growth_per_turn`) bounds expansion further. |
| `MOEPT_AGENTIC__MAX_CONTEXT_GROWTH_PER_TURN` | `1500` | Hard ceiling on how much the OPTIMIZED context may GROW in a single turn (tokens). Prevents one turn from jumping straight to the dynamic budget cap and forcing a large mid-body rewrite that breaks backend prefix-cache reuse (the v0.7.18 turn-13 regression). When the next turn's optimized size would exceed the previous turn's size by more than this, the optimizer trims back to `prev_size + max_context_growth_per_turn` instead of the full budget, so growth is gradual and the cached prefix stays valid. Set to `0` to disable the growth cap. |
| `MOEPT_AGENTIC__MAX_CONTEXT_SHRINK_PER_TURN` | `0` (AUTO) | Hard floor on how much the OPTIMIZED context may SHRINK in a single turn (tokens). Symmetric to the growth ceiling: bounds one-shot front-eviction so the backend's cached KV for the body head stays valid (the v0.7.21 turn-13 break was an 8.5K→2K tok collapse that wiped the whole body). When `0` (AUTO), the cap is derived as `max(current_size * shrink_context_fraction, max_context_growth_per_turn, shrink_min_tokens)` — proportional to the CURRENT lean context size (the target is a lean context, not the model's full window), so a 12K-tok context may shrink ~1.8K/turn while a 2K-tok context only ~300/turn. The compactor (Step 7) and `_trim_to_budget` both honor this floor; remaining over-budget tokens are shed gradually on later turns. |
| `MOEPT_AGENTIC__SHRINK_CONTEXT_FRACTION` | `0.15` | Fraction of the CURRENT lean context size used to derive the AUTO per-turn shrink cap when `max_context_shrink_per_turn=0`. The cap scales with what we are actually carrying, not the model's full window. |
| `MOEPT_AGENTIC__SHRINK_MIN_TOKENS` | `800` | Absolute floor on the AUTO per-turn shrink cap (tokens). Guarantees even a tiny lean context still has a bounded, non-trivial shrink rate so the optimizer can return to budget. |
| `MOEPT_AGENTIC__ROLLING_SUMMARY_BUDGET_FRACTION` | `0.25` | Fraction of the dynamic context budget used as the SATURATING ceiling for the adaptive rolling-summary cap. The summary cap grows with the number of folded turns up to this ceiling, so a long session keeps a denser summary than a short one without ever eating into the verbatim recent window. 0.25 of a ~15.7K budget ≈ 3.9K tokens. |
| `MOEPT_AGENTIC__TOOL_OUTPUT_COMPRESSION_BUDGET_FRACTION` | `0.10` | Fraction of the LEAN dynamic context budget (not the raw window) used as the tool/assistant output boundary-compression threshold. The optimized context is already capped at 6% of the window, so this stays tiny even on huge windows — the proxy keeps the context lean. `MOEPT_AGENTIC__TOOL_OUTPUT_COMPRESSION_MAX_CHARS` is the hard floor (chars). |
| `MOEPT_AGENTIC__USER_PASTE_COMPRESSION_BUDGET_FRACTION` | `0.10` | Fraction of the lean dynamic budget used as the large user-code-paste compression threshold. `MOEPT_AGENTIC__USER_PASTE_COMPRESSION_MAX_CHARS` is the hard floor (chars). |
| `MOEPT_AGENTIC__STATE_STEPS_BUDGET_FRACTION` | `0.025` | Fraction of the lean dynamic budget (tokens) used as the per-session `AgentStateStore` step cap. `MOEPT_AGENTIC__MAX_STATE_STEPS` is the hard floor. |
| `MOEPT_AGENTIC__ANCHOR_MAX_CONSTRAINTS_BUDGET_FACTOR` | `0.001` | Per-token factor of the lean dynamic budget that sets the cap on accumulated quality-anchor constraints. The static floor is 5 constraints. |
| `MOEPT_CODE_CHUNKING__CHUNK_BUDGET_FRACTION` | `0.05` | Fraction of the lean dynamic budget used as the max size of a single tree-sitter code chunk during RAG retrieval. `MOEPT_CODE_CHUNKING__CHUNK_MAX_CHARS` is the hard floor (chars). |
| `MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO` | `0.45` | Ratio of the (effective) max tokens where proactive top-only trimming starts (base default; the `balanced` profile raises this to `0.6`) |
| `MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO` | `0.75` | Ratio of max tokens where compaction/compression starts (base default; the `balanced` profile raises this to `0.85`) |
| `MOEPT_AGENTIC__THINKING_PROTECT_RECENT` | `2` | Keep full thinking for last N steps |
| `MOEPT_AGENTIC__SESSION_TIMEOUT` | `3600` | Session inactivity timeout in seconds |
| `MOEPT_AGENTIC__MAX_SESSIONS` | `256` | Max concurrently tracked sessions; LRU eviction over the cap (`0` disables) |
| `MOEPT_AGENTIC__USE_TOKEN_BUDGET` | `true` | Use token-based budget enforcement |
| `MOEPT_AGENTIC__FAST_PATH_ENABLED` | `true` | Bypass expensive transformations for contexts already under budget |
| `MOEPT_AGENTIC__RAG_ENABLED` | `true` | Enable state-based RAG injection for long/over-budget sessions |
| `MOEPT_AGENTIC__OPTIMIZE_CODE_BLOCKS` | `true` | Run tree-sitter code-block optimization (chunk dedup). Budget-gated: only fires when the context exceeds the proactive trim threshold, so lean contexts keep exact code and avoid proxy latency. |
| `MOEPT_AGENTIC__CODE_SKELETON_ENABLED` | `false` | Compress large code blocks to skeletons under context pressure. Base default is `true`, but the `balanced` quality profile (the default) sets it `false` so the model keeps full code to edit/extend; only the `aggressive` profile enables skeletonization. |
| `MOEPT_AGENTIC__CODE_LEDGER_MAX_SIGS` | `40` | Max code-signature lines carried forward into the evicted-turn code ledger. When front-eviction drops a code-bearing turn, its function/class signatures are accumulated into a compact `[Evicted-turn code index]` system message appended to the protected tail, so the model keeps awareness of code that lived in dropped turns (fixes `has_code_proxy=0` / `code_block_loss`). Capped to bound the ledger's own size. |
| `MOEPT_AGENTIC__PROMPT_TEMPLATE_ENABLED` | `false` | Apply task-template specialization to rewrite the user prompt. OFF by default: for agentic coding the client's exact wording matters and template rewrites risk changing it (cache guide DONT #4). Gated behind the proactive threshold. |
| `MOEPT_AGENTIC__DELTA_ENCODE_INJECT` | `true` | When a file is re-read after an edit, replace the full re-read file body with a compact unified diff against the prior snapshot (review §3.4). ON by default, but injection is decided **dynamically per re-read**: it only fires when the prior version of the file is already present in the context (verified by substring), so the model can apply the diff to a file it already sees. On a first read, or when the prior version was evicted/summarized out of context, the full current code is kept verbatim so edits stay correct. Set to `false` to always forward the full re-read body. |
| `MOEPT_AGENTIC__ATTENTION_SINKS_ENABLED` | `false` | Inject model-visible attention-sink markers |
| `MOEPT_AGENTIC__STATIC_LAYER_ALIGNMENT_ENABLED` | `false` | Pad static layer to cache-block boundaries |
| `MOEPT_AGENTIC__REASONING_PRESEED_ENABLED` | `false` | Inject reasoning scaffolding into user messages |
| `MOEPT_AGENTIC__MTP_BOUNDARY_ALIGNMENT_ENABLED` | `false` | Pad final context to an MTP prediction boundary |
| `MOEPT_AGENTIC__IMMUTABLE_PREFIX_ENABLED` | `true` | Freeze the system prompt verbatim across turns so the backend's automatic prefix cache can reuse it. The first user message is not frozen (it is deterministically compressed and stays stable on its own). |
| `MOEPT_AGENTIC__MAX_STATE_STEPS` | `200` | Hard floor for the per-session `AgentStateStore` step cap. When the live backend window is known, the effective cap is derived from `STATE_STEPS_BUDGET_FRACTION * dynamic budget`; this value is the floor so a tiny/unknown window never starves goal context. |
| `MOEPT_AGENTIC__GOAL_RELEVANCE_THRESHOLD` | `2.0` | Minimum relevance score for a step to survive task-aware pruning (review §10). Steps scored below this threshold are evicted from the evictable body after goal decomposition. Set to `0.0` to disable. |
| `MOEPT_AGENTIC__LIVE_ZONE_COMPRESSION_ENABLED` | `true` | Live-zone compression (review §5.1/§7.1). Tracks a content hash of the frozen stable prefix and only re-runs expensive stages (tree-sitter code optimization, tool-output filter/compression) on the *live zone* — messages new or changed since the previous turn. Keeps the prefix byte-identical across turns (guaranteeing backend prefix-cache reuse) and cuts per-turn CPU by avoiding redundant parsing of unchanged code blocks. |
| `MOEPT_AGENTIC__QUALITY_PROFILE` | `balanced` | Optimization preset trading token savings against response fidelity (review03.md §10). One of `quality` (no summarization/RAG/anchor/code-skeleton; only lossless boundary compression — maximizes similarity to the direct baseline), `balanced` (defaults), `aggressive` (lower token cap, more top-only eviction, earlier compaction). Applied on top of any explicit field overrides, so individual fields can still be tuned. |
| `MOEPT_AGENTIC__EXPLAIN_MODE_ENABLED` | `false` | Dry-run / explain mode (review03.md §10). When enabled, the proxy attaches the optimized prompt it would send to the backend as the `X-MOEPT-Optimized-Messages` response header (base64 JSON) so operators can inspect exactly what the proxy changed. A single request can also opt in via the `X-MOEPT-Explain: true` request header, which works regardless of this flag. Off by default to avoid leaking full context in response headers. |

### Code Chunking

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CODE_CHUNKING__CHUNK_MAX_CHARS` | `1500` | Hard floor (chars) for a single tree-sitter code chunk during RAG retrieval. When the live backend window is known, the effective chunk size is derived from `CHUNK_BUDGET_FRACTION * dynamic budget`; this value is the floor. |
| `MOEPT_CODE_CHUNKING__TOP_K_CHUNKS` | `5` | Number of top relevant chunks to retrieve |
| `MOEPT_CODE_CHUNKING__MIN_CHUNK_SCORE` | `0.05` | Minimum embedding similarity score |
| `MOEPT_CODE_CHUNKING__EMBEDDING_DIM` | `384` | Embedding vector dimension |

### Cache

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CACHE__EMBED_CACHE_MAX` | `512` | Maximum embeddings in memory cache |
| `MOEPT_CACHE__LANCEDB_PATH` | `~/.moeptimizer/lancedb` | LanceDB vector database data directory |

### Speculative Decoding

> Client-proxy speculative decoding is non-functional (review03.md §2.1). The only effective path is a backend with native MTP support, enabled via `MOEPT_V050__NATIVE_MTP_PASSTHROUGH` (auto-detected by `MOEPT_V050__NATIVE_MTP_AUTODETECT`). The old `MOEPT_SPECULATIVE__*` variables were removed.

### v0.5.x Optimizations

> **Auto-detection (default):** with `MOEPT_V050__CAPABILITY_AUTODETECT=true`, the proxy probes the *live* backend's own metadata on a TTL — the active device and per-model `backend_url` from `/health`, native `/slots`, native `/tokenize`, and the model's checkpoint/labels — and enables slot pinning, native MTP passthrough, and the exact tokenizer only when the **currently active device** actually supports them. This is NPU↔GPU hot-swap aware: a session is never pinned to an `id_slot` on a device that has no slots. The manual `SLOT_PINNING_ENABLED` / `NATIVE_MTP_PASSTHROUGH` flags below are **force-on overrides**; leave them `false` to let auto-detection decide. All probes are best-effort and never block startup or requests.

| Variable | Default | Description |
|---|---|---|
| `MOEPT_V050__STATIC_PREFIX_KV_ENABLED` | `true` | Enable static prefix memo — stores the prompt *text* (NOT real KV tensors; a client proxy cannot read backend KV). It only short-circuits the pipeline when the incoming prefix is byte-identical and already under budget; the backend's own prefix cache does the real KV reuse. |
| `MOEPT_V050__STATIC_PREFIX_KV_MAX_ENTRIES` | `64` | Max entries in static prefix KV-cache |
| `MOEPT_V050__TOKEN_AWARE_TRUNCATION_ENABLED` | `true` | Enable token-aware truncation with tiktoken |
| `MOEPT_V050__CAPABILITY_AUTODETECT` | `true` | Live, device-aware capability detection. Probes the backend's own metadata on a TTL and drives slot pinning, MTP passthrough, and tokenizer selection from what the **active device** (NPU vs GPU) actually supports. Best-effort; failures never block startup or requests. |
| `MOEPT_V050__CAPABILITY_PROBE_TTL_SECONDS` | `30.0` | How long a capability snapshot is trusted before a live re-check. Re-checks are cheap (cached between TTLs) and let device hot-swaps be picked up without restarting the proxy. |
| `MOEPT_V050__REMOTE_TOKENIZE_ENABLED` | `true` | When the active backend exposes an exact native `/tokenize` (llama.cpp), prefer it for authoritative whole-prompt token counts (fingerprint-cached, never on the per-fragment hot path). Falls back to the local/tiktoken counter + backend `prompt_tokens` calibration otherwise (e.g. NPU device). Only takes effect with `CAPABILITY_AUTODETECT` on. |
| `MOEPT_V050__SLOT_PINNING_ENABLED` | `false` | **Force-on override** for slot pinning: pin each session to a stable llama.cpp `id_slot` so the backend reuses the whole conversation prefix across turns instead of re-prefilling. With `CAPABILITY_AUTODETECT` on, this is enabled automatically on devices that expose `/slots` (e.g. GPU/llama.cpp) and skipped when the active device has none (e.g. NPU); set this `true` to force it unconditionally. `id_slot` is sent via `extra_body`, so non-llama.cpp backends ignore it. |
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
| `MOEPT_V050__NATIVE_MTP_PASSTHROUGH` | `false` | **Force-on override** for MTP passthrough: forward MTP/speculative `extra_body` keys to the backend instead of stripping them, so a backend with native MTP speculative decoding (e.g. llama.cpp `--speculative` / Qwen3-MTP heads) uses the model's own 2–3× decode-speed feature. With `CAPABILITY_AUTODETECT` on, this is enabled automatically from backend metadata (a `mtp` label, a speculative slot, or `--spec-type ...mtp` launch args). |
| `MOEPT_V050__NATIVE_MTP_AUTODETECT` | `true` | Last-resort chat-probe for native MTP when metadata is inconclusive. Best-effort and bounded by a timeout, so startup is never blocked. |
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
| `GET` | `/v1/agent/sessions/{id}/debug` | Per-session debug snapshot: live-zone boundary, real prefix-cache outcome + token savings, embedding circuit-breaker state, and per-session metrics (review §11.6) |
| `DELETE` | `/v1/agent/session/{id}` | Delete a session |
| `POST` | `/v1/cache/clear` | Clear caches |
| `GET` | `/v1/metrics` | Proxy metrics: per-turn `cached_tokens`, `prompt_tokens`, `saved_tokens`, `latency_ms`, aggregate token savings / latency, a `backend_errors` count (turns the backend failed to serve, e.g. a truncated tool call), plus a per-session breakdown under `sessions{}` (review03.md §10) |
| `POST` | `/v1/metrics/reset` | Reset the process-wide metrics counters |

## Observability & Operations (review03.md §10)

### Metrics endpoint

The proxy exposes a process-wide metrics aggregate (lock-protected) fed from
both the streaming and non-streaming completion paths. Each turn records
`cached_tokens`, `prompt_tokens`, `saved_tokens`, and `latency_ms` against the
backend. A per-session breakdown (bounded by an internal LRU) is also returned
under `sessions{}` so you can see cache-hit rate and prefix-cache reuse per
conversation, not just in aggregate.

```bash
curl -s http://127.0.0.1:8080/v1/metrics | jq
# -> { "requests": 30, "cache_hits": ..., "cache_misses": ..., "cache_hit_rate": ...,
#      "total_cached_tokens": ..., "total_prompt_tokens": ..., "prefix_cache_reuse_ratio": ...,
#      "total_saved_tokens": ..., "total_latency_ms": ..., "avg_latency_ms": ...,
#      "backend_errors": ...,
#      "sessions": { "<session_id>": { "requests": ..., "cache_hit_rate": ...,
#                    "total_cached_tokens": ..., "prefix_cache_reuse_ratio": ...,
#                    "total_saved_tokens": ..., "avg_latency_ms": ..., "backend_errors": ... }, ... } }

curl -s -X POST http://127.0.0.1:8080/v1/metrics/reset   # reset counters
```

### Quality profiles

Pick a preset that trades token savings against response fidelity to the
un-proxified baseline. Set `MOEPT_AGENTIC__QUALITY_PROFILE` (or
`config.agentic.quality_profile`):

- **`quality`** — no middle-history mutation; only lossless boundary compression
  of oversized tool/assistant output. Maximizes similarity to the direct
  baseline (highest token cost).
- **`balanced`** (default) — the preset applied at app-build time: 12000-token / 48000-char cap, keep 8 full steps, RAG + cache-stable summary on, code skeletonization off. This is the effective default (the base `AgenticConfig` field defaults are overridden by this profile).
- **`aggressive`** — lower token cap, earlier compaction, more top-only
  eviction. Maximum token savings (lowest fidelity).

The preset is applied at app-build time and layered on top of any explicit
env/field overrides, so individual fields can still be tuned. An unknown
profile name falls back to `balanced` with a warning.

### Dry-run / explain mode

Inspect exactly what the proxy would send to the backend without changing
behavior. Enable globally with `MOEPT_AGENTIC__EXPLAIN_MODE_ENABLED=true`, or
opt in per request with the `X-MOEPT-Explain: true` request header (works
regardless of the flag). The proxy attaches two response headers:

- `X-MOEPT-Explain: true`
- `X-MOEPT-Optimized-Messages: <base64 JSON>` — the optimized message list the
  proxy built for the backend.

The headers are set before the backend call, so they survive backend 500s.

### Config sanity-check CLI

Validate the resolved configuration and surface risky / contradictory settings
that would silently hurt prefix-cache reuse or response quality. Exits non-zero
on any ERROR-level issue, so it can gate CI / deploy:

```bash
moeptimizer-config-check          # console script
python -m moeptimizer --check-config
# [ERROR] bad_budget: max_optimized_tokens must be > 0.
# exit code 1
```

Severities: `ERROR` (blocks deploy), `WARN` (prefix-cache killers, e.g.
`attention_sinks_enabled`,
`reasoning_preseed_enabled`), `INFO` (legacy aliases, phantom / non-functional
subsystems and backend-compatibility notes).

## Benchmarking

The benchmark script compares direct Lemonade vs moeptimizer proxy performance.
Every scenario runs as an **OpenCode-style agentic harness by default**: each
turn sends a real agent payload — the user task plus assistant `tool_calls` and
the corresponding `tool` results (file reads, test/lint/build logs) — and the
OpenAI `tools` schema is forwarded to the backend, exactly like a production
coding client. The proxy boundary-compresses large tool outputs (terminal logs,
file dumps) via `ToolOutputCompressor` before they enter the stable prefix, so
the benchmark exercises that path too. Pass `--no-agentic` to fall back to plain
user messages.

```bash
# Run with defaults (proxy on 8080, lemonade on localhost:13305)
python scripts/benchmark.py

# Real-life coding scenarios (all agentic / OpenCode-harness by default)
python scripts/benchmark.py --scenario debug --turns 15
python scripts/benchmark.py --scenario debug_long --turns 30
python scripts/benchmark.py --scenario refactor_long --turns 30
python scripts/benchmark.py --scenario feature_long --turns 30
python scripts/benchmark.py --scenario default_long --turns 30
python scripts/benchmark.py --scenario feature --turns 20

# OpenCode-harness replay of the real fixture project (user task + tool calls +
# real tool outputs read from scripts/fixtures/). The run_command log is >4k
# chars, so the proxy's ToolOutputCompressor fires on it.
python scripts/benchmark.py --scenario fixtures --turns 30
python scripts/benchmark.py --scenario opencode --turns 30

# Plain user messages instead of agent payloads
python scripts/benchmark.py --scenario debug_long --turns 30 --no-agentic

# Stress test with context eviction
python scripts/benchmark.py --turns 50 --budget 8000

# Aggressive token-savings profile (top-only eviction, 4000-token cap)
python scripts/benchmark.py --scenario refactor_long --turns 30 --profile aggressive --json > report.json

# Quality-fidelity profile (maximize similarity to the direct baseline)
python scripts/benchmark.py --scenario refactor_long --turns 30 --profile quality --json > report.json

# Regression gate: fail (exit 2) if mean semantic similarity drops below 0.85
python scripts/benchmark.py --scenario all --turns 10 --min-similarity 0.85
python scripts/benchmark.py --scenario refactor_long --turns 30 --min-similarity 0.80 --json > report.json

# JSON output for analysis
python scripts/benchmark.py --turns 20 --json > report.json 2> benchmark.log

# Dump full response pairs
python scripts/benchmark.py --turns 10 --dump-responses

# TTFT measurement and per-turn prefix-cache hit capture are ON by default
# (the benchmark streams responses via SSE but keeps each side's multi-turn
# conversation fully contiguous; only the transport changes). Per-round
# /v1/metrics snapshots are recorded for cache-reuse trend analysis.
# Disable with --no-measure-ttft if you must use the non-streaming path.
python scripts/benchmark.py --scenario refactor_long --turns 30 --no-measure-ttft

# Complete example 
python scripts/benchmark.py --scenario opencode --json > benchmark_opencode_10_5.json 2> benchmark_opencode_10_5.log
```

> **Rounds default is 3** (was 1). Run at least 3 rounds so per-round variance
> is observable and the regression gate can use the robust round-mean-of-means
> instead of a single noisy sample. Override with `--rounds N`.

### Benchmark Parameters

| Parameter | Default | Description |
|---|---|---|
| `--turns` | `10` | Number of conversation turns per round. |
| `--rounds` | `3` | Number of full conversation rounds (run ≥3 so per-round variance is observable and the regression gate can use the robust round-mean-of-means). |
| `--max-tokens` | `1024` | Max tokens per response (realistic for agentic coding; `256` understates proxy savings). |
| `--port` | `8080` | Proxy server port. |
| `--json` | off | Output the report as JSON to stdout. |
| `--dump-responses` | off | Print direct vs proxy response pairs for quality inspection. |
| `--budget` | unset | Override `MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS` (char budget); eviction triggers when context exceeds this. |
| `--profile` | `balanced` | Context optimization profile: `quality` (max fidelity, no summarization/RAG), `balanced` (default), `aggressive` (max token savings, top-only eviction). |
| `--min-similarity` | unset | Regression gate (review03.md §10): exit `2` if the mean semantic similarity (proxy vs direct) falls below this threshold. Use in CI to block quality regressions. |
| `--scenario` | `default` | Real-life coding scenario: `debug`, `debug_long`, `refactor`, `refactor_long`, `feature`, `feature_long`, `default`, `default_long`, `fixtures`, `opencode`, or `all`. |
| `--temperature` | `0.0` | Sampling temperature for both direct and proxy runs. `0` is deterministic so quality metrics are reproducible; raise only to stress-test nondeterminism. |
| `--no-agentic` | off (agentic on) | Send plain user messages instead of OpenCode-style agent payloads (user task + `tool_calls` + tool outputs). Agentic mode is the default for every scenario. |
| `--no-measure-ttft` | off (TTFT on) | Disable TTFT measurement and per-turn prefix-cache hit capture. By default the benchmark streams responses (SSE) to measure time-to-first-token and record per-round `/v1/metrics` snapshots; conversations stay contiguous, only the transport changes. |

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
| `fixtures` | **Alias of `opencode`** — OpenCode-harness replay of the real `scripts/fixtures/` project: each turn reads a real fixture file and runs a realistic test/lint/build log. |
| `opencode` | OpenCode-harness replay of the real `scripts/fixtures/` project (user task + `tool_calls` + real tool outputs). Always agentic. `fixtures` is an alias of this scenario. |

All scenarios are agentic by default. `--no-agentic` sends plain user messages
instead of agent payloads. `--profile` selects the optimization preset
(`quality` / `balanced` / `aggressive`, default `balanced`); `--min-similarity
<float>` makes the run exit `2` if the mean semantic similarity to the direct
baseline falls below the threshold (regression gate).

### Metrics Collected

- **Latency**: Direct vs proxy response times (mean, median, p95). **TTFT**
  (time to first token, ms) is captured per turn via the streaming path by
  default (disable with `--no-measure-ttft`).
- **Token Usage**: Prompt tokens, cached tokens, token savings percentage
- **Context Window**: Final utilization percentage
- **Prefix-cache reuse**: The proxy's authoritative prefix-cache hit count per
  turn (`X-Prefix-Cache-Hit-Tokens`, surfaced as a response header in
  non-streaming and an SSE comment when streaming). By default the benchmark
  records a per-round `/v1/metrics` snapshot (`cache_hit_rate`,
  `prefix_cache_reuse_ratio`, `total_cached_tokens`) for cache-reuse trend
  analysis (disable with `--no-measure-ttft`).
- **Response Quality**: Compares proxy-optimized responses against the direct
  Lemonade baseline. Similarity metrics are reported on a 0–1 scale, where `1.0`
  is the perfect score; they are not percentages, so `0.99` means near-perfect
  alignment against the 1.0 top score. Exceptions are called out below.

  The quality block is split into **headline** (the non-redundant, robust
  signals the regression gate reasons about) and **secondary** (correlated /
  redundant overlap signals, kept for deep inspection only). `semantic_similarity`
  is reported as its own informational key — the `embed-gemma-300m-FLM` embedder
  is weak on code, so it must not sit in the headline block.

  - **Headline**: `rouge_l_f1`, `token_jaccard`, `code_block_ratio`,
    `edit_similarity`, `length_ratio`, `code_syntax_validity`.
  - **Secondary**: `trigram_overlap`, `markdown_structure_similarity`,
    `vocabulary_richness_delta`, `rouge_l_precision`, `rouge_l_recall`,
    `response_stability`, `code_structure_consistency`, `has_code_direct`,
    `has_code_proxy`.
  - **Semantic similarity** (informational): Embedding cosine similarity. Higher
    is better; it means the proxy response preserves the same meaning even if
    wording differs. Weak on code, so it is not a headline/gate metric.
  - **Token Jaccard**: Word-set overlap between proxy and baseline responses. Higher is better; it indicates shared vocabulary/content coverage.
  - **ROUGE-L F1**: Longest common subsequence overlap between proxy and baseline text. Higher is better; it rewards shared wording and ordering.
  - **Edit similarity**: Normalized longest-common-subsequence edit similarity. Higher is better; it means fewer insertions, deletions, or rewrites are needed to transform one response into the other.
  - **Code block ratio**: Fraction of baseline code blocks preserved by the proxy response. Higher is better; 1.0 means all baseline code blocks were preserved or no baseline code blocks existed.
  - **Code syntax validity**: Fraction of fenced `python` code blocks in the proxy response that parse with `ast.parse`. Higher is better; `1.0` means no syntactically broken code was emitted as a side effect of optimization (boundary compression, summarization, eviction). This is a hard correctness signal the embedding/lexical metrics cannot catch.
  - **Markdown structure similarity**: Jaccard similarity of markdown structural elements such as headings, lists, code fences, and blockquotes. Higher is better.
  - **Length ratio**: Proxy response length divided by baseline response length. Closer to 1.0 is better; `<0.5` flags severe truncation and `>2.0` flags verbosity inflation.
  - **Vocabulary richness delta**: Absolute difference in type-token ratio between proxy and baseline. Lower is better; 0 means identical vocabulary diversity.
  - **MTP stability**: Proxy preservation of baseline code-block and thinking-tag structure. Higher is better; it indicates stable MTP-style response structure.
  - **Syntax consistency**: Similarity of code-structure keywords preserved in code blocks. Higher is better; it indicates the proxy kept similar code constructs when code was present.
- **MTP Stability**: Code block preservation, syntax consistency
- **Eviction**: Two distinct signals. `raw_exceeds_optimized_target` (informational) counts turns whose raw input exceeded the proxy's post-optimization char budget (`MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS`) — expected for long scenarios, **not** a failure. The real eviction signal is `compaction_triggered`: turns where the proxy actually sent strictly fewer prompt tokens to the backend than the direct baseline (i.e. real compaction/optimization happened). The per-turn detail table flags eviction as `proxy.prompt_tokens < direct.prompt_tokens`.
- **Long-horizon signals** (cross-turn, reported under `long_horizon` in the JSON
  report and as dedicated dashboard cards). These measure how the proxy behaves
  over a *whole* multi-turn conversation, not per-turn quality:
  - **Context drift / fact recall**: a small set of anchor facts is prepended to
    Turn 1's user message and a recall probe is appended as the final turn of
    every scenario (no separate scenario, no flag). The final-turn response is
    graded against the facts via embedding cosine similarity (threshold `0.35`);
    `fact_recall` is the fraction of facts preserved. `None` when the embedding
    model is unavailable. The anchor lands in user content, never the system
    prompt, so the frozen prefix is untouched.
  - **Self-contradiction rate**: `contradictions` counts turns whose assertions
    flip a prior assertion's polarity (conservative negation-flip heuristic, a
    lower bound — no extra backend calls). Lower is better.
  - **Context-window wall**: `context_window_wall` is the first turn where
    `code_block_ratio < 0.5` OR `semantic_similarity < 0.3` (the conversation
    "falls off the cliff"). Reported per side as `{"proxy": int|None, "direct":
    int|None}`; `null` means no wall was hit within the run.
  - **Latency compounding**: the dashboard also plots a per-turn **TTFT growth**
    curve (existing per-turn TTFT data, proxy vs direct) to show how
    time-to-first-token compounds across turns under each path.

## Development

```bash
./scripts/dev.sh    # Install deps, run tests, lint, type-check
pytest tests/ -v    # Run tests only
ruff check src/ tests/  # Lint
mypy src/moeptimizer/  # Type-check
```

## License

MIT
