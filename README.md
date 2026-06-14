# MOE-ptimizer

Transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turns agentic tasks.

![img](moe2.jpg)

## Features

- **Scratchpad Compaction** — Front-Loading Eviction for MTP head protection.
- **Thinking Preservation** — Protects recent `<thinking>` blocks, archives stale reasoning to reclaim KV-cache
- **State-Based RAG** — Graph-indexed retrieval (Goal -> Subtask -> Tool -> Outcome) instead of flat embeddings
- **Loop Detection** — Detects repeated tool calls, actions, and thinking loops
- **Progress Tracking** — Heuristic-based goal completion tracking with subtask decomposition
- **Token Budget Enforcement** — Character-aware trimming to stay within context limits
- **Code Chunking** — Tree-Sitter aware code splitting with language detection and NPU-based relevance ranking
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

## Architecture

```
Client (OpenAI SDK) → moeptimizer:8080 → Lemonade NPU:13305
                                │
                                ├── SessionManager (per-session isolation)
                                ├── AgentStateStore (KV graph)
                                ├── ScratchpadCompactor
                                ├── ThinkingPreserver
                                ├── StateBasedRAG
                                │   └── SymbolIndex (fuzzy symbol lookup)
                                ├── LoopDetector
                                ├── ProgressTracker
                                ├── PromptTemplateManager (task classification)
                                │   └── ContextTemplateMatcher (template matching)
                                ├── AttentionSinkManager (long context stability)
                                ├── ExpertRoutingCache (MoE routing cache)
                                ├── CacheKeyRegistry (hit prediction)
                                ├── ContextAligner (block alignment)
                                ├── ContextCanonicalizer (formatting normalization)
                                ├── SelectiveTruncator (duplicate removal)
                                ├── PatternInjector (section markers)
                                ├── DependencyOrderer (import ordering)
                                ├── IncrementalUpdater (cache preservation)
                                ├── CacheAwareChunker (aligned chunking)
                                ├── ContextCompressor (skeleton compression)
                                └── EmbeddingService (LanceDB + NPU)
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

Environment variables use the `MOEPT_` prefix with `__` for nested config:

### Proxy Server

| Variable | Default | Description |
|---|---|---|
| `MOEPT_PORT` | `8080` | Proxy listen port |

### Lemonade NPU Server

| Variable | Default | Description |
|---|---|---|
| `MOEPT_SERVER__URL` | `http://localhost:13305/api/v1` | Base URL of the Lemonade server API |
| `MOEPT_SERVER__LLM_MODEL` | `Qwen3.6-35B-A3B-MTP-GGUF` | LLM model identifier for completions |
| `MOEPT_SERVER__EMBED_MODEL` | `embed-gemma-300m-FLM` | Embedding model for semantic search |

### Agentic Loop Tuning

| Variable | Default | Description |
|---|---|---|
| `MOEPT_AGENTIC__KEEP_FULL_STEPS` | `3` | Last N steps kept in full detail |
| `MOEPT_AGENTIC__ARCHIVE_THRESHOLD` | `3` | Steps before this index get compressed |
| `MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS` | `12000` | Hard cap on optimized context window (chars) |
| `MOEPT_AGENTIC__THINKING_PROTECT_RECENT` | `2` | Keep full thinking for last N steps |
| `MOEPT_AGENTIC__SESSION_TIMEOUT` | `3600` | Session inactivity timeout (seconds) |

### Code Chunking

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CODE_CHUNKING__CHUNK_MAX_CHARS` | `1500` | Maximum characters per code chunk |
| `MOEPT_CODE_CHUNKING__TOP_K_CHUNKS` | `5` | Number of top relevant chunks to retrieve |
| `MOEPT_CODE_CHUNKING__MIN_CHUNK_SCORE` | `0.05` | Minimum embedding similarity score |

### Cache

| Variable | Default | Description |
|---|---|---|
| `MOEPT_CACHE__EMBED_CACHE_MAX` | `512` | Maximum embeddings in memory cache |
| `MOEPT_CACHE__LANCEDB_PATH` | `~/.moeptimizer/lancedb` | LanceDB vector database data directory |

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

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (proxy + optimize) |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings (proxy to NPU) |
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
python scripts/benchmark.py --scenario refactor --turns 10 --live
python scripts/benchmark.py --scenario feature --turns 20 --live

# Stress test with context eviction
python scripts/benchmark.py --turns 50 --budget 8000 --live

# JSON output for analysis
python scripts/benchmark.py --turns 20 --json > report.json

# Dump full response pairs
python scripts/benchmark.py --turns 10 --dump-responses
```

### Benchmark Scenarios

| Scenario | Description |
|----------|-------------|
| `debug` | Debugging session with error analysis and fix suggestions |
| `refactor` | Code refactoring with performance optimization and type hints |
| `feature` | Feature implementation with API design and testing |
| `default` | General coding conversation (Fibonacci example) |

### Metrics Collected

- **Latency**: Direct vs proxy response times (mean, median, p95)
- **Token Usage**: Prompt tokens, cached tokens, token savings percentage
- **Context Window**: Final utilization percentage
- **Response Quality**: Semantic similarity, ROUGE-L, trigram overlap, edit similarity
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
