# MOE-ptimizer Architecture Review (ling)

**Date:** 2026-07-31  
**Target model:** Qwen3.6-35B-A3B-MTP-GGUF (MoE + MTP)  
**Proxy version:** 0.7.26 (pyproject.toml)  
**Benchmark scenario:** opencode, 30 turns, 1 round  
**Benchmark file:** `scripts/benchmark_opencode_30_1_0.7.26_fix2.json`

---

## 1. Executive Summary

The MOE-ptimizer proxy achieves **66.2% token savings** against the raw input prompt, but at a significant latency cost: the proxy is **slower than direct** on 16 of 30 turns (mean +6629ms, median +1674ms). TTFT is **2x worse** for the proxy (26562ms vs 13311ms). Quality metrics are poor (ROUGE-L F1 mean 0.247, token Jaccard 0.237, semantic similarity 0.116). The cache cliff at turn 14 persists despite the frozen-prefix fix, and the working-tree cliff fix is incomplete — fold turns still break prefix-cache reuse.

The core problem is that the optimization pipeline adds overhead (compaction, tree-sitter parsing, budget computation) that exceeds the latency saved by sending fewer tokens to the backend. The pipeline is also too aggressive in evicting content, causing quality degradation.

---

## 2. Architecture Overview

### 2.1 Request Flow

```
Client (OpenAI SDK)
  → FastAPI proxy (app.py, :8080)
    → AgentContextOptimizer (optimizer.py)
      → parse messages
      → ScratchpadCompactor (compactor.py)
      → ThinkingPreserver (preserver.py)
      → Tree-Sitter code chunking (code_chunking.py)
      → MoE budget calculation
      → static layer alignment
      → _update_stable_prefix (frozen prefix)
    → BackendClient → Lemonade server (:13305)
  → Client
```

### 2.2 Key Components

| Component | File | Role |
|-----------|------|------|
| `AgentContextOptimizer` | `src/moeptimizer/optimizer.py` | Main pipeline orchestrator |
| `ScratchpadCompactor` | `src/moeptimizer/compactor.py` | Compresses scratchpad messages |
| `ThinkingPreserver` | `src/moeptimizer/preserver.py` | Preserves thinking/reasoning blocks |
| `TreeSitterChunker` | `src/moeptimizer/code_chunking.py` | Language-aware code chunking |
| `MoEBudgetAllocator` | `src/moeptimizer/optimizer.py` | Allocates token budget per MoE layer |
| `_ProxyMetrics` | `src/moeptimizer/app.py` | Tracks cache hits/misses, tokens |
| `AppConfig` | `src/moeptimizer/config.py` | All configuration (pydantic-settings) |

### 2.3 Configuration (AppConfig)

All settings use `MOEPT_` prefix with `__` nesting. Key optimization settings:

- `MOEPT_AGENTIC__QUALITY_PROFILE` — quality vs savings tradeoff
- `MOEPT_AGENTIC__KEEP_FULL_TURNS` — max recent turns kept in full (default 6)
- `MOEPT_OPTIMIZATION__CHAR_BUDGET` — per-turn character budget (default 12000)
- `MOEPT_OPTIMIZATION__FOLD_MARGIN_TURNS` — DRIFT fold margin (working tree, default 0)
- `MOEPT_OPTIMIZATION__FOLD_WINDOW_FRACTION` — space-based folding threshold (working tree, default 0.0)

---

## 3. Key Findings

### Finding 1: Proxy is Net Negative on Latency (CRITICAL)

| Metric | Direct | Proxy | Delta |
|--------|--------|-------|-------|
| Mean latency | 37411ms | 44040ms | **+6629ms (+17.7%)** |
| Median latency | 37128ms | 39179ms | +1674ms (+4.5%) |
| P90 latency | 57911ms | 80368ms | +49961ms (+86.3%) |
| P95 latency | 68710ms | 87106ms | +54878ms (+80.0%) |
| Mean TTFT | 13311ms | 26562ms | **+13251ms (+99.5%)** |
| Median TTFT | 14205ms | 18802ms | +4597ms (+32.3%) |

The proxy adds ~2x TTFT overhead and ~18% mean latency overhead. The optimizer pipeline (tree-sitter parsing, compaction, budget calculation) is too expensive relative to the token savings achieved.

**Root cause:** The optimization work (compaction + tree-sitter + budget allocation) runs synchronously in the request handler before forwarding to the backend. Even though the optimizer uses a ThreadPoolExecutor, the total wall-clock time includes both the optimization overhead AND the backend processing time for the reduced prompt.

### Finding 2: Cache Cliff Persists at Turn 14 (CRITICAL)

The frozen-prefix fix (storing only the frozen prefix as `_last_raw_prefix`, excluding the rolling summary block) was applied but the cache cliff at turn 14 (cached=881) persists. The direct conversation on the same backend continues to cache normally, ruling out backend-level causes.

**Root cause (from CLIF_RESEARCH.md):** The budget/keep-window mismatch causes the compactor to evict content that the backend had already cached. When the live zone is split and the frozen prefix is updated, the backend's KV cache entries for the evicted content are invalidated. The 6 eviction mechanisms (compactor, preserver, chunker, budget allocator, prefix updater, summary builder) are fighting each other.

### Finding 3: Quality Degradation is Severe (HIGH)

| Quality Metric | Value | Assessment |
|----------------|-------|------------|
| ROUGE-L F1 (mean) | 0.247 | Poor — proxy responses share only ~25% of content with direct |
| Token Jaccard (mean) | 0.237 | Poor — low token overlap |
| Semantic similarity (mean) | 0.116 | Very poor — responses are semantically distant |
| Response stability (mean) | 0.897 | Acceptable |
| Code structure consistency | 0.801 | Acceptable |
| Truncation count | 14/30 | High — nearly half the turns are truncated |
| Low semantic similarity turns | 27/30 | Critical — 90% of turns have poor semantic match |
| Low token Jaccard turns | 28/30 | Critical — 93% of turns have low token overlap |
| Code block loss turns | 9/30 | High — 30% of turns lose code blocks |
| Code syntax invalid turns | 1/30 (turn 11) | Low but notable |
| Foreign markers | 0 | Clean — no model-visible markers leaked |

The proxy is losing critical content during compaction. The 66.2% token savings comes at the cost of dropping important information.

### Finding 4: Working-Tree Cliff Fix is Incomplete (HIGH)

The working tree has an incomplete cliff fix in `optimizer.py` that:
- Compares `_last_raw_prefix` against frozen prefix only (not live zone start)
- Gates the compactor to skip when the prefix hasn't changed
- Skips futile eviction attempts

However, **fold turns still break** at ~0.25 reuse. The `fold_margin_turns` and `fold_window_fraction` config fields in `config.py` (working tree) implement option (A) from CLIF_RESEARCH.md — space-based folding makes folds rare by design — but they are not yet integrated into the optimizer pipeline.

### Finding 5: Token Savings vs Quality Tradeoff is Unfavorable (HIGH)

The proxy achieves 66.2% token savings but the quality metrics indicate the savings come from **dropping content** rather than **intelligently compressing** it. The ROUGE-L recall (0.348) is higher than precision (0.226), meaning the proxy includes some relevant content but misses a lot. The length ratio mean of 0.89 suggests proxy responses are slightly shorter, but the median of 0.59 indicates many responses are much shorter.

### Finding 6: Context Utilization is Extremely Low (MEDIUM)

Final prompt tokens: 10703 out of 262144 context window (4.08% utilization). The char budget of 12000 is too aggressive — it triggers compaction on nearly every turn (30/30 turns) and eviction starting at turn 7. The budget is so tight that the compactor is forced to drop content aggressively.

### Finding 7: Dead/Redundant Modules (MEDIUM)

From REVIEW.md, several modules are dead or redundant:
- `hierarchical_index.py` — not wired into the main pipeline
- `dependency_orderer.py` — not used in the current pipeline
- `state_rag.py` — not integrated
- `prompt_templates.py` — templates exist but are not actively used for optimization

### Finding 8: Prefix Cache Reuse Ratio is Low (MEDIUM)

The prefix cache reuse ratio is 0.4787 (47.87%). This means less than half of the prompt tokens benefit from prefix cache reuse. The frozen-prefix mechanism is working (29/30 cache hits), but the 1 cache miss at turn 1 and the low reuse ratio indicate that the prefix is being invalidated too frequently.

---

## 4. Benchmark Data Summary

### 4.1 Latency Distribution

| Percentile | Direct (ms) | Proxy (ms) | Delta (ms) |
|-----------|-------------|------------|------------|
| min | 4936 | 978 | -60799 |
| mean | 37411 | 44040 | +6629 |
| median | 37128 | 39179 | +1674 |
| p90 | 57911 | 80368 | +49961 |
| p95 | 68710 | 87106 | +54878 |
| p99 | 70835 | 103379 | +68733 |
| max | 71573 | 109421 | +73054 |

### 4.2 Token Metrics

| Metric | Direct | Proxy | Raw Input |
|--------|--------|-------|-----------|
| Total prompt tokens | 759846 | 256801 | 582866 |
| Per-turn mean | 25328 | 8560 | 19429 |
| Cached tokens mean | — | 4098 | — |
| Token savings vs raw | — | 55.94% | — |
| Token savings vs direct | — | 66.2% | — |

### 4.3 Quality Metrics

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| ROUGE-L F1 | 0.247 | 0.221 | 0.078 | 0.559 |
| Token Jaccard | 0.237 | 0.224 | 0.105 | 0.418 |
| Edit similarity | 0.203 | 0.163 | 0.044 | 0.543 |
| Trigram overlap | 0.339 | 0.332 | 0.150 | 0.598 |
| Markdown structure sim | 0.327 | 0.385 | 0.000 | 1.000 |
| Vocabulary richness delta | 0.140 | 0.111 | 0.047 | 0.397 |
| Prompt faithfulness | 0.942 | 0.991 | 0.757 | 0.996 |
| Evicted content recall | 0.996 | 0.997 | 0.988 | 0.998 |
| Response stability | 0.897 | 1.000 | 0.500 | 1.000 |
| Semantic similarity | 0.116 | 0.012 | -0.136 | 0.965 |

### 4.4 Eviction & Compaction

- Char budget: 12000
- Total chars before optimization: 2,068,332
- Turns exceeding optimized target: 24/30 (starting turn 7)
- Compaction triggered: 30/30 turns
- Eviction triggered: 24/30 turns (starting turn 7)

### 4.5 Long-Horizon Quality

- Contradictions: proxy 40 vs direct 19 (proxy has 2x more)
- Fact recall at turn 30: both 1.0 (perfect)
- Context window wall hits: both 5

---

## 5. Implementation Plan

### Phase 1: Stabilize the Cache (P0 — must fix before any further optimization)

**Goal:** Eliminate the cache cliff at turn 14 and achieve >90% prefix cache reuse.

1. **Integrate `fold_window_fraction` into the optimizer pipeline** — The config field exists in the working tree but is not wired into `optimizer.py`. When `fold_window_fraction > 0`, the turn-count DRIFT trigger should be disabled and the rolling summary should fold only when emitted context exceeds the fraction of the live backend window.

2. **Fix the fold-turn break** — The working-tree cliff fix handles the frozen-prefix comparison but does not handle the case where the live zone is split across a fold boundary. The `_partition_for_budget` method needs to be aware of fold boundaries and avoid splitting cached content.

3. **Reduce compaction aggressiveness** — The char budget of 12000 triggers compaction on every turn. Increase the default or make it adaptive based on the backend's actual context window usage.

4. **Add a dryrun gate** — Before any benchmark run, the dryrun diagnosis script (`scripts/diag_dryrun_opencode.py`) must confirm that prefix cache reuse is stable across all turns. The dryrun should check:
   - Cache hit rate per turn
   - Prefix cache reuse ratio
   - No cliff at any turn

### Phase 2: Reduce Optimization Overhead (P1)

**Goal:** Bring proxy TTFT and latency below direct levels.

1. **Cache optimization results** — The optimizer output for a given prefix should be cached so that repeated requests with the same prefix don't re-run the full pipeline. The `_last_raw_prefix` mechanism already tracks this, but it needs to be extended to cache the actual optimization result (not just the prefix hash).

2. **Lazy optimization** — Defer optimization until the backend actually needs it (i.e., when the context exceeds a threshold). Currently, optimization runs on every request regardless of whether it's needed.

3. **Parallelize optimization with backend communication** — The optimizer currently blocks the request handler. Use the existing `_OPTIMIZER_EXECUTOR` to run optimization in parallel with any backend warm-up or connection setup.

4. **Simplify the pipeline for common cases** — For turns where the prefix hasn't changed (cache hit), skip the full pipeline and just append the new message to the existing optimized context.

### Phase 3: Improve Quality (P2)

**Goal:** Achieve ROUGE-L F1 > 0.5 and semantic similarity > 0.3.

1. **Implement semantic-aware compaction** — The current compactor uses heuristic rules (importance scoring, keyword matching). Replace or augment with a lightweight semantic similarity check that ensures critical information is preserved.

2. **Preserve code blocks intact** — 9/30 turns lose code blocks. The tree-sitter chunker should mark code blocks as "do not compress" and preserve them in full.

3. **Reduce truncation** — 14/30 turns are truncated. The char budget needs to be increased or the compaction needs to be smarter about what to keep.

4. **Fix the quality regression gate** — `scripts/benchmark_gate.py` should include quality metrics (ROUGE-L F1, semantic similarity) in the regression check, not just token savings.

### Phase 4: Remove Dead Code (P3)

**Goal:** Reduce maintenance burden and improve code clarity.

1. Remove or archive `hierarchical_index.py`, `dependency_orderer.py`, `state_rag.py` if they are truly unused.
2. Consolidate `prompt_templates.py` usage or remove if templates are not actively used.
3. Remove legacy benchmark support from `scripts/benchmark.py` (already done per AGENTS.md constraints).

### Phase 5: Observability (P4)

**Goal:** Make the proxy's optimization decisions transparent.

1. Add per-turn optimization metrics to the proxy response headers or a debug endpoint.
2. Expose the optimizer's decision log (why content was kept/evicted) via `MOEPT_DIAG_DUMP=1`.
3. Add a `/metrics` endpoint that exposes cache hit rate, token savings, and optimization latency as Prometheus metrics.

---

## 6. Reference: CLIF_RESEARCH.md Key Findings

The cache cliff investigation (`CLIF_RESEARCH.md`, 312 lines) identified:

1. **Root cause:** Budget/keep-window mismatch — the compactor evicts content that the backend has already cached, invalidating the KV cache entries.
2. **6 eviction mechanisms** are fighting each other: compactor, preserver, chunker, budget allocator, prefix updater, summary builder.
3. **Frozen-prefix fix** (storing only the frozen prefix as `_last_raw_prefix`) was applied but the cliff persists at turn 14.
4. **Fold turns** are an inherent design limitation — when the live zone is folded, the backend's cached entries for the folded content are lost.
5. **Option (A)** from the research (space-based folding via `fold_window_fraction`) makes folds rare by design and is the recommended approach. This is now partially implemented in the working tree config.

---

## 7. Reference: REVIEW.md Key Findings

The comprehensive architecture review (`REVIEW.md`, 1064 lines) covers:

1. **Full pipeline analysis** of all optimization stages
2. **Benchmark comparison** between 0.7.26_fix and 0.7.26_fix2
3. **Reference project techniques** from other context optimization systems
4. **Dead module inventory** identifying unused code
5. **Implementation plan** with phased approach

---

## 8. Recommendations

1. **Do not run the benchmark until the dryrun is greenlit.** The current proxy is slower than direct and degrades quality. The dryrun diagnosis script must confirm stable cache reuse before any benchmark is run.

2. **Prioritize Phase 1 (cache stability).** Without stable prefix cache reuse, all other optimizations are wasted — the backend re-processes evicted content anyway.

3. **Integrate `fold_window_fraction` into the optimizer pipeline immediately.** This is the most impactful single change: it makes folds rare by design, which directly addresses the cache cliff.

4. **Increase the default char budget.** 12000 is too aggressive for a 262144 context window. A budget of 20000-30000 would reduce compaction frequency while still achieving meaningful savings.

5. **Add quality metrics to the regression gate.** Token savings alone is not a sufficient metric — the regression gate must also check ROUGE-L F1 and semantic similarity.

6. **Consider whether the proxy should be disabled for short conversations.** If the conversation is under 5 turns, the optimization overhead exceeds any benefit. Add a `MOEPT_OPTIMIZATION__MIN_TURNS` config flag.

---

## 9. Appendix: Working Tree Changes

### config.py (unstaged)

Added two new fields to the optimization config:

- `fold_margin_turns` (int, default=0): DRIFT fold margin — allows the live zone to drift past the keep window before a batch fold pulls it back. Larger values = rarer folds = better prefix-cache reuse but larger context between folds.
- `fold_window_fraction` (float, default=0.0): Space-based folding — when >0, turn-count DRIFT is disabled and the rolling summary folds only when emitted context exceeds this fraction of the live backend window. Start around 0.25-0.5 and benchmark the tradeoff.

### optimizer.py (unstaged)

Working-tree cliff fix changes:
- Frozen-prefix-only comparison for `_last_raw_prefix` (excludes rolling summary block)
- Compactor gate to skip when prefix hasn't changed
- Futile-eviction skip when the live zone is already within budget

### diag_dryrun_opencode.py (improved)

The dryrun script has been improved with:
- Proxy lifecycle management (auto-start on port 8080)
- Internal process management
- `MOEPT_DIAG_DUMP=1` writes `/tmp/diag_*.json` files
- `MOEPT_DIAG_STAGE=1` enables `[DIAG]` stderr output

---

*This review was generated from the project's existing documentation (README.md, notes.md, cache_preservation_guide.md, CLIF_RESEARCH.md, REVIEW.md) and the latest benchmark results (benchmark_opencode_30_1_0.7.26_fix2.json).*
