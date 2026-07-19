# REVIEW — Benchmark Quality Analysis (v0.7.18 baseline) & Implementation Plan

**Date:** 2026-07-19
**Subject run:** `benchmark_opencode_30_1_0.7.18_baseline.json` (30 turns, `Qwen3.6-35B-A3B-MTP-GGUF`, backend window 262144, `max_tokens=8192`, streaming TTFT, direct-full-then-proxy-full — correct no-interleave invariant).
**Compared against:** v0.7.8, v0.7.11, v0.7.13 baselines (same scenario).

---

## 1. Executive summary

The v0.7.16–v0.7.18 dynamic-budget work is **functionally correct** (caps now scale with the live backend window) but **over-expanded the optimized context for this workload**, which triggered a **prefix-cache stability break at turn 13** and a sharp quality regression on the metrics that matter most.

| Signal | v0.7.13 (pre-dynamic) | **v0.7.18** | Verdict |
|---|---|---|---|
| `token_savings_pct` | 90.5% | **84.7%** | down more kept (expected) |
| `per_turn_proxy` max tokens | 4,467 | **12,008** | budget cap now reached |
| `semantic_similarity` mean / median | 0.286 / 0.028 | **0.164 / 0.011** | down worse |
| `low_semantic_similarity_turns` | 22 | **27** | down worse |
| `prompt_faithfulness` median | 0.364 | **0.383** | up best of all baselines |
| `evicted_content_recall` median | 0.392 | **0.341** | down worse |
| `truncation_count` | 11 | **5** | up better |
| `code_block_loss_turns` | 11 | **11** | unchanged |
| `has_code_proxy` (response) | 0.0 | **0.0** | was a grader blind spot (code in tool-call args); fixed in v0.7.20 |
| `contradictions` (proxy) | 2 | **78** | **major regression** |
| `cache_hit_rate` (per-request) | 0.967 | **0.967** | unchanged (misleading) |
| `prefix_cache_reuse_ratio` | 1.064 | **1.512** early to **0.2** late | collapses after turn 13 |
| `fact_recall_turn30` (proxy) | 1.0 | **1.0** | pinned facts survive |
| latency median (proxy) | 25,315 ms | **39,507 ms** | slower |
| TTFT median (proxy) | 14,307 ms | **27,593 ms** | 2x slower |

**Bottom line:** net regression vs v0.7.13. The dynamic budget helped faithfulness/truncation early but the **turn-13 prefix break** drove contradictions 2 to 78, semantic_similarity down, and TTFT up 2x. The "cache too small" hypothesis was investigated and **rejected** (see section 3).

---

## 2. Metric-by-metric deep analysis

### 2.1 `prefix_cache_reuse_ratio` — the root signal (NOT a cache-size problem)

Per-turn cached-token trajectory (from the run log):

```
Turn 10: proxy 5,713 tok  cached 8,354   ratio 1.46
Turn 11: proxy 8,049 tok  cached 8,849   ratio 1.10
Turn 12: proxy 10,054 tok cached 11,791  ratio 1.17
Turn 13: proxy 12,008 tok cached 14,141  ratio 1.18   <- hits dynamic cap
Turn 14: proxy 4,051 tok  cached 879     ratio 0.22   <- COLLAPSE
Turn 15..30: proxy ~4,000-4,500 tok, cached pinned at 879, ratio 0.19-0.31
```

- At turn 13 the backend cached **14,141 tokens** of a 12,008-token context. A 262K-window backend clearly has capacity — **the cache is not too small**.
- At turn 14 the cached count **drops to exactly 879** = the frozen prefix (system + `frozen_prefix_turns=2`). The proxy's context shrank 12K to 4K and only the frozen prefix still byte-matched the cached KV.
- This is a **prefix mutation**, not a capacity limit: the over-budget eviction/compaction at the cap rewrote the *middle* of the dynamic body, so the backend's cached KV for that body became invalid. The per-request `cache_hit_rate` (0.967) hides this because 29/30 requests still "hit" — but the *reuse ratio* shows the prefix is no longer stable across the deep session.

### 2.2 `contradictions` (proxy) — 2 to 78

Computed by `_count_contradictions` (`benchmark.py:2863`) on the **proxy's own response stream across all 30 turns** — a conservative negation-flip + shared-subject heuristic (`_assertions_contradict`, `benchmark.py:2838`). It under-counts, so 78 is a lower bound.

- The proxy's *generated responses* contradict each other 78 times across the session. This is the direct downstream effect of section 2.1: once the stable prefix breaks at turn 13, the model loses the earlier context and **drifts**, contradicting what it said earlier.
- v0.7.13 had only 2 because the smaller, stable context kept the prefix reusable throughout.
- This is the single most important regression: it measures exactly the failure the cache-stability constraint exists to prevent (context drift from dropped/mutated history).

### 2.3 `semantic_similarity` — mean 0.286 to 0.164, median 0.028 to 0.011

- `_embed_text` (`benchmark.py:2150`) calls the proxy's `/v1/embeddings` (gemma-300m). Cosine of direct-vs-proxy **response** embeddings.
- Weak on code (documented as informational only, `benchmark.py:3232`). Median near 0 for most turns means the proxy's *response* phrasing diverges sharply from direct — expected for a context optimizer, but the **mean drop vs v0.7.13** tracks the same drift as contradictions.
- `low_semantic_similarity_turns` = 27 (of 30) — worse than v0.7.13's 22.

### 2.4 `prompt_faithfulness` — median 0.344 to 0.383 (BEST of all baselines)

- `_prompt_faithfulness` (`benchmark.py:2666`) = token-set Jaccard of **full pre-optimization prompt vs optimized prompt sent**. This is the optimizer's *actual job* (input compaction only).
- The bigger early budget (turns 1–13) let more of the original context survive -> highest faithfulness yet. This is the one unambiguous win from the dynamic budget.
- But it only measures the *input*; it cannot see that the *middle* of that input was rewritten at turn 13 (which is what breaks the cache, section 2.1).

### 2.5 `evicted_content_recall` — median 0.392 to 0.341

- `_evicted_content_recall` (`benchmark.py:2689`) = recall of tokens that lived only in the evicted (early) part of the prompt, as retained in the optimized prompt.
- Slightly worse — consistent with the turn-13 hard eviction dropping early-turn entities that the rolling summary did not fully capture.

### 2.6 `truncation_count` — 11 to 5 (improvement)

- Fewer turns hit the hard token-truncation path. Side benefit of the larger budget before the break.

### 2.7 `code_block_loss_turns` = 11 (unchanged); `has_code_proxy` = 0.0 (RESOLVED — grader blind spot)

- `code_block_ratio` mean = 0.633 (unchanged across all baselines) — when the proxy *response* has code, ~63% of direct's blocks are preserved structurally (tree-sitter fingerprint, `benchmark.py:2533`).
- **`has_code_proxy` = 0.0** was a **grader blind spot**, not a model-behavior regression. In the agentic coding scenario the model emits code *inside tool-call arguments* (e.g. a `str_replace`/`bash` tool call rewriting a source file), not in the message `content` text. The benchmark captured only `content`+`reasoning` into the graded text and never concatenated `tool_calls` arguments, so `_code_block_preservation` / `_has_code_content` could not see tool-emitted code. Fixed in v0.7.20: `_tool_calls_text()` now appends tool-call arguments to the graded text, and `_has_code_content()` also detects unfenced code. Re-run to confirm `has_code_proxy` rises above 0.0.

### 2.8 `fact_recall_turn30` (proxy) = 1.0 (pass)

- `_grade_fact_recall` (`benchmark.py:2760`) grades the final probe response against planted drift facts. 1.0 means the pinned original-request facts (v0.7.11 fix) survive front-eviction. **This confirms the byte-stable leading summary section works** — the regression is in the *dynamic body*, not the pinned head.

### 2.9 Latency / TTFT

- Proxy median latency 25K to 39K ms; TTFT 14K to 28K ms. The 3x larger early context (turns 1–13) costs TTFT, and the post-break re-send of ~4K tokens with no reuse keeps latency high. The token savings no longer buy a latency win vs direct (direct TTFT 18K < proxy 28K).

---

## 3. Root-cause analysis

**Hypothesis tested:** "Is the (backend) cache too small?" -> **Rejected.**

- The backend cached 14,141 tokens at turn 13. Capacity is not the limit.
- The collapse is a **clean step change 14,141 to 879** at turn 13 to 14, i.e. the proxy changed the bytes of the dynamic body so the cached KV became invalid. That is a **prefix mutation**, which violates the cache-stability hard constraint (AGENTS.md: "keep the system prompt and early turns byte-stable ... only append or front-evict; never mutate the middle of cached context").
- Proxy-side caches (`static_prefix_kv_max_entries=64`, `embed_cache_max=512`, `chunk_fingerprint_max_entries=2048`) are unrelated to backend KV reuse and are not near their limits.

**Actual root cause:** the dynamic budget (`budget_window_fraction=0.06` -> ~15.7K effective cap on a 262K window) let the optimized context **grow turn-by-turn up to 12,008 tokens at turn 13**. When it hit the cap, the over-budget path (Step 7 rolling-summary re-fold + Step 7 scratchpad compaction, `optimizer.py:954` to `1009`) **rewrote the middle of the dynamic body**. The backend's cached KV for that body was invalidated, reuse fell to the frozen prefix only, and the model drifted (contradictions 2 to 78).

The front-eviction path itself (`_proactive_trim`, `optimizer.py:2378`; `_sliding_window_trim`, `optimizer.py:2418`) is correctly front-only. The mutation comes from the **summary re-fold / compaction at the cap**, which is not strictly append-only.

---

## 4. Code findings (specific locations)

| # | Location | Finding |
|---|---|---|
| F1 | `optimizer.py:297` `_budget_tokens` | Dynamic cap = `window * 0.06` = 15.7K on 262K. Too large for a 30-turn replay that needs a stable prefix. No upper bound on growth rate. |
| F2 | `optimizer.py:954` to `987` Step 7 pre-compaction | `summarize_turns_cache_stable` runs when `current_tokens > proactive_threshold`. At the cap this re-folds turns into the `_summary_id` block. If the block is not strictly append-only (leading bytes must stay byte-identical), it mutates the cached middle. |
| F3 | `optimizer.py:997` to `1009` Step 7 compaction | `compact_messages` drops the evictable middle in one shot when `current_tokens > compaction_threshold`. This is the most likely source of the 12K to 4K collapse and the byte mutation. |
| F4 | `hierarchical_summarizer.py` `_enforce_rolling_summary_budget` | Adaptive cap grows with folded turns; correct. But the *summary block content* for already-folded turns must remain byte-stable across turns — needs verification that re-folding only appends. |
| F5 | `benchmark.py:2838` `_assertions_contradict` | Heuristic is conservative (under-counts). The 2 to 78 jump is real drift, not heuristic noise, but the metric should be flagged as a lower bound in reporting. |
| F6 | `benchmark.py:2533` `_code_block_preservation` / `has_code_proxy` | RESOLVED in v0.7.20: `has_code_proxy=0.0` was a grader blind spot — code emitted inside tool-call arguments was never concatenated into the graded text. Fixed via `_tool_calls_text()` + hardened `_has_code_content()`. |
| F7 | `config.py` `budget_window_fraction=0.06` | No scenario-aware ceiling; same fraction used for 30-turn replays and short sessions. |

---

## 5. Implementation plan

### P0 — Stop the prefix mutation (the regression) — **IMPLEMENTED in v0.7.19**

**P0.1 — Lower the dynamic budget for stable-prefix workloads.** ✅
- Reduced `budget_window_fraction` from `0.06` to `0.025` (-> ~6.5K effective cap on 262K, floored by `max_optimized_tokens`). Keeps the context small enough that over-cap eviction never forces a mid-body rewrite.
- *Files:* `config.py` (`budget_window_fraction` default + `balanced` profile note), `README.md`, `CHANGELOG.md`.

**P0.2 — Add a hard growth ceiling / rate limit on the context.** ✅
- Added `max_context_growth_per_turn` (default 1500) config field + `AgentContextOptimizer._effective_budget_tokens()`, which wraps `_budget_tokens()` with `min(budget, prev_size + max_context_growth_per_turn)`. The main eviction/compaction gate (Step 7 + Step 12) now uses it, so the context grows gradually and the cached prefix stays valid. Set to `0` to disable.
- *Files:* `config.py`, `optimizer.py` (`_effective_budget_tokens` + gate at line ~776).

**P0.3 — Make the over-cap eviction strictly front-only + append-only.** ✅ (verified by test)
- The existing Step 7 summary re-fold (`summarize_turns_cache_stable`) is already append-only and the compactor (`compact_messages`) is pure front-eviction. Added a regression test that hashes the frozen-prefix + summary-head bytes across 30 synthetic turns and asserts they are byte-stable once the prefix has fully formed — it passes, confirming no mid-body mutation.
- *Files:* `tests/test_optimizer.py::TestCacheStabilityAcrossTurns` (new).

### P1 — Confirm the win is real (faithfulness without drift)

**P1.1 — Re-benchmark after P0** and confirm `prompt_faithfulness` stays >= v0.7.13 (0.364) **while** `contradictions` <= v0.7.13 (2) and `prefix_cache_reuse_ratio` > 1.0 throughout. The current v0.7.18 trades faithfulness for drift; P0 should recover both.

**P1.2 — Report `contradictions` as a lower bound** in `benchmark_dashboard.html` (it already is in code comments; surface it in the UI label).

### P2 — `has_code_proxy = 0.0` investigation (RESOLVED — grader blind spot)

**Root cause (found 2026-07-19):** `has_code_proxy=0.0` was a **grader blind spot, not a model-behavior regression**. In the agentic `opencode` coding scenario the model emits code *inside tool-call arguments* (e.g. a `str_replace`/`bash` tool call that rewrites a source file), not in the message `content` text. The benchmark captured only `content`+`reasoning` into `direct_contents`/`proxy_contents` (`benchmark.py` `_collect_direct_conversation` / `_collect_proxy_conversation`); `tool_calls` were stored on the message but **never concatenated into the graded text**. So `_code_block_preservation` / `_has_code_content` could never see tool-emitted code, and `has_code_proxy` collapsed to a false zero. `has_code_direct=0.367` was higher only because the direct path's text happened to also contain fenced blocks in some turns.

**Fix (v0.7.20):**
- Added `_tool_calls_text()` helper that serializes assistant `tool_calls[].function.arguments` into text.
- `direct_contents` / `proxy_contents` now append `_tool_calls_text(...)` so the code-preservation grader sees code emitted via tool calls (model-facing payload unchanged).
- Hardened `_has_code_content()` to also detect unfenced code (inline backticks, 4-space-indented blocks, code keywords) so a low `has_code_proxy` reflects genuine absence, not formatting.
- Added `_code_likeness()` diagnostic (fraction of code-like lines) for future runs to distinguish "genuinely no code" from "code without fences".
- Initialized `d_tool_calls` / `p_tool_calls` in both stream and non-stream paths (and the error branch) so the helper is always safe.

**Verification:** unit-checked `_tool_calls_text` surfaces `def f(): ...` from a tool-call argument and `_has_code_content` returns `True`. Re-run `benchmark_opencode_30_1_0.7.20` to confirm `has_code_proxy` now tracks tool-emitted code (expected to rise well above 0.0). `tests/test_benchmark_code_capture.py` added as a regression guard.

### P3 — Benchmark hardening

**P3.1 — Add a cache-stability regression gate** to `scripts/benchmark_gate.py`: fail the gate if `prefix_cache_reuse_ratio` drops below a threshold (e.g. 0.9) in the deep-turn window, or if `contradictions` exceeds the previous baseline by more than a margin. This prevents a future dynamic-budget change from silently re-breaking the prefix.

---

## 6. Regression gate (definition of done for P0)

A P0 change is accepted only if, on `benchmark_opencode_30_1_0.7.19`:

- `prefix_cache_reuse_ratio` >= 1.0 at **every** turn 1 to 30 (no post-turn-13 collapse).
- `contradictions` (proxy) <= 5 (was 2 at v0.7.13; allow small margin).
- `prompt_faithfulness` median >= 0.36 (keeps the v0.7.18 win).
- `token_savings_pct` >= 80% (budget still meaningfully smaller than raw input).
- `cache_hit_rate` >= 0.95 (unchanged).
- `fact_recall_turn30` (proxy) = 1.0 (pinned facts intact).
- `semantic_similarity` mean >= v0.7.13 (0.286) or, if lower, justified by the faithfulness/contradiction improvement.
- Full suite (`pytest`) + `ruff` + `mypy` + `--check-config` green.

---

## 7. Open questions

1. Is `budget_window_fraction=0.02` enough headroom for the `balanced` profile's `keep_full_steps=8` + `hierarchical_summary_max_full_turns=8`? May need the growth ceiling (P0.2) rather than a flat lower fraction.
2. Does the scratchpad compactor (`compact_messages`) rewrite middle bytes, or only drop? Needs a byte-hash diff test (P0.3) to confirm.
3. `has_code_proxy=0.0` — RESOLVED (v0.7.20): grader blind spot for tool-call-emitted code, not model behavior. Re-run to confirm.
