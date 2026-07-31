# REVIEW — MOE-ptimizer Architecture Review & Implementation Plan

**Reviewer role:** senior LLM-inference architect (vLLM / llama.cpp / OpenAI API /
OpenCode / Qwen3-MoE-MTP / tree-sitter / embeddings / prefix-caching / KV-cache /
agentic coding systems).

**Target:** a transparent, OpenAI-compatible proxy (`:8080`) that compacts the
**input** context sent to a llama.cpp/Lemonade backend (`:13305`) serving
`Qwen3.6-35B-A3B-MTP-GGUF` (a MoE model with Multi-Token-Prediction heads) on
local, hardware-limited infrastructure. Mission: improve **TTFT and TPS** while
**not degrading response quality**, by keeping the context as lean as possible
**without** triggering backend KV-cache refills.

**Version reviewed:** `0.7.26` (`pyproject.toml`), working tree containing the
2026-07-30 eviction-cliff fix (unstaged).

---

## Implementation status (Phase 1 + 2 + 3 — done, validated)

The following items (§5) are **implemented and validated** (pytest 440 passed / 11
pre-existing failures — zero new; ruff 1 pre-existing; mypy 59, zero new;
`--check-config` exit 0; dry-run cache-stability preserved — breaks steady at 7):

**Phase 1 — correctness, leaks, adaptive budget, cleanup** (commit `4b79eb5`):

| Item | Status | Evidence |
|---|---|---|
| §4.8.8 `tokenize_count` never-awaited | ✅ fixed | remote `/tokenize` works in async app; running-loop calls no longer disable remote counting |
| §4.8.3 `_volatile_turn` leak | ✅ fixed | `_*` keys scrubbed at the backend boundary in `app.py` |
| §4.10.1 `_SLOT_MAP` unbounded | ✅ fixed | LRU `OrderedDict`, cap 512 |
| §4.2.5 `OutputShaper` constraint violation | ✅ disabled | removed from proxy path; 2 tests converted to constraint guards |
| §4.8.1 embedding deadlock + dead RAG | ✅ fixed | genuinely-async fetch on a dedicated loop; `CircuitBreaker.call_async`; one initialized service shared per optimizer; degradation marker |
| §4.8.2 cross-session summarizer/delta | ✅ fixed | per-optimizer `HierarchicalSummarizer` + `CodeDeltaEncoder`; isolation regression test |
| §4.9.1 construction on the event loop | ✅ fixed | `get_or_create` offloaded to the optimizer executor |
| §3.1/§4.2.2 **adaptive budget** (skeleton) | ✅ implemented | horizon-growing, window-capped, monotonic-floor budget; dry-run breaks **10 → 7**; 4 new tests |
| §7 dead/phantom modules | ✅ deleted | 9 modules + 2 config flags removed; docs synced; dry-run breaks **unchanged** (no-op stages had zero effect) |
| §4.12.2 CI cache gate | ✅ added | `diag_dryrun_opencode.py --max-breaks` + backend-conditional `dev.sh` step |

**Phase 2 — TTFT/TPS throughput**:

| Item | Status | Evidence |
|---|---|---|
| §4.9.2/§4.9.3/§4.9.5 probing | ✅ fixed | sub-probes parallelized within dependency pairs; single-flight on TTL expiry (no thundering herd); long-lived httpx client closed in lifespan |
| §4.12.1 real TTFT metric | ✅ fixed | first CONTENT chunk timestamped; `avg_ttft_ms` (global + per-session) distinct from `avg_latency_ms`; dashboard relabeled |
| §4.4.2 O(n²) floor pass | ✅ fixed | running suffix total instead of re-tokenizing the growing candidate; dry-run avg turn time **443 → 201 ms**, breaks unchanged |
| §4.4.3 per-text token memo | ✅ fixed | local `count()` caches per-text so unchanged prefix messages aren't re-tokenized on a list miss |
| §4.9.4 CPU on the event loop | ✅ fixed | `_serialize_messages_text` length pre-check (no build-then-discard); `get_session_state()` offloaded to the executor |
| §4.12.7 `context_window_wall` artifact | ✅ fixed | requires `min_turn=5` + `sustained=2` consecutive breaches; 3 new tests |

**Phase 3 — context efficiency & quality**:

| Item | Status | Evidence |
|---|---|---|
| §4.1.1 **error-aware tool-output compression** | ✅ implemented | `ToolOutputCompressor` now keeps error/stack-frame/file:line/summary lines and drops passing-test/progress noise (failure log 2.1K→0.65K with diagnostics intact); pure-success logs collapse to the one-line verdict; `ToolOutputFilter` no longer destroys failure detail; 6 new tests; dry-run breaks unchanged |
| §4.2.6 redundant volatile anchor | ✅ gated | `volatile_quality_anchor_enabled` (default `true` = current behavior); the anchor is a 3rd copy of the original request (also in frozen prefix + summary head) — set `false` to drop ~900 chars/over-budget-turn once benchmark quality confirms; test added |
| §4.3.3 evicted-code skeleton index | ✅ already present | the `code_ledger` (`CODE_LEDGER_MAX_SIGS`) already accumulates evicted-turn code signatures into a compact index |

**Adaptive budget — code-density term (implemented, but see caveat):**
`adaptive_code_density_factor` (default 0.25) grows the budget by a fraction of the
request's code tokens (§3.1 task-complexity signal). Implemented + tested +
cache-stable. **Caveat (measured):** on the 262K-window opencode benchmark it has
**zero effect** — the budget ceiling (~39K) is already far above the ~22K context
(4% utilization), so the budget is non-binding and the term changes nothing. The
folds that drive the reuse regression come from the turn-count-based **DRIFT
trigger** (`live_count > keep + fold_margin` in `hierarchical_summarizer.py`), which
a budget term cannot influence. The code-density term only matters when the budget
is binding (a smaller window or a much larger working set).

**Fold-geometry experiments (implemented + swept, dry-run opencode/30):** the binding
constraint on context size is the fold, not the budget, so two fold controls were
added and swept (both default OFF = current behavior; both cache-stable, tested):

- `fold_margin_turns` (DRIFT margin) and `fold_window_fraction` (space-based
  folding: disables the turn-count DRIFT trigger and folds only when the context
  exceeds a fraction of the live window; the budget ceiling rises to match and the
  scratchpad compactor + front-evictors are skipped so the cliff is not re-created).

| Config | Breaks | turn-30 context |
|---|---|---|
| baseline (margin=keep=6) | 7 `[16,17,20,21,24,27,28]` | 22.3K |
| `fold_margin_turns=18` (3×) | **6** `[18,22,23,25,27,28]` | 23.0K |
| `fold_margin_turns=30` (5×) | 6 (same) | 23.0K |
| `fold_window_fraction=0.1` | 7 `[20,21,23,24,26,27,29]` | 23.0K |
| `fold_window_fraction=0.25` | 7 `[22,23,26,27,28,29,30]` | 31.8K |
| `fold_window_fraction=0.5` | 7 (same) | 31.8K |

**Findings:** (1) widening the DRIFT margin gives only a **marginal** improvement
(7→6 breaks). (2) Space-based folding does **not** reduce the break *count* for a
30-turn session — it still breaks 7 times, just later, and grows the context
(31.8K vs 22.3K, worse for savings). The first space-based attempt re-created the
every-turn cliff (11 consecutive breaks) because disabling DRIFT left no rolling
summary to gate the scratchpad compactor; that is now fixed (compactor +
front-evictors skipped under space-based folding). (3) **The ~7 breaks are
structural** — they persist regardless of fold trigger, so they are not purely
fold-driven; the remaining breaks likely come from the boundary transforms /
volatile tail interacting with the backend's cache, and need per-turn backend-cache
instrumentation to pin down. Space-based folding is still the right design for
**very long** sessions (folds only near the window) but is not a win at 30 turns.

**Conclusion:** neither fold knob meaningfully recovers reuse at 30 turns. The
reuse regression vs the pre-Phase-1 baseline (0.81→0.48) is partly single-round
variance and partly the leaner context; recovering it further needs per-turn
backend-cache instrumentation to attribute the residual ~7 breaks, then a targeted
fix. Both fold controls are committed as tunable, validated options.

**§4.8.4 quality-profile-vs-env precedence — DONE.** `apply_quality_profile` now
fills only fields the user left at their default (`field not in
target.model_fields_set`), so an explicit env/`.env`/constructor value wins over the
preset (e.g. `MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS=24000` is no longer reset to
12000). Tests updated to isolate the preset (clear `MOEPT_*` env); new
`test_explicit_env_wins_over_profile`. **Environment caveat:** the precedence is
correct, but a stale exported `MOEPT_*` shell environment overrides the `.env`. The
shipped `.env` had pre-profile values (`KEEP_FULL_STEPS=3`, `MAX_OPTIMIZED_TOKENS=3000`,
…) that would now win and re-introduce the turn-12 cliff; those fields are commented
out in `.env` so the balanced profile applies. **The user must also unset stale
exported `MOEPT_*` shell vars (or start a fresh shell) for the balanced profile to
take effect** — with them cleared, the dry-run shows 5 breaks (no cliff).

**Deferred to a benchmark-validated follow-up:**
- §3.1 adaptive budget **amortization trigger** — sweep confirmed fold knobs don't
  recover reuse at 30 turns; needs per-turn backend-cache instrumentation first.
- §4.4.1 zero-copy SSE passthrough — a rewrite of the working streaming path; the
  proxy must inspect every chunk (usage / thinking / content), so the "minimal"
  benefit is limited. Needs careful design.
- §4.1.2 reversible compression + retrieval handles — a content-addressed store +
  `expand(id)` tool (MCP-style); a substantial new feature changing the interaction
  model. Needs dedicated design.
- §4.5.3 syntactic code slicing — **primitive done** (`slice_code_to_query` in
  `code_block_optimizer.py`, 11 tests): keeps the imports header + every top-level
  def named in the query, collapses the rest, strictly fail-open / never-expands.
  **Pipeline wiring still deferred** (needs a config gate + a compression point that
  has the current query, and must compress on *first appearance* to avoid a new
  prefix break). Blocked on the tree-sitter finding below.

**New finding — the tree-sitter AST path is, and has been, silently disabled.** The
installed `tree_sitter` is **0.25.2**, whose API exposes `tree.root_node`,
`node.type`, `node.child_count`, `node.byte_range` as **properties** (and
`byte_range` is a plain `(start, end)` tuple). But `code_chunking.py` calls them as
**methods** (`tree.root_node()`, `node.kind()`, `node.child_count()`,
`node.byte_range()` — its docstring even asserts this is "the tree-sitter >= 0.25
API"). Every one of those raises inside the `try/except Exception` guard, so
`chunk_code_with_treesitter` **always falls back to line-based `chunk_text_fallback`**
— verified byte-identical output, and the AST path's signature "prepend imports to
every chunk" behavior is absent. Net: the project's "language-aware AST chunking"
has been a silent no-op. This is a concrete instance of the "silently swallowed
exceptions" weakness (§4.5 #5): uvicorn at `log_level=warning` + blanket
`except Exception` means a permanently-failing stage leaves no test-visible signal.
**Resolved:** `code_chunking.py` is ported to the correct 0.25.2 property API
(`parse(bytes)`, `tree.root_node`, `node.type`, `node.child_count`, `node.byte_range`
as a `(start, end)` tuple, byte-offset slicing on the encoded bytes) and the AST path
now genuinely runs — verified output diverges from the line fallback and the imports
header is prepended to every chunk. `test_ast_path_actually_runs`
(`tests/test_code_chunking.py`) locks this in by asserting the imports land in a
non-first chunk (which the fallback never does). `slice_code_to_query` uses the same
correct API. **Observed wart (pre-existing, now visible):** chunk 0 emits the imports
twice — once via the header prefix and once as the file's leading top-level nodes;
only chunk 0 is affected. Left as-is to keep the change scoped to the API fix;
candidate for a small follow-up (skip header-kind nodes in the body loop). **Because
this changes chunking output, it needs a benchmark round to confirm no regression in
token savings / quality / prefix-cache reuse.**

**Phase 4 — god-object decomposition (started).** Extracted `BudgetGovernor`
(`budget_governor.py`) out of the 3.3K-line `optimizer.py` (~250 lines moved): it
owns the token-calibration, horizon, code-density, and last-optimized-count state
plus all budget/growth/shrink/dynamic-cap math; the optimizer keeps thin delegates.
Behavior-preserving (dry-run break pattern identical). Remaining god-object work:
extract the eviction/trim machinery and the stage runner.

**Per-turn break attribution (done — verbose dry-run, post-fix breaks `[14,19,20,23,26]`):**
the residual breaks have two distinct causes, visible in the per-turn `msg[i]
content changed` + `byte_diff first_diff_at_char`:
- **Fold turns (14, 20, 23, 26):** the first divergent message is the rolling-summary
  block (`msg[16]` becomes `Context summary (rolling):…` and grows each fold), which
  shifts the entire live zone after it. This is the summary-in-middle geometry — the
  durable fix is the compaction-geometry redesign (below), not a knob.
- **Late tool-output compression (19):** a tool output is compressed *after* first
  appearance (`msg[26]` pytest output `4998 → 15` chars, `first_diff_at_char=12432`).
  The output entered the prefix uncompressed, then the boundary filter compressed it
  a turn later, breaking the prefix at that message. **Fixable:** compress tool
  outputs on *first appearance* (so the compressed form is what gets cached), which
  removes this class of break entirely. This is a concrete, safe win worth doing.

**Still deferred:** §4.1.2 reversible compression + retrieval handles, §4.4.1
zero-copy SSE, and the compaction-geometry redesign (summary placement) that would
remove the fold-turn breaks.

---

## Latest benchmark — `0.7.26_fix2` (post-Phase-1/2/3)

Single-round opencode run (30 turns, `balanced`, 262K window) after Phases 1–3,
compared against the pre-Phase-1 `0.7.26_fix` baseline. **The result is a clear
tradeoff — quality and token savings improved; prefix-cache reuse and TTFT
regressed.**

| Metric | `fix` (pre-P1) | `fix2` (post-P3) | Direction |
|---|---|---|---|
| ROUGE-L F1 (mean) | 0.139 | **0.247** | ✅ +78% |
| token-Jaccard (mean) | 0.138 | **0.237** | ✅ +72% |
| `length_ratio` (mean) | 2.02 | **0.888** | ✅ verbosity fixed (proxy was 2× too long) |
| `code_syntax_validity` | 0.95 | 0.967 | ✅ |
| Token savings | 58.0% | **66.2%** | ✅ |
| Final context / utilization | 18.6K / 7.1% | **10.7K / 4.1%** | ✅ leaner |
| **Prefix-cache reuse ratio** | 0.81 | **0.48** | ❌ −41% |
| per-turn `cached` (mean) | 8,810 | **4,098** | ❌ |
| TTFT mean (proxy / direct) | 23.2 / 14.3 s | **26.6 / 13.3 s** | ❌ proxy ~2× direct |
| TTFT median (proxy / direct) | 8.6 / 16.5 s | 18.8 / 14.2 s | ❌ (median regressed) |
| Contradictions (proxy / direct) | 33 / 39 | **40 / 19** | ❌ proxy now > direct |

**Root cause of the reuse/TTFT regression (from the per-turn log):** the proxy's
backend `cached` collapses to **~864 tokens (frozen prefix only)** on ~12 break
turns, with long consecutive runs (16–18, 21–23, 27–30) where the rolling-summary
**fold fires every turn**, re-prefilling the whole ~12–15K body each time. The
dry-run (proxy-side common-prefix) shows only 7 breaks, so the backend is breaking
cache *more* than the optimized prompt changes — the fold is firing too aggressively
in the back half of the session. This is precisely what the **deferred
adaptive-budget amortization trigger** (§3.1) is meant to fix: fold only when the
cumulative prefill saved over the remaining horizon exceeds the one-time re-prefill
cost. At 4% window utilization there is no space pressure justifying a fold every
turn.

**What improved and why:** the error-aware tool-output compression (§4.1.1) collapses
passing-test/progress noise and keeps diagnostics, so tool outputs are much smaller
(leaner context, +8pt savings) and the model sees cleaner input (ROUGE-L/Jaccard up,
verbosity normalized). The quality gains are real and substantial.

**Honest assessment:** the mission is TTFT **+** quality. Phase 1–3 delivered the
quality half but regressed the TTFT half via over-aggressive folding. The
highest-value next work is the **amortization trigger** to make folds rare (the
context is at 4% of the window — folds are buying nothing in space terms), which
should recover reuse/TTFT *while keeping* the quality gains. The contradictions
uptick (proxy 40 vs direct 19) also argues for folding less (less lossy
summarization).

---

## 0. Method — what was actually validated

This is not a read-only opinion. The following were executed against the live tree:

- **Dry-run diagnosis** (`scripts/diag_dryrun_opencode.py --persistent-session
  --turns 30`, pycache cleared, proxy auto-managed). Result:
  `breaks: [13,14,15,16,18,20,22,24,26,28]`. Non-fold turns (17,19,21,23,25,27,
  29,30) read **REUSED**; fold turns (every 2nd from 18) and the post-first-fold
  settling turns (13–16) **BREAK**. Context sawtooths at **16–18K tokens**
  (turn 30 = 18,607), i.e. **well above the stated 12K budget** — the budget is
  not actually enforced. This matches `CLIF_RESEARCH.md` exactly.
- **Fresh full benchmark** (`scripts/benchmark_opencode_30_1_0.7.26_fix.json` +
  `.log`, **post-cliff-fix**, 30 turns × 1 round, opencode scenario, `balanced`
  profile) — the primary data source for §1. Per-turn `cached`/`prefix_hits` in the
  `.log` show the live cache behavior. (The older pre-fix
  `benchmark_opencode_30_1_diag_v2.json` is kept only as a before/after contrast.)
- **Three deep code audits** of `optimizer.py` (3,416 lines), the runtime/serving
  layer (`app.py`, `embedding.py`, `backend_capabilities.py`, `session_manager.py`,
  …), and 23 "optimization-stage" modules, with `file:line` verification.
- **Reference projects** reviewed: headroom, snip, rtk, lean-ctx, swe-pruner
  (techniques in §6).

Backend confirmed live (NPU device, `embed-gemma-300m-FLM` loaded).

> **Refresh (this revision):** §1, §3, §4.2.2, §4.2.5, §4.7, §4.8 and §5 were
> updated against the fresh `0.7.26_fix` benchmark and re-framed around
> **dynamic, smartly-assigned budgets** (not a fixed hard cap) per maintainer
> direction. The headline conclusion changed: the cliff fix **largely fixed cache
> reuse (22 % → 81 %)** and **TTFT median now beats direct**, but it traded away
> token savings and **regressed quality**, because the context is still governed by
> a fixed 12 K cap that is *neither enforced nor appropriate* — see §4.2.2.

---

## 1. Executive summary

The 2026-07-30 cliff fix was a real win: **prefix-cache reuse jumped from 22 % to
81 %** and **proxy TTFT median now beats direct** (8.6 s vs 14.8 s). But the fix
worked by *letting the context grow* — and that exposes the deeper problem this
refresh focuses on: the context is governed by a **fixed 12 K cap that is neither
enforced nor appropriate**. The context actually runs **16–18.5 K tokens** (the cap
is ignored), token savings **fell 69 % → 58 %**, and **quality regressed**
(ROUGE-L F1 0.27 → 0.14, token-Jaccard 0.25 → 0.14, and proxy responses are now
**2× longer** than direct, `length_ratio` mean 2.02, max 12.3). Fresh
`0.7.26_fix` benchmark, 30 turns, opencode:

| Metric | Direct | Proxy | Verdict |
|---|---|---|---|
| TTFT mean | 14.3 s | 23.2 s | still worse (fold-turn spikes, p90 89 s) |
| **TTFT median** | 14.8 s | **8.6 s** | ✓ **proxy now faster** (was 3.4× worse pre-fix) |
| End-to-end latency mean | 42.0 s | 41.9 s | ✓ parity (Δ mean −55 ms; 17 faster / 13 slower) |
| Prompt tokens / turn (mean) | 25,814 | 10,825 | proxy **2.4× leaner** ✓ |
| Token savings | — | **58.07 %** | ✓ (down from 69 % pre-fix) |
| **Prefix-cache reuse ratio** | — | **0.814 (81 %)** | ✓ **fixed** (was 22 % pre-fix) |
| Cache hit rate / `cached` mean | — | 0.967 / 8,810 tok | ✓ |
| Final context / window utilization | — | 18,577 tok / **7.09 %** | ✗ **93 % of the window idle** |
| ROUGE-L F1 (mean) | — | **0.139** | ✗ regressed (was 0.27) |
| Token-Jaccard (mean) | — | **0.138** | ✗ regressed (was 0.25) |
| `length_ratio` (mean / max) | — | **2.02 / 12.3** | ✗ proxy responses balloon |
| `code_syntax_validity` | — | 0.95 (turns 8,16 invalid) | ✗ proxy emitted broken code twice |
| Context-window wall (turn) | 1 | 1 | both fall off immediately (metric is strict) |

**The new core problem — over-compaction against an idle window.** The proxy now
reuses 81 % of its prefix and yet sits at **7 % of a 262 K window**. There is no
engineering reason to fold every 2–3 turns (each fold drops `cached` to **882** —
the frozen prefix only — and re-prefills the whole 16 K body, the p90 TTFT spikes)
in order to hug a 12 K cap the code doesn't even enforce. On this window the proxy
could carry **3–4× more *cached* context**, fold far less often, and improve
**both** TTFT (fewer cache breaks) **and** quality (less lossy summarization, more
verbatim code retained). The fixed cap is the wrong abstraction: the right governor
is **dynamic** — it grows with conversation horizon, task complexity and codebase
size, and treats a fixed token count as a *last-resort ceiling near the real
window*, not a target (§4.2.2). **Cache reuse, not token count, is the real
constraint — and right now the proxy sacrifices both quality and cache stability to
hit a number it doesn't even reach.**

**The before/after, in one line:** the cliff fix turned a *cache-instability*
failure (22 % reuse, TTFT 3× worse) into a *budget-policy* failure (81 % reuse but
over-compaction, quality regression, unenforced cap). The next gains come from
**adaptive budgeting**, not more eviction gating.

**Root causes, in priority order:**

1. **The budget is a fixed hard cap that is neither enforced nor appropriate
   (the headline issue).** A static `max_optimized_tokens=12000` /
   `max_optimized_chars=12000` governs the pipeline, but the context actually runs
   16–18.5 K (the cap is ignored — §4.2.2) while the backend window is 93 % idle
   (7 % utilization). The right size depends on **conversation horizon, task
   complexity and codebase size**, none of which a constant captures. The fix is an
   **adaptive budget** that grows with those signals and treats a token count as a
   last-resort ceiling near the real window — with **cache reuse**, not a token
   count, as the governing constraint (§4.2.2). This single change subsumes much of
   the eviction churn below.
2. **Six independent mechanisms delete/fold old turns** (1 rolling-summary fold +
   5 front-evictors), each gated by a *different* predicate. When the immutable
   zones (frozen prefix + summary + keep-window) exceed the budget — which they do
   by default — they fight and slide the post-summary body, breaking the prefix
   cache. This is the bug the project has chased across 6+ point releases; a dynamic
   budget that keeps `reserved < budget` removes the condition that makes them fight
   (§4.2 / §4.7).
3. **The embedding/RAG subsystem is silently dead.** The per-session
   `EmbeddingService` is never initialized, so every embedding call hits an
   `assert`, the circuit breaker returns a **zero vector**, and semantic code-chunk
   ranking returns chunks **unranked**. `RAG_ENABLED=true` does nothing today, and
   the same code is a **latent event-loop deadlock** one refactor away (§4.8 / §4.10).
4. **Heavy per-session optimizer construction (incl. a possible HuggingFace
   tokenizer download) runs on the event loop under a global lock.** The first
   request of every new session stalls *all* concurrent requests (§4.9).
5. **Cross-session shared state:** the `HierarchicalSummarizer` (and `DeltaEncoder`
   snapshots) are module-global singletons, so concurrent sessions contaminate each
   other's rolling summary and code deltas (§4.8).
6. **`OutputShaper` violates the project's own hard constraint** ("proxy must NOT
   tune response verbosity") by clamping `max_tokens`/`reasoning_effort` and
   injecting a "be terse" system instruction — and varying those params turn-to-turn
   violates `cache_preservation_guide.md` DONT #2. The fresh benchmark shows the
   verbosity it was meant to fix got **worse** (`length_ratio` 2.02, max 12.3)
   (§4.2.5).
7. **~Half of the advertised "optimization stages" are dead weight** — 11 of 23
   audited modules are no-ops, phantoms (fabricate KV/MTP/slot data a client proxy
   cannot deliver), or never called (§4.2 / §7).

The good news: the architecture's *intent* is correct (freeze prefix, append-only
live zone, lossless boundary compression, capability auto-detection), the benchmark
harness is mature, and the codebase is unusually honest in its comments. The fixes
are mostly deletion and consolidation, not invention.

---

## 2. Critique of `REVIEW_gemini.md` (be critical — it is partly obsolete/wrong)

The Gemini review is a reasonable *generic* LLM-proxy review but several of its
central claims are **outdated or incorrect** against the current tree:

| Gemini claim | Status | Evidence |
|---|---|---|
| "Turns 13,14,15,16,18,20,22,24,26,28 trigger prefix-cache invalidation" (its headline) | **Partly OBSOLETE.** The every-turn cliff is fixed; non-fold turns now REUSE at ~0.98. Only fold turns (18,20,22,24,26,28) + settling (13–16) break. | dry-run 2026-07-30; `CLIF_RESEARCH.md` |
| References `expert_cache.py` "fabricating expert routing masks" | **WRONG — file does not exist.** The real module is `attention_sink.py` (a no-op). Hallucinated. | `ls src/moeptimizer/` |
| Phase 1.1: "append updated summaries exclusively to the latest user turn (volatile tail) instead of mid-prefix" | **NAIVE / likely WRONG.** The authors already analyzed trailing placement: the fold removes the live zone's *head*, so trailing placement diverges immediately after the frozen prefix and loses even the summary's cache. Middle placement (`[frozen][summary][live]`) is the deliberate, documented correct choice. The real fix is rarer/batched folds + a single size governor, not moving the summary. | `hierarchical_summarizer.py:466-473`; `optimizer.py:2956-2971`; `CLIF_RESEARCH.md` "Summary at the tail" |
| "Align truncation to 16/32/64-token PagedAttention block boundaries" | **Mostly a RED HERRING for a client proxy.** The proxy cannot see or control backend KV page allocation; llama.cpp/vLLM page internally regardless of prompt length. Rounding the prompt to a multiple of 32 buys ~nothing and is not worth a feature. | backend paging is internal |
| XGBoost hit-prediction / slot-tracking / MTP-state are phantoms | **CORRECT** — matches this audit (§7). | `hit_prediction_model.py`, `kv_slot_tracker.py`, `mtp_state.py` |
| `EmbeddingService` nested `ThreadPoolExecutor` "thrashing" | **CORRECT but understated.** It is worse: a latent event-loop **deadlock** (`run_coroutine_threadsafe` onto a loop that is never run) and, today, **silent zero vectors** that make RAG a no-op (§4.8). | `embedding.py:108-124` |
| Zero-copy streaming; async post-turn indexing | **CORRECT** — matches §4.4. | `app.py:723-776` |
| "Unaligned memory truncation" / "30–80 ms CPU overhead" | Directionally right but vague; the real TTFT costs are event-loop-blocking construction (§4.9), ~15–18 full re-tokenizations/turn + an O(n²) transform pass (§4.9), and SSE re-serialization (§4.4). | below |

**What Gemini missed entirely** (the higher-value findings): the cross-session
summarizer singleton (§4.8), the unbounded `_SLOT_MAP` leak (§4.10), the
embedding deadlock + dead RAG (§4.8), event-loop-blocking optimizer construction
(§4.9), the `OutputShaper` hard-constraint violation (§4.2), the quality-profile
silently clobbering env overrides (§4.8/§4.11), the five-evictor gate inconsistency
(§4.2 — Gemini blamed only the summarizer), the mislabeled "TTFT" metric (§4.12),
and the `_volatile_turn` flag leaking to the backend (§4.8).

---

## 3. The central architectural tension (read this before the findings)

For a prefix-caching backend, **any** change to the token sequence invalidates the
KV cache **from the first divergent token onward**:

- **Append-only at the tail** → full prefix reuse, only new tokens prefilled. ✓
- **Modify/delete anything in the middle** → cache broken from that point. ✗
- **Delete from the front (after the system prompt)** → cache broken for almost
  everything (worst case). ✗✗

Therefore **"keep the context lean" and "keep the prefix cached" are fundamentally
in tension the moment you mutate history.** You cannot delete old turns *and* keep
the prefix cached — deleting changes the sequence. The honest design space is:

- **(A) Append-only + backend-managed eviction.** The proxy does only *lossless*
  compression of **new** content (tool outputs, code, user pastes), never touches
  history, and lets llama.cpp `--context-shift` / native eviction bound the window.
  Maximum cache reuse; context fills the window (not "lean"), but on a 262K window
  that is fine and TTFT-optimal because every turn is a pure tail append.
- **(B) Periodic compaction with an amortized prefill.** Append-only most of the
  time; *rarely* (every N turns) do one big compaction and accept **one** full
  re-prefill, amortized over the N cheap turns. This is what the fold attempts —
  but it fires **every 2 turns** and the summary-in-middle makes it expensive.
- **(C) Compact into a new session/slot** and pay one prefill at the switch.

The current design is a hybrid that gets the worst of both: context is **not lean**
(16–18K, budget unenforced) **and** the cache breaks frequently (folds every 2 turns
+ the five evictors). The single most impactful architectural decision is to
**commit to (A) as the default for cache-stable mode and reserve (B) for genuinely
window-bound sessions**, with **one** size governor instead of six.

### 3.1 The budget must be dynamic, not a hard limit

The deeper lesson from the fresh benchmark: **"keep it lean" was over-corrected into
a fixed 12 K cap that makes no sense on a 262 K window.** Lean is a *means* (avoid
cache-breaking prefill); the *end* is TTFT/TPS + quality. A constant cap confuses the
two. The right size is a **derived, time-varying ceiling** that responds to:

- **Conversation horizon** — early turns need little; long sessions accumulate. The
  ceiling should grow with turn index (the rolling-summary cap already grows with
  folded-turn count — generalize this to the whole budget).
- **Task complexity / code density** — a multi-file refactor needs a larger verbatim
  code window than a Q&A. Signals already available in the proxy: `code_block_ratio`,
  distinct-file count from `delta_encoder` snapshots, tool-output volume, tree-sitter
  symbol count.
- **Codebase size** — number of distinct files/symbols touched; larger surface →
  larger working set worth keeping cached.
- **The real backend window** (already probed) — the absolute ceiling. The current
  `budget_window_fraction=0.025` (2.5 % of 262 K ≈ 6.5 K) is absurdly small; a sane
  ceiling is a *much larger* fraction that grows toward the window as the session
  lengthens.
- **Cache-break amortization (the economic signal that should actually drive
  compaction).** A fold costs one full re-prefill of the post-frozen body (~16 K
  tokens here). It is only worth doing when the **cumulative prefill saved over the
  expected remaining horizon exceeds that one-time cost**. So compact when
  `marginal_per_turn_prefill × remaining_turns > re_prefill_cost`, **not** when a
  token counter crosses a constant. At 7 % window utilization with 30-turn horizons,
  that inequality says "almost never fold" — which is exactly what the TTFT spikes
  and quality regression are telling us.

Expressed and enforced in **calibrated backend tokens** (not chars — the char/token
mismatch is part of why "12 K" is meaningless), an adaptive governor looks like:

```
budget(t) = clamp(
    base
    + horizon_growth · turn_index                 # conversation horizon
    + code_density_k · recent_code_tokens         # task complexity
    + codebase_k · distinct_files_touched,        # codebase size
    floor = reserved_immutable_zones,             # never below frozen+summary+keep
    ceiling = backend_window · adaptive_fraction  # grows toward the window; << 1.0
)
compact_only_when(budget exceeded AND amortized_savings > re_prefill_cost)
```

This is the single change that reconciles the whole design: it keeps
`reserved < budget` (so the six governors stop fighting — §4.2.1), it folds rarely
(so the cache stays hot — §4.7), and it retains more verbatim context (so quality
stops regressing — §4.12). **The fixed `max_optimized_tokens`/`max_optimized_chars`
become last-resort safety valves near the real window, not the operating point.**

Note also a subtlety in `cache_preservation_guide.md`: its "DO slice exclusively
from the top when truncating" is about *correctness* (no mid-log gaps), **not**
cache reuse — top eviction is the *worst* case for prefix reuse. The guide
conflates the two; the proxy should treat top-only eviction as a last-resort
safety valve, not a cache strategy.

---

## 4. Findings by focus area

Severity: **CRIT** = breaks a core feature / unbounded leak / hang; **HIGH** =
measurable TTFT/TPS/quality/correctness regression; **MED** = latent risk / waste.

### 4.1 Missing optimizations

1. **Error-aware tool-output compression (HIGH).** `ToolOutputCompressor`
   (`tool_output_compressor.py`) does ANSI-strip + repeated-line/stack-frame
   collapse + **head/tail truncation** ("never drop the head"). It is **not
   error-aware**: a compiler error, failing-test name, or `file:line` in the
   *middle* of a long log is truncated away. snip/rtk get 60–90 % savings by doing
   the opposite — **failures-only** for tests/builds/lint, keep `error|failed|
   panic|fatal` lines + stack frames + `file:line`, collapse passing tests to a
   count, group lint by file/rule, dedup repeated lines with counts. This is the
   single highest-value missing optimization and directly improves quality (the
   model keeps the diagnostic signal) *and* token count. → Phase 2.
2. **Reversible compression with retrieval handles (HIGH).** headroom/lean-ctx
   replace large payloads with a compact placeholder + a handle, store the original
   locally, and let the model request it via a tool (`headroom_compress`/
   `headroom_retrieve`; lean-ctx cached re-reads ≈ 13 tokens). The proxy currently
   compresses **irreversibly** — once a tool output is boundary-truncated or folded
   into a summary, the detail is gone, which is a driver of the low semantic
   similarity and `code_block_loss`. A content-addressed store + an MCP-style
   `expand(id)` tool would let the proxy keep the context lean *without* permanent
   information loss. → Phase 3 (larger).
3. **Cached re-read collapse (MED).** lean-ctx collapses a repeated file read to a
   ~13-token reference. `delta_encoder.py` already diffs a re-read against the prior
   snapshot, but only when the prior version is still in context; it could go
   further and emit a stable `[file X unchanged since turn N]` reference when the
   content hash matches, regardless of eviction. → Phase 3.
4. **Volatile-field relocation (MED).** lean-ctx/headroom move dates, UUIDs, commit
   SHAs, and timestamps **out of the cacheable prefix**. The proxy has a "volatile
   anchor" but it is a *quality* anchor (a 3rd copy of the original request — see
   §4.2.6), not a volatile-field scrubber. Detecting and neutralizing
   high-entropy/changing tokens in otherwise-stable messages would raise reuse. → Phase 3.
5. **Effort routing — already present but mis-scoped (see §4.2.5).** headroom's
   "lower reasoning_effort on routine tool-result turns" is the *right* idea but the
   proxy's implementation (`OutputShaper`) violates the hard constraint; do it via
   client-side defaults or not at all.

### 4.2 Design weaknesses

1. **Six size governors with four different predicates (CRIT, the core bug).**
   There is **1 fold + 5 front-evictors**, each gated differently:
   | Mechanism | Call site | Gate |
   |---|---|---|
   | Rolling-summary fold | `optimizer.py:1247` | `_cache_stable_summary and cache_stable_mode` |
   | Scratchpad compactor (Step 7) | `optimizer.py:1290` | `current_tokens > compaction_threshold and not _skip_compactor` |
   | Proactive trim (Step 11) | `optimizer.py:1512` | `> proactive_threshold and not _prefix_drift and not skip_front_eviction` |
   | Sliding window (Step 11.8) | `optimizer.py:1574` | `> 0.8*max_tokens and not _prefix_drift and not skip_front_eviction` |
   | `_trim_to_budget` (Step 12) | `optimizer.py:1585` | `> max_tokens` (guard keys on `cache_stable_mode` only) |
   | `token_aware_truncator` (Step 12b) | `optimizer.py:1588` | `> max_tokens and not skip_front_eviction` |
   `_skip_compactor` needs `has_rolling_summary()`; `skip_front_eviction` needs
   `summary_armed and (has_rolling_summary() or <= compaction_threshold)`;
   `_trim_to_budget`'s guard needs only `cache_stable_mode`. On any turn where
   `cache_stable_mode=True` but `has_rolling_summary()` is `False` and the context
   is over the compaction threshold (e.g. the first over-budget turn, or any turn
   where the fold refused to store because the summary budget is full —
   `hierarchical_summarizer.py:425-429`), **up to four evictors fire in one turn**,
   each sliding the post-summary body further → exactly the per-turn prefix slide
   the comments claim to have fixed. There are also **three near-identical
   `_partition_for_budget`** (`optimizer.py:2401`, `token_aware_truncator.py:212`,
   `compactor.py:206`) and **three `_evict_for_budget`** implementations.
   **Fix:** collapse to a **single `BudgetGovernor`** + a single shared
   `PrefixLayout` (frozen/summary/live partition). The fold owns sizing in
   cache-stable mode; delete `_proactive_trim` and `_sliding_window_trim` from the
   pipeline; keep `_trim_to_budget` only as a hard-cap safety valve behind the
   *same* predicate. Add the `evictable_budget <= 0 → return unchanged` guard to
   `token_aware_truncator._evict_for_budget` (`token_aware_truncator.py:298`) and
   the compactor (`compactor.py:131`) — today only `_trim_to_budget` has it. → Phase 1.
2. **The budget is a fixed hard cap that is neither enforced nor appropriate
   (CRIT, the headline design change).** Two facts from the fresh benchmark:
   (a) the context settles at **16–18.5 K** (dry-run turn 30 = 18,607; benchmark
   final 18,577) vs `max_optimized_tokens=12000` — because the immutable zones
   exceed the cap, `evictable_budget == 0` and no evictor can reach it, so **the cap
   is a fiction**; and (b) that 18.5 K is only **7.09 % of the 262 K window** — the
   proxy is **over-compacting against a 93 %-idle window**, folding every 2–3 turns
   (each fold drops `cached` to 882 and re-prefills the whole body → the p90 TTFT
   spikes) to hug a number it doesn't even reach. A constant cannot encode what the
   right size actually depends on — **conversation horizon, task complexity,
   codebase size**. **Fix — replace the fixed cap with an `AdaptiveBudgetGovernor`**
   (full design in §3.1): a per-turn ceiling derived from the live backend window,
   turn index (horizon), code-density / distinct-files (complexity & codebase size),
   floored at the immutable-zone reservation (`reserved < budget` always, which also
   stops the six governors fighting — item 1), with compaction triggered by
   **cache-break amortization** (`marginal_per_turn_prefill × remaining_turns >
   re_prefill_cost`), not a token threshold. Keep `max_optimized_tokens`/`_chars`
   only as last-resort safety valves near the real window, enforced in **calibrated
   tokens** (not chars). Expected effect on this benchmark: far fewer folds → higher
   reuse + lower TTFT p90, more verbatim context → quality recovers, and the
   "budget" becomes a measured outcome rather than an ignored constant. → Phase 1
   (governor skeleton) + Phase 2 (adaptive policy + amortization trigger).
3. **God object (HIGH, maintainability).** `optimizer.py` is 3,416 lines, one class
   with ~50 instance attributes and a ~210-line constructor; `_optimize_messages_locked`
   is ~700 lines; step numbering is non-sequential with fractional sub-steps
   (5.1.5, 7.25, 11.7, 14.12…), **Step 7 and Step 11.7 each appear twice**, and the
   comments are an archaeological record of regressions ("turn-11/12/13 cliff",
   "v0.7.18/19/21/22") where each fix added a new gate instead of consolidating —
   which is *why* there are six governors. **Fix:** extract `BudgetGovernor`,
   `PrefixLayout`, and a `StageRunner` (ordered, uniformly-guarded stages). → Phase 4.
4. **Dead/phantom stages wired into the hot path (HIGH).** See §7 — 11 of 23 audited
   modules are dead/no-op/phantom, several still constructed per session and "called"
   (e.g. `thinking_preserver.process_messages` is a pure copy at `optimizer.py:1013`;
   `incremental_updater.update_context(optimized, "")` is always a no-op at
   `optimizer.py:1336` but still triggers a token recount at `:1337`). → Phase 1/4.
5. **`OutputShaper` violates the project's hard constraint (HIGH).** AGENTS.md:
   *"the proxy must NOT try to tune/compress response verbosity."* Yet
   `output_shaper.py` appends a "Be concise…" system instruction and clamps
   `max_tokens` and `reasoning_effort` per turn-class, and its own docstring admits
   it was added because "proxy responses are 3.6× longer than direct." Problems:
   (a) it is exactly the forbidden response-verbosity tuning; (b) `reasoning_effort`
   /`max_tokens` vary **per turn-class** (NEW_QUESTION vs TOOL_RESULT vs ERROR), so
   generation params change turn-to-turn — `cache_preservation_guide.md` DONT #2
   ("modifying generation properties mid-chat forces llama.cpp to re-evaluate
   sampling layers"); (c) it mutates the system message that the optimizer's
   immutable-prefix guard is trying to freeze (order-of-operations tension); (d) it
   chases a *symptom* and is **losing**: the fresh benchmark shows proxy responses
   are now **2× longer than direct** (`length_ratio` mean **2.02**, max **12.3**,
   `model_verbosity_delta_turns=7`) — *worse* than the 3.6× that motivated the
   module — despite the shaper being in the path. Verbose proxy responses are a
   downstream effect of the **lossy, over-compacted context** (the model re-asks /
   re-explains when it can't see prior detail), and the "be terse" instruction is
   injected only into the proxy path, which also biases the direct-vs-proxy quality
   comparison. **Fix:** remove `OutputShaper` from the proxy path (or make it a
   no-op default behind an explicit opt-in flagged as constraint-violating); fix the
   *cause* — better context fidelity via the adaptive budget (§4.2.2) and
   error-aware/reversible compression (§4.1) — instead of clamping the output. → Phase 1.
6. **The volatile quality anchor is redundant churn (MED).** `_append_volatile_context`
   (`optimizer.py:1821`) appends a trailing user turn with the "Original request"
   text — which is **already** present verbatim in the frozen first user turn *and*
   pinned into the rolling-summary head (`seed_original_request`,
   `hierarchical_summarizer.py:178`) — i.e. a **third copy**, plus up to ~900 chars
   of accumulated constraints (also extracted into the summary). Its careful
   monotonic-append stability is **wasted** because the same trailing turn also
   carries RAG context recomputed from the last assistant message every turn
   (`optimizer.py:1399-1404`), so the turn's bytes change every turn regardless.
   Net: ~900 chars/turn of uncached, duplicate prefill for marginal quality gain.
   **Fix:** drop the duplicated anchor; keep only genuinely-volatile RAG/warnings in
   the trailing turn (and reconcile with the `_volatile_turn` leak, §4.8.3). → Phase 3.

### 4.3 Better alternatives

1. **Default to append-only + native context-shift (HIGH).** For cache-stable mode
   on a large window (262K here), stop deleting history entirely; do only lossless
   boundary compression of *new* content and let `llama.cpp --context-shift` bound
   the window. Every turn becomes a pure tail append → near-100 % reuse → TTFT
   collapses. Reserve compaction for genuinely window-bound sessions. This is the
   biggest lever for the mission (§3, option A). → Phase 2 (after Phase 1 cleanup).
2. **Native slot pinning is already done well — keep it (positive).**
   `backend_capabilities.py` device-aware auto-detection of `/slots`, native MTP
   passthrough, and exact tokenizer is sound and NPU/GPU hot-swap aware
   (`backend_capabilities.py:128-135`, `app.py:560-577`). The `id_slot` pinning is
   the correct mechanism for whole-prefix reuse; just fix the probing issues (§4.9).
3. **Symbol/skeleton index for evicted code (MED).** When a code-bearing turn is
   evicted, accumulate a compact signature index (the `code_ledger` at
   `optimizer.py:2585` is a start). Extend to a scope-graph skeleton (class/method
   signatures + line spans, lean-ctx `signatures` mode) so the model keeps API
   awareness without full bodies. → Phase 3.

### 4.4 Additional throughput improvements

1. **Zero-copy SSE passthrough (HIGH).** `app.py:723-776` parses every backend SSE
   chunk into pydantic objects and re-emits a fresh `json.dumps(...)` per chunk —
   the hot decode path, capping TPS on a high-TPS MTP backend. The only reasons it
   can't passthrough are minting its own `id`/`created`/`model`, usage capture, and
   thinking capture. **Fix:** add a raw-byte passthrough fast path (httpx
   `aiter_bytes`/`aiter_raw`) for turns needing no delta inspection; tap only the
   final usage chunk. At minimum, forward the SDK's raw line for deltas that need no
   rewrite. → Phase 2.
2. **O(n²) token counting in the floor-bounded transform (HIGH).**
   `_apply_transform_with_floor` (`optimizer.py:2622`) builds `candidate =
   [*result, transformed]` and calls `count_messages(candidate)` **per message**
   (each a distinct fingerprint → cache miss, each O(n)) → O(n²) per transform;
   `_apply_boundary_transforms` runs **three** such transforms on essentially every
   request (incl. fast path). **Fix:** maintain a running token total (count each
   message once, subtract back-to-front, stop at the floor) → O(n). → Phase 2.
3. **~15–18 full re-tokenizations per turn (HIGH).** `TokenCounter.count_messages`
   memoizes by a SHA-1 of the *whole list* (`token_counter.py:404-419`), but the
   fingerprint is recomputed O(n) on every call and any mutation misses; the full
   prompt is recounted at ~18 sites in one pass (`optimizer.py:1024,1122,1154,…,1846`),
   most after a stage mutated the list (misses). With the remote `/tokenize` path
   each miss is a **synchronous HTTP round-trip**. **Fix:** count once per stage
   boundary and thread an integer through the gates; key the memo **per-message**
   (not per-whole-list) so unchanged messages are never re-tokenized. → Phase 2.
4. **Post-stream CPU on the event loop (MED).** After the stream ends the generator
   runs `record_cache_outcome`, `capture_thinking`, `count_messages`,
   `set_token_calibration`, `calibrate_remote_overhead` on the loop thread
   (`app.py:838-885`). **Fix:** offload to the optimizer executor. → Phase 2.
5. **Ad-hoc executors / nested pools (MED).** `embedding.py:149,157` create a fresh
   `ThreadPoolExecutor` per call/batch, each task doing `asyncio.run(...)` (a new
   event loop per embedding). **Fix:** reuse the single bounded `AsyncIOStage`
   executor; make embedding fetch genuinely async (§4.8.1). → Phase 1.

### 4.5 Additional context-efficiency improvements

1. **Error-aware tool-output compression** — see §4.1.1 (the biggest win).
2. **Global content-addressed code dedup (MED).** A SHA-256 registry across turns so
   a file printed twice (`cat` in turn 2, `read_file` in turn 5) collapses the second
   to a reference. `chunk_fingerprint.py` exists but is per-chunk; extend to
   whole-file identity. → Phase 3.
3. **Syntactic code slicing (MED) — primitive done, wiring pending.** For a
   multi-hundred-line read, keep only the target function/class referenced by the
   user query and stub siblings with `# ... [N definitions collapsed]`.
   `slice_code_to_query` (`code_block_optimizer.py`) implements this against the
   correct tree-sitter 0.25.2 property API, fail-open + never-expands, 11 tests.
   Note: the pre-existing "tree-sitter plumbing" in `code_chunking.py` was in fact
   broken (silent line-fallback — see the tree-sitter finding in the status section);
   the slicer does not depend on it beyond the parser cache. Remaining: wire into the
   pipeline behind a config gate, compressing on first appearance. → Phase 3.
4. **Stop double/triple-copying the original request** (§4.2.6) — ~900 chars/turn.

### 4.6 Additional MTP-preservation techniques

1. **Native MTP passthrough is the only real lever — already implemented (positive).**
   Client-proxy speculative decoding is impossible (the proxy can't run the draft
   heads); the correct path is forwarding MTP `extra_body` to a backend with native
   MTP (`v050.native_mtp_passthrough`, auto-detected). Keep this; **delete the dead
   client-side MTP scaffolding** (`mtp_state.py`, `mtp_speculative` refs — §7).
2. **Preserve exact ChatML/tool-schema/code-fence structure (MED).** MTP draft
   acceptance drops when formatting deviates. The proxy mostly preserves this, but
   the boundary transforms and the `_volatile_turn`/`_code_ledger` injections are
   places where structure can shift; ensure injected messages are well-formed
   ChatML and never split a code fence. → Phase 3 (hardening).
3. **Verbatim `<think>` preservation (positive).** `thinking_preserver` is a no-op
   *because* thinking is already echoed back by the client; this is correct — but
   `capture_thinking`/re-injection in `app.py` should be verified to round-trip
   `reasoning_content` byte-for-byte (any normalization invalidates the cache). → Phase 3.

### 4.7 Additional model KV-cache preservation techniques

1. **Single size governor + append-only default** (§4.2.1, §4.3.1) — the dominant lever.
2. **Drive folds by cache-break amortization, not a token threshold (HIGH).** The
   fresh `.log` shows the cost directly: on fold turns (14, 17, 20, 22, 25) `cached`
   collapses to **882** (frozen prefix only) and the whole ~16 K body is re-prefilled
   — those are exactly the p90 TTFT outliers (proxy TTFT p90 89 s vs direct 24 s).
   Between folds, reuse is near-total. So fold frequency *is* the TTFT knob. Trigger
   a fold only when `marginal_per_turn_prefill × expected_remaining_turns >
   re_prefill_cost` (§3.1) — at 7 % window utilization that is "rarely," which both
   raises reuse and removes the spikes. Widen the hysteresis
   (`growth_budget = max(2048, target//3)`, `hierarchical_summarizer.py`) to ≥ the
   observed max single-turn growth as an interim step. `CLIF_RESEARCH.md` #28 found
   "rarer folds" negative only because it also raised context size; with an adaptive
   ceiling (§4.2.2) the tradeoff disappears. → Phase 1 (hysteresis) + Phase 2 (amortization).
3. **Make the summary a first-class typed region (MED).** Today the summary is
   re-recognized each turn by a **content marker** (`ROLLING_SUMMARY_MARKER`) because
   `_strip_internal_flags` removes `_summary_id`; there are many scattered
   `if self._is_summary_block(msg): continue` guards (`optimizer.py:1424,1664,3338,
   3356,3375`), each a place the invariant can break (the documented "turn-11 cliff").
   Return the summary as a typed region from the partition so stages skip it
   structurally. → Phase 4.
4. **Volatile-field relocation** (§4.1.4) and **byte-stable tools array** (the
   benchmark forwards the OpenAI `tools` schema; ensure the proxy never reorders or
   re-serializes it — `pin_tools` exists; verify it is byte-stable). → Phase 3.

### 4.8 Potential bugs

1. **CRIT — embedding subsystem: latent deadlock + silently dead RAG.**
   `embedding.py:108-124` `_fetch` does `asyncio.run_coroutine_threadsafe(post,
   _get_sync_loop()).result()`, but `_get_sync_loop()` (`embedding.py:24-35`) creates
   a loop and **never runs it** → `.result()` blocks forever. On the async path this
   hangs the uvicorn loop; on the sync path it self-deadlocks. The circuit breaker
   gives **zero** protection (a hang is not an exception). It is not a *live* hang
   today only because the per-session `EmbeddingService` (`optimizer.py:194`) is
   **never `initialize()`d**, so `_http_client is None` and the `assert` fires first
   → breaker returns a **zero vector**. Consequence: `_rank_chunks`
   (`optimizer.py:2293-2296`) sees `norm_q == 0` and returns chunks **unranked** —
   **the entire semantic RAG ranking is a silent no-op** (`RAG_ENABLED=true` does
   nothing). It is a landmine one refactor away from a production freeze. **Fix:**
   delete the `_get_sync_loop`/`run_coroutine_threadsafe` machinery; `get_embedding`
   is already `async` — `await self._http_client.post(...)` directly; give
   `CircuitBreaker` an `async call_async`; inject **one initialized** `EmbeddingService`
   shared across optimizers; add an explicit degradation marker when the breaker is
   open so operators can see ranking is disabled. → Phase 1.
2. **CRIT — cross-session shared state.** `SessionManager` promises per-session
   isolation and builds a fresh optimizer per session, but the optimizer obtains the
   summarizer via `get_hierarchical_summarizer()` (`optimizer.py:137`) — a
   **module-global singleton** (`hierarchical_summarizer.py:943-951`). All sessions
   share `_rolling_summary_texts`, `_summarized_turn_count`, `_original_request_facts`,
   `_last_fold_emitted_tokens`. Session A's first request pins `Key facts (from
   original request)` that session B can never replace (`if self._original_request_facts:
   return`); B's fold boundaries are computed against A's history. Same hazard for
   `get_delta_encoder` snapshots keyed by `f"inline:{lang}"` (`optimizer.py:1668,2105`),
   **not** per session → code deltas against another session's file version. Silent,
   intermittent, invisible in single-session tests. **Fix:** construct
   `HierarchicalSummarizer` and `DeltaEncoder` per optimizer (drop the `get_*` global
   cache) or key their state by session id. → Phase 1.
3. **HIGH — `_volatile_turn` flag leaks to the backend.** The volatile turn is
   appended at Step 14.12 (`optimizer.py:1875`) as `{..., "_volatile_turn": True}`,
   **after** `_strip_internal_flags` (Step 13, `optimizer.py:1605`); nothing strips it
   afterward and `app.py` assigns the optimizer output straight into the request
   (`app.py:1530`). A non-OpenAI key is sent to the backend on every over-threshold
   turn — violates the transparent contract and the "no model-visible markers"
   constraint; a strict backend/SDK could reject the message. **Fix:** append before
   the strip, or strip `_`-prefixed keys in `_finalize_optimized`. → Phase 1.
4. **HIGH — quality profile silently clobbers explicit env overrides.**
   `apply_quality_profile` (`config.py:740-770`) unconditionally `setattr`s the
   profile's values, so `MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS=24000` is overwritten
   back to 12000 by the default `balanced` profile (`CLIF_RESEARCH.md` #26 confirmed
   this). The README claims "individual fields can still be tuned" — the code does
   the opposite. **Fix:** apply the profile as *defaults* and let explicit env/field
   values win (track which fields were set explicitly), or apply the profile before
   env parsing. → Phase 1.
5. **MED — silently swallowed exceptions.** Almost every stage is
   `try/except Exception → logger.warning` (`optimizer.py:1146,1175,…,1834`) and
   uvicorn runs at `log_level="warning"`; a stage that consistently fails (e.g.
   tree-sitter parser missing) degrades quality with no test-visible signal. The
   `_trim_to_budget` failure (`optimizer.py:1594`) being swallowed means a
   budget-enforcement crash can leave an over-window context the backend hard-rejects.
   **Fix:** surface persistent degradation via the `/v1/metrics` `backend_errors`/
   degradation counters and a test assertion. → Phase 4.
6. **MED — calibrated vs uncalibrated token-count drift in eviction.**
   `_evict_for_budget` sums per-pair `count_messages(pair)` (`optimizer.py:2532`)
   while the outer loop compares calibrated whole-list counts (`calibrated_token_count`,
   `optimizer.py:537`); mixing the two across the partition/evict boundary can leave
   the result off by up to the calibration factor (~2×, `optimizer.py:533`). The
   "trim to budget" guarantee is fuzzy. **Fix:** use one consistent (calibrated)
   count throughout the governor. → Phase 1 (with the governor).
7. **MED — same-session concurrent requests race shared optimizer state.** The
   endpoint/generator call `get_session_state`, `record_cache_outcome`,
   `capture_thinking`, `set_token_calibration` (`app.py:838-885,1530`) without
   obviously holding the optimizer lock that `optimize_messages` uses
   (`optimizer.py:892`). Low impact (agentic turns are sequential) but calibration/
   thinking state can tear. **Fix:** route mutators through the lock. → Phase 4.
8. **HIGH — remote `/tokenize` is silently dead in the async app (observed in the
   fresh benchmark log).** `tokenize_count_sync` (`backend_capabilities.py:293-306`)
   does `asyncio.run(self.tokenize_count(text))`; called from a running event loop
   that raises `RuntimeError` and `return None`, but the coroutine argument was
   already created and is never awaited — the log shows exactly this:
   `RuntimeWarning: coroutine 'BackendCapabilityProbe.tokenize_count' was never
   awaited … return None`. So `REMOTE_TOKENIZE_ENABLED` never actually fires in the
   FastAPI path; every "authoritative" count silently falls back to approximate
   tiktoken, and token calibration leans entirely on backend `prompt_tokens`
   feedback. This also means the budget is enforced (when it is enforced) against an
   *estimate*, which matters once the budget becomes adaptive in calibrated tokens
   (§4.2.2). **Fix:** async callers should `await probe.tokenize_count(...)`
   directly; reserve the sync wrapper for genuinely-sync code and don't construct the
   coroutine eagerly (`coro = self.tokenize_count(text)` only inside the `try`). Add
   a test that asserts a non-`None` remote count when the backend exposes `/tokenize`. → Phase 1.

### 4.9 Performance bottlenecks

1. **CRIT — optimizer construction on the event loop under a global lock.**
   `app.py:1352` `session_manager.get_or_create(session_id)` is a synchronous call in
   the `async` handler; `get_or_create` (`session_manager.py:64-77`) holds a global
   `RLock` and, on a new session, builds ~25 components incl. `cache_registry.load_from_disk()`
   (disk I/O) and `TokenCounter` whose init can call `AutoTokenizer.from_pretrained(...)`
   (`token_counter.py:48-52`) — a HuggingFace load that may hit network/disk for
   **seconds**. The **first request of every new session stalls the entire event
   loop** and serializes all session threads. The single biggest new-conversation
   TTFT cliff. **Fix:** construct optimizers via `run_in_executor`; load the
   tokenizer **once process-wide** and share it; narrow the session-manager lock to
   map mutation only (build outside the lock, insert under it with a double-check). → Phase 1.
2. **HIGH — sequential startup capability probes block `yield`.** In `lifespan`
   (`app.py:1185-1235`), `capability_probe.get(force=True)` awaits four sub-probes
   **sequentially** (`backend_capabilities.py:148-152`), each `probe_timeout=4.0`s →
   ~16 s against a degraded backend, + a 5 s MTP chat probe, all **before `yield`**
   (server accepts no connections). The docstring claims probes "never block startup."
   **Fix:** `asyncio.gather` the sub-probes; wrap startup probing in a bounded
   `asyncio.wait_for`; or serve with cached defaults and probe lazily after `yield`. → Phase 2.
3. **HIGH — thundering herd on probe TTL expiry.** `BackendCapabilityProbe.get`
   (`backend_capabilities.py:116-141`) releases the lock before `await self._probe()`;
   when the 30 s TTL lapses, every concurrent request fires its own full probe (no
   single-flight). Every 30 s → N×(4 HTTP probes) at the backend + N×TTFT inflation.
   **Fix:** single-flight via a shared in-flight `asyncio.Task` (use `asyncio.Lock`,
   not `threading.Lock`, across `await`s). → Phase 2.
4. **HIGH — per-request inline CPU on the loop.** After offloading the optimize step,
   the endpoint still runs on the loop: `_serialize_messages_text` (`app.py:1490`)
   renders the *entire* prompt to a string and `.replace("\n","\\n")` then checks
   `len <= 32000` — building a 200 KB string just to discard it; `get_session_state()`
   (`app.py:1530`) `json.dumps` the whole store + `goal_decomposer.decompose`; a
   fallback `count_messages` for the token header (`app.py:1466-1470`). **Fix:** gate
   `_serialize_messages_text` behind a length pre-check; move session-state/header
   serialization into the offloaded step. → Phase 2.
5. **MED — fresh `httpx.AsyncClient` per probe / per `tokenize_count`**
   (`backend_capabilities.py:147,~305`). **Fix:** one long-lived client on the probe
   object, opened/closed with the lifespan. → Phase 2.
6. **MED — only 2 optimizer-executor workers + 300 s backend timeout.** A wedged
   backend holds one of 2 workers (`config.py:389`) for up to 300 s; two stuck
   sessions queue every subsequent request indefinitely (`app.py:1362`), and a hung
   embedding thread can permanently drain the 4-thread async_io pool (`config.py:633`).
   **Fix:** make embedding async (§4.8.1); add real cancellation; isolate embedding
   on its own executor. → Phase 1/2.

### 4.10 Memory leaks

1. **CRIT — `_SLOT_MAP` grows without bound.** `app.py:538` `_SLOT_MAP: dict[str,int]`
   is only ever inserted into (`app.py:577`), never `pop`/`clear`/evicted. Session ids
   derive from client-controlled conversation fingerprints, so every distinct
   conversation creates a permanent process-lifetime entry — a slow memory-exhaustion
   vector. Note `PROXY_METRICS._per_session` *is* correctly LRU-capped at 512
   (`app.py:64,151-153`), so this is an oversight. **Fix:** bound it identically
   (`OrderedDict` + `move_to_end` + `popitem(last=False)`), and evict on
   `delete_session`/expiry. Trivial. → Phase 1.
2. **HIGH — per-session `EmbeddingService` (cache + breaker), never cleaned.**
   `optimizer.py:194` builds a private `EmbeddingService` per session (each an
   `OrderedDict` cache cap 512 + a `CircuitBreaker`); with `max_sessions=256` → up to
   ~200 MB of embedding cache worst-case, and the app *also* builds a separate
   initialized service (`app.py:1133`) and a separate embed client (`app.py:1136`) —
   three independent stacks. **Fix:** share one initialized `EmbeddingService`
   (also fixes §4.8.1). → Phase 1.
3. **MED — lazy session expiry; no reaper.** `_cleanup_expired` runs only inside
   `get_or_create` (`session_manager.py:138-146`); idle expired optimizers linger
   until the next create, so the memory high-water mark tracks peak concurrent
   sessions forever (`session_timeout=3600`). **Fix:** a lightweight background reaper
   started in `lifespan`. → Phase 2.
4. **MED — `AgentStateStore.goals` never pruned** (`state_store.py:156-160`); old
   `GoalNode`s accumulate and are re-serialized into every `get_session_state`.
   **Fix:** keep only the current goal. → Phase 4.
5. **Positive:** `cache_registry.py` (cap 1000/map), `cache.py` LRU, and
   `_ProxyMetrics._per_session` are correctly bounded.

### 4.11 New UX features

1. **Fix the config-override contract (HIGH).** Make explicit env/field values win
   over the quality profile (§4.8.4); document the precedence clearly. Today an
   operator cannot tune the budget without also setting the profile, which is a
   foot-gun discovered the hard way in `CLIF_RESEARCH.md`.
2. **Real TTFT + cache-reuse telemetry (HIGH).** The proxy's `/v1/metrics` "TTFT" is
   mislabeled (it is end-to-end latency, §4.12); expose a *true* first-token time and
   a per-turn `(prompt_tokens − cached_tokens)` "fresh prefill" series so operators
   can see the cache→TTFT link directly. The `/v1/agent/sessions/{id}/debug` endpoint
   exists — add the live-zone/frozen/summary token breakdown there (Gemini §2.11 had
   this right).
3. **Make degradation visible (MED).** Surface the embedding-breaker state and any
   consistently-failing stage in `/v1/metrics` and the `X-MOEPT-Optimization-Degraded`
   header (today the dead RAG never records degradation). → Phase 1.
4. **Per-tool compression budgets + escape hatch (MED).** Borrow snip/rtk: per-tool
   result token budgets and a `full_output` passthrough flag so the agent can request
   the uncompressed original when a compressed result is insufficient (pairs with
   reversible compression, §4.1.2). → Phase 3.
5. **Savings analytics by tool/command (LOW).** Track which tool outputs waste the
   most tokens (snip/rtk `gain` reports) to guide filter tuning. → Phase 3.

### 4.12 Benchmark improvements

1. **The "TTFT" metric is mislabeled (HIGH).** In `app.py` the `latency_ms` recorded
   per turn is `(time.time() − turn_start)` measured **after the stream completes**
   (`app.py:866-872` streaming, `:1015` non-streaming), yet the dashboard renders it
   as "Avg TTFT (ms)". It is total request duration (dominated by full generation),
   not time-to-first-token — operators tuning TTFT are looking at a number that
   barely correlates with it. **Fix:** timestamp the first yielded content chunk and
   report that as TTFT; keep total latency separate. (The *benchmark script* measures
   TTFT itself via streaming and is fine; this is the proxy's own metric.) → Phase 2.
2. **Greenlight + CI-gate the dry-run (HIGH).** Integrate
   `diag_dryrun_opencode.py` into `dev.sh`/CI as a prefix-reuse gate (fail if
   non-fold turns drop below a reuse threshold, e.g. 0.8). Today it is not gated and
   not greenlit (fold turns break). Use the **common-prefix reuse ratio** (the real
   backend metric) not strict `startswith` (the volatile tail differs by design). → Phase 1.
3. **Add a cache→TTFT correlation metric (HIGH).** Report per-turn
   `fresh_prefill_tokens = prompt_tokens − cached_tokens` alongside TTFT, and a
   correlation/plot, to *prove* the cache-instability→TTFT mechanism quantitatively
   and prevent regressions. → Phase 2.
4. **Drop the weak code embedder from the headline gate (MED).** README admits
   `embed-gemma-300m-FLM` is weak on code; `semantic_similarity` median 0.037 is
   noise. The regression gate (`--min-similarity`) should lean on the robust headline
   (`rouge_l_f1`, `token_jaccard`, `code_block_ratio`, `edit_similarity`,
   `code_syntax_validity`), which the README already separates — make the *gate* use
   the headline block, not semantic similarity. → Phase 2.
5. **Make the direct-vs-proxy comparison fair (MED).** If any output-shaping
   instruction is injected into the proxy path only (§4.2.5), the quality comparison
   is biased. Remove `OutputShaper` from the proxy path (recommended) or apply the
   same instruction to the direct baseline. → Phase 1.
6. **The quality regression is now the gate concern (HIGH).** The fresh post-fix
   benchmark traded quality for cache reuse: ROUGE-L F1 0.27 → **0.14**, token-Jaccard
   0.25 → **0.14**, `length_ratio` 0.87 → **2.02** (max 12.3), and
   `code_syntax_invalid_turns: [8,16]` (proxy emitted broken code). Add an explicit
   **quality regression gate** alongside the reuse gate: fail if headline
   `rouge_l_f1` / `token_jaccard` / `code_syntax_validity` drop below the previous
   round, and flag `length_ratio` outside [0.5, 2.0]. The adaptive budget (§4.2.2)
   and error-aware/reversible compression (§4.1) are what should move these back up;
   the benchmark must measure that they do. → Phase 2 gate.
7. **`context_window_wall` at turn 1 for both sides is a metric bug, not a result
   (MED).** The wall triggers on `code_block_ratio < 0.5 OR semantic_similarity <
   0.3`; turn 1 (a short non-code answer) trips it immediately for *both* direct and
   proxy, so it reports "wall=1" and conveys nothing. Require a minimum turn index
   (e.g. ≥ 5) and a sustained (2+ consecutive turns) breach before declaring a wall. → Phase 2.
8. **Re-benchmark after the adaptive budget (process).** With the fresh
   `0.7.26_fix` baseline in hand (reuse 0.81, TTFT median 8.6 s, ROUGE-L 0.14),
   re-run `--scenario opencode --turns 30 --rounds 3` after Phase 1–2 and confirm:
   reuse stays ≥ 0.8, TTFT **mean** (not just median) drops below direct as fold
   spikes disappear, token savings recover toward 65 %+ *and* ROUGE-L/Jaccard climb
   back above the 0.7.26_fix baseline. Use ≥3 rounds (this run was 1) so variance is
   visible. → Phase 2 gate.
9. **Multi-file agentic replay (LOW).** Expand `scripts/fixtures` to multi-file
   edits + real test/lint/compiler failures (exercises error-aware compression). → Phase 3.

---

## 5. Prioritized implementation plan

Ordered by impact/effort. Each phase ends with the stated gate. **Do not run the
full benchmark until the dry-run is greenlit** (per `CLIF_RESEARCH.md`).

### Phase 1 — Correctness, leaks, and the cache-break root cause (highest ROI)
*Goal: stop the silent corruption and leaks; unify eviction; greenlight the dry-run.*

1. **Single size governor** (§4.2.1, §4.7.2): introduce `BudgetGovernor` +
   `PrefixLayout`; delete `_proactive_trim` and `_sliding_window_trim` from the
   pipeline; fold owns sizing in cache-stable mode; `_trim_to_budget` is the only
   hard-cap safety valve; add `evictable_budget <= 0 → return` to
   `token_aware_truncator` + compactor; widen fold hysteresis ≥ max single-turn growth.
2. **Adaptive budget — skeleton** (§4.2.2, §3.1): introduce the
   `AdaptiveBudgetGovernor` interface; first make the ceiling **window-relative and
   horizon-growing** instead of the fixed 12 K (raise `budget_window_fraction` from
   0.025 to a sane value, grow with turn index), **floored at `reserved`** so
   `reserved < budget` always (this alone stops the six governors fighting and makes
   the cap honest). Enforce in **calibrated tokens**. Keep `max_optimized_tokens`/
   `_chars` only as last-resort valves. *Gate: dry-run non-fold turns ≥ 0.9 reuse;
   fold turns are the only breaks; the realized context size is a deliberate,
   logged outcome of the governor (not an ignored constant).*
3. **Fix the embedding subsystem** (§4.8.1, §4.10.2): make `get_embedding` truly
   async (delete `_get_sync_loop`/`run_coroutine_threadsafe`), inject **one
   initialized** shared `EmbeddingService`, add `CircuitBreaker.call_async`, record a
   degradation marker when the breaker opens. *Gate: a unit test asserts ranking is
   non-trivial (non-zero query norm) when the embedder is up, and a test asserts no
   event-loop hang.*
4. **Per-session summarizer/delta-encoder** (§4.8.2): construct
   `HierarchicalSummarizer` + `DeltaEncoder` per optimizer (drop `get_*` globals) or
   key state by session id. *Gate: concurrency test — two sessions don't share
   rolling-summary text.*
5. **Strip `_volatile_turn`** (§4.8.3); **fix profile-vs-env precedence** (§4.8.4);
   **fix `tokenize_count_sync` never-awaited** so remote `/tokenize` actually works
   in the async app (§4.8.8) — needed before the budget is enforced in calibrated tokens.
6. **Bound `_SLOT_MAP`** (§4.10.1); **offload optimizer construction** + share one
   tokenizer + narrow the session lock (§4.9.1).
7. **Remove `OutputShaper` from the proxy path** (§4.2.5) — restores the hard
   constraint and a fair benchmark.
8. **Delete the dead/phantom modules** (§7): `kv_slot_tracker`, `mtp_state`,
   `attention_sink`, `pattern_injector`, `dependency_orderer`, `hierarchical_index`,
   `goal_relevance_scorer`, `incremental_updater`, `context_template_matcher`; strip
   dead branches from `state_rag`, `thinking_preserver`, `prompt_templates`,
   `selective_truncator`, `context_canonicalizer`; remove their config flags.
9. **CI-gate the dry-run** (§4.12.2).
   *Phase gate: `bash scripts/dev.sh` green; dry-run greenlit; new regression tests
   for governor/embedding/session-isolation pass.*

### Phase 2 — TTFT/TPS + adaptive budget policy (the mission)
*Goal: make the proxy faster than direct on TTFT mean (not just median), and recover
the quality the cliff fix traded away.*

1. **Adaptive budget — full policy** (§4.2.2, §3.1): add the task-complexity /
   code-density and codebase-size signals to the ceiling, and switch compaction to
   the **cache-break amortization trigger** (`marginal_per_turn_prefill ×
   remaining_turns > re_prefill_cost`). This is what removes the fold-turn TTFT
   spikes (cached→882 re-prefills) and stops the quality regression (more verbatim
   context retained).
2. **Append-only default for cache-stable mode** (§4.3.1): lossless compression of
   new content only; let `--context-shift` bound the window; reserve compaction for
   genuinely window-bound sessions.
3. **Zero-copy SSE passthrough** (§4.4.1).
4. **Token-counting overhaul** (§4.4.2, §4.4.3): per-message memo + running totals +
   count-once-per-stage; remove the O(n²) floor pass.
5. **Probing fixes** (§4.9.2, §4.9.3, §4.9.5): gather sub-probes, bound startup,
   single-flight on TTL expiry, one long-lived httpx client.
6. **Move post-stream + per-request CPU off the loop** (§4.4.4, §4.9.4).
7. **Real TTFT metric + cache→TTFT correlation + quality gate** (§4.12.1, §4.12.3,
   §4.12.6); fix the `context_window_wall` turn-1 artifact (§4.12.7).
   *Phase gate: re-run `--scenario opencode --turns 30 --rounds 3` (≥3 rounds);
   prefix-cache reuse stays ≥ 0.8; proxy TTFT **mean** ≤ direct (fold spikes gone);
   token savings recover toward 65 %+; and ROUGE-L F1 / token-Jaccard /
   `code_syntax_validity` climb back **above** the `0.7.26_fix` baseline (0.14 / 0.14
   / 0.95) with `length_ratio` back inside [0.5, 2.0].*

### Phase 3 — Context efficiency & quality (the missing optimizations)
1. **Error-aware tool-output compression** (§4.1.1): failures-only for tests/builds/
   lint, keep error/stack/`file:line`, group by file/rule, dedup-with-counts,
   per-tool budgets + `full_output` escape hatch.
2. **Reversible compression + retrieval handles** (§4.1.2): content-addressed store +
   `expand(id)` tool (MCP-style); pair with cached re-read collapse (§4.1.3) and
   global code dedup (§4.5.2).
3. **Syntactic code slicing + evicted-code skeleton index** (§4.5.3, §4.3.3).
4. **Volatile-field relocation** (§4.1.4); drop the redundant anchor (§4.2.6).
5. **MTP/ChatML hardening** (§4.6.2, §4.6.3); multi-file fixtures (§4.12.7).
   *Phase gate: token savings up AND headline quality (esp. `code_block_ratio`,
   `rouge_l_f1`) up vs Phase 2 baseline.*

### Phase 4 — Maintainability (reduce the chance of regressions)
1. **Decompose the god object** (§4.2.3): `BudgetGovernor`/`PrefixLayout`/`StageRunner`.
2. **Typed summary region** (§4.7.3) replacing scattered content-marker guards.
3. **Surface persistent stage failures** (§4.8.5, §4.8.7); prune `AgentStateStore.goals`
   (§4.10.4); session reaper (§4.10.3).
   *Phase gate: `optimizer.py` < ~1500 lines; no stage with two gate predicates;
   `dev.sh` green.*

---

## 6. Borrowable techniques from reference projects (ranked)

| # | Technique | Source | Where it fits |
|---|---|---|---|
| 1 | **Error-aware, failures-only tool-output compression** (keep error/stack/file:line; collapse passing tests to counts; group by file/rule; dedup-with-counts; deterministic, no LLM) | snip, rtk | §4.1.1 / Phase 3 — biggest win |
| 2 | **Reversible compression + retrieval handles** (placeholder + store original + `expand` tool; cached re-read ≈ 13 tokens) | headroom, lean-ctx | §4.1.2 / Phase 3 |
| 3 | **Cache-safe live-zone compression + volatile-field relocation** (move dates/UUIDs/SHAs out of the prefix; CacheAligner warns on cache-busting volatile content) | headroom, lean-ctx | §4.1.4, §4.7 / Phase 3 |
| 4 | **Outline/signatures-first code views** with targeted line-range expansion | lean-ctx, rtk, swe-pruner | §4.3.3, §4.5.3 / Phase 3 |
| 5 | **Task-aware code pruning** (infer the current goal; keep task-relevant lines; stub the rest) — heuristic or tiny local skimmer | swe-pruner | §4.5.3 / Phase 3 (optional) |
| 6 | **Per-tool token budgets + savings analytics + passthrough escape hatch** | snip, rtk | §4.11.4 / Phase 3 |
| 7 | **Effort routing** (lower reasoning_effort on routine tool-result turns) — *client-side or not at all; the proxy must not own response verbosity* | headroom | §4.2.5 (caution) |

Central lesson across all five: **treat tool output as a first-class, deterministic,
command-aware compression layer** (60–90 % savings, no extra model call, no added
latency) — and **never delete information irreversibly** without a retrieval handle.
Both are gaps in the current proxy.

---

## 7. Appendix — dead / phantom / no-op inventory (verified)

Of 23 audited stage modules, **11 (~48 %) are dead weight**. A client proxy cannot
read backend KV tensors, MTP hidden state, or expert routing; anything claiming to
is structurally impossible.

**Delete outright (dead/phantom/pure no-op):**
| Module | Verdict | Evidence |
|---|---|---|
| `kv_slot_tracker.py` | Phantom — `build_slot_map` return value discarded (`optimizer.py:1115`); hints on the strip-list | proxy can't address backend KV slots |
| `mtp_state.py` | Dead — instantiated (`optimizer.py:198`), `save_state` never called; `restore` always `None` | client-proxy spec-decode impossible |
| `attention_sink.py` | No-op — `apply_attention_sinks` returns input unchanged (`:131-141`) | behind disabled WARN flag |
| `pattern_injector.py` | Dead — instantiated (`optimizer.py:194`), no method ever called | markers stripped anyway |
| `dependency_orderer.py` | Dead — "instantiated but never called" (`optimizer.py:192`); `_reconstruct` returns unchanged | — |
| `hierarchical_index.py` | Dead — instantiated (`optimizer.py:197`), no method called | — |
| `goal_relevance_scorer.py` | Dead — only consumer `state_store.prune_by_relevance` never called | — |
| `incremental_updater.py` | No-op as called — `update_context(optimized, "")` always returns unchanged (`optimizer.py:1336`) | still triggers a recount at `:1337` |
| `context_template_matcher.py` | No-op — `apply_template` returns unchanged (`:78-86`) | behind disabled flag |
| `thinking_preserver.py` | Mostly no-op — `process_messages` is a pure copy (`:40-48`); preseed is a flagged-harmful mutation | simplify |
| `prompt_templates.py` | Mostly no-op — `apply_template` returns unchanged (`:139-150`) | keep `classify_task` only if wanted |

**Simplify (dead branches inside live modules):** `hit_prediction_model.py` (drop
XGBoost/persistence — one label source is circular, `optimizer.py:1641`; keep the
static heuristic), `state_rag.py` (delete the never-called dependency-graph branch),
`symbol_index.py` (remove the optimizer's unused instance — `StateBasedRAG` builds
its own), `selective_truncator.py` (keep `remove_duplicates`, delete the rest),
`context_canonicalizer.py` (drop dead methods).

**Genuinely real — keep:** `static_prefix_kv.py` (honest text-memo fast path; rename
away from "KV"), `cache_aware_chunker.py`, `loop_detector.py`, `progress_tracker.py`,
`delta_encoder.py`, `chunk_fingerprint.py`, `goal_decomposer.py` (last feeds
observability, not the backend request).

**Config flags controlling dead/phantom subsystems (remove):**
`agentic.attention_sinks_enabled`, `agentic.reasoning_preseed_enabled`,
`v050.enable_experimental_backend_hints`, `agentic.prompt_template_enabled`.
(`MTP_BOUNDARY_ALIGNMENT_ENABLED` / `STATIC_LAYER_ALIGNMENT_ENABLED` from older
reviews **no longer exist** in `config.py`.)

**The real, load-bearing pipeline is small:** compactor + hierarchical_summarizer +
token_aware_truncator + context_compressor + code_chunking/code_block_optimizer +
tool_output_compressor/filter + state_rag(`get_context_for_step`) + loop_detector +
delta_encoder + chunk_fingerprint + static_prefix_kv fast path. The rest is
scaffolding that obscures the six-governor bug and should go.

---

### Bottom line

**Status after the cliff fix (fresh `0.7.26_fix` benchmark):** the cache-instability
crisis is largely resolved — **prefix reuse 22 % → 81 %**, **TTFT median now beats
direct (8.6 s vs 14.8 s)**, end-to-end latency at parity. But the fix worked by
letting the context grow against a **fixed 12 K cap that is neither enforced
(context runs 16–18.5 K) nor appropriate (7 % of a 262 K window idle)**, and that
trade **regressed quality** (ROUGE-L 0.27 → 0.14, responses 2× longer, broken code on
2 turns). The bottleneck moved from *cache stability* to *budget policy*.

**The headline lever now is adaptive budgeting** (§3.1, §4.2.2): replace the fixed
cap with a ceiling that grows with **conversation horizon, task complexity and
codebase size**, floored at the immutable-zone reservation, and drive compaction by
**cache-break amortization** instead of a token threshold. That single change keeps
`reserved < budget` (so the six governors stop fighting), folds rarely (so the cache
stays hot and the TTFT p90 spikes vanish), and retains more verbatim context (so
quality recovers) — turning the fixed `max_optimized_tokens` into a last-resort
valve near the real window.

The rest is still **deletion and consolidation, not invention**: a real async
embedding service (the dead RAG + latent deadlock), off-loop construction + zero-copy
streaming + per-message token counting (TTFT/TPS), the `_SLOT_MAP`/session-isolation
fixes, the `tokenize_count` never-awaited bug, and error-aware + reversible
tool-output compression (the main missing context win). Phase 1 is mostly low-risk
deletions/guards plus the budget skeleton and should greenlight the dry-run; Phase 2
— the adaptive policy + amortization trigger — is where faster-than-direct TTFT
*mean* and recovered quality are won.
