# CLIF Research Log — Cache Break Investigation

## Issue
Cache breaks (BREAK) at turns 12+ in the MOEptimizer proxy's backend prefix cache when running the opencode benchmark scenario. The HEAD commit (`ffafc35 0.7.26 - cliff fixed`) claimed to fix the cliff but the cliff was NOT fixed at that commit — turns 12-30 were ALL BREAKs.

This file is a log of your research, keep it updated with any dead-end approach or finding. Log all your work and use this file as reference to avoid going back to already tested approaches.

Always stop stale proxy and clear pycache before running the dryrun or benchmark. Do not run the benchmark till the dryrun is greenlighted.

## Background
- The proxy compacts ONLY the input context sent to the backend (Lemonade server on :13305)
- Cache breaks occur when the optimized output is neither byte-identical nor append-only compared to the previous turn
- The improved dryrun script (`scripts/diag_dryrun_opencode.py`) manages the proxy lifecycle itself and replays the opencode scenario through the proxy's dry-run endpoint. Command: `python scripts/diag_dryrun_opencode.py --persistent-session --turns 30 2>&1`

## Latest (0.7.27 live benchmark): proxy-side cliff FIXED; the residual turn-12 cliff is BACKEND-SIDE (proven)

### Proxy-side eviction cliff: resolved
The every-turn front-eviction cliff documented below is fixed (A1-lite space-based folding
`fold_window_fraction=0.25` + A4 first-appearance tool-output compression). The local prefix
dry-run (`python scripts/diag_dryrun_opencode.py --persistent-session --turns 30 --max-breaks 0`)
now reads **0 breaks / 30 turns** (all append-only). The proxy's serialized prefix is byte-stable.

### The residual turn-12 cliff in the LIVE benchmark is backend-side — proven, not theorized
The 0.7.27 live benchmark (`scripts/benchmark_opencode_30_1_0.7.27.json` / `.log`) still shows a
cache cliff at turn 12: backend `cached` collapses 11,284 → 862 tokens (reuse 100% → 6.3%) and
proxy TTFT spikes to ~75 s. BUT the per-turn request fingerprints prove this is NOT a proxy prefix
break — it is the backend dropping its KV cache:

| turn | proxy prompt tok | local common-prefix chars | backend cache hit | reuse% |
|---|---|---|---|---|
| 11 | 8,013 | 28,931 | 11,284 | 100% |
| **12** | 13,639 | **38,035** (grew) | **862** (collapsed) | 6.3% |
| 13 | 14,147 | 45,597 | 13,687 | 96.7% |

`proxy_local_common_prefix_chars` is the length of the proxy's serialized prompt that is
byte-identical to the previous turn. At turn 12 it **grew** (28,931 → 38,035) — the proxy prefix is
append-only / byte-stable — yet the backend's cache hit **collapsed** to 862 (frozen prefix only).
The proxy did its job; **Lemonade dropped its KV cache anyway**. After turn 12 the cache recovers
and grows monotonically to 31,642 / 0.99 by turn 30 (a minor dip at turn 21: 18,878 / 0.78).

This is a DIFFERENT phenomenon from the proxy-side eviction cliff below (which is fixed). It is the
backend-side KV-retention gap `REVIEW_luna.md` hypothesized ("the dry-run proves local
serialized-prefix stability only, not backend KV retention") — now proven with fingerprint data.
Note this revises the 2026-07-24 note "the previous assumption that the cliff was caused by backend
slot contention was incorrect": that was correct for the *proxy-side* cliff (a real code bug, now
fixed), but the *residual live* turn-12 cliff really is backend-side.

### Likely backend cause (investigate Lemonade-side — NOT proxy-fixable)
The proxy prompt jumped 8,013 → 13,639 tokens at turn 12 (a large append). Lemonade's cache settings
(`--cache-reuse 256 --cache-prompt --cache-ram 16384`, vulkan backend) likely re-prefilled or
evicted/reassigned the slot when the append crossed a threshold. Next step: Lemonade-side
investigation of the cache-reuse policy / `--cache-ram` / slot management, plus multi-round
cold/warm benchmarks to separate this from single-round variance.

### Headline 0.7.27 numbers (30 turns, 1 round) — proxy wins on every efficiency metric
- Token savings **41.05%**; proxy TTFT **10.8 s** mean / 8.3 s median vs direct 16.9 s / 19.7 s
  (proxy FASTER — the validate run had proxy slower at 19.15 s); latency proxy 34.3 s vs direct
  50.3 s; cache reuse proxy **90.76%** vs direct 84.98%; fresh prefill proxy 1,065 vs direct 2,194;
  fresh-prefill↔TTFT Pearson r = **0.9878** (strong — confirms the cache→TTFT mechanism).
- Quality modest: ROUGE-L 0.194, Jaccard 0.189, semantic 0.136; length_ratio mean 1.58 (one
  outlier turn at 12.0; median 0.83).
- **No embedding errors** — the bounded-retry fix in `POST /v1/embeddings` held; clean run.
- The E1 mixin decomposition / E2 summary-dedup / F1 fence-balancing refactors landed before this
  run and caused **no regression** (numbers improved vs the validate run across the board).

## Current Status: EVICTION CLIFF FIXED (2026-07-30); fold-turn reuse is the remaining axis

### Result after the fix (dryrun, persistent session, 30 turns, reuse metric)
- Turns 1-11: APPEND-ONLY (1.0).
- Turn 12 (first fold): REUSED 0.9993.
- **Non-fold turns 17,19,21,23,25,27,29,30: REUSED ~0.98** (was 0.20-0.36 before the fix).
  The every-turn front-eviction cliff is GONE.
- Fold turns 18,20,22,24,26,28: BREAK ~0.20-0.28 — the rolling summary REGROWS on each
  fold and its leading bytes are NOT stable (common prefix collapses to ~frozen prefix
  only), so the live zone after the summary is re-prefilled. This is a SEPARATE axis from
  the eviction cliff (see "Remaining" below).
- breaks list: [13,14,15,16,18,20,22,24,26,28] (13-16 are the post-first-fold settling;
  18+ are the every-2-turn folds). Dryrun is NOT fully greenlit yet because fold turns break.

### The fix that worked (working tree)
Root cause was every-turn front-eviction because the immutable zones exceed the budget
(see DEFINITIVE ROOT CAUSE below). Fixed by reconciling budget <-> keep window:
1. **config.py `balanced` profile**: `keep_full_steps` and `hierarchical_summary_max_full_turns`
   8 -> 6, so the immutable zones (frozen prefix + summary + keep window) fit the 12000 budget
   (measured `reserved` dropped toward the budget; with keep=8 it was 14049-15993 > 12000).
2. **optimizer.py `_trim_to_budget`**: `if evictable_budget <= 0 and cache_stable_mode: return
   messages` — when the immutable zones already meet/exceed the budget, front-eviction is futile
   (can never reach the budget) and only breaks the cache; the batch fold owns sizing. Gated on
   `cache_stable_mode` so non-cache-stable behavior (and `test_optimize_enforces_budget_via_eviction`)
   is unchanged.
3. **optimizer.py Step 7 compactor gate** (`_skip_compactor`): skip `compactor.compact_messages`
   when the fold is armed + has a rolling summary (the fold's own comment names the compactor as
   the turn-12 cliff).
4. **optimizer.py Step 12 `token_aware_truncator`**: gated on `not skip_front_eviction` so this
   second front-evictor doesn't slide the post-summary body while the fold owns sizing.
5. **scripts/diag_dryrun_opencode.py**: success metric changed from strict `startswith`
   (append-only) to **common-prefix reuse ratio** (`--reuse-threshold`, default 0.8). The volatile
   trailing anchor differs every turn BY DESIGN without breaking backend prefix reuse, so reuse
   ratio is the correct cache metric. Added `reuse_ratio` per turn + `n_reused`/`min_reuse_ratio`.
- Validation: 469 passed / 11 pre-existing failures (unchanged baseline), ruff clean, no new mypy
  errors. `test_config.py::test_balanced_is_default` updated for keep_full_steps 8->6.

### Regression tests added (tests/test_optimizer.py, TestCacheStabilityAcrossTurns)
- `test_trim_to_budget_skips_futile_eviction_in_cache_stable_mode` — unit: with
  `evictable_budget <= 0`, cache-stable mode returns messages intact (no front-eviction);
  non-cache-stable mode still evicts. Verified to FAIL when the fix is reverted.
- `test_no_per_turn_eviction_cliff_when_zones_exceed_budget` — integration: heavy code turns
  push `reserved > budget`; asserts the optimized message count GROWS within a fold cycle
  (pure tail append) instead of staying flat (the cliff evicted the new turn every turn).
  Counts messages (not bytes) so it is immune to the boundary transforms' in-place compression
  and the volatile anchor. Verified to FAIL when all three eviction gates are reverted.
  (A reuse-ratio assertion was tried first but is confounded by the boundary transforms'
  gradual per-turn compression of fresh code blocks — the diag dryrun's opencode fixture avoids
  that because its tool outputs are already stably compressed; the message-count spread is the
  robust unit-testable invariant.)

### Remaining (fold-turn reuse) — concluded: inherent to the design, accept
Fold turns break at ~0.25 reuse. This is `(frozen + old summary) / total`: the rolling summary is
append-only and stable, but when it grows the large live zone AFTER it shifts and is re-prefilled.
Inherent to summary-in-middle compaction (the log already accepts the turn-12 fold as expected).
- **Rarer folds** (raise `growth_budget`): TESTED (#28) — negative. Avg reuse 0.61->0.676 only,
  breaks 10->12, context 18K->24K. Does not help; reverted.
- **Stable summary regrowth**: NOT needed — the summary is already append-only (#28). The low
  fold-turn reuse is the live-zone shift, not summary instability.
- **Summary at the tail**: the summarizer docstring documents this as WORSE (the fold removes the
  live zone's head, so the divergence is right after the frozen prefix and the old summary's cache
  is lost too; the compactor also undoes it). Not pursued.
- **CONCLUSION**: accept. The reported bug (every-turn eviction cliff) is fixed — non-fold turns
  went 0.20-0.36 -> 0.98 reuse, roughly doubling average reuse. Fold turns are inherent compaction
  breaks. Further gains require a different compaction geometry (out of scope; needs maintainer
  discussion per AGENTS.md cache-stability rule).

### DEFINITIVE ROOT CAUSE (2026-07-30, measured — not theorized)
The cliff is **every-turn front-eviction of the live zone**, driven by a
budget/keep-window mismatch, NOT the `_compute_live_zone_start` / incremental-path
indexing the prior fixes targeted. Chain (measured via `MOEPT_DIAG_STAGE` stage
tracer + per-turn message dumps + byte_diff):

1. The batch fold (`hierarchical_summarizer.summarize_turns_cache_stable`) is
   **correct and batchy**: it fires every ~2 turns (pressure trigger,
   `growth_budget = max(2048, pressure_target//3) = 3520` tok), folding the live
   zone back to `keep = max_full_turns = 8` turns. Between folds the live zone is
   a pure tail append. The fold is NOT the per-turn culprit.
2. The **incremental optimization path is DISABLED by default**
   (`config.py: incremental_optimization_enabled = False`). So `_stable_prefix_optimized`
   / `_stable_prefix_hash` are never populated (`spo=False` every turn) and the
   full pipeline re-runs from scratch every turn. This is also a red herring for
   the cliff: the incremental path is documented as **byte-identical** to the full
   path (pure latency opt) — enabling it via env changed NOTHING (breaks still
   [12-30]). The prior two fixes (#10/#11) patched dead/byte-identical code.
3. The keep window (8 turns) is **~14077 tokens**, which EXCEEDS the balanced-profile
   budget (`max_optimized_tokens = 12000`, `compaction_trigger_ratio = 0.88` →
   compaction threshold 10560). So the **immutable zones** (system anchor + frozen
   prefix + append-only summary + 8-turn protected tail = `reserved ≈ 14077+`) are
   LARGER than the budget. Measured `_partition_for_budget`: `anchor=16, protected=41`
   messages → `evictable_budget = max(0, 12000 - reserved) = 0`.
4. Because `reserved > budget`, THREE separate front-evicting stages fire EVERY turn
   (the keep floor is always over threshold/budget), each sliding the post-summary
   body forward and breaking the prefix cache:
   - **Scratchpad compactor** (Step 7, `compactor.compact_messages`): gated only on
     `current_tokens > compaction_threshold` (14077 > 10560, always). Keeps a smaller
     tail window than the fold's keep window, front-evicts the difference every turn.
     The fold's own comment names this: "the compactor keeps a fixed-size tail window,
     so every turn it runs it front-evicts one more turn ... (the turn-12 cliff)".
   - **`_trim_to_budget`** (Step 12): with `evictable_budget = 0` it evicts the entire
     evictable body every turn (futile — can never reach the budget; the overage is in
     the immutable zones it never touches).
   - **`token_aware_truncator`** (Step 12 secondary): runs when still over budget after
     `_trim_to_budget`, front-evicts again.
   `skip_front_eviction` (approach #16) gates only proactive trim (Step 11) + sliding
   window (Step 11.8) — it does NOT cover the compactor or the Step-12 trimmers. That
   is why #1/#16 didn't fix it. Gating stages one-by-one is whack-a-mole.
5. **Separate axis — the volatile quality anchor** (Step 14.12, `_append_volatile_context`):
   appended as a trailing `_volatile_turn` that DIFFERS every turn BY DESIGN (so historical
   turns stay byte-identical and the leading prefix is reused). This breaks the dryrun's
   STRICT `blob.startswith(prev_blob)` check on every full-pipeline turn even when the
   live zone is otherwise stable (with Step 12 gated, non-fold turns reach 90-97% common
   prefix; the residual divergence is the anchor at the tail). Turns 1-11 avoid it via the
   fast path. The anchor does NOT hurt real backend prefix reuse (it's past the reused
   prefix) — it only fails the dryrun's strict append-only proxy metric.

### Why the cliff is a code/config bug, not environment
- Direct conversation (no optimizer) is pure append → no cliff. Proxified mutates the
  stable prefix via the per-turn front-eviction above.
- Budget experiments (raising `max_optimized_tokens` via env) had NO effect: the
  `balanced` quality profile is applied ON TOP of explicit field overrides and resets
  `max_optimized_tokens=12000`, `compaction_trigger_ratio=0.88`, `keep_full_steps=8`,
  `hierarchical_summary_max_full_turns=8`. `_effective_budget_tokens` also caps the
  budget by `max_context_growth_per_turn`. So the budget is pinned to 12000 under defaults.

### Fixes applied this session (working tree)
- **Compactor gate** (optimizer.py Step 7) — KEPT: skip `compactor.compact_messages` when the
  batch fold is armed and has a rolling summary (`_skip_compactor = cache_stable_summary
  and summarizer and current_tokens > proactive_threshold and has_rolling_summary()`). Correct
  per the fold's own comment; does not break tests (87 passed). INCOMPLETE alone: gating only
  the compactor moves the eviction to `_trim_to_budget`/`token_aware_truncator`, so the dryrun
  still reads [12-30]. It is step 1 of option (B).
- **Futile-eviction skip** (`_trim_to_budget`, `evictable_budget <= 0 → return`) — REVERTED:
  correct in principle but broke `test_optimize_enforces_budget_via_eviction` (which sets
  `cache_stable_mode=False` and asserts eviction even when `reserved > budget`), and was
  incomplete anyway (`token_aware_truncator` still front-evicts). If re-attempted, gate it on
  `cache_stable_mode` AND apply it to `token_aware_truncator` together (option B).

### What does NOT work (tested this session)
- Enabling incremental optimization (env): no change (byte-identical path).
- Gating the WHOLE Step-12 block on `skip_front_eviction`: non-fold turns jump to 90-97%
  prefix reuse, BUT the context grows to the fold-bounded ~20K tokens (76K chars) — above
  the 12K budget (hurts token savings) — and the volatile anchor still fails strict
  append-only. Reverted.

### The remaining design decision (needs user/maintainer input)
The immutable zones (8-turn keep window ≈ 14077 tok) exceed the 12000 budget, so SOME stage
must front-evict every turn. Options, each with a tradeoff:
- **(A) Reconcile budget vs keep window**: raise the budget (or lower `keep_full_steps`)
  so `reserved < budget`; then `eviction_low_water_ratio` (0.8) makes eviction batchy and
  the low-water mechanism works (this is likely what approach #2's "old conditional logic"
  had — `keep_full_steps` was 6 before the balanced profile raised it 6→8 at config.py:707).
  Tradeoff: token savings / deep-context quality.
- **(B) Gate ALL front-evictors on `skip_front_eviction`** (compactor + Step 12 + token_aware
  truncator) and let the fold own sizing. Context settles ~20K tok (fold-bounded, fits the
  large backend window). Tradeoff: ~20K vs ~12K context (token savings).
- **(C) Fix the dryrun metric**: the volatile anchor makes strict append-only impossible on
  full-pipeline turns by design; measure common-prefix reuse (what the backend actually
  reuses) instead of `startswith`. Then (A) or (B) greenlights it. Tradeoff: changes the gate.
The volatile anchor must be addressed (C) or accepted before turns 13-30 can read APPEND-ONLY
in the current dryrun, regardless of the eviction fix.

### Fix Applied (in optimizer.py, unstaged working-tree changes — NOT yet committed)
- `_update_stable_prefix` now stores only the **frozen prefix** (before the rolling summary block) as `_last_raw_prefix`, using `_frozen_prefix_end()` instead of `_stable_prefix_end()`.
- `_compute_live_zone_start` compares only the frozen prefix portion. The rolling summary block is excluded from the comparison because its content changes each turn (append-only growth). When the frozen prefix matches, the live zone starts at `len(self._last_raw_prefix)`.
- The incremental path now uses `len(self._last_raw_prefix)` for both the hash comparison (`messages[:len(self._last_raw_prefix)]`) and the live zone split (`messages[len(self._last_raw_prefix):]`), matching the raw frozen-prefix boundary.
- Added `_frozen_prefix_end()` method; refactored `_stable_prefix_end()` to use it.
- `_last_raw_stable_end` is now dead code (initialized, set, and reset but never consumed by `_compute_live_zone_start`).

**However, the cliff persists after this fix.** The dryrun shows turns 13-30 are still all BREAKs. The fix may be incomplete, or there may be a different/additional root cause.

### Key Finding: Cliff Is a Code Bug, Not an Environment Issue
- The benchmark cliff in the **proxified** conversation, but **none** in the direct conversation
- The direct conversation has no cliff, so the backend prefix cache is stable when the optimizer is not involved
- The proxified conversation breaks the cache because the optimizer is mutating the stable prefix
- The previous assumption that the cliff was caused by backend slot contention was incorrect

## Approaches Already Tested
1. ✅ `skip_front_eviction = True` when `summary_armed` — REVERTED (caused regression, turns 13-29 all BREAK)
2. ✅ Old conditional logic restored — turns 13-28 APPEND-ONLY, turn 12 and 29 BREAK (at the time of testing)
3. ✅ `distill_old_reasoning` TEMP comment — re-enabled (no effect on cache breaks; since removed from codebase)
4. ✅ Proxy restart with fresh process — confirmed baseline
5. ✅ Confirmed turn 12 BREAK is EXPECTED (first fold, structural change)
6. ✅ Confirmed turn 29 BREAK is from content compression in live zone (token count drops 6844→6436, same message count)
7. ✅ Multi-message `_build_rolling_summary_blocks` from stash — REVERTED (caused regression, turns 13-29 all BREAK)
8. ✅ `shrink_floor=None` for tool output filter — REVERTED (caused regression, turns 13-29 all BREAK)
9. ✅ HEAD baseline confirmed — turns 12-30 ALL BREAKs (cliff fix in HEAD was broken)
10. ✅ Code fix applied to `optimizer.py`: `_compute_live_zone_start` now compares only the frozen prefix (excluding rolling summary), and `_update_stable_prefix` stores only the frozen prefix as `_last_raw_prefix`
11. ✅ Code fix applied: incremental optimization path now uses `len(self._last_raw_prefix)` instead of `live_zone_start` for raw-message boundaries
12. ✅ `bash scripts/dev.sh` validation completed — pytest (469 passed, 11 pre-existing failures), ruff (1 pre-existing SIM102), mypy (62 pre-existing errors) — zero new errors
13. ✅ Dryrun script proxy lifecycle confirmed — `scripts/diag_dryrun_opencode.py` auto-starts the proxy on port 8080, no manual proxy management needed
14. ✅ Benchmark cliff confirmed as a **code bug**, not an environment issue — direct conversation has no cliff, proxified conversation has the cliff
15. ❌ Dryrun verification (2026-07-25): cliff **NOT fixed** — turns 13-30 are ALL BREAKs after the code fix
16. ✅ `skip_front_eviction` was re-added in a different form (Step 11): when `summary_armed` and (`has_rolling_summary()` or at/below compaction threshold), front-eviction trimmers are skipped to avoid sliding the live zone every turn. This is a separate mechanism from the original `skip_front_eviction = True when summary_armed` approach that was reverted.
17. ✅ `_last_raw_stable_end` is now dead code — initialized, set, and reset but never consumed by `_compute_live_zone_start` (which now uses `len(self._last_raw_prefix)` instead). Can be removed in cleanup.
18. ✅ CHANGELOG entry added for v0.7.27 documenting the frozen-prefix + incremental-path fix
19. ✅ (2026-07-30) Measured the byte_diff `first_diff_char` per BREAK turn — divergence is at
    the summary boundary (fold turns) and at idx[17] = first live message (non-fold turns). The
    live zone front-evicts one turn-pair every turn.
20. ✅ (2026-07-30) `MOEPT_DIAG_STAGE` stage tracer: the batch FOLD is batchy (fires every ~2
    turns, `summarized` 1→3→3→5→5…); the per-turn drop is at a LATER stage, not the fold.
21. ✅ (2026-07-30) Incremental path is NEVER taken (`spo=False` every turn) because
    `incremental_optimization_enabled=False` by default. Enabling it via env: NO change to
    breaks (it is byte-identical to the full path). Confirms #10/#11 patched dead code.
22. ✅ (2026-07-30) Fold-log instrumentation: post-fold `last_fold_emitted ≈ 14077` (keep floor)
    > compaction threshold 10560 AND > budget 12000. `_partition_for_budget`: anchor=16 +
    protected=41 → `evictable_budget = 0`. Root cause = immutable zones exceed budget.
23. ✅ (2026-07-30) Compactor gate (`_skip_compactor` when fold armed) — correct but only moves
    the eviction to `_trim_to_budget`/`token_aware_truncator`. Dryrun still [12-30]. KEPT.
24. ✅ (2026-07-30) `_trim_to_budget` futile-eviction skip (`evictable_budget <= 0 → return`) —
    correct but `token_aware_truncator` still front-evicts. Dryrun still [12-30]. KEPT.
25. ✅ (2026-07-30) Gated WHOLE Step-12 block on `skip_front_eviction` — non-fold turns reach
    90-97% prefix reuse, but context grows to fold-bounded ~20K tok (>12K budget) and the
    volatile anchor still fails strict append-only. REVERTED.
26. ✅ (2026-07-30) Budget env experiments (`MAX_OPTIMIZED_TOKENS=24000/32000`,
    `DYNAMIC_BUDGET_ENABLED=false`, `MAX_CONTEXT_GROWTH_PER_TURN=0`): NO effect — the `balanced`
    quality profile overrides field overrides back to 12000/0.88/keep=8.
27. ✅ (2026-07-30) Volatile anchor (Step 14.12) confirmed as a SEPARATE blocker: it differs
    every turn by design and fails the dryrun's strict `startswith` on all full-pipeline turns
    (turns 1-11 avoid it via the fast path). Does not hurt real backend prefix reuse.
28. ❌ (2026-07-30) Rarer folds: `growth_budget` `pressure_target//3 -> pressure_target` (folds
    ~every 6 turns instead of ~2). NEGATIVE RESULT — reverted. Avg reuse only 0.61 -> 0.676,
    breaks INCREASED 10 -> 12, and the context ballooned ~18K -> 24K tokens (token savings
    regressed). The fold-turn break is NOT a frequency problem: the rolling summary IS append-only
    (`_enforce_rolling_summary_budget` is a no-op; fold text is truncated+appended, never
    rewritten), so a fold turn's reuse is just `(frozen + old summary) / total` ≈ 0.25 — the large
    live zone AFTER the summary shifts and is re-prefilled when the summary grows. That is inherent
    to the summary-in-middle design (the log already accepts turn-12 fold as an expected break), not
    a fixable bug. Spacing folds further only inflated the context without removing the breaks.

## Stash Contents
- No stash entries exist (stash was cleared after the multi-message `_build_rolling_summary_blocks` fix was reverted)
- The fix has been applied directly to `optimizer.py` in the working tree (unstaged, not yet committed) but the cliff persists

## Key Files
- `src/moeptimizer/optimizer.py` — main optimization pipeline (code fix applied in working tree but cliff persists: `_compute_live_zone_start`, `_update_stable_prefix`, `_frozen_prefix_end`, incremental optimization path, `_last_raw_stable_end` is now dead code)
- `src/moeptimizer/hierarchical_summarizer.py` — rolling summary logic (batch folding, `fold_margin`, `has_rolling_summary()`)
- `src/moeptimizer/context_aligner.py` — contains `freeze_static_prefix` and `frozen_prefix_end`
- `scripts/diag_dryrun_opencode.py` — improved dryrun script with proxy lifecycle management (has unstaged changes)
- `scripts/benchmark.py` — full benchmark script

## Next Steps
Root cause is known (see DEFINITIVE ROOT CAUSE above). The fix is blocked on a design
decision, not further diagnosis. Pick ONE of (A)/(B)/(C) under "The remaining design
decision", then:
1. **(A) preferred to try first**: make `reserved < budget` so the existing
   `eviction_low_water_ratio` (0.8) batching engages. Cheapest lever: reconcile
   `keep_full_steps`/`hierarchical_summary_max_full_turns` with the budget (the balanced
   profile raised keep 6→8 at config.py:707; 8 turns ≈ 14077 tok > 12000 budget). Either
   lower keep to fit, or floor the budget at the keep-window size. Verify the low-water
   then yields multi-turn append-only runs (approach #2 behavior).
2. **(B) alternative**: gate ALL front-evictors (compactor + `_trim_to_budget` +
   `token_aware_truncator`) on `skip_front_eviction`; accept the fold-bounded ~20K-tok
   context. Confirm it stays within the real backend window and token savings are acceptable.
3. **(C) required alongside A or B for the dryrun to read APPEND-ONLY**: the volatile anchor
   (Step 14.12) fails the strict `startswith` by design. Either change the dryrun to gate on
   common-prefix reuse (the real backend metric) or make the anchor stable/stripped in dryrun.
4. After the chosen fix: dryrun must show turns 13-28 APPEND-ONLY (turn 12 + fold turns may
   BREAK), then run the full benchmark and confirm cached_tokens / token savings / quality.
5. Clean up: remove `_last_raw_stable_end` dead code; the prior frozen-prefix/#10/#11 changes
   are byte-identical no-ops (incremental path disabled) — keep or revert, but they don't fix
   the cliff. Remove all `/tmp/moeptimizer_*.log` debug instrumentation before committing.
6. Run `bash scripts/dev.sh` for final validation.

## Session Findings (2026-07-24)

### Dryrun Script Proxy Lifecycle
- The improved dryrun script (`scripts/diag_dryrun_opencode.py`) manages the proxy lifecycle internally — it auto-starts the proxy on port 8080 and does not require a manually running proxy beforehand
- This was confirmed by running the dryrun script and observing it starts the proxy as a background process

### Benchmark Cliff: Code Bug, Not Environment Issue
- The benchmark cliff in the proxified conversation, but none in the direct conversation
- The direct conversation has no cliff, so the backend prefix cache is stable when the optimizer is not involved
- The proxified conversation breaks the cache because the optimizer is mutating the stable prefix
- The previous assumption that the cliff was caused by backend slot contention was incorrect

### Validation Results
- `bash scripts/dev.sh` completed successfully:
  - pytest: 469 passed, 11 pre-existing failures (no new failures)
  - ruff: 1 pre-existing SIM102 error (no new errors)
  - mypy: 62 pre-existing errors (no new errors)
- `pip install -e ".[dev]"` succeeded for dev.sh
- The fix introduces zero new test failures, lint errors, or type-check errors
- The code fix is in the working tree (unstaged) but NOT yet committed

### Incremental Optimization Path Fix (root cause of the persistent cliff)
- The frozen prefix comparison fix in `_compute_live_zone_start` was necessary but not sufficient
- The incremental optimization path (review §4) had a second bug: it used `live_zone_start` (the optimized-output boundary, which includes the summary block) as the raw-message boundary for hash comparison and live zone splitting
- Raw messages do NOT contain the summary block, so using `live_zone_start` as the split point was incorrect — it split the raw messages at the wrong index, causing the hash comparison to include the summary block content (which grows each turn) and always fail after the first fold
- Fix: the incremental path now uses `len(self._last_raw_prefix)` for both the hash comparison (`messages[:len(self._last_raw_prefix)]`) and the live zone split (`messages[len(self._last_raw_prefix):]`), matching the raw frozen-prefix boundary used by `_compute_live_zone_start`
- This ensures the incremental path's raw-to-optimized comparison is consistent with the main path's frozen-prefix-only comparison
- `_last_raw_stable_end` is now dead code (set/reset but never consumed) — can be removed in cleanup

### Dryrun Verification (2026-07-25)
- Ran `python scripts/diag_dryrun_opencode.py --persistent-session --turns 30`
- Result: cliff **NOT fixed** — turns 13-30 are ALL BREAKs
- The code fix in optimizer.py did not resolve the cliff; further investigation needed
- The fix is in the working tree (unstaged) but not yet committed
