# REVIEW.md — Context-Optimization Architecture Review (v0.7.8)

**Reviewer role:** Senior LLM inference architect (vLLM / llama.cpp / OpenAI API / Qwen3.6-35B-A3B-MTP / tree-sitter / embeddings / MoE routing / prefix caching / KV-cache / MTP / agentic coding).

**Scope:** Existing transparent OpenAI-API proxy (`moeptimizer`) that sits between an OpenCode-style client and a Lemonade/llama.cpp backend serving `Qwen3.6-35B-A3B-MTP` (MoE + MTP). Goal: keep the multi-turn agentic context as lean as possible while preserving response quality and avoiding KV-cache refills.

**Inputs reviewed:** `README.md`, `notes.md`, `cache_preservation_guide.md`, `scripts/benchmark_opencode_30_1_0.7.8_baseline.json` (+ the `0.7.7_fix1` baseline), `src/moeptimizer/*.py` (optimizer, compactor, context_compressor, token_aware_truncator, hierarchical_summarizer, tool_output_compressor, context_aligner, app.py).

---

## 0. TL;DR — the headline problem

The latest 30-turn `opencode` benchmark (`benchmark_opencode_30_1_0.7.8_baseline.json`) shows the proxy is **destroying response quality while saving tokens**:

| Metric | Direct | Proxy | Verdict |
|---|---|---|---|
| `token_savings_pct` | — | **94.68%** | excellent on paper |
| `semantic_similarity` (mean) | — | **0.248** (median 0.0175) | **catastrophic** |
| `code_block_ratio` (mean) | — | 0.733 (8/30 turns lose code) | poor |
| `has_code_proxy` (mean) | 0.267 | **0.0** | **proxy emits NO code at all** |
| `fact_recall_turn30` | 0.0 | 0.0 | both fail (anchor not in system prompt — see §6) |
| `context_window_wall` | turn 2 | turn 2 | both hit the wall at turn 2 |
| `contradictions` | 30 | 0 | proxy "agrees" with everything (low-information) |
| `total_proxy_prompt` | — | 42,405 vs 797,584 direct | 30× smaller |

**Interpretation:** The proxy compresses the context so aggressively (and with such lossy transforms) that the model no longer has the code, the file contents, or the prior decisions it needs to produce a useful answer. The 94.68% token saving is real but **worthless** — the proxy is trading correctness for token count. `has_code_proxy = 0.0` is the smoking gun: the model is being asked to "implement a REST API" with no file context and no prior turns, so it produces prose instead of code.

This is the single most important finding. **Before adding any new optimization, the proxy must be re-tuned so that `semantic_similarity` and `code_block_ratio` stay in the 0.85+ range that the regression gate claims to enforce.** The current `balanced` profile is not balanced — it is closer to the old `aggressive` profile in destructiveness.

The rest of this review covers (1) the root causes of the quality collapse, (2) how well the architecture follows the cache-preservation guide, (3) missing optimizations, (4) design weaknesses / bugs / leaks, and (5) a concrete implementation plan.

---

## 1. Root cause of the quality collapse

The pipeline (`optimizer.py::_optimize_messages_locked`) applies, in order, on **every** over-budget turn:

1. `context_compressor.compress` — **skeletonizes code in the newest user message** (Step 5.7).
2. `compactor.compact_messages` — **front-evicts whole turns** (Step 7) *or* folds them into a rolling summary.
3. `hierarchical_summarizer.summarize_turns_cache_stable` — **folds old turns into a constraint-only summary** (Step 8.5).
4. `context_compressor` again via `_optimize_code_block_content` (Step 10) — **skeletonizes code in ALL messages** when over the proactive threshold.
5. `_proactive_trim` + `_sliding_window_trim` + `_trim_to_budget` (Steps 11, 11.8, 12) — **drop whole turns from the top**.

For a 30-turn agentic coding session the context is over budget from turn ~7 onward (see `eviction_triggered_at_turns` = 7..30). So from turn 7 the proxy:

- **Skeletonizes every code block** (Step 10 runs on all messages, not just the newest — contradicting the docstring of `ContextCompressor.compress` which says "only the newest user message"). The tree-sitter skeleton keeps only `def`/`class` signatures + `...` bodies. For a coding task this removes the *implementation* the model must edit/extend.
- **Front-evicts the early turns** that contain the file reads and the user's original task, replacing them with a rolling summary that retains only lines containing "don't"/"must not"/"avoid" keywords (`_CONSTRAINT_HINTS`). Real code, real file contents, and real prior decisions are gone.
- **Drops the middle of history** via sliding-window trim, so the model loses the conversation thread.

Net effect: by turn 30 the proxy sends ~1,400 tokens of (system + first user + 2 frozen turns + a keyword summary + recent turns with skeletonized code). The model has no file to edit and no context, so it answers in prose → `has_code_proxy = 0.0`, `semantic_similarity ≈ 0`.

### 1.1 Specific defects driving the collapse

- **`ContextCompressor.compress` is mis-scoped.** Its docstring says "Compress only the newest user message," but `optimizer.py` Step 10 (`_optimize_code_block_content`) loops over **all** messages when `current_tokens > proactive_threshold_tokens`. So historical code (already in the stable prefix) gets re-skeletonized every turn. This both destroys quality *and* breaks the byte-stability guarantee the cache guide demands (DO #1 / "freeze the structure").
- **The rolling summary is keyword-gated and lossy.** `_extract_constraints` only keeps lines containing a constraint hint, plus a 2-sentence topic. For a debug/refactor task the *actual bug* and *actual code* are almost never in a "don't" sentence, so they are dropped. This is the wrong summarization strategy for agentic coding — you need the *state* (current file contents, current error, current plan), not a list of prohibitions.
- **`compactor.compact_messages` double-folds.** When `hierarchical_summarizer` is present it folds the evictable body into the rolling summary AND the optimizer later calls `summarize_turns_cache_stable` again (Step 8.5). Two different summary mechanisms run on overlapping turns → the evicted content is summarized twice and the protected tail is what survives, but the *summary content* is near-empty (keyword-only), so the net information retained is tiny.
- **Budget is far too small for agentic coding.** `max_optimized_tokens = 3000` (default) with `proactive_trim_ratio = 0.45` → proactive trim at 1,350 tokens. A single file read in the fixture scenario is >4k chars. The proxy starts evicting/skeletonizing almost immediately. The benchmark `char_budget = 12000` (≈3,000 tokens) is the binding cap. For a 262k context window this is absurdly conservative — the whole point of the proxy is to stay *under* the window to avoid prefill, but 3k tokens vs 262k is 1% of the window; you are paying the full quality cost for a prefill saving that does not exist at this scale.
- **`has_code_proxy = 0.0` is also a measurement artifact** (see §9): the benchmark grades `has_code` on the *final assistant response*, and a model with no code context produces no code. But it is still a real signal that the proxy removed the code the model needed.

### 1.2 Fix direction (must-do before anything else)

Re-tune so the proxy preserves **task-critical state** instead of skeletonizing it:

1. **Never skeletonize code that is the current edit target.** Only compress code that is *reference/background* (e.g. library source the model is not editing), and keep the *active file* verbatim. This requires a notion of "active file" — derivable from the most recent `read_file`/`edit` tool calls.
2. **Replace keyword summary with a state summary.** The rolling summary should retain: current file path + current file contents (or a diff), the last error message, and the current plan/goal. Not a list of "don't" sentences.
3. **Raise the budget to a sane fraction of the window** (e.g. 8k–16k tokens for a 262k window) so eviction only triggers when genuinely needed, and gate skeletonization behind a much higher threshold.
4. **Make `balanced` actually balanced**: the current defaults behave like `aggressive`. Re-baseline the three profiles against the regression gate (A≥0.88) on the `opencode` 30-turn scenario.

---

## 2. Conformance to `cache_preservation_guide.md`

The guide is the architectural contract. Score each DO/DONT:

### DOs — mostly followed, two important gaps

- **DO #1 Append volatile data to the last turn only** — ✅ Followed. `Step 14.12` appends the volatile anchor/RAG/loop-warning as a single trailing user turn; historical turns are never mutated. Good.
- **DO #2 Echo every token of the thinking process** — ⚠️ **Partial / risky.** `app.py` *forwards* `reasoning_content` on the way out (streaming path copies `d.reasoning_content` to the client), but on the **request** path the proxy does **not** guarantee the client's next-turn `messages` carry the assistant's `reasoning_content`/`<think>` block back. If the OpenCode client strips `reasoning_content` (common — many clients only persist `content`), the proxy's `optimize_messages` receives an assistant message without the thinking block, so the *next* turn's prefix differs from what the backend cached → forced prefill. **Action:** the proxy should *reconstruct* the assistant message it sends to the backend to include the thinking block it observed (it has it from the prior streaming response), independent of whether the client echoed it. This is the single highest-value cache-stability fix.
- **DO #3 Keep the system prompt immutable** — ✅ `freeze_static_prefix` freezes the system prompt verbatim. Good. (Note the anchor is deliberately *not* in the system prompt — see §6 — which is correct for cache stability but hurts fact recall.)
- **DO #4 Use the structural Chat Completions API** — ✅ The proxy forwards the OpenAI `messages` array; it never touches the raw `/completion` template. Good.
- **DO #5 Freeze the structure/order of available tools** — ⚠️ **Not enforced.** The proxy forwards `tools` from the client but does **not** verify the `tools` array is byte-stable across turns. If the client re-sends tools in a different order (or with different schema), the prefix shifts. **Action:** cache the first-seen `tools` schema per session and re-emit it verbatim (same order, same dict) on every turn, ignoring client reordering. This is a cheap, high-value cache win.
- **DO #6 Slice exclusively from the top when truncating** — ✅ `TokenAwareTruncator` and `_sliding_window_trim` drop whole turns from the front. Good. (But see §3 — the *summary* insertion still shifts offsets; mitigated by trailing placement.)

### DONTs — one real violation

- **DONT #1 Append trailing whitespace randomly** — ✅ `_strip_internal_flags` and the compressor are whitespace-stable. Good.
- **DONT #2 Server-side parameter variations mid-chat** — ⚠️ **Latent risk.** The proxy does not currently change `temperature`/`top_k` per turn, but `OutputShaper` clamps `max_tokens`/`reasoning_effort` per turn-class. If those clamps change the *requested* sampling params turn-to-turn, llama.cpp re-evaluates sampling layers (the guide notes this does not wipe the string cache but does force re-eval). **Action:** keep generation params constant per session; only clamp on the first turn or via a stable policy.
- **DONT #3 Strip hidden object IDs on session reload** — ✅ Session state is serialized/deserialized as-is. Good.
- **DONT #4 Mix text strings and multi-modal/tool block formats** — ✅ The proxy preserves the client's content format (string vs list) per message; it does not convert historical strings to `{"type":"text"}` blocks. Good.
- **DONT #5 Allow background async tasks to interleave into the active context slot** — ⚠️ **Real risk.** `async_io_stage` offloads tree-sitter compression + embedding ranking to a thread pool. If two turns of the *same* session race through the optimizer (the optimizer holds a per-instance `RLock`, so `optimize_messages` is serialized per optimizer — good), but the **embedding service / LanceDB** is shared process-wide and could be hit by a background maintenance task. Also `cache_registry.save_to_disk` and `hierarchical_summarizer.save_to_disk` write to the same files from any session. **Action:** ensure all per-session optimizer state is confined to the session's optimizer instance; only truly global, read-only services (tokenizer) should be shared. Verify no background thread mutates a session's message list.
- **DONT #6 (System-Level Ephemeral Insertion)** — ✅ The summary is appended as a trailing turn (not mid-history), and updates are incremental (append-only). This matches the guide's "fixed summary slot / incremental triggers / let llama.cpp handle eviction" advice. Good — but the *content* of the summary is wrong (§1.1), not its *placement*.

**Verdict:** Cache-stability *placement* is well done. The two highest-value gaps are **DO #2 (reconstruct thinking blocks)** and **DO #5 (freeze tools schema)**.

---

## 3. Missing optimizations (higher ROI than what exists)

1. **Thinking-block reconstruction (DO #2).** As above — reconstruct the assistant `<think>`/`reasoning_content` the backend cached, so the next turn's prefix matches. This alone can recover a large fraction of the "0% real reuse" the notes mention.
2. **Tools-schema pinning (DO #5).** Cache first-seen `tools` and re-emit verbatim. Cheap, high cache win.
3. **Active-file awareness.** Track the most recent `read_file`/`edit`/`write` target per session and keep that file's content verbatim (never skeletonize/evict it). This is the single biggest quality lever for agentic coding and is currently missing.
4. **Delta/semantic diff instead of skeleton for edited files.** When a file is re-read after an edit, send a *unified diff* (or the `DeltaEncoder` output that already exists but is never surfaced to the model) instead of the full new file. The `delta_encoder.py` module exists but its snapshots are never injected — it only stores them. Wire it: replace a re-read full file with `diff(old, new)` when the old snapshot exists.
5. **Per-turn prefix-cache *verification*, not just prediction.** The proxy predicts hit rate via `cache_registry`/`hit_prediction`, but the authoritative signal is the backend's `usage.prompt_tokens_details.cached_tokens` (already captured in `app.py`). Use the *real* cached-token ratio per turn to dynamically raise/lower the eviction aggressiveness: if real reuse is high, you can evict more boldly; if reuse is low (prefix shifted), stop mutating the prefix. This closes the loop the guide implies.
6. **MTP boundary alignment is a no-op placeholder.** `mtp_state.py` is explicitly non-functional (cannot capture real MTP hidden states from an OpenAI client). The only real MTP lever is **native MTP passthrough** (already autodetected) — keep it, drop the dead `mtp_state` bookkeeping, and add **prompt-length padding to the MTP draft multiple** (e.g. pad the final user turn so total prompt tokens is a multiple of the MTP `draft_len`, typically 2–4) so the draft heads align. This is a legitimate, model-visible-free optimization (padding is just trailing whitespace the tokenizer collapses, or a no-op trailing token) that improves draft acceptance. **Flag:** must be validated against the real backend; pad only when native MTP is confirmed.
7. **Context-shift / `--context-shift` delegation.** The guide recommends letting llama.cpp handle eviction. The proxy currently does its own front-eviction *and* the backend may also context-shift, causing double-eviction (proxy drops turn 1, backend also drops turn 1 → the proxy's eviction was wasted work and may have *caused* a prefix shift). **Action:** when the backend exposes `/slots` + context-shift, the proxy should *reduce* its own eviction and let the backend shift, only evicting when the proxy's budget is the tighter bound. Coordinate, don't compete.
8. **Speculative/MTP draft on the *client* side is impossible** — correctly noted as non-functional. Do not revive it.
9. **Prompt caching *hints* for llama.cpp.** When `ENABLE_EXPERIMENTAL_BACKEND_HINTS` is on, send `cache_prompt: true` / slot-pinning `id_slot` so the backend pins the prefix. Already partially present (`slot_pinning_enabled`); extend to also set `cache_prompt` on the stable prefix request. Validate it does not break non-llama.cpp backends (it is sent via `extra_body`, so safe).
10. **Token-budget-aware RAG retrieval.** `StateBasedRAG` retrieves chunks but does not account for the remaining token budget after eviction; it can push the context back over budget. Gate RAG injection by `remaining_budget = max_tokens - current_tokens` and retrieve only what fits.

---

## 4. Design weaknesses

1. **Two overlapping summary mechanisms.** `ScratchpadCompactor` (folds evictable body into rolling summary when `hierarchical_summarizer` present) *and* `optimizer` Step 8.5 (`summarize_turns_cache_stable`) both summarize. They share `self.hierarchical_summarizer._rolling_summary_text` state, so they interact in subtle ways (the compactor feeds turns into the summarizer's state, then Step 8.5 folds *more*). This is fragile and the source of the double-fold in §1.1. **Action:** pick one summary path. Recommend: compactor does pure front-eviction (drop turns), and a *single* cache-stable summary step folds the evicted turns into a *state* summary. Delete the compactor's summarization branch.
2. **`ContextCompressor.compress` scope vs docstring mismatch.** Step 10 calls `_optimize_code_block_content` on all messages, contradicting the module's "newest user message only" contract and breaking prefix stability. **Action:** either rename/redocument, or restrict Step 10 to the live zone only (it already has `live_zone_start` plumbing — use it).
3. **`is_lean_context` is computed once and used as a gate for RAG/summary, but the context can cross the threshold mid-pipeline** (compression/eviction change token counts). The gate is stale by Step 8.5. Minor, but contributes to inconsistent behavior.
4. **`static_prefix_kv` early-exit can skip compaction but the comment admits it must not bypass compaction when over budget** — correctly handled. Good. But the *static prefix KV* stores the *optimized* prefix text, not real KV tensors (correctly documented as a no-op short-circuit). It is fine; just don't mistake it for real cache reuse.
5. **`hit_prediction` model trains on `self._last_static_prefix_hit`** (the proxy's own memo hit), which is a weak label. The app layer *can* override with the backend's real `cached_tokens` via `record_cache_outcome`, but `record_cache_outcome` is only called in the streaming path (app.py line 832) — verify the non-streaming path also calls it (it does at line ~973). OK, but the *default* label when the app hasn't reported yet is the proxy's own hit, which biases training. **Action:** only train `hit_prediction` from the authoritative backend signal; ignore the proxy-memo hit.
6. **`goal_relevance_scorer.prune_by_relevance` evicts low-relevance steps from the *store*, not the messages** — but the store is re-ingested from `messages` every turn (`_ingest_messages`), so pruning the store has **no effect** on the optimized output (the messages are the source of truth). This is a dead optimization. Either prune from `messages` directly or drop it. Currently it burns CPU and claims savings it does not deliver.
7. **`classify_and_template` / `context_template_matcher`** rewrite the prompt with task templates. For an agentic coding client this can *change the user's wording* (the guide's DONT #4 risk) and hurt quality. These are gated behind the proactive threshold (good) but should be **disabled by default for agentic scenarios** — the client's exact wording matters for coding tasks. Recommend defaulting `prompt_template` specialization off unless the scenario is non-agentic.

---

## 5. Potential bugs

1. **`compactor.compact_messages` returns `system_anchor + protected_tail + [summary_block]`** — the summary is placed *after* the protected tail. But `optimizer` Step 8.5 (`summarize_turns_cache_stable`) places the summary as a *trailing* turn after `keep_recent`. So the two summaries can end up in different positions, and the compactor's summary is inside the "protected tail" region while Step 8.5's is after it. The prefix boundary (`frozen_prefix_end`) is computed *before* Step 8.5, so the compactor's summary (if it lands before `frozen_end`) would be frozen into the stable prefix and then *re-summarized* next turn → content drift. **Action:** unify summary placement (trailing only) and compute `frozen_end` after all summary steps.
2. **`_append_volatile_context` dedup check** compares `result[-1].content == content`. If the volatile turn is not the last element (e.g. a summary was appended after it), the dedup fails and a duplicate volatile turn accumulates. The guard `if result and result[-1].get("role")=="user" and result[-1].get("content")==content` is too narrow. **Action:** scan for any trailing `_volatile_turn` and replace it, not just the last element.
3. **`TokenAwareTruncator._partition_for_budget` hardcodes `keep = 3`** instead of reading `self` config (`keep_full_steps`). The optimizer wires `keep_full_steps` into `ScratchpadCompactor` but the truncator ignores it. Inconsistent eviction boundaries between the two stages can drop turns the compactor kept. **Action:** pass `keep_full_steps` into the truncator.
4. **`freeze_static_prefix` (cache-stable mode) freezes `original[:n]` verbatim but `original` is the *optimized* list passed by the caller** — the caller passes `optimized` as both args, so it freezes the optimized early turns. Fine. But it then does `frozen[first_user] = optimized[first_user]` to keep the compressed first user. If `first_user` index in `original` differs from `optimized` (they are the same list here, so OK). Edge case: if a prior step inserted a message before the first user, indices shift. Low risk but worth an assertion.
5. **`seed_token_calibration` / `set_token_calibration` clamp to [0.5, 2.0]** — if the backend tokenizer ratio is genuinely outside this (e.g. Qwen vs tiktoken on code can be ~0.3–3.0), the clamp *distorts* the budget. The benchmark shows `per_turn_proxy` median 1,596 tokens vs `per_turn_direct` median 24,703 — a 15× difference that is *not* calibration (it is real eviction), but the clamp could hide a genuine mis-count on small contexts. **Action:** widen the clamp or make it adaptive (track a rolling median ratio).
6. **`app.py` fallback on optimizer exception** calls `_fallback_optimized_messages(messages, keep_full_steps)` — this returns *recent* turns only, which is correct, but it does **not** strip internal `_` flags, so a failed optimization could leak `_volatile_turn` / `_summary_id` markers to the backend. **Action:** strip internal flags in the fallback too.

---

## 6. Fact-recall / long-horizon signal is broken by design

`long_horizon.fact_recall_turn30` is `0.0` for **both** proxy and direct. The anchor facts are "prepended to Turn 1's user message" (README line 443-449), but the benchmark JSON shows `fact_recall` is `None`/0 because the embedding model grades the *final* response against facts — and the facts were prepended to the *user* message, not the system prompt. The guide explicitly says **do not put volatile data in the system prompt** (DO #3 is about *immutability*, not about injecting facts). The current design puts the anchor in the *user* turn (correct for cache stability) but then the proxy's front-eviction **drops Turn 1** once over budget (Turn 1 is in the evictable body after the frozen prefix of 2 turns). So the facts are evicted by turn ~9 and the model can never recall them. **Action:** pin the fact-anchor facts into the *frozen prefix* (or the rolling summary's "key decisions" block) so they survive eviction. This is a measurement+design fix: the fact-recall probe is only meaningful if the facts are retained by *both* paths; currently neither retains them because the direct path also evicts? No — direct path sends the full history, so direct *should* recall. The `0.0` for direct suggests the benchmark's fact probe is mis-graded (embedding threshold 0.35 too high, or facts not actually injected). **Action:** verify the fixture actually injects the anchor facts; if not, fix the benchmark, not the proxy.

---

## 7. Performance bottlenecks

1. **`optimize_messages` runs on a single worker thread per request** (`_OPTIMIZER_EXECUTOR`) but the pipeline does a *lot* of redundant token counting: `self.token_counter.count_messages(optimized)` is called **~15 times** per turn (Steps 5, 5.5, 5.7, 6, 7, 8, 8.5, 10, 10.5, 11, 11.8, 12, …). Each call re-tokenizes the *entire* message list with tiktoken (or the HF Qwen tokenizer if local). For a 30-turn context this is the dominant CPU cost and the reason `notes.md` reported +68% latency on `refactor_long`. **Action:** count once at the top, cache the count, and only recompute after a stage that *mutates* content (compression, eviction, summary). Pass the cached count into each gated stage.
2. **`async_io` offloads compression + embedding, but the *token counting* and *partitioning* (the hot loops) run on the event-loop-adjacent executor synchronously.** Move the count-cache above to avoid this.
3. **`cache_registry.predict_hit_rate` + `hit_prediction.should_early_exit`** run ML inference (XGBoost) every turn even when the context is over budget (they are gated by `total_tokens <= proactive_threshold_tokens` — good — but the *call* still happens before the gate in Step 5.1/5.1.5). Cheap to skip entirely when over budget.
4. **`static_prefix_kv.get` does a content hash + dict lookup every turn** — fine, but `put` re-pickles the whole prefix. Gated by change-detection (good).
5. **LanceDB embedding queries** are the other big cost. `StateBasedRAG` + `code_chunking` embed on every over-budget turn. The `async_io` offload helps, but embeddings are also cached (`embed_cache_max=512`) — for a 30-turn session the cache is tiny and thrashes. **Action:** increase embed cache or make it LRU with a larger cap; batch embed (already has `embedding_batch_size`).

---

## 8. Memory leaks / unbounded growth

1. **`AgentStateStore` is bounded** (`MAX_STATE_STEPS=200`, `_prune_if_needed`) — good.
2. **`cache_registry` on-disk size** — `notes.md` flagged this as "verify bounded." `register_context` keys by exact context hash, which changes every turn, so the registry grows unbounded on disk across a long session unless `_register_save_counter % save_every` throttles writes (it does) — but the *in-memory* dict still grows every turn (new key per context). **Action:** cap `cache_registry` in-memory size (LRU) and persist only the hit-rate model, not every context.
3. **`chunk_fingerprint` cache** (`max_entries=2048`) — bounded, good.
4. **`delta_encoder` snapshots** (`max_snapshots=100`) — bounded, good.
5. **`hierarchical_summarizer._summaries`** — bounded to 100, good. But `_rolling_summary_text` grows **unbounded** across a session (append-only by design). For a 200-turn session this is a large string re-built every turn. **Action:** cap `_rolling_summary_text` length and prune oldest constraints when over a char budget (already caps constraint *count* to 5 in the anchor, but the summary text itself is uncapped).
6. **`self._tool_output_cache`** (`max=1024`) — bounded, good.
7. **`optimizer._last_stable_prefix` / `_stable_prefix_optimized`** hold full message lists — bounded to one session's prefix, fine, but they are per-optimizer-instance and the optimizer is per-session, so they are freed on session expiry. Verify `SessionManager` actually drops optimizers on `MAX_SESSIONS` LRU eviction (it should `del` them). **Action:** confirm `session_manager` eviction calls `optimizer` cleanup / drops the reference so the GC reclaims the stored message lists.

---

## 9. Benchmark / measurement issues (affect the review itself)

1. **`has_code_proxy = 0.0`** is partly a *consequence* of the proxy removing code context (real bug) but also means the quality gate's `code_block_ratio` is the better signal — and it is 0.733, not 0. The benchmark correctly separates them; just don't over-read `has_code`.
2. **`semantic_similarity` uses `embed-gemma-300m-FLM`**, which the README itself says is "weak on code." A mean of 0.248 with median 0.0175 means the *distribution* is bimodal: some turns near 1.0, many near 0. The median (0.0175) is the more honest number and it is terrible. **Action:** add a *code-aware* similarity (e.g. AST diff or a code embedding) as the headline gate metric, not just lexical/embed similarity. The current headline block already has `code_block_ratio` + `code_syntax_validity` — good — but the regression gate (`--min-similarity`) keys off `semantic_similarity`, which is the wrong metric to gate on for code. **Action:** gate on `code_block_ratio` + `rouge_l_f1` + `edit_similarity` composite, not raw embedding cosine.
3. **`rounds=1` in the `0.7.8` baseline** — the README says rounds default to 3 for variance. A single round is noisy. Re-run with `--rounds 3` before declaring the collapse real (though 94.68% savings + 0.0 code is unambiguous regardless).
4. **The `0.7.7_fix1` baseline is missing its quality block in the snippet** — compare against it to confirm the collapse is a *regression* from 0.7.7, not a long-standing issue. (The notes show 0.7.7-era 30-turn runs had `semantic_similarity 0.92-0.98` on 6-turn scenarios, but those were 6 turns. The 30-turn `opencode` is the first real long agentic run.) **Action:** diff `0.7.7_fix1` vs `0.7.8` on the same 30-turn `opencode` scenario to isolate what changed.

---

## 10. New UX / operability features

1. **Per-session "what was dropped" header.** The proxy already emits `X-MOEPT-Optimization-Degraded` and an eviction SSE comment. Add `X-MOEPT-Evicted-Turns: <n>` and `X-MOEPT-Summary-Chars: <n>` so the client/operator can see how much context was lost. Already partially present (`_last_evicted_turns`); surface it.
2. **Dry-run diff endpoint** — already exists (`X-MOEPT-Dry-Run`). Extend it to return a token-level *diff* (which turns dropped, which code blocks skeletonized) as JSON, not just the messages. High value for tuning.
3. **Live "context budget" gauge** in `/v1/metrics` — show `optimized_tokens / max_tokens` per session so operators can see when the proxy is thrashing (evicting every turn).
4. **Quality-profile A/B** — let the proxy run two profiles (e.g. `balanced` vs `quality`) on alternate sessions and report which saved more tokens at equal quality. Useful for tuning.
5. **`/v1/agent/sessions/{id}/debug`** already exists (good). Add a "replay last turn optimized vs raw" so an operator can see exactly what the model received.
6. **Config sanity-check should WARN on `max_optimized_tokens` being unrealistically small for the context window** (e.g. <5% of `context_window`). The current `config_check.py` does not catch this, and it is the root cause of the over-aggressive eviction.

---

## 11. Implementation plan (priority order)

### P0 — Stop the quality collapse (do first, block release)
- [x] **P0.1** Raise `max_optimized_tokens` default to a sane value (e.g. `8192`) and `max_optimized_chars` to `32000`; document that the budget should be a *fraction* of the context window, not 1%. Re-baseline `balanced` so 30-turn `opencode` hits `code_block_ratio ≥ 0.9` and `semantic_similarity ≥ 0.85`.
- [x] **P0.2** Restrict `ContextCompressor` / Step 10 to the **live zone only** (use `live_zone_start`); never re-skeletonize the stable prefix. Fix the docstring/scope mismatch.
- [x] **P0.3** Replace the keyword-only rolling summary (`_extract_constraints`) with a **state summary**: retain current file path + current file contents (or diff), last error, current plan. Keep it append-only and trailing.
- [x] **P0.4** Delete the `ScratchpadCompactor` summarization branch; let the compactor do *pure front-eviction* and a single summary step fold evicted turns. Unify summary placement (trailing) and compute `frozen_prefix_end` *after* summary steps.
- [x] **P0.5** Add **active-file tracking**: keep the most-recent `read_file`/`edit` target verbatim; never skeletonize/evict it.
- [x] **P0.6** Gate the regression on `code_block_ratio` + `rouge_l_f1` + `edit_similarity` composite, not raw `semantic_similarity`. Re-run `opencode` 30-turn with `--rounds 3` and confirm the gate passes.

### P1 — Cache-stability wins (DO #2, DO #5)
- [x] **P1.1** Reconstruct assistant `reasoning_content`/`<think>` blocks on the request path from the proxy's memory of the prior streaming response, so the next turn's prefix matches the backend's cache regardless of whether the client echoed thinking. (Highest-value cache fix.)
- [x] **P1.2** Pin the `tools` schema per session: cache first-seen `tools`, re-emit verbatim (same order/dict) every turn.
- [x] **P1.3** Use the *real* backend `cached_tokens` ratio per turn to adapt eviction aggressiveness (close the loop; stop training `hit_prediction` on the proxy-memo hit).
- [ ] **P1.4** When backend exposes `/slots` + context-shift, *reduce* proxy eviction and let the backend shift (avoid double-eviction / prefix shift).

### P2 — Efficiency / correctness
- [x] **P2.1** Count tokens **once** per turn; cache and pass the count; only recompute after mutating stages. (Biggest TTFT win.)
- [x] **P2.2** Wire `DeltaEncoder` to inject `diff(old, new)` for re-read files instead of full re-reads.
- [x] **P2.3** Fix `TokenAwareTruncator` to use `keep_full_steps` (not hardcoded 3); fix `_append_volatile_context` dedup to scan for any trailing volatile turn; strip internal flags in the optimizer-exception fallback.
- [x] **P2.4** Remove dead `goal_relevance_scorer.prune_by_relevance` (prunes the store, not messages) or make it prune messages. Default `prompt_template` specialization off for agentic scenarios.
- [x] **P2.5** Bound `cache_registry` in-memory size (LRU); cap `_rolling_summary_text` length.

### P3 — MTP / speculative (validate before enabling)
- [ ] **P3.1** Keep native MTP passthrough (autodetected). Drop dead `mtp_state` bookkeeping.
- [ ] **P3.2** Add optional prompt-length padding to the MTP draft multiple when native MTP confirmed; validate draft-acceptance improvement on the benchmark before defaulting on.
- [ ] **P3.3** Send `cache_prompt: true` (llama.cpp) on the stable-prefix request via `extra_body` when experimental hints on.

### P4 — UX / ops
- [ ] **P4.1** Surface `X-MOEPT-Evicted-Turns` + `X-MOEPT-Summary-Chars`; extend dry-run to a JSON diff.
- [ ] **P4.2** Add context-budget gauge to `/v1/metrics`.
- [ ] **P4.3** `config_check.py` WARN when `max_optimized_tokens` < 5% of `context_window`.

---

## 12. Summary judgment

The architecture is **well-structured for cache stability** (trailing volatile turns, frozen prefix, top-only eviction, live-zone incremental optimization, native MTP passthrough) and follows most of the guide's DOs. The **fatal flaw is the optimization *aggressiveness* and *lossiness***: the current `balanced` profile skeletonizes all code, evicts the task context, and replaces it with a near-empty keyword summary, producing a 94.68% token saving that is worthless because the model can no longer do the task (`has_code_proxy = 0.0`, `semantic_similarity = 0.248`).

**Fix the quality collapse (P0) before shipping or adding any new feature.** The two cache-stability gaps (reconstruct thinking blocks, pin tools schema) are the highest-ROI follow-ups. The rest is efficiency and polish.

**Do NOT** enable `attention_sinks`, `reasoning_preseed`, `static_layer_alignment`, or `mtp_boundary_alignment` (all already default-off and correctly flagged as WARN-level config issues in `config_check.py`) — they violate the guide and add no value for this backend.
