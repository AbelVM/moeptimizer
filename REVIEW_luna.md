# MOE-ptimizer Senior Architecture Review

## Executive verdict

The project has a strong and unusually explicit cache-preservation goal, and several important fixes are already in place: bounded async work, MTP capability probing, calibrated token accounting, error-aware tool compression, persistent degradation counters, and cache-stability dry-run coverage.

The latest live validation is nevertheless a **failed release candidate**. It reports 36.02% token savings, 31,458 final proxy prompt tokens, a cache collapse at turn 12 with intermittent later resets, repeated `diff` parser failures, and oversized session-state headers. The result is not explained by token savings alone: fresh prefill tokens correlate strongly with TTFT (`r=0.9817` in the captured run), so backend cache loss is the direct performance risk.

A subsequent persistent-session dry-run passed: 30/30 turns were append-only, with no proxy-side breaks and approximately 4.4K -> 13.5K optimized tokens. This is useful but not exculpatory. The dry-run proves local serialized-prefix stability only; it does not prove that Lemonade/llama.cpp retains the corresponding KV cache. Do not claim the regression is fixed until backend cache telemetry and repeated benchmark rounds agree.

**Release recommendation: hold.** First fix observability and the HTTP diagnostic surface, then isolate backend cache behavior, then optimize context geometry. Keep quality-preserving compression enabled only when its first-appearance and backend-cache behavior are demonstrated.

## Scope and evidence

Target: Python 3.11+, FastAPI/uvicorn, OpenAI-compatible proxy, Lemonade/llama.cpp backend, `Qwen3.6-35B-A3B-MTP-GGUF`.

Governing invariant: the proxy may compact input context, but must preserve model quality and byte-stable cacheable prefixes. The backend controls response length; the proxy must not shape response verbosity.

Evidence reviewed:

- `scripts/benchmark_opencode_30_1_0.7.27_validate.raw.txt`: latest failed validation.
- `scripts/benchmark_opencode_30_1_0.7.27_phaseA.json` and `scripts/benchmark_opencode_30_1_0.7.27_chunkfix.json`: single-round comparison artifacts.
- `scripts/diag_dryrun_opencode.py`: local prefix-stability gate.
- `src/moeptimizer/optimizer.py`, `hierarchical_summarizer.py`, `compactor.py`, `code_chunking.py`, `app.py`, and `backend_client.py`.
- Existing tests and the project guidance in `AGENTS.md`.

The live artifacts are single-round measurements. Their TTFT and latency confidence intervals are wide, so they establish a regression signal, not a stable performance estimate. The cache cliff, parser failure, and header overflow are deterministic engineering findings and should be treated separately from noisy latency means.

## Measured status

| Signal | Latest validation | Interpretation |
|---|---:|---|
| Token savings | 36.02% | Below the reported 54.10% baseline/gate target |
| Final proxy context | 31,458 tokens | Context continues growing despite a 262K window |
| Cached tokens | Mean 14,040 | Not enough to explain the intended TTFT advantage |
| Proxy TTFT | 19.15 s mean | Worse than direct in the captured run |
| Cache behavior | Live 12-turn run: proxy cached tokens rose from 0 to 10,358 with no backend errors or cliff | This run did not reproduce the prior cache cliff; backend slot events remain unobservable |
| `diff` parser | Repeated lookup failures | Code/delta optimization silently degrades |
| Session-state header | Up to about 273 KB | Unsafe HTTP diagnostic design |
| Dry-run gate | 0 breaks / 30 turns | Latest persistent local prefix dry-run is append-only; average turn time was 173 ms; the run emitted no backend cache telemetry |
| Quality | Live 12-turn run: syntax validity 1.0, but 10/12 low semantic-similarity turns, 11/12 low-Jaccard turns, and 1 code-block-loss turn | Release gate still fails despite valid syntax and high prompt-faithfulness/evicted-content recall |

The Phase-A and chunk-fix artifacts also show that one-turn latency comparisons are noisy. Use multiple rounds, randomized or interleaved only where the benchmark invariant permits, and report confidence intervals plus per-turn cache state.

## Findings

### P0: Backend KV-cache reuse is not observable enough to debug

The proxy reports a local cache key and estimated hit, while the backend reports the authoritative cached prefix. These are different measurements. A local key can remain present while llama.cpp changes slot, evicts the KV cache, rejects a prompt layout, or receives a different serialized request.

Required instrumentation per request:

- session id and backend `id_slot`;
- exact tokenized prompt length;
- backend `prompt_eval_count` / cached-prefix count, if available;
- local common-prefix token count and first differing token index;
- model, tokenizer identity, context size, MTP/speculative settings, and backend restart/slot events;
- whether the request was a dry run, cache hit, refill, or error.

Store this as bounded structured metrics, not a giant response header. Add a diagnostic endpoint or JSONL trace explicitly intended for local benchmarking.

**Acceptance:** repeated 30-turn runs show backend cached tokens equal to or above the local common-prefix estimate except for documented backend quantization/slot behavior. Any divergence is visible at the first affected turn.

### P0: Diagnostic data is transported in unbounded HTTP response headers

`app.py` builds base64 JSON and full optimized prompt text for response headers. Header-safe encoding prevents invalid bytes, but it does not solve size limits. A 273 KB state value can be rejected by proxies, clients, or the server and can make diagnostics perturb the request path.

Replace full-state headers with:

1. a short request/debug id;
2. bounded scalar headers only, such as optimized tokens, cached estimate, and degradation count;
3. a debug endpoint or bounded file/JSONL trace keyed by request id;
4. explicit opt-in for full prompt dumps, with size limits and redaction.

Never place the full session state or optimized prompt in a normal OpenAI-compatible response. This is both an operational reliability issue and a measurement-contamination risk.

### P1: The live regression is not explained by the current dry-run result

The latest dry-run is green and append-only, while the live benchmark previously showed cache resets. This means at least one of the following differs: backend state, slot assignment, environment/configuration, prompt serialization at the backend boundary, tokenizer accounting, benchmark session lifecycle, or backend cache policy.

Do not merge a context rewrite based only on dry-run results. Add a paired trace that records the exact outbound request body hash and token prefix alongside backend usage for every turn. Compare turn 11/12 first.

The dry-run gate should remain a prerequisite, but it must be named accurately: `proxy_prefix_stability`, not `backend_cache_stability`.

### P1: Summary-in-the-middle remains a structural cache hazard

`HierarchicalSummarizer` places the rolling summary immediately after the frozen prefix. When it changes, every later live message shifts. The current batch and space-based controls reduce the frequency of this event, but they cannot make a fold free.

For window-bound sessions, redesign the layout so the mutable summary is outside the long-lived prefix, or use two explicit regions:

- immutable system/early-task prefix;
- append-only live transcript;
- bounded historical state represented by a stable handle or a separately versioned summary region.

A fold may still require a refill, but it should not rewrite unrelated cached bytes. Test the geometry at near-window occupancy, not only the 4% utilization opencode case.

### P1: Context budget and fold policy are separate governors

The budget governor, rolling summary, scratchpad compactor, proactive trim, and sliding-window trim can each make independent decisions. This creates policy conflicts: one stage measures a calibrated token budget, another uses a structural turn count, and another can front-evict messages after the summary stage.

Define one `PrefixLayout`/budget decision for each turn:

- protected prefix tokens;
- mutable summary tokens;
- live-zone tokens;
- evictable tokens;
- target, hard ceiling, and hysteresis.

Stages should consume the decision rather than independently deciding whether to evict. Keep the refactor incremental and preserve the existing dry-run regression tests.

### P1: Code optimization has a silent-failure path

Tree-sitter failures and parser lookup failures have historically been swallowed behind broad exception handling. The AST API mismatch was especially dangerous because the advertised AST path silently fell back to line chunking.

The tree-sitter 0.25 property API is now corrected, but the general failure mode remains. Every optional optimizer stage should emit:

- a stage failure counter;
- a short reason code;
- input size and language;
- whether the stage failed open or changed behavior.

Avoid logging full source or prompts. At warning level, emit an aggregate transition, not one log line per tool output.

`diff` is particularly important: if its parser is unavailable, code delta compression should explicitly report `parser_unavailable` and use a tested, bounded fallback. Never imply that a delta was produced when the full file was retained.

### P1: Quality metrics are too easy to overread

Syntax validity and lexical overlap do not prove agent success. The current artifacts show high syntax validity but low semantic-similarity/token-Jaccard counts and code-block-loss turns. The proxy can return valid code that is incomplete or solves the wrong subtask.

Add task-level checks:

- test/lint/compiler exit status and failing diagnostics;
- required file/function/symbol presence;
- patch application success;
- exact constraint retention for user-provided constants;
- tool-call validity and argument equivalence;
- answer-groundedness against the final task, not only the direct response.

Gate on worst-case and per-task metrics, not only means. A mean quality improvement must not hide a small number of catastrophic code omissions.

### P1: Tool output compression must be monotonic from first appearance

A tool output that first enters the backend prompt in full and is compressed on a later turn necessarily changes a previously cacheable prefix. The current cache memoization and error-aware compressor are good foundations, but the invariant must be explicit: normalize/compress before the message is first emitted to the backend.

Use tool-specific policies:

- preserve compiler/test failures, stack frames, file:line locations, and final summaries;
- aggressively remove repetitive progress and passing-test noise;
- preserve raw output behind a bounded content-addressed handle;
- keep the placeholder deterministic and stable;
- never compress a summary block again.

Measure per-tool input/output chars and tokens, retained diagnostic lines, and quality outcomes.

### P2: Reversible handles are promising but change the interaction contract

Content-addressed storage and an `expand_content` tool can recover information lost by compression, but they add tool-schema tokens, continuation latency, permissions, lifecycle, and failure modes. They must remain gated until tested through both streaming and non-streaming paths.

Required tests: missing handle, expired handle, cross-session access denial, bounded expansion, tool-call buffering, client-visible stream correctness, and backend refusal to call the tool.

### P2: MTP support is incomplete without backend throughput evidence

Capability probing and passthrough are implemented, but accepting an extra body field does not prove that MTP is active or beneficial. Record accepted/rejected settings, draft acceptance rate, speculative tokens accepted, decode TPS, and fallback rate. Compare MTP on/off with identical prompts and cache state.

Do not tune response `max_tokens` or output verbosity in the proxy to make MTP look better. The proxy mission is input compaction only.

### P2: Session state needs bounded lifecycle and stable handles

The session manager and reaper address memory lifecycle, but state transport should use a short session id and server-side bounded storage. Enforce caps on messages, summaries, tool handles, debug traces, and per-session counters. Return 413 or a clear degradation status when a requested dump exceeds limits; do not silently truncate diagnostic JSON in a way that looks complete.

### P2: Executor and cancellation behavior can cap throughput

The bounded CPU/I/O stages and dedicated embedding loop are sound, but a small shared executor plus long timeouts can serialize unrelated sessions. Instrument queue wait, active workers, stage duration, cancellation latency, and timeout count. Propagate cancellation through CPU-bound boundaries where possible. Scale only after queue wait is measured; do not add workers blindly on hardware-limited deployments.

## Missing optimizations, ranked

1. Backend-authoritative per-turn KV/prefix telemetry and request fingerprinting.
2. First-appearance, tool-specific normalization with deterministic content handles.
3. Task-aware code selection using the current subtask and dependency closure, not only the original request.
4. Cache-safe summary geometry for genuinely window-bound conversations.
5. One budget/layout governor replacing competing trim paths.
6. Token counting once per stage with tokenizer identity and calibrated-count diagnostics.
7. Backend MTP acceptance-rate and decode-throughput telemetry.
8. Multi-file replay fixtures with executable quality checks.
9. Bounded debug traces instead of prompt-sized headers.
10. Repeated-round benchmark statistics with cache warm/cold labeling.

Deferred or lower priority until evidence supports them: zero-copy SSE, large god-object extraction, native context-shift, speculative deduplication, and broad reversible-compression rollout.

## Benchmark protocol

### Gate 0: environment and dry run

- Start a clean proxy and record commit, Python/dependency versions, model id, tokenizer id, backend version, context size, and all `MOEPT_*` settings.
- Verify backend health and empty/restarted KV state.
- Run `python scripts/diag_dryrun_opencode.py --persistent-session --turns 30 --max-breaks 0`.
- Save the JSON trace, not only console output.
- Fail on any proxy break, stage failure, parser failure, or unexpected message mutation.

### Gate 1: backend cache equivalence

- Run a short 12-turn fixture with deterministic requests and no response generation, if backend supports it.
- Compare local token common-prefix and backend cached-prefix count per turn.
- Repeat with the same session and with a new slot/session.
- Run the turn-11/12 prompt pair with raw outbound request dumps and hashes.

### Gate 2: repeated quality/performance benchmark

- At least 5 rounds for each arm: direct, proxy baseline, proxy candidate.
- Preserve the required execution order: complete proxy conversation, then complete direct conversation, each contiguous and sorted.
- Record per-turn prompt tokens, cached tokens, fresh prefill tokens, TTFT, decode TPS, total latency, errors, and output tokens.
- Report median, p90, confidence interval, paired deltas, and worst-turn values.
- Run cold-backend and warm-backend conditions separately.

### Gate 3: quality

- Require no statistically meaningful regression in task success, patch validity, tool-call validity, required-symbol retention, or executable checks.
- Require token savings improvement only after quality and cache gates pass.
- Treat any cache cliff, oversized-header error, parser failure, or silent fallback as a failed candidate even if the mean TTFT improves.

## Phased implementation plan

### Phase 0: instrumentation and operational safety

1. Replace full prompt/session headers with request ids and bounded scalar headers.
2. Add bounded JSONL/debug storage keyed by request id.
3. Add backend cache fields, slot id, request fingerprint, tokenizer, and MTP telemetry.
4. Rename the dry-run gate to distinguish local prefix stability from backend cache reuse. **Done:** the tool now labels this as a local prefix dry run; backend KV reuse remains third-party-authoritative.
5. Make optional-stage failures visible through metrics and reason codes.

Gate: dry run green; no large headers; turn-12 trace identifies whether the divergence is local or backend-side.

### Phase 1: monotonic context transforms

1. Normalize tool output before first backend emission.
2. Keep deterministic per-tool budgets and failure-preserving filters.
3. Validate content-store handles and session isolation.
4. Add current-subtask code relevance and dependency-closed slicing, fail-open when uncertain.

Gate: no new local prefix breaks, no task-quality regression, and backend cache does not fall below the local prefix estimate without an explained backend event.

### Phase 2: layout and budget consolidation

1. Introduce a typed `PrefixLayout`.
2. Make one governor own protected, summary, live, and evictable regions.
3. Disable competing front-evictors when cache-stable folding owns sizing.
4. Test near-window occupancy and multi-file code-heavy sessions.

Gate: cache-safe folds at window pressure or an explicit documented refill budget; bounded context; quality retained.

### Phase 3: MTP and throughput tuning

1. Measure draft acceptance and decode TPS before changing MTP settings.
2. Profile token-counting and executor queue wait.
3. Consider zero-copy SSE only for requests proven not to require body inspection.

Gate: lower fresh-prefill and TTFT at equal quality, with no increase in p95/p99 failure behavior.

## Implementation plan

Status icons: ✅ done and validated; 🟡 partial, gated, or blocked on wiring/evidence;
⬜ not started or deferred; ➖ resolved by deletion; 🆕 new finding; ❌ rejected / won't do.

| Priority | Risk | ROI | Effort | Task / description | Implementation status |
|---|---|---|---|---|---|
| P0 | High | Very high | S | Backend-authoritative KV/cache telemetry and local prefix comparison | 🟡 Proxy-owned portion complete: bounded `/v1/debug/requests` records request id, optimized-prompt hash, slot, prompt/cache/fresh-prefill tokens, TTFT, and latency; benchmark `TurnMetrics` now retains request id, prompt hash, and slot per proxy turn; Lemonade slot/cache events remain unavailable |
| P0 | High | High | S | Bounded diagnostic transport and state handles | ✅ Prompt diagnostics are opt-in and capped at 8 KB with an explicit limit header; session state is served through bounded endpoint-backed APIs rather than response headers |
| P0 | High | High | S | Bounded diagnostic headers and opt-in prompt diagnostics | ✅ Implemented: normal responses omit prompt/session dumps; benchmark diagnostics are explicit and capped at 8 KB |
| P1 | High | High | S | Process-wide degradation counts and per-turn failure reason codes | ✅ Implemented and tested: `/v1/metrics` aggregates newly observed optimizer stage failures once per completed turn |
| P1 | High | Very high | M | Backend cache divergence trace and repeated validation | 🟡 Bounded proxy trace and benchmark-visible request fingerprints are implemented and tested; repeated cold/warm runs and backend-vs-local first-difference confirmation remain operational evidence tasks, blocked by unavailable backend cache events |
| P1 | High | High | M | Monotonic, first-appearance tool-output normalization | ✅ Implemented and tested: boundary transforms compress new tool output before it enters the stable prefix |
| P1 | High | High | L | Unified budget and prefix-layout decisions | 🟡 Partial, intentionally paused: `BudgetGovernor` and `PrefixLayout` cover unit selection, remaining-budget arithmetic, and shared immutable-zone measurement; the near-window prefix experiment did not produce a safe fix, so the broader competing-trim refactor remains deferred |
| P1 | High | High | L | Cache-safe summary geometry for window-bound sessions | ❌ No safe local implementation is justified without a reproducible backend/cache regression and quality fixture |
| P2 | Medium | High | M | Task-aware code slicing and dependency-closed selection | ✅ Implemented and tested: query-aware slicing retains direct and transitive top-level helpers while failing open when uncertain |
| P2 | Medium | High | M | MTP acceptance, fallback, and decode-throughput telemetry | 🟡 Proxy now records optional accepted/draft/fallback/decode fields in bounded request traces and `/v1/metrics`; Lemonade currently omits those fields, so authoritative rates remain unavailable |
| P2 | Medium | Medium | M | Executor queue, cancellation, and token-counting profiling | 🟡 Profiling implemented for optimizer/token-count timing, queue depth/wait, backend errors, and degradation events; stop here until measured queue pressure justifies further work, since forcibly stopping running Python work is unsupported |
| P2 | Medium | Medium | M | Multi-file agentic replay fixtures with executable quality gates | ✅ Local gate implemented and tested: required files, task constraints, language fences, fixture tests, CLI smoke, and quality wiring are covered; defer live backend replay assertions |
| P1 | High | High | S | Worst-case quality gates and dashboard visibility | ✅ Local benchmark/dashboard gate implemented: per-metric minima, worst headline quality floor, fresh-prefill-by-turn chart, bounded local prompt fingerprints, and near-window prefix regression are covered; defer live task-success evidence until a backend replay is available |
| P3 | Medium | Medium | M | Reversible compression handles through repeated streaming benchmarks | 🟡 Handle storage, deterministic placeholders, endpoint retrieval, repeated expansion, eviction, and streaming/non-streaming continuation tests are covered; backend-refusal coverage and repeated live benchmarks remain unavailable without a runnable backend fixture; keep the feature opt-in |
| P3 | Medium | Low | M | Zero-copy SSE where body inspection is unnecessary | ❌ The proxy currently needs request-body inspection for optimization, and no measured no-inspection request path exists |

## Test matrix

| Area | Required coverage |
|---|---|
| Prefix layout | append-only turn, fold, tool normalization, summary growth, near-window pressure |
| Headers/debug | large prompt, Unicode, proxy limits, opt-in full dump, redaction, bounded response |
| Parsers | tree-sitter available/unavailable, `diff` missing, malformed code, fallback reason code |
| Tool compression | success log, compiler failure, stack trace, repeated output, first appearance |
| Handles | retrieve, missing, expired, wrong session, size limit, streaming tool call |
| Sessions | concurrent sessions, reaper, state cap, cancellation, executor saturation |
| MTP | supported, rejected, timeout, accepted draft, fallback, decode TPS |
| Quality | multi-file patch, syntax, tests, lint, required symbols, task constraints |
| Benchmark | 5+ rounds, cold/warm backend, direct/proxy ordering, per-turn prompt/cache/TTFT metrics, and bounded proxy request fingerprints |

## Final decision

Local observability, bounded diagnostics, request fingerprinting, optional MTP telemetry, first-appearance normalization, reversible-handle continuation tests, task-aware slicing, fixture quality checks, and worst-case dashboard reporting are implemented and validated. The review is complete for proxy-owned work. Release remains held for external evidence: the quality gate still fails, Lemonade omits authoritative MTP/slot/cache-event fields, and repeated cold/warm backend rounds are still required. Unified near-window layout, cache-safe summary geometry, backend-refusal replay, and zero-copy SSE remain intentionally deferred because this workload does not exercise those paths.
