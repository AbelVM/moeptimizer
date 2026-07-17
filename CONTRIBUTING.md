# Contributing to MOE-ptimizer

Thanks for your interest in improving MOE-ptimizer! This document covers how to set
up a development environment, the conventions we follow, the architectural boundaries
you must respect, and how to get changes merged.

## What MOE-ptimizer is

MOE-ptimizer is a transparent OpenAI-compatible API proxy that optimizes context for
MoE + MTP models in multi-turn agentic tasks. It sits between an OpenAI-SDK client and
a backend (Lemonade server) and **compacts the input context** sent to the backend
while keeping byte-stable prefixes so the backend's native prefix cache is reused.

It is *not* a model server and does not generate responses — it only transforms the
request context on the way in and passes the response straight through.

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- A running backend (Lemonade server) on `:13305` for end-to-end testing
- `pip` (the project uses `hatchling` for builds)

### Setup

```bash
# Clone and enter the repo
git clone <your-fork>
cd moeptimizer

# Create a virtualenv (recommended)
python -m venv .venv && source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Configure the backend
cp .env.example .env
# edit .env: MOEPT_SERVER__URL, MOEPT_SERVER__LLM_MODEL, etc.

# Start the proxy
bash scripts/run.sh
```

### Verify the install

```bash
curl http://127.0.0.1:8080/v1/health
python -m moeptimizer --check-config
```

## Development workflow

1. **Branch** off `main` with a descriptive name (`feat/...`, `fix/...`, `chore/...`).
2. **Make changes** following the conventions below.
3. **Run the full check suite** before opening a PR:

   ```bash
   bash scripts/dev.sh
   ```

   This runs `pytest`, `ruff`, and `mypy --strict`. All three must pass.
4. **Validate config** if you touched `config.py` or `.env.example`:

   ```bash
   python -m moeptimizer --check-config
   ```

   The CLI exits non-zero on any `ERROR`-level issue (e.g. a non-positive token
   budget) so it can gate CI / deploy. `WARN` issues are prefix-cache killers
   (e.g. `ATTENTION_SINKS_ENABLED`, `REASONING_PRESEED_ENABLED`); `INFO` issues are
   legacy aliases and non-functional subsystems.
5. **Update docs**: add user-visible changes to `CHANGELOG.md`; update the `README.md`
   config tables if you added/changed a setting. Keep `README.md` concise and link to
   the changelog rather than duplicating history.
6. **Open a PR** following the pull-request rules below.

## Pull request rules

All PRs must satisfy these rules before they are reviewed or merged. PRs that violate
the hard rules below may be closed without review.

### Hard rules (non-negotiable)

- **No fully AI-generated PRs.** A PR authored entirely by an AI agent with no human
  oversight or verification will be automatically discarded. Every PR must be reviewed
  and owned by a human who understands the change and stands behind it.
- **One issue per PR.** The scope of a PR must be limited to a single issue or a single
  logical change. Do not tackle several unrelated issues in the same PR — split them
  into separate PRs so each can be reviewed, tested, and reverted independently.
- **No huge PRs.** Keep PRs small and reviewable. If a change is large, break it into a
  sequence of focused PRs (e.g. one for the interface, one for the implementation, one
  for tests/docs) rather than dumping everything at once.
- **Pair the PR with an issue.** A PR is much easier to review when it is coupled with
  an issue that describes the problem, the motivation, and the expected outcome. Open
  an issue first and link it (`Fixes #123` / `Related to #456`); drive-by PRs without a
  described problem are discouraged.

### Before you open a PR

- **Branch from `main`** with a descriptive name: `feat/...`, `fix/...`,
  `chore/...`, `docs/...`, `refactor/...`, `test/...`.
- **Pass the full check suite locally**: `bash scripts/dev.sh` (pytest + ruff +
  `mypy --strict`). All three must be green.
- **Validate config** if you touched `config.py` or `.env.example`:
  `python -m moeptimizer --check-config` must exit 0 (no `ERROR`-level issues).
- **Add or update tests** for any behavior change. New pipeline stages require a
  matching `tests/test_<module>.py`. Bug fixes should include a regression test.
- **Update docs**: user-visible changes go in `CHANGELOG.md`; config additions/changes
  update the `README.md` config tables and `.env.example`. Keep `README.md` concise.
- **Do not reformat unrelated code.** Limit diffs to the change at hand; style-only
  churn in untouched files belongs in its own PR.

### PR description

The PR body must include:

- **What** changed and **why** (motivation / problem being solved).
- **How** it was implemented at a high level (key modules / stages touched).
- **How it was tested** (commands run, scenarios, benchmark before/after if relevant).
- **Config / behavior impact**: any new or changed `MOEPT_*` settings, defaults, or
  user-visible behavior, and migration notes if a default changed.
- **Links** to related issues (`Fixes #123`, `Related to #456`).

### Review & merge requirements

- **CI must pass**: tests, lint, and type-check are required. The `--check-config`
  gate (exit non-zero on `ERROR`) must be clean.
- **No regressions**: if the change affects token savings, latency, or response
  quality, run the benchmark and confirm the regression gate holds
  (`--min-similarity`, exit 2 on failure). Do not silently reduce quality below the
  agreed threshold.
- **Respect hard constraints**: the PR must not violate the architectural constraints
  (input-compaction-only, cache stability, benchmark integrity, no model-visible
  markers, capability auto-detection defaults). Call these out explicitly if a PR
  intentionally touches them.
- **Approval**: at least one maintainer approval is required. Address review feedback
  with follow-up commits (do not force-push over review context unless asked).
- **Squash or rebase** onto an up-to-date `main` before merge; keep the final history
  clean and the commit message descriptive of the net change.
- **Conventional, scoped commits** are preferred (e.g. `fix(compactor): ...`,
  `feat(config): ...`, `docs(readme): ...`).

## Code conventions

- **Language**: Python 3.11+. Type-hint everything; `mypy --strict` is enforced
  (`warn_return_any = true`, `warn_unused_configs = true`).
- **Lint**: `ruff` (line-length 100, select `E,F,I,N,W,UP,B,SIM,RUF`; `E501` ignored).
  Run `ruff check src/ tests/`.
- **Format**: follow the existing style; let `ruff` flag issues. Do not reformat
  unrelated code in the same PR.
- **Imports**: prefer `from moeptimizer import <Symbol>` (re-exported from
  `__init__.py`) over deep submodule imports in new code.
- **Tests**: add a `tests/test_<module>.py` for new pipeline stages. The suite uses
  `pytest` with `asyncio_mode = "auto"`, so async tests need no explicit marker.
  Shared fixtures live in `tests/conftest.py`. Integration-level tests are
  `test_e2e.py` and `test_optimizer.py`.
- **Config**: new settings go in `src/moeptimizer/config.py` as pydantic-settings
  fields on `AppConfig` with the `MOEPT_` prefix and `__` nesting. Document them in
  `README.md` and add defaults to `.env.example`.
- **Logging**: use the standard `logging` module. The proxy runs uvicorn at
  `log_level="warning"` by default, so INFO/DEBUG logs are hidden unless raised.
- **Commits**: keep them focused and descriptive. Reference issues where relevant.

### Commit message conventions

- Use **Conventional Commits**: `<type>(<scope>): <subject>` where `type` is one of
  `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`; `scope`
  is the module or area (e.g. `compactor`, `config`, `benchmark`).
- Subject line ≤ 50 chars, lowercase, no trailing period, imperative mood
  ("add" not "added").
- Add a body only when the **why** is not obvious from the subject. Explain the
  motivation, not the diff (the diff is self-explanatory).
- Reference the issue it closes: `Fixes #123`.
- Do not commit secrets, `.env`, large artifacts, or generated caches
  (`__pycache__/`, `.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/` are git-ignored).

### Security & secrets

- **Never commit secrets.** Do not put real tokens, API keys, passwords, or a populated
  `.env` in a PR. `.env.example` must contain only safe defaults/placeholders.
- Redact any `MOEPT_*` values that could leak credentials when sharing logs, configs,
  or benchmark output in issues or PRs.
- Treat backend URLs and model identifiers as non-sensitive, but redact anything that
  could identify an internal host or account.

## Architectural constraints

These are intentional design boundaries — please respect them in PRs. Violating them
silently breaks prefix-cache reuse or response quality and will not be caught by tests
alone.

- **Input compaction only.** The proxy optimizes ONLY the input context sent to the
  backend. It must not attempt to tune or compress the model's response verbosity;
  that is the backend's responsibility.
- **Cache stability.** Keep the system prompt and early turns byte-stable (frozen
  prefix) so the backend's native prefix cache is reused. Never mutate the middle of
  cached context; only append (incremental update) or front-evict.
- **Benchmark integrity.** `scripts/benchmark.py` runs each round as a complete proxy
  conversation followed by a complete direct conversation — never interleave the two.
  Only the new benchmark format is supported (no legacy logs).
- **No model-visible markers.** Internal cache hints (attention sinks, pattern
  markers, reasoning preseed) are stripped before the model sees the prompt.
- **Capability auto-detection.** Slot pinning, native MTP passthrough, and tokenizer
  selection are driven by live, device-aware auto-detection (`MOEPT_V050__CAPABILITY_AUTODETECT`).
  The explicit `SLOT_PINNING_ENABLED` / `NATIVE_MTP_PASSTHROUGH` flags are force-on
  overrides; leave them `false` unless you specifically need to override detection.

## Pipeline orientation

The main pipeline lives in `src/moeptimizer/optimizer.py` (`AgentContextOptimizer`).
It orchestrates one module per stage, e.g.:

- `session_manager.py` — per-session isolation + stable anonymous session resolver.
- `compactor.py` — scratchpad compaction (front-loading eviction for MTP protection).
- `thinking_preserver.py` — protects recent `<think>` blocks, archives stale reasoning.
- `state_store.py` / `state_rag.py` — KV graph + graph-indexed retrieval.
- `loop_detector.py` — detects repeated tool calls / actions / thinking loops.
- `code_chunking.py` — tree-sitter-aware code splitting + language detection.
- `context_compressor.py` / `context_canonicalizer.py` / `selective_truncator.py` —
  newest-user-turn-only compression and trimming.
- `hierarchical_summarizer.py` — cache-stable rolling-summary compaction.
- `static_prefix_kv.py` / `cache_registry.py` / `hit_prediction_model.py` — prefix
  reuse and cache-hit prediction.
- `backend_client.py` / `backend_capabilities.py` — backend communication + probing.

When adding a stage, create the module, wire it into `AgentContextOptimizer`,
re-export the public symbol from `__init__.py`, and add `tests/test_<module>.py`.

## Benchmarking your change

If your change affects token savings, latency, or response quality, run the benchmark
and compare against a baseline:

```bash
# Baseline
python scripts/benchmark.py --scenario opencode --turns 30 --json > before.json 2> before.log

# After your change
python scripts/benchmark.py --scenario opencode --turns 30 --json > after.json 2> after.log

# Regression gate (fails with exit 2 if mean semantic similarity < 0.85)
python scripts/benchmark.py --scenario all --turns 10 --min-similarity 0.85
```

Long runs may need to be launched as a background task to avoid command timeouts;
progress is written to stderr. The report UI is `scripts/benchmark_dashboard.html`.

## Reporting bugs

Open an issue using this template so maintainers can reproduce quickly:

```markdown
**Environment**
- MOE-ptimizer version: (pyproject.toml / `python -m moeptimizer --check-config`)
- Backend model: (MOEPT_SERVER__LLM_MODEL)
- Backend URL: (MOEPT_SERVER__URL, redacted if internal)
- Python version:

**Describe the bug**
A clear and concise description of what went wrong.

**To reproduce**
Steps or a minimal request payload that triggers the issue.

**Expected behavior**
What you expected to happen.

**Observed behavior / logs**
Relevant output, and `/v1/metrics` if the issue is about token savings,
cache reuse, or latency. Redact any secrets or internal hostnames.

**Additional context**
Config overrides (`MOEPT_*`), quality profile, benchmark scenario, etc.
```

## Good first contributions

New to the codebase? Good starting points:

- **Add or tighten a test** for an existing stage (`tests/test_<module>.py`).
- **Improve docs** — clarify a config flag's behavior in `README.md` or add a
  changelog entry.
- **Small, well-scoped bug fixes** with a regression test.
- **Benchmark scenario or metric** improvements that keep the no-legacy /
  no-interleave invariants.

Avoid as a first PR: large pipeline refactors, changes to the cache-stability or
prefix-reuse logic, or anything that touches the architectural constraints — those
need deep context and maintainer discussion first.

## Changelog & release notes

- User-visible changes go in `CHANGELOG.md` under the relevant version section.
  Keep `README.md` concise and link to the changelog rather than duplicating history.
- Write changelog entries in **imperative, user-facing** language describing the
  benefit, not the implementation: "Add `MOEPT_AGENTIC__X` to control Y" rather than
  "Edited config.py to add X".
- Group entries by kind where natural: Added / Changed / Fixed / Removed.
- The version of record is `pyproject.toml` (`version = "0.7.4"`). Bump it following
  semantic versioning when you open a release PR, and keep `__init__.__version__`
  in sync if it is referenced.

## License

By contributing, you agree that your contributions will be licensed under the MIT
License.
