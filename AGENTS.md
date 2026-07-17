# AGENTS.md

Guidance for AI coding agents working in this repository. Read this file before
making changes — it encodes the project's conventions, hard constraints, and the
commands that must pass before any change is considered done.

## Project

MOE-ptimizer is a transparent OpenAI-compatible API proxy that optimizes context
for MoE + MTP models in multi-turn agentic tasks. It sits between a client (OpenAI
SDK) and a backend (Lemonade server) and compresses/compacts the **input** context
sent to the backend while keeping byte-stable prefixes so the backend's native
prefix cache is reused.

- Proxy listens on `:8080` (start with `bash scripts/run.sh`).
- Backend (Lemonade) listens on `:13305` (`MOEPT_SERVER__URL`).
- Python 3.11+, FastAPI, pydantic-settings, tree-sitter, LanceDB.
- Current version: `0.7.4` (see `pyproject.toml`; note `__init__.__version__` may
  lag — treat `pyproject.toml` as the source of truth).

## Repository layout

- `src/moeptimizer/` — package source. `app.py` (FastAPI app + streaming),
  `optimizer.py` (the main context-optimization pipeline), `config.py`
  (pydantic-settings config), plus one module per pipeline stage
  (`compactor.py`, `context_compressor.py`, `code_chunking.py`, `state_rag.py`, …).
  Public API surface is re-exported from `__init__.py` (prefer importing from the
  package root, e.g. `from moeptimizer import AgentContextOptimizer`).
- `tests/` — pytest suite, one `test_*.py` per module. `asyncio_mode = "auto"`.
  `conftest.py` holds shared fixtures. `test_e2e.py` and `test_optimizer.py` are the
  integration-level tests.
- `scripts/` — `run.sh` (start proxy), `dev.sh` (install + test + lint + type-check),
  `benchmark.py` (direct-vs-proxy benchmark), `benchmark_dashboard.html` (report UI),
  `benchmark_all.sh`, `fixtures/` (real project replayed by the `opencode`/`fixtures`
  benchmark scenario).
- `CHANGELOG.md` — version-by-version feature history (NOT README).
- `README.md` — concise overview + config tables; links to CHANGELOG.
- `AGENTS.md` — this file. `CONTRIBUTING.md` — human contributor guide.
  `llm.txt` — LLM-oriented project map.

## Conventions

- **Code style**: `ruff` (line-length 100, `E,F,I,N,W,UP,B,SIM,RUF`; `E501` ignored).
  Run `ruff check src/ tests/`.
- **Types**: `mypy --strict` over `src/moeptimizer/`. Keep functions typed; avoid
  `Any` returns where feasible (`warn_return_any = true`). New public functions and
  methods should have full type annotations.
- **Tests**: `pytest tests/ -v`. New pipeline stages should get a matching
  `test_<module>.py`. Async tests work without explicit markers (`asyncio_mode=auto`).
- **Config**: all settings are pydantic-settings with `MOEPT_` prefix and `__`
  nesting (e.g. `MOEPT_AGENTIC__QUALITY_PROFILE`). Add new settings in `config.py`
  (the `AppConfig` model) and document them in `README.md` + `.env.example`.
- **Changelog**: add user-visible changes to `CHANGELOG.md`; keep `README.md` concise.
- **Imports**: prefer `from moeptimizer import <Symbol>` (re-exported) over deep
  submodule imports in new code, matching `__init__.py`.
- **Logging**: use the standard `logging` module; the proxy runs uvicorn at
  `log_level="warning"` by default, so INFO/DEBUG logs are hidden unless raised.

## Hard constraints (do not violate)

These are intentional design boundaries. Violating them silently breaks prefix-cache
reuse or response quality and will not be caught by tests alone.

- **Proxy scope = input compaction only.** The proxy compacts ONLY the input context
  sent to the backend. The backend/model fully controls response size — the proxy
  must NOT try to tune/compress response verbosity.
- **No legacy benchmark support.** `scripts/benchmark.py` and
  `benchmark_dashboard.html` support only the new benchmark format (changed quality
  metrics). Do not add legacy/old-format log handling.
- **Benchmark invariant**: never interleave direct and proxified requests. Each round
  runs the COMPLETE proxy conversation (all turns) first, then the COMPLETE direct
  conversation as its own full, contiguous, sorted session.
- **Cache stability**: keep the system prompt and early turns byte-stable (frozen
  prefix) so the backend prefix cache is reused. Do not mutate the middle of cached
  context; only append (incremental update) or front-evict.
- **No model-visible markers**: internal cache hints (attention sinks, pattern
  markers, reasoning preseed) are stripped before the model sees the prompt. The
  `ATTENTION_SINKS_ENABLED` / `REASONING_PRESEED_ENABLED` flags are WARN-level config
  issues for this reason.
- **Dashboard charts**: when adding charts to `benchmark_dashboard.html`, choose chart
  types per the FT Visual Vocabulary (https://github.com/ft-interactive/visual-vocabulary).

## Common commands

```bash
bash scripts/run.sh                 # start proxy on :8080
bash scripts/dev.sh                 # install + pytest + ruff + mypy
pytest tests/ -v                    # tests only
ruff check src/ tests/              # lint
mypy src/moeptimizer/               # type-check
python -m moeptimizer --check-config  # validate resolved config
python scripts/benchmark.py --scenario opencode --turns 30 --json > report.json 2> run.log
```

## Before committing

Run `bash scripts/dev.sh` (tests + lint + types) and ensure it passes. The
`--check-config` CLI exits non-zero on ERROR-level config issues and can gate CI.

If you changed `config.py`, `.env.example`, or any default, re-run
`python -m moeptimizer --check-config` and update `README.md`'s config tables.

If you changed pipeline behavior, run a benchmark round and confirm token savings and
quality metrics did not regress (see `README.md` "Benchmarking" for the regression gate).

### Commit conventions

- **Conventional Commits**: `<type>(<scope>): <subject>` — `type` ∈
  `feat|fix|docs|refactor|test|chore|perf|build|ci`; `scope` is the module/area
  (e.g. `compactor`, `config`, `benchmark`). Subject ≤ 50 chars, lowercase, no trailing
  period, imperative mood. Add a body only when the **why** is not obvious. Reference
  the issue: `Fixes #123`.
- **Never commit secrets.** Do not put real tokens, API keys, passwords, or a populated
  `.env` in a change. `.env.example` holds only safe defaults/placeholders. Redact any
  `MOEPT_*` values that could leak credentials in logs, configs, or benchmark output.
- Do not commit generated caches (`__pycache__/`, `.ruff_cache/`, `.mypy_cache/`,
  `.pytest_cache/` — all git-ignored).

### PR rules (summary)

When opening a PR, follow `CONTRIBUTING.md`. The hard rules: a PR must be **human-owned
and reviewed** (not 100% AI-generated), scoped to a **single issue**, **small and
reviewable** (split large changes), and **paired with an issue** describing the problem.
Also: pass `dev.sh`, keep `README.md`/`CHANGELOG.md`/`.env.example` in sync, and respect
the architectural constraints above.

### Changelog & version

- Add user-visible changes to `CHANGELOG.md` (imperative, user-facing language;
  group as Added / Changed / Fixed / Removed). Keep `README.md` concise.
- The version of record is `pyproject.toml` (`version = "0.7.4"`). Bump it per
  semantic versioning on a release PR; treat `pyproject.toml` as the source of truth
  (note `__init__.__version__` may lag).

### Good first tasks

Prefer small, well-scoped work: add/tighten a `tests/test_<module>.py`, clarify a
config flag in `README.md`, or fix a bug with a regression test. Avoid as a first
change: large pipeline refactors or anything touching cache-stability / prefix-reuse
logic — those need maintainer discussion first.

## Where to look for common tasks

- **Add a config flag**: `src/moeptimizer/config.py` (`AppConfig`), then `.env.example`
  and the README config table.
- **Add a pipeline stage**: new module in `src/moeptimizer/`, wire it into
  `optimizer.py` (`AgentContextOptimizer`), re-export from `__init__.py`, add
  `tests/test_<module>.py`.
- **Change an endpoint**: `src/moeptimizer/app.py` (`create_app`).
- **Change backend communication**: `src/moeptimizer/backend_client.py` and
  `backend_capabilities.py`.
- **Change benchmark**: `scripts/benchmark.py` (keep the no-legacy / no-interleave
  invariants) and `scripts/benchmark_dashboard.html`.
