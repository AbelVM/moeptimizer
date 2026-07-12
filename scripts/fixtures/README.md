# Benchmark fixtures: a real use case

These files are a small but **real, runnable** Python project that the
`scripts/benchmark.py` `fixtures` scenario replays as a 30-turn agentic-coding
session. They are the concrete "real use case" the synthetic `*_long` scenarios
approximate: a JSONL-backed user-analytics service that grows from a single
module into a typed, tested, packaged, dockerized service.

## What's here

| Path | Role in the session |
|------|---------------------|
| `users/models.py` | Starting point: `User` dataclass + `summarize`. |
| `users/repository.py` | JSONL IO, schema validation, strict mode, retry. |
| `users/config.py` | Env-loaded `Config` dataclass with fail-fast validation. |
| `users/service.py` | `SummarizerService` with dependency injection. |
| `users/metrics.py` | In-memory `Metrics` + structured `log_event`. |
| `users/cli.py` | `argparse` entry point over the service. |
| `users/__init__.py` | Public `__all__` surface. |
| `tests/test_users.py` | pytest suite (happy path, missing file, strict mode, config). |
| `pyproject.toml` | Packaging metadata. |
| `Dockerfile` | Minimal runtime image. |
| `users.jsonl` | Realistic 100-row fixture (incl. 2 malformed rows for strict mode). |

## How the benchmark uses them

`scripts/fixtures/loader.py` discovers these files in build order and builds a
cumulative multi-turn scenario: each turn pastes the **current project state**
(all files added so far) and asks the agent to add the next real file or
refinement. Because the pasted context genuinely grows turn-over-turn from real
source, the proxy's cache-stable summarization and front-eviction behave like
they would in production.

Run it with:

```bash
python scripts/benchmark.py --scenario fixtures --turns 30
```
