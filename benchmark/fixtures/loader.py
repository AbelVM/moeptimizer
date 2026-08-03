"""Build a realistic multi-turn benchmark scenario from the fixture project.

The fixture files under this directory are a small, real, runnable Python
project. This loader replays building that project as an agentic-coding
session: each turn pastes the *current project state* (every file added so far)
and asks the agent to add the next real file or apply the next real refinement.

Because the pasted context genuinely grows turn-over-turn from real source, the
proxy's cache-stable summarization and front-eviction behave like production
instead of against a synthetic static snippet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Build order: a realistic session grows a single module into a packaged service.
# Each entry either ADDS a new file (context grows) or REFINES an existing one
# (context stays put that turn, like a real edit pass).
_MANIFEST: list[dict] = [
    {"add": "users/models.py",
     "instruction": "Start by turning this into a typed, testable module with a User dataclass and a summarize() helper."},
    {"add": "users/repository.py",
     "instruction": "Add a UserRepository that reads users from a JSONL file, validates the schema per row, and returns User objects."},
    {"refine": "users/repository.py",
     "instruction": "Harden the repository: add a strict mode that collects every malformed row instead of stopping at the first, and a retry wrapper around file reads."},
    {"add": "users/config.py",
     "instruction": "Introduce a Config dataclass that loads from environment variables and fails fast if input_path is missing."},
    {"add": "users/service.py",
     "instruction": "Refactor the summarizer into a SummarizerService with dependency injection so tests can swap in a fake repository."},
    {"refine": "users/service.py",
     "instruction": "Optimize the summary step using collections.Counter to avoid repeated scans."},
    {"add": "users/metrics.py",
     "instruction": "Add lightweight in-memory Metrics (load count, parse errors, summary latency) and a structured log_event helper."},
    {"refine": "users/repository.py",
     "instruction": "Emit a log_event on each load and record parse errors into Metrics so the service is observable."},
    {"add": "users/cli.py",
     "instruction": "Add an argparse CLI entry point (--input, --output, --dry-run) that wires Config -> Repository -> Service."},
    {"add": "users/__init__.py",
     "instruction": "Add a package __init__ that exports the public surface via __all__."},
    {"add": "tests/test_users.py",
     "instruction": "Add a pytest suite covering the happy path, missing file, invalid JSONL in strict mode, and Config.from_env."},
    {"add": "pyproject.toml",
     "instruction": "Add packaging metadata (pyproject.toml) declaring the users package and pytest config."},
    {"add": "Dockerfile",
     "instruction": "Add a minimal Dockerfile that installs the package and runs the CLI against a mounted users.jsonl."},
    {"add": "users.jsonl",
     "instruction": "Add a realistic users.jsonl fixture (include a couple of malformed rows so strict mode is exercised)."},
    {"refine": "users/service.py",
     "instruction": "Add an active_count helper to the service and expose it through the public API."},
    {"refine": "users/cli.py",
     "instruction": "Make the CLI write its summary to --output when provided, otherwise print JSON to stdout."},
    {"refine": "users/repository.py",
     "instruction": "Add a streaming variant that yields User objects one at a time instead of loading everything into memory."},
    {"refine": "users/config.py",
     "instruction": "Add a log_level field to Config and thread it through to the logger."},
    {"refine": "users/metrics.py",
     "instruction": "Add a reset() method to Metrics and a trace_id to log_event for request correlation."},
    {"refine": "users/service.py",
     "instruction": "Add a migration helper that returns the legacy dict-based representation for old callers."},
    {"refine": "tests/test_users.py",
     "instruction": "Add a test for the streaming repository variant and one for the legacy migration helper."},
    {"refine": "users/cli.py",
     "instruction": "Add a --version flag and graceful handling of empty input files."},
    {"refine": "users/repository.py",
     "instruction": "Add a register_validator hook so external validators can run during loading."},
    {"refine": "users/service.py",
     "instruction": "Group the summarize/active_count helpers into a small stats submodule without changing behavior."},
    {"refine": "pyproject.toml",
     "instruction": "Pin the Python requirement to >=3.11 and add a [project.optional-dependencies] test extra."},
    {"refine": "Dockerfile",
     "instruction": "Add a non-root user and a healthcheck that runs the CLI in dry-run mode."},
    {"refine": "users/__init__.py",
     "instruction": "Tighten type hints across the public API and re-export the new stats submodule."},
    {"refine": "tests/test_users.py",
     "instruction": "Add a regression test that replays a 30-turn session by loading the fixture project."},
    {"refine": "users/models.py",
     "instruction": "Do a final cleanup pass: remove dead code and make the module easier to navigate."},
    {"instruction": "Finish by summarizing the build, listing the remaining risks, and suggesting the next production hardening step."},
]

# Files the loader itself should never treat as project source.
_SKIP = {"loader.py", "__init__.py", "README.md"}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _format_project(files: list[tuple[str, str]]) -> str:
    """Render the current project state as fenced file blocks."""
    blocks = []
    for rel, content in files:
        suffix = Path(rel).suffix.lower()
        language = {
            ".jsonl": "json",
            ".toml": "toml",
            ".py": "python",
        }.get(suffix, "dockerfile" if Path(rel).name == "Dockerfile" else "text")
        blocks.append(f"# {rel}\n```{language}\n{content.rstrip()}\n```")
    return "\n\n".join(blocks)


def _apply_refinement(rel: str, content: str, instruction: str, turn: int) -> str:
    """Carry a deterministic working-tree change into the next replay turn."""
    return f"{content.rstrip()}\n# benchmark refinement {turn}: {instruction}\n"


def _variant_suffix(seed: int | None, turn: int) -> str:
    if seed is None:
        return ""
    variants = (
        "Keep the implementation incremental and favor explicit error handling.",
        "Prioritize compatibility with the existing public surface and tests.",
        "Call out any behavior change that could affect operational reliability.",
    )
    return f"\n\nScenario variant {seed} (turn {turn}): {variants[(seed + turn) % len(variants)]}"


def build_fixture_tasks(
    max_turns: int | None = None,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """Return (role, content) tasks replaying the fixture project build.

    Each turn pastes the cumulative project and asks for the next real change.
    Safe to call at import time: returns a minimal fallback on any error so the
    benchmark module always imports.
    """
    try:
        tasks: list[tuple[str, str]] = []
        current: list[tuple[str, str]] = []
        for turn, entry in enumerate(_MANIFEST, 1):
            instruction = entry["instruction"] + _variant_suffix(seed, turn)
            if "add" in entry:
                rel = entry["add"]
                current.append((rel, _read(rel)))
            elif "refine" in entry:
                rel = entry["refine"]
                files = dict(current)
                current = [
                    (path, _apply_refinement(path, files[path], instruction, turn))
                    if path == rel
                    else (path, content)
                    for path, content in current
                ]
            project = _format_project(current)
            tasks.append((
                "user",
                f"{instruction}\n\n"
                "Conversation constraints:\n"
                "- Preserve the existing public API unless the request explicitly asks to change it.\n"
                "- Prefer small, incremental patches over broad rewrites.\n"
                "- Show the key updated sections; you may omit unchanged boilerplate.\n"
                "- Mention any tradeoff that affects latency, cache stability, or testability.\n\n"
                "Current project state (files added in prior turns):\n\n"
                f"{project}\n\n"
                "Please apply the requested change.",
            ))
        if max_turns is not None and max_turns < len(tasks):
            tasks = tasks[:max_turns]
        return tasks
    except Exception:
        # Never break benchmark import; fall back to a single realistic task.
        return [(
            "user",
            "Build a small JSONL-backed user-analytics service with a typed User model, "
            "a repository, a service layer, and a CLI. Start with the model.",
        )]


def _agentic_constraints() -> str:
    return (
        "Conversation constraints:\n"
        "- Preserve the existing public API unless the request explicitly asks to change it.\n"
        "- Prefer small, incremental patches over broad rewrites.\n"
        "- Show the key updated sections; you may omit unchanged boilerplate.\n"
        "- Mention any tradeoff that affects latency, cache stability, or testability."
    )


def read_fixture_file(rel: str) -> str | None:
    """Return the text of a fixture file, or None if it cannot be read."""
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except Exception:
        return None


def agent_log_output(has_tests: bool, seed: int | None = None, turn: int = 0) -> str:
    """Realistic agent ``run_command`` output (verbose test + lint + build log).

    When the suite exists this is a long multi-tool log (>4k chars) so the
    proxy's boundary compressor actually fires on it in the benchmark; the
    file-read tool outputs stay under the threshold and are forwarded verbatim
    (quality-safe), which mirrors real agentic traffic where terminal logs are
    the dominant bloat, not file contents.
    """
    if not has_tests:
        return (
            "collected 0 items / 1 error\n"
            "ERROR tests/test_users.py: No module named 'users'\n"
            "(run `pip install -e .` then retry)\n"
        )
    seeded_failure = seed is not None and (seed + turn) % 5 == 0
    if seeded_failure:
        return (
            "============================= test session starts ==============================\n"
            "collected 5 items\n\n"
            "tests/test_users.py::test_service_summarize FAILED\n\n"
            "E   AssertionError: expected active user count to remain stable\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_users.py::test_service_summarize - AssertionError\n"
            "========================= 1 failed, 4 passed in 0.41s =========================\n"
        )
    lines: list[str] = []
    lines.append("======================== test session starts ========================")
    lines.append("platform linux -- Python 3.14.0, pytest-8.3.4, pluggy-1.5.0")
    lines.append("rootdir: /workspace")
    lines.append("plugins: cov-5.0.0, asyncio-0.24.0, timeout-2.3.1")
    lines.append("collected 5 items")
    lines.append("")
    _tests = [
        "tests/test_users.py::test_load_happy_path",
        "tests/test_users.py::test_missing_file_raises",
        "tests/test_users.py::test_invalid_jsonl_strict_mode",
        "tests/test_users.py::test_service_summarize",
        "tests/test_users.py::test_config_from_env",
    ]
    for i, t in enumerate(_tests, 1):
        pct = i * 20
        lines.append(f"{t} PASSED                       [{' ' if pct < 100 else ''}{pct:>3}%]")
    lines.append("")
    lines.append("======================== 5 passed in 0.34s ========================")
    lines.append("")
    lines.append("$ ruff check .")
    lines.append("All checks passed!")
    lines.append("")
    lines.append("$ mypy users")
    lines.append("Success: no issues found in 7 source files")
    lines.append("")
    lines.append("$ pytest --cov=users --cov-report=term-missing -q")
    cov = [
        ("users/__init__.py", 3, 0, "100%", ""),
        ("users/models.py", 45, 2, "96%", "45-48"),
        ("users/repository.py", 62, 4, "94%", "45-48, 60-62"),
        ("users/config.py", 30, 1, "97%", "22"),
        ("users/service.py", 55, 3, "95%", "30-32"),
        ("users/metrics.py", 40, 2, "95%", "18-19"),
        ("users/cli.py", 70, 5, "93%", "40-44"),
    ]
    lines.append("Name                 Stmts   Miss  Cover   Missing")
    lines.append("----------------------------------------------------")
    for name, stmts, miss, cover, missing in cov:
        lines.append(f"{name:<20} {stmts:>5} {miss:>5} {cover:>6}   {missing}")
    lines.append("----------------------------------------------------")
    lines.append("TOTAL                  305    17    94%")
    lines.append("")
    lines.append("$ pip install -e .")
    lines.append("Obtaining file:///workspace")
    lines.append("  Installing build dependencies: started")
    lines.append("  Installing build dependencies: finished with status 'done'")
    lines.append("  Getting requirements to build wheel: started")
    lines.append("  Getting requirements to build wheel: finished with status 'done'")
    lines.append("  Preparing metadata (pyproject.toml): started")
    lines.append("  Preparing metadata (pyproject.toml): finished with status 'done'")
    lines.append("Requirement already satisfied: pydantic>=2.0 in /usr/local/lib/python3.14/site-packages (from users==0.1.0)")
    lines.append("Requirement already satisfied: click>=8.0 in /usr/local/lib/python3.14/site-packages (from users==0.1.0)")
    lines.append("Building wheels for collected packages: users")
    lines.append("  Building wheel for users (pyproject.toml): started")
    lines.append("  Building wheel for users (pyproject.toml): finished with status 'done'")
    lines.append("  Created wheel for users: filename=users-0.1.0-py3-none-any.whl size=5432 sha256=9f2c...")
    lines.append("  Stored in directory: /root/.cache/pip/wheels/ab/12/...")
    lines.append("Successfully installed users-0.1.0")
    lines.append("")
    lines.append("$ docker build -t users-service:dev .")
    lines.append("Sending build context to Docker daemon  14.21kB")
    lines.append("Step 1/9 : FROM python:3.12-slim")
    lines.append(" ---> 3a7c9b2d1e4f")
    lines.append("Step 2/9 : ENV PYTHONUNBUFFERED=1")
    lines.append(" ---> Running in 7b1f0c2a9d33")
    lines.append(" ---> 5e8d4c1b0a22")
    lines.append("Step 3/9 : WORKDIR /app")
    lines.append(" ---> Running in 2c9a4e7b1f08")
    lines.append(" ---> 0d3b6c2a8e91")
    lines.append("Step 4/9 : COPY pyproject.toml .")
    lines.append(" ---> 9c1e2d3b4a50")
    lines.append("Step 5/9 : RUN pip install --no-cache-dir .")
    lines.append(" ---> Running in 4f7a1c9e2b60")
    lines.append("Processing /app")
    lines.append("  Installing build dependencies: started")
    lines.append("  Installing build dependencies: finished with status 'done'")
    lines.append("  Getting requirements to build wheel: started")
    lines.append("  Getting requirements to build wheel: finished with status 'done'")
    lines.append("  Preparing metadata (pyproject.toml): started")
    lines.append("  Preparing metadata (pyproject.toml): finished with status 'done'")
    lines.append("Building wheels for collected packages: users")
    lines.append("  Building wheel for users (pyproject.toml): finished with status 'done'")
    lines.append("  Created wheel for users: filename=users-0.1.0-py3-none-any.whl")
    lines.append("Successfully built users-0.1.0-py3-none-any.whl")
    lines.append("Step 6/9 : COPY users ./users")
    lines.append(" ---> 1a2b3c4d5e60")
    lines.append("Step 7/9 : COPY tests ./tests")
    lines.append(" ---> 6b5c4d3e2a10")
    lines.append("Step 8/9 : RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app")
    lines.append(" ---> Running in 8c7d6e5f4a32")
    lines.append(" ---> 2e1d0c9b8a77")
    lines.append("Step 9/9 : HEALTHCHECK CMD python -m users --dry-run")
    lines.append(" ---> Running in 3f2e1d0c9b88")
    lines.append(" ---> 7a6b5c4d3e29")
    lines.append("Successfully tagged users-service:dev")
    lines.append("")
    lines.append("real 0m12.481s")
    lines.append("user 0m8.902s")
    lines.append("sys 0m1.337s")
    lines.append("")
    lines.append("$ trivy image --severity HIGH,CRITICAL users-service:dev")
    lines.append("2024-01-15T10:22:31.447Z\tINFO\tVulnerability scanning is enabled")
    lines.append("2024-01-15T10:22:31.448Z\tINFO\tSecret scanning is enabled")
    lines.append("2024-01-15T10:22:33.102Z\tINFO\tDetected OS: debian")
    lines.append("2024-01-15T10:22:33.102Z\tINFO\tDetecting Debian vulnerabilities...")
    lines.append("2024-01-15T10:22:35.771Z\tINFO\tNumber of language-specific files: 1")
    lines.append("2024-01-15T10:22:35.771Z\tINFO\tDetecting python-pkg vulnerabilities...")
    lines.append("users-service:dev (debian 12.4)")
    lines.append("========================================================")
    lines.append("Total: 2 (HIGH: 1, CRITICAL: 1)")
    lines.append("")
    lines.append("┌─────────────────┬──────────┬──────────┬──────────────────────┬───────────────┬──────────────────────────┐")
    lines.append("│ Library         │ Severity │ CVE ID   │ Installed Version      │ Fixed Version │         Title            │")
    lines.append("├─────────────────┼──────────┼──────────┼──────────────────────┼───────────────┼──────────────────────────┤")
    lines.append("│ pydantic        │ CRITICAL │ CVE-2023-12345 │ 2.5.2          │ 2.5.3         │ ReDoS in pydantic        │")
    lines.append("│ click           │ HIGH     │ CVE-2023-67890 │ 8.1.3          │ 8.1.4         │ Command injection in CLI │")
    lines.append("└─────────────────┴──────────┴──────────┴──────────────────────┴───────────────┴──────────────────────────┘")
    lines.append("")
    lines.append("Recommendation: bump pydantic>=2.5.3 and click>=8.1.4 in pyproject.toml")
    return "\n".join(lines)


def build_fixture_agentic_tasks(
    max_turns: int | None = None,
    seed: int | None = None,
) -> list[list[dict]]:
    """Return OpenCode-style agentic turns replaying the fixture project build.

    Each turn is a realistic agent payload: the user task (no pasted code — the
    code lives in tool outputs, like a real coding client), followed by assistant
    ``tool_calls`` and the *real* tool results read from the fixture files
    (``read_file`` returns the actual file content; ``run_command`` returns a
    believable pytest result). Safe to call at import time.
    """
    try:
        import json

        tasks: list[list[dict]] = []
        current: list[tuple[str, str]] = []
        for idx, entry in enumerate(_MANIFEST):
            instruction = entry["instruction"] + _variant_suffix(seed, idx + 1)
            read_path: str | None = None
            read_content: str | None = None
            if "add" in entry:
                rel = entry["add"]
                content = _read(rel)
                read_path = rel
                read_content = content
                current.append((rel, content))
            elif "refine" in entry:
                rel = entry["refine"]
                read_path = rel
                files = dict(current)
                read_content = files.get(rel, _read(rel))
                updated = _apply_refinement(rel, read_content, instruction, idx + 1)
                current = [
                    (path, updated if path == rel else content)
                    for path, content in current
                ]
            # A summary-only entry (no add/refine) reads no file; it just runs
            # the suite and asks for a wrap-up, like the end of a real session.
            has_tests = any(r == "tests/test_users.py" for r, _ in current)
            turn = idx + 1
            messages: list[dict] = [
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{_agentic_constraints()}",
                }
            ]
            if read_path is not None:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{turn}_0",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": read_path}),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{turn}_0",
                            "name": "read_file",
                            "content": (read_content or "").rstrip(),
                        },
                    ]
                )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{turn}_1",
                                "type": "function",
                                "function": {
                                    "name": "run_command",
                                    "arguments": json.dumps(
                                        {"command": "python -m pytest -q users"}
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{turn}_1",
                        "name": "run_command",
                        "content": agent_log_output(has_tests, seed=seed, turn=turn),
                    },
                ]
            )
            tasks.append(messages)
        if max_turns is not None and max_turns < len(tasks):
            tasks = tasks[:max_turns]
        return tasks
    except Exception:
        return [
            [
                {
                    "role": "user",
                    "content": "Build a small JSONL-backed user-analytics service. Start with the model.",
                }
            ]
        ]


def fixture_root() -> Path:
    return ROOT


def materialize_fixture_workspace(
    destination: Path,
    max_turns: int | None = None,
    seed: int | None = None,
) -> Path:
    """Write the manifest's current project state into an isolated workspace."""
    current: dict[str, str] = {}
    limit = len(_MANIFEST) if max_turns is None else min(max_turns, len(_MANIFEST))
    for turn, entry in enumerate(_MANIFEST[:limit], 1):
        if "add" in entry:
            rel = entry["add"]
            current[rel] = _read(rel)
        elif "refine" in entry:
            rel = entry["refine"]
            instruction = entry["instruction"] + _variant_suffix(seed, turn)
            current[rel] = _apply_refinement(rel, current[rel], instruction, turn)
    for rel, content in current.items():
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return destination


def apply_fixture_patch(destination: Path, patch_text: str) -> bool:
    """Apply a unified diff only when it passes a zero-fuzz dry run."""
    command = ["patch", "--batch", "--forward", "--fuzz=0", "-p0"]
    check = subprocess.run(
        [*command, "--dry-run"],
        cwd=destination,
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        return False
    return subprocess.run(
        command,
        cwd=destination,
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0


def run_fixture_acceptance(
    destination: Path,
    max_turns: int | None = None,
    max_retries: int = 0,
    generated_patches: dict[int, str] | None = None,
    repair_patches: dict[int, str] | None = None,
    seed: int | None = None,
) -> list[dict[str, object]]:
    """Materialize each state and run the real fixture tests when available."""
    def bounded_output(result: object) -> str:
        output = "\n".join(
            str(getattr(result, name, "") or "")
            for name in ("stdout", "stderr")
        )
        return output[-2000:]

    records: list[dict[str, object]] = []
    limit = len(_MANIFEST) if max_turns is None else min(max_turns, len(_MANIFEST))
    for turn in range(1, limit + 1):
        materialize_fixture_workspace(destination, turn, seed=seed)
        generated_patch_applied = False
        if generated_patches and turn in generated_patches:
            generated_patch_applied = apply_fixture_patch(destination, generated_patches[turn])
        has_tests = (destination / "tests" / "test_users.py").exists()
        if not has_tests:
            records.append({
                "turn": turn,
                "action": "edit",
                "verified": False,
                "generated_patch_applied": generated_patch_applied,
            })
            continue
        attempts = 0
        repair_applied = False
        while True:
            attempts += 1
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests"],
                cwd=destination,
                env={**os.environ, "PYTHONPATH": str(destination)},
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 and not repair_applied and repair_patches and turn in repair_patches:
                repair_applied = apply_fixture_patch(destination, repair_patches[turn])
            if result.returncode == 0 or attempts > max_retries:
                break
        records.append({
            "turn": turn,
            "action": "edit_and_test",
            "verified": result.returncode == 0,
            "returncode": result.returncode,
            "attempts": attempts,
            "generated_patch_applied": generated_patch_applied,
            "repair_applied": repair_applied,
            "output_tail": bounded_output(result),
        })
    return records


def available_files() -> list[str]:
    """Files the loader would discover, for inspection/tests."""
    out = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel in _SKIP or rel == "loader.py":
            continue
        out.append(rel)
    return out


def load_generated_patches(patch_directory: Path, max_turns: int | None = None) -> dict[int, str]:
    """Load optional per-turn patches named ``turn-N.patch``."""
    limit = len(_MANIFEST) if max_turns is None else min(max_turns, len(_MANIFEST))
    return {
        turn: (patch_directory / f"turn-{turn}.patch").read_text(encoding="utf-8")
        for turn in range(1, limit + 1)
        if (patch_directory / f"turn-{turn}.patch").is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated fixture acceptance tests.")
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--patch-dir", type=Path, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="moeptimizer-fixture-") as workspace:
        records = run_fixture_acceptance(
            Path(workspace),
            max_turns=args.turns,
            max_retries=args.max_retries,
            generated_patches=(
                load_generated_patches(args.patch_dir, args.turns)
                if args.patch_dir is not None
                else None
            ),
            seed=args.seed,
        )
    print(json.dumps(records, indent=2))
    return int(any(record.get("action") == "edit_and_test" and not record.get("verified")
                   for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
