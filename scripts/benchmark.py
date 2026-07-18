#!/usr/bin/env python3
"""Multi-turn benchmark: direct Lemonade vs moeptimizer proxy.

Compares latency, token usage, context-window efficiency, and response quality
across realistic multi-turn agentic-coding conversations that grow the context
window.

Every scenario runs as an OpenCode-style harness by default: each turn sends a
real agent payload — the user task plus assistant ``tool_calls`` and the
corresponding ``tool`` results (file reads, test/lint/build logs) — and the
OpenAI ``tools`` schema is forwarded to the backend, exactly like a production
coding client. The proxy boundary-compresses large tool outputs (terminal
logs, file dumps) via ToolOutputCompressor before they enter the stable prefix,
so the benchmark exercises that path too. Pass ``--no-agentic`` to fall back to
plain user messages.

The proxy is auto-started if not already running on the target port (checked via /v1/health).

Usage:
    # Run with defaults (proxy on 8080, lemonade on localhost:13305)
    python scripts/benchmark.py

    # Custom ports / turns / rounds
    python scripts/benchmark.py --port 9090 --turns 20 --rounds 3

    # JSON output for downstream analysis
    python scripts/benchmark.py --json > report.json

    # Dump full response pairs with all quality metrics
    python scripts/benchmark.py --dump-responses

    # Real-life coding scenarios (all agentic / OpenCode-harness by default)
    python scripts/benchmark.py --scenario debug --turns 15
    python scripts/benchmark.py --scenario debug_long --turns 30
    python scripts/benchmark.py --scenario refactor_long --turns 30
    python scripts/benchmark.py --scenario feature_long --turns 30
    python scripts/benchmark.py --scenario default_long --turns 30

    # OpenCode-harness replay of the real fixture project (user task + tool
    # calls + real tool outputs read from scripts/fixtures/)
    python scripts/benchmark.py --scenario fixtures --turns 30
    python scripts/benchmark.py --scenario opencode --turns 30

    # Plain user messages instead of agent payloads
    python scripts/benchmark.py --scenario debug_long --turns 30 --no-agentic

    # Run all scenarios
    python scripts/benchmark.py --scenario all --turns 10

    # Stress test with large context
    python scripts/benchmark.py --turns 50 --budget 8000
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEMONADE_URL = os.environ.get("MOEPT_SERVER__URL", "http://localhost:13305/api/v1")
MODEL_ID = os.environ.get(
    "MOEPT_SERVER__LLM_MODEL", "Qwen3.6-35B-A3B-MTP-GGUF"
)
MOEPT_PORT = int(os.environ.get("MOEPT_PORT", "8080"))

# Realistic agentic-coding system prompt. This is the frozen-prefix anchor that
# the proxy keeps byte-stable across turns, so it should resemble what a real
# coding client (e.g. OpenCode) actually sends: tool-use framing, conciseness
# guidance, and an instruction to preserve prior context.
SYSTEM_PROMPT = (
    "You are an autonomous coding agent operating inside a developer's editor. "
    "You have access to tools for reading files, running shell commands, editing "
    "code, and searching the repository. Follow these rules:\n"
    "1. Think step by step, but keep reasoning concise and never repeat what the "
    "user or a previous turn already established.\n"
    "2. When the user pastes the current module, treat it as the source of truth "
    "and apply the requested change incrementally.\n"
    "3. Prefer small, well-scoped edits over broad rewrites unless asked.\n"
    "4. Show the key updated sections; you may omit unchanged boilerplate.\n"
    "5. Mention any tradeoff that affects latency, cache stability, or testability."
)

# OpenAI-compatible tool schemas an agentic coding client (e.g. OpenCode) sends
# on every request. The benchmark includes these in the request body whenever it
# runs in agentic (OpenCode-harness) mode so the payload matches what a real
# client ships to the backend — and so the proxy must forward `tools` alongside
# the `tool_calls` / `tool` messages below.
OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply a search/replace or unified-diff edit to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "edits": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of edits to apply.",
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "timeout": {"type": "integer", "description": "Optional timeout in seconds."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the workspace for a pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Optional path to search within."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (defaults to workspace root)."},
                },
                "required": [],
            },
        },
    },
]


# Fallback run_command log used only if the fixture loader is unavailable. It is
# deliberately >4k chars so the proxy's tool-output compression still fires.
_FALLBACK_AGENT_LOG = "\n".join(
    [
        "======================== test session starts ========================",
        "platform linux -- Python 3.14.0, pytest-8.3.4, pluggy-1.5.0",
        "collected 12 items",
        "",
        "tests/test_users.py::test_load_happy_path PASSED                 [  8%]",
        "tests/test_users.py::test_missing_file_raises PASSED            [ 16%]",
        "tests/test_users.py::test_invalid_jsonl_strict_mode PASSED      [ 25%]",
        "tests/test_users.py::test_service_summarize PASSED              [ 33%]",
        "tests/test_users.py::test_config_from_env PASSED                [ 41%]",
        "tests/test_users.py::test_streaming_repository PASSED           [ 50%]",
        "tests/test_users.py::test_legacy_migration PASSED              [ 58%]",
        "tests/test_users.py::test_cli_output_flag PASSED               [ 66%]",
        "tests/test_users.py::test_register_validator PASSED            [ 75%]",
        "tests/test_users.py::test_stats_submodule PASSED               [ 83%]",
        "tests/test_users.py::test_pyproject_extra PASSED               [ 91%]",
        "tests/test_users.py::test_docker_healthcheck PASSED            [100%]",
        "",
        "======================== 12 passed in 0.34s ========================",
        "",
        "$ ruff check .",
        "All checks passed!",
        "",
        "$ mypy users",
        "Success: no issues found in 7 source files",
        "",
        "$ pytest --cov=users --cov-report=term-missing -q",
        "TOTAL                  305    17    94%",
        "",
        "$ docker build -t users-service:dev .",
        "Successfully tagged users-service:dev",
    ]
    + ["DEBUG worker heartbeat ok"] * 200
)


def _extract_code_block(text: str) -> str | None:
    """Return the first fenced code block in *text*, or None if there is none."""
    import re

    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).rstrip() if match else None


def _synthetic_run_log(turn_index: int) -> str:
    """A realistic, per-turn-varying test/lint/build log (>4k chars).

    Each turn gets a distinct log (turn-tagged header + varying heartbeat count)
    so the synthetic agent payloads are not byte-identical across turns, while
    still being large enough to trigger the proxy's tool-output compression.
    """
    loader = _get_fixture_loader()
    base = loader.agent_log_output(True) if loader is not None else _FALLBACK_AGENT_LOG
    heartbeats = "\n".join(
        f"DEBUG worker heartbeat ok seq={turn_index}:{i}"
        for i in range(180 + (turn_index * 7) % 60)
    )
    return f"$ python -m pytest -q  # turn {turn_index}\n{base}\n{heartbeats}"


def _synthetic_grep_output(turn_index: int, read_content: str | None) -> str:
    """A realistic, per-turn-varying grep result (>4k chars).

    Real agents grep the repo constantly; the matches differ every turn as the
    search target or the codebase changes. Large enough to trigger the proxy's
    tool-output compression, like a real multi-file search would.
    """
    files = [
        "src/users/repository.py", "src/users/service.py", "src/users/schema.py",
        "tests/test_users.py", "tests/test_service.py", "src/api/routes.py",
        "src/cli.py", "src/config.py", "src/metrics.py", "src/logging_utils.py",
    ]
    symbols = [
        "def load", "def summarize", "class User", "def _parse_row",
        "async def load", "def build_cli", "def validate_config", "def stream_users",
    ]
    lines = [
        f"{files[i % len(files)]}:{10 + (i * 7 + turn_index * 3) % 400}:"
        f"{(symbols[(i + turn_index) % len(symbols)])}(self, ...):"
        for i in range(220)
    ]
    return "\n".join(lines)


def _synthetic_edit_output(turn_index: int) -> str:
    """A realistic, per-turn-varying edit confirmation / unified diff (>4k chars)."""
    header = (
        f"Applied 1 edit to module.py (turn {turn_index})\n"
        "--- a/module.py\n+++ b/module.py\n"
    )
    hunks = []
    for i in range(120):
        ln = 20 + i
        hunks.append(f"@@ -{ln},{ln + 2} +{ln},{ln + 2} @@")
        hunks.append(f"-    result = compute_legacy(items, seed={i})")
        hunks.append(f"+    result = compute(items, seed={i}, turn={turn_index})")
        hunks.append("     return result")
    return header + "\n".join(hunks)


def _synthetic_list_output(turn_index: int) -> str:
    """A realistic, per-turn-varying directory listing (>4k chars)."""
    entries = []
    roots = ["src", "tests", "scripts", "docs", "config"]
    for r in roots:
        for i in range(40):
            kind = "d" if (i + turn_index) % 5 == 0 else "f"
            entries.append(f"{kind} {r}/{r}_{i:03d}.py")
    return "\n".join(entries)


def _synthetic_tool_outputs(
    turn_index: int,
    read_path: str | None,
    read_content: str | None,
) -> list[dict]:
    """Build a realistic, turn-varying set of agent tool calls + results.

    Real coding agents interleave many tool types (grep, edit_file, list_files,
    multiple reads/commands) and vary the call count per turn; the synthetic
    scenarios previously emitted the identical ``read_file`` + ``run_command``
    pair every turn, which under-exercises the proxy's tool-output compression
    path and is unrepresentative. This rotates through a few believable tool
    mixes so each turn's payload differs in shape and content (grep hits, edit
    diffs, file listings, build/lint logs) while keeping the ``read_file`` result
    coherent with the task when a module is supplied.
    """
    run_content = _synthetic_run_log(turn_index)
    read_result = read_content or "(current file contents would be returned here by the harness)"
    read_tool = {
        "name": "read_file",
        "arguments": {"path": read_path or "src/module.py"},
        "content": read_result,
    }
    grep_tool = {
        "name": "grep",
        "arguments": {"pattern": r"def \w+|class \w+", "path": read_path or "src"},
        "content": _synthetic_grep_output(turn_index, read_content),
    }
    edit_tool = {
        "name": "edit_file",
        "arguments": {
            "path": read_path or "src/module.py",
            "edits": [{"old": "    pass", "new": "    return result"}],
        },
        "content": _synthetic_edit_output(turn_index),
    }
    list_tool = {
        "name": "list_files",
        "arguments": {"path": "."},
        "content": _synthetic_list_output(turn_index),
    }
    cmd_tool = {
        "name": "run_command",
        "arguments": {"command": "python -m pytest -q"},
        "content": run_content,
    }
    # Rotate the mix so the shape varies turn-to-turn (2-3 calls, different tools).
    mix = turn_index % 4
    if mix == 0:
        return [read_tool, cmd_tool]
    if mix == 1:
        return [grep_tool, read_tool]
    if mix == 2:
        return [edit_tool, cmd_tool]
    return [list_tool, grep_tool, cmd_tool]


def _agentic_exchange(
    user_content: str,
    turn_index: int,
    tool_outputs: list[dict] | None = None,
    read_path: str | None = None,
    read_content_override: str | None = None,
) -> list[dict]:
    """Build one OpenCode-style agentic turn as a list of messages.

    The turn is a realistic agent payload: the user task, followed by assistant
    ``tool_calls`` and the corresponding ``tool`` results. ``tool_outputs`` is a
    list of ``{"name", "arguments", "content"}`` describing the tool calls the
    agent makes this turn and their (already-computed) results. When omitted, a
    turn-varying mix of tool calls (read_file / grep / edit_file / list_files /
    run_command) is synthesized via :func:`_synthetic_tool_outputs` so even the
    synthetic scenarios emit a believable, non-repetitive agent payload and
    exercise the proxy's tool-output compression path across varied content
    types (grep hits, edit diffs, file listings, build/lint logs).

    When ``read_path`` is given, the ``read_file`` result is the scenario's own
    current module so the tool output is coherent with the task. By default the
    module is extracted from the fenced code block in ``user_content`` (the
    long scenarios paste the growing module each turn). ``read_content_override``
    lets a caller supply the module directly — used by the short scenarios,
    whose tasks carry no code block, so they would otherwise fall back to a
    placeholder string and produce an incoherent agent payload. The
    ``run_command`` log varies per turn. When ``read_path`` is omitted the
    exchange falls back to reading a real fixture file (whole-project scenarios).
    """
    if tool_outputs is None:
        if read_path is not None:
            # Prefer an explicit override (e.g. the scenario's base module) so the
            # tool output stays coherent even when the user task has no code block.
            read_content = (
                read_content_override
                if read_content_override is not None
                else _extract_code_block(user_content)
            )
        else:
            loader = _get_fixture_loader()
            read_path = "users/repository.py"
            read_content = (
                loader.read_fixture_file("users/repository.py")
                if loader is not None
                else None
            )
        # Vary the tool mix per turn so the synthetic agent payloads resemble
        # real OpenCode-harness traffic (interleaved grep/edit/list/read/run
        # calls) instead of the identical read_file + run_command every turn.
        tool_outputs = _synthetic_tool_outputs(turn_index, read_path, read_content)
    msgs: list[dict] = [{"role": "user", "content": user_content}]
    for i, tool in enumerate(tool_outputs):
        call_id = f"call_{turn_index}_{i}"
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "arguments": json.dumps(tool["arguments"]),
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool["name"],
                "content": tool["content"],
            }
        )
    return msgs


def _append_assistant_message(messages: list[dict], msg: dict) -> None:
    """Append an assistant response, synthesizing tool results if it emitted tool_calls.

    Keeps the conversation valid for the next turn even if the backend chooses to
    call tools: each ``tool_call`` gets a placeholder tool result so the history
    never ends on a dangling assistant ``tool_calls`` entry.
    """
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""
    if tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": (tc.get("function") or {}).get("name", "tool"),
                    "content": "(tool executed by harness; result omitted in benchmark replay)",
                }
            )
    else:
        messages.append({"role": "assistant", "content": content})


# ---------------------------------------------------------------------------
# Long benchmark scenarios
# ---------------------------------------------------------------------------

# Realistic long-benchmark scenarios model an *accumulating* codebase: each turn
# the user pastes the current module, which has grown with every prior edit. The
# pasted code therefore genuinely grows turn-over-turn (instead of re-pasting a
# static file), which is what makes older turns become stale and lets the proxy's
# cache-stable summarization / front-eviction behave like in production.

import re as _re_mod

# Top-level definition matcher (def / async def / class) at column 0.
_DEF_NAME_RE = _re_mod.compile(r"^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)", _re_mod.MULTILINE)


def _split_top_level_blocks(text: str) -> list[str]:
    """Split a module into top-level blocks.

    A block is a maximal run of lines that starts at column 0 (or with a
    decorator ``@``) and continues through indented lines, the ``def`` /
    ``class`` line that immediately follows a decorator, and blank lines that
    occur *inside* a definition body. A new top-level block starts only at a
    column-0 line that is neither a decorator nor directly preceded by one.

    This matters because some steps (e.g. ``DEFAULT_STEPS[4]``'s ``FibService``)
    contain a blank line between methods; a naive blank-line split would orphan
    the later method as a separate top-level block and corrupt the merge.
    """
    blocks: list[str] = []
    cur: list[str] = []
    prev_was_decorator = False
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                cur.append(line)
            prev_was_decorator = False
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.lstrip()
        is_decorator = stripped.startswith("@")
        starts_new = (
            indent == 0
            and not is_decorator
            and not prev_was_decorator
        )
        if not cur or starts_new:
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
        prev_was_decorator = is_decorator
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _defined_names(block: str) -> set[str]:
    """Return the set of top-level def/class names declared in *block*."""
    names: set[str] = set()
    for m in _DEF_NAME_RE.finditer(block):
        names.add(m.group(1) or m.group(2))
    return names


def _merge_modules_last_def_wins(parts: list[str]) -> str:
    """Concatenate module parts, keeping only the last definition of each name.

    Some code steps are *replacements* of a function the base (or an earlier
    step) already defines — e.g. ``DEFAULT_STEPS`` redefines ``fibonacci`` and
    ``fibonacci_gen``. A naive cumulative append would paste the same name
    twice, producing an incoherent (and for a real agent, impossible) "current
    module". Last-def-wins keeps each top-level name exactly once, matching
    what an agent's actual file would contain.
    """
    blocks: list[str] = []
    for part in parts:
        if part and part.strip():
            blocks.extend(_split_top_level_blocks(part))
    last_index: dict[str, int] = {}
    for i, blk in enumerate(blocks):
        for name in _defined_names(blk):
            last_index[name] = i
    kept = [
        blk
        for i, blk in enumerate(blocks)
        if not _defined_names(blk)
        or all(i == last_index[name] for name in _defined_names(blk))
    ]
    return "\n\n".join(kept)


def _cumulative_code(base: str, steps: list[str], index: int) -> str:
    """Return the module state after applying code steps 0..index (capped).

    Real agentic coding accumulates state: each turn the user pastes the
    current file, which has grown with every prior edit. We model that by
    appending each step's delta to the base so the pasted code genuinely
    grows turn-over-turn (instead of re-pasting a static file). This is what
    makes older turns become stale and lets the proxy's cache-stable
    summarization / front-eviction behave like in production.

    Replacement-style steps (a step that redefines a name the base or an
    earlier step already defined) are merged last-def-wins so the pasted
    "current module" never contains duplicate top-level definitions.
    """
    if not steps:
        return base
    idx = min(index, len(steps) - 1)
    applied = [s for s in steps[: idx + 1] if s.strip()]
    if not applied:
        return base.rstrip()
    return _merge_modules_last_def_wins([base.rstrip(), *applied])


def _build_long_tasks(
    instructions: list[str], base_code: str, code_steps: list[str]
) -> list[str]:
    """Build long-benchmark tasks: each turn pastes the *current* (cumulative) code."""
    return [
        f"""{instruction}

Conversation constraints:
- Preserve the existing public API unless the request explicitly asks to change it.
- Prefer small, incremental patches over broad rewrites.
- Show the key updated sections; you may omit unchanged boilerplate.
- Mention any tradeoff that affects latency, cache stability, or testability.

Current code (module state after prior turns):

```python
{_cumulative_code(base_code, code_steps, index)}
```

Please apply the requested change."""
        for index, instruction in enumerate(instructions)
    ]


BASE_REFACTOR_CODE = """from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(slots=True)
class User:
    id: int
    name: str
    active: bool = True

class UserRepository:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[User]:
        users = []
        with self.path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                raw = json.loads(line)
                users.append(User(id=int(raw["id"]), name=str(raw["name"]), active=bool(raw.get("active", True))))
        return users

def summarize(users: list[User]) -> dict[str, int | bool]:
    return {"count": len(users), "active": sum(1 for user in users if user.active)}
"""

# One code delta per LONG_REFACTOR_INSTRUCTIONS entry. Empty string = the turn
# asks for an artifact that lives outside this module (tests, Dockerfile, CI,
# docs, changelog), so the pasted module is unchanged that turn.
REFACTOR_STEPS = [
    # 0: typed, testable module + entry point
    '''__all__ = ["User", "UserRepository", "summarize"]

if __name__ == "__main__":
    repo = UserRepository(Path("users.jsonl"))
    print(summarize(repo.load()))''',
    # 1: schema validation for malformed rows
    '''class UserSchemaError(ValueError):
    """Raised when a JSONL row fails schema validation."""

def _parse_row(line: str, line_no: int) -> User:
    raw = json.loads(line)
    if "id" not in raw or "name" not in raw:
        raise UserSchemaError(f"row {line_no}: missing id/name")
    return User(id=int(raw["id"]), name=str(raw["name"]), active=bool(raw.get("active", True)))''',
    # 2: Config dataclass loaded from env
    '''import os

@dataclass(slots=True)
class Config:
    input_path: Path = Path("users.jsonl")
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            input_path=Path(os.environ.get("USERS_INPUT", "users.jsonl")),
            dry_run=os.environ.get("USERS_DRY_RUN", "0") == "1",
        )''',
    # 3: service class with dependency injection
    '''class SummarizerService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def summarize(self) -> dict[str, int | bool]:
        return summarize(self.repository.load())''',
    # 4: structured logging
    '''import logging

logger = logging.getLogger("users")

def log_event(step: str, **fields: object) -> None:
    logger.info(json.dumps({"step": step, **fields}))''',
    # 5: CLI entry point
    '''import argparse

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize users")
    parser.add_argument("--input", default="users.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser''',
    # 6: pytest suite (separate file; module unchanged)
    "",
    # 7: async support with aiofiles
    '''import aiofiles

class AsyncUserRepository:
    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> list[User]:
        users: list[User] = []
        async with aiofiles.open(self.path) as fh:
            async for line in fh:
                users.append(_parse_row(line, 0))
        return users''',
    # 8: package layout (structural; module unchanged)
    "",
    # 9: Dockerfile (separate; module unchanged)
    "",
    # 10: GitHub Actions workflow (separate; module unchanged)
    "",
    # 11: lightweight benchmark script
    '''import time

def benchmark_load(path: Path, rows: int = 10_000) -> float:
    start = time.perf_counter()
    _ = UserRepository(path).load()
    return time.perf_counter() - start''',
    # 12: documentation (module unchanged)
    "",
    # 13: changelog entry (module unchanged)
    "",
    # 14: harden against malformed JSONL with strict mode
    '''def load_strict(self) -> list[User]:
    users: list[User] = []
    errors: list[str] = []
    with self.path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                users.append(_parse_row(line, line_no))
            except UserSchemaError as exc:
                errors.append(str(exc))
    if errors:
        raise UserSchemaError(f"{len(errors)} bad rows: {errors[:3]}")
    return users''',
    # 15: in-memory metrics emission
    '''@dataclass
class Metrics:
    loads: int = 0
    parse_errors: int = 0
    summary_ms: float = 0.0

    def record_load(self, n: int) -> None:
        self.loads += n''',
    # 16: observability trace id
    '''import uuid

@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def bind(self, **fields: object) -> dict[str, object]:
        return {"trace_id": self.trace_id, **fields}''',
    # 17: migration path from the old dict API
    '''def legacy_load(path: Path) -> list[dict]:
    """Backwards-compatible dict-based loader for old callers."""
    return [vars(u) for u in UserRepository(path).load()]''',
    # 18: release notes (module unchanged)
    "",
    # 19: final cleanup pass
    '''def active_count(users: list[User]) -> int:
    return sum(1 for u in users if u.active)''',
    # 20: streaming variant
    '''def stream_users(path: Path) -> Iterator[User]:
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            yield _parse_row(line, line_no)''',
    # 21: retry logic with exponential backoff
    '''import time

def load_with_retry(path: Path, retries: int = 3) -> list[User]:
    for attempt in range(retries):
        try:
            return UserRepository(path).load()
        except OSError:
            time.sleep(2 ** attempt)
    raise OSError(f"failed after {retries} retries")''',
    # 22: localization for user-facing errors
    '''class Localizer:
    def __init__(self, locale: str = "en") -> None:
        self.locale = locale

    def error(self, key: str) -> str:
        return {"missing_file": "input file not found"}.get(key, key)''',
    # 23: Counter-based summary optimization
    '''from collections import Counter

def summarize(users: list[User]) -> dict[str, int | bool]:
    counts = Counter(u.active for u in users)
    return {"count": len(users), "active": counts.get(True, 0)}''',
    # 24: plugin hook for external validators
    '''_validators: list[Callable[[User], None]] = []

def register_validator(fn: Callable[[User], None]) -> None:
    _validators.append(fn)''',
    # 25: config validation that fails fast
    '''def validate_config(cfg: Config) -> None:
    if cfg.input_path is None:
        raise ValueError("input_path is required")''',
    # 26: group helpers (module unchanged)
    "",
    # 27: architecture diagram (module unchanged)
    "",
    # 28: final replay test (module unchanged)
    "",
    # 29: final summary (module unchanged)
    "",
]


LONG_REFACTOR_INSTRUCTIONS = [
    "Expose the public API via `__all__` and add a `__main__` guard that loads users.jsonl and prints the summary.",
    "Add a `UserSchemaError` and a `_parse_row` helper that validates each JSONL row and raises on missing id/name.",
    "Introduce a `Config` dataclass loaded from environment variables with a `dry_run` flag.",
    "Refactor the summarizer into a `SummarizerService` with dependency injection so tests can swap in a fake repository.",
    "Add structured logging so each step emits a compact JSON event object.",
    "Please add a CLI entry point that accepts `--input`, `--output`, and `--dry-run`.",
    "Now add a pytest suite that covers happy path, missing file, invalid JSONL, and dry-run behavior.",
    "Add async support with `aiofiles` for the repository and a small async wrapper around the CLI path.",
    "Refactor the code into a package layout with `src/`, `tests/`, and `pyproject.toml`.",
    "Add a Dockerfile that installs the package and runs the CLI against a mounted input file.",
    "Add a GitHub Actions workflow that runs formatting, linting, and tests on push.",
    "Add a lightweight benchmark script that measures repository load time and summary latency on a 10k-row fixture.",
    "Add documentation for the package: usage, config, CLI flags, and a short architecture note.",
    "Add a changelog entry for the refactor and a release checklist.",
    "Harden the repository against malformed JSONL rows by adding row-level error reporting and a `strict` mode.",
    "Add metrics emission for load count, parse errors, and summary duration using a simple in-memory metrics object.",
    "Add observability hooks so the service can export a trace id and propagate it through logs.",
    "Add a migration path from the old dict-based API to the new typed API.",
    "Add release notes that explain the new data model, CLI, and async support.",
    "Do a final cleanup pass: remove dead code, tighten type hints, and make the package easier to navigate.",
    "Add a streaming API variant that yields processed users one at a time instead of loading everything into memory.",
    "Add retry logic around file reads with exponential backoff and a max retry count.",
    "Add localization support for user-facing CLI errors and show how the code chooses a locale.",
    "Add a performance optimization for the summary step by using `collections.Counter` and avoiding repeated scans where possible.",
    "Add a plugin hook that allows external validators to be registered and run during repository loading.",
    "Add config validation so invalid environment variables fail fast with clear messages.",
    "Add a final refactor that groups related helpers into small modules without changing behavior.",
    "Add a short architecture diagram in text form and explain how data flows from input to output.",
    "Add a final test that simulates a 30-turn conversation by replaying the refactor steps against the package.",
    "Finish by summarizing the refactor, listing the remaining risks, and suggesting the next production hardening step.",
]

LONG_REFACTOR_TASKS = _build_long_tasks(LONG_REFACTOR_INSTRUCTIONS, BASE_REFACTOR_CODE, REFACTOR_STEPS)


BASE_DEBUG_CODE = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

class Item(BaseModel):
    name: str
    quantity: int

@app.post("/items")
def create_item(item: Item):
    if item.quantity < 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    return {"name": item.name, "total": item.quantity * 10}
"""

# One code delta per DEBUG_LONG_INSTRUCTIONS entry. Empty = artifact outside the
# module (tests, CLI, docs, diagram); the pasted module is unchanged that turn.
DEBUG_STEPS = [
    # 0: diagnose the IndexError -> safe indexing helper
    '''def _safe_get(items: list[Item], i: int) -> Item | None:
    if not items:
        return None
    return items[i % len(items)]''',
    # 1: fix off-by-one + guard empty input
    '''def _normalize(items: list[Item]) -> list[Item]:
    if not items:
        return []
    return items[:-1]''',
    # 2: validation for malformed records
    '''class ItemSchemaError(ValueError):
    """Raised when a record fails validation."""

def _validate(item: Item) -> None:
    if item.quantity < 0:
        raise ItemSchemaError("quantity must be non-negative")''',
    # 3: logging around the failing section
    '''def log_failure(step: str, **fields: object) -> None:
    logger.error(json.dumps({"step": step, **fields}))''',
    # 4: retry wrapper around the read path
    '''import time

def read_with_retry(path: str, retries: int = 3) -> list[Item]:
    for attempt in range(retries):
        try:
            return _read(path)
        except OSError:
            time.sleep(2 ** attempt)
    raise OSError(f"read failed after {retries} retries")''',
    # 5: reusable error-handling helper
    '''def handle_error(exc: Exception) -> dict[str, str]:
    return {"error": type(exc).__name__, "detail": str(exc)}''',
    # 6: unit test for the retry path (separate file; module unchanged)
    "",
    # 7: consistent error shape for validation failures
    '''from pydantic import BaseModel as _BM

class ErrorResponse(_BM):
    error: str
    detail: str''',
    # 8: CLI smoke test (separate; module unchanged)
    "",
    # 9: separate pure logic from FastAPI plumbing
    '''def parse_items(raw: list[dict]) -> list[Item]:
    return [Item(**r) for r in raw]''',
    # 10: timeout around the read path
    '''READ_TIMEOUT = 5.0

def read_with_timeout(path: str) -> list[Item]:
    with timeout(READ_TIMEOUT):
        return _read(path)''',
    # 11: metrics object
    '''@dataclass
class Metrics:
    successes: int = 0
    failures: int = 0

    def record(self, ok: bool) -> None:
        if ok:
            self.successes += 1
        else:
            self.failures += 1''',
    # 12: structured logging + trace id
    '''import uuid

def bind_trace() -> str:
    return uuid.uuid4().hex''',
    # 13: harden against oversized payloads
    '''MAX_PAYLOAD_BYTES = 1_000_000

def _check_size(payload: bytes) -> None:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")''',
    # 14: compatibility shim for old dict format
    '''def legacy_create(payload: dict) -> dict:
    item = Item(name=payload["name"], quantity=payload.get("qty", 0))
    return create_item(item)''',
    # 15: exponential backoff with jitter
    '''import random

def read_with_jitter(path: str, retries: int = 3) -> list[Item]:
    for attempt in range(retries):
        try:
            return _read(path)
        except OSError:
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))  # noqa: B608
    raise OSError("read failed")''',
    # 16: test rejects negative quantities (separate; module unchanged)
    "",
    # 17: benchmark fixture
    '''def benchmark_parse(rows: int = 10_000) -> float:
    import time
    start = time.perf_counter()
    _ = parse_items([{"name": "x", "quantity": 1} for _ in range(rows)])
    return time.perf_counter() - start''',
    # 18: documentation (module unchanged)
    "",
    # 19: final cleanup pass
    '''def active_count(items: list[Item]) -> int:
    return sum(1 for i in items if i.quantity > 0)''',
    # 20: streaming variant
    '''def stream_items(raw: list[dict]) -> Iterator[Item]:
    for r in raw:
        yield Item(**r)''',
    # 21: localization for user-facing errors
    '''class Localizer:
    def error(self, key: str) -> str:
        return {"negative_qty": "quantity must be positive"}.get(key, key)''',
    # 22: plugin hook for custom validators
    '''_validators: list[Callable[[Item], None]] = []

def register_validator(fn: Callable[[Item], None]) -> None:
    _validators.append(fn)''',
    # 23: config validation that fails fast
    '''def validate_config(cfg: dict) -> None:
    if "secret" not in cfg:
        raise ValueError("secret is required")''',
    # 24: group helpers (module unchanged)
    "",
    # 25: architecture diagram (module unchanged)
    "",
    # 26: replay test (module unchanged)
    "",
    # 27: summarize the fix (module unchanged)
    "",
    # 28: observability hook
    '''def record_duration(trace_id: str, ms: float) -> None:
    logger.info(json.dumps({"trace_id": trace_id, "duration_ms": ms}))''',
    # 29: release note (module unchanged)
    "",
]


DEBUG_LONG_INSTRUCTIONS = [
    "Add a safe indexing helper `_safe_get` that wraps list access and avoids the IndexError.",
    "Fix the off-by-one in the list handling and guard against empty input with a `_normalize` helper.",
    "Add an `ItemSchemaError` and a `_validate` helper for malformed records.",
    "Add logging around the failing section via a `log_failure` helper.",
    "Add a retry wrapper `read_with_retry` around the file read path and keep the public API stable.",
    "Refactor the error handling into a small `handle_error` helper that can be reused across endpoints.",
    "Add a unit test for the retry path using a fake file object.",
    "Make the API return a consistent `ErrorResponse` shape for validation failures.",
    "Add a CLI smoke test that exercises the failing path end to end.",
    "Refactor the module so the pure logic is separated from FastAPI plumbing via `parse_items`.",
    "Add a timeout `read_with_timeout` around the file read path and document the tradeoff.",
    "Introduce a small `Metrics` object that counts successful and failed parses.",
    "Add structured logging for every request and include the trace id in the response via `bind_trace`.",
    "Harden the endpoint against oversized payloads with a `_check_size` guard without changing the happy path.",
    "Add a compatibility shim `legacy_create` for clients that still send the old dict format.",
    "Refactor the retry logic to use exponential backoff with jitter.",
    "Add a test that verifies the endpoint rejects negative quantities.",
    "Add a small benchmark fixture `benchmark_parse` that measures the debug path on 10k rows.",
    "Add documentation for the new error contract and retry behavior.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming variant `stream_items` that yields parsed items one at a time.",
    "Add localization support for user-facing error messages via a `Localizer`.",
    "Add a plugin hook `register_validator` for custom validators that can be registered at startup.",
    "Add config validation so bad environment variables fail fast.",
    "Add a final refactor that groups related helpers into small modules.",
    "Add a short architecture diagram in text form and explain the data flow.",
    "Add a final test that replays the debug session against the package.",
    "Finish by summarizing the fix, remaining risks, and next hardening step.",
    "Add a small observability hook `record_duration` that exports request duration and parse errors.",
    "Add a final release note that explains the bug fix and the new safeguards.",
]


BASE_FEATURE_CODE = """from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt

app = FastAPI()
SECRET_KEY = "dev-secret"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(payload: LoginRequest) -> TokenResponse:
    if payload.password != "secret":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": payload.username, "exp": datetime.utcnow() + timedelta(hours=1)}, SECRET_KEY, algorithm="HS256")
    return TokenResponse(access_token=token)
"""

# One code delta per FEATURE_LONG_INSTRUCTIONS entry. Empty = artifact outside
# the module (tests, Dockerfile, CI, docs, changelog); module unchanged that turn.
FEATURE_STEPS = [
    # 0: API shape + data models
    '''class RefreshRequest(BaseModel):
    refresh_token: str''',
    # 1: token creation helper
    '''def create_token(sub: str) -> str:
    return jwt.encode({"sub": sub, "exp": datetime.utcnow() + timedelta(hours=1)}, SECRET_KEY, algorithm="HS256")''',
    # 2: config object for JWT settings
    '''@dataclass(slots=True)
class JWTConfig:
    secret: str = "dev-secret"
    token_ttl_hours: int = 1''',
    # 3: auth service with injectable signer
    '''class AuthService:
    def __init__(self, config: JWTConfig):
        self.config = config

    def login(self, username: str, password: str) -> TokenResponse:
        if password != "secret":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return TokenResponse(access_token=create_token(username))''',
    # 4: rate limiting
    '''class RateLimiter:
    def __init__(self, max_hits: int = 5, window: int = 60) -> None:
        self.max_hits = max_hits
        self.window = window

    def allow(self, key: str) -> bool:
        return True  # placeholder for a real token-bucket''',
    # 5: dependency-injection layer
    '''def get_auth_service() -> AuthService:
    return AuthService(JWTConfig())''',
    # 6: test suite (separate file; module unchanged)
    "",
    # 7: async support around refresh
    '''async def refresh(payload: RefreshRequest) -> TokenResponse:
    return TokenResponse(access_token=create_token("user"))''',
    # 8: package layout (structural; module unchanged)
    "",
    # 9: Dockerfile (separate; module unchanged)
    "",
    # 10: CI workflow (separate; module unchanged)
    "",
    # 11: benchmark for login throughput
    '''def benchmark_login(users: int = 1_000) -> float:
    import time
    svc = AuthService(JWTConfig())
    start = time.perf_counter()
    for i in range(users):
        svc.login(f"u{i}", "secret")
    return time.perf_counter() - start''',
    # 12: documentation (module unchanged)
    "",
    # 13: changelog (module unchanged)
    "",
    # 14: harden malformed/oversized requests
    '''def _check_request(payload: LoginRequest) -> None:
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="missing fields")''',
    # 15: metrics for login outcomes
    '''@dataclass
class AuthMetrics:
    success: int = 0
    failure: int = 0
    rate_limited: int = 0''',
    # 16: observability trace id
    '''import uuid

def bind_trace() -> str:
    return uuid.uuid4().hex''',
    # 17: migration from session-cookie flow
    '''def legacy_session_login(cookie: str) -> TokenResponse:
    return TokenResponse(access_token=create_token(cookie))''',
    # 18: release notes (module unchanged)
    "",
    # 19: final cleanup pass
    '''def active_sessions(tokens: list[str]) -> int:
    return len([t for t in tokens if t])''',
    # 20: streaming token refresh
    '''def stream_refresh(tokens: list[str]) -> Iterator[TokenResponse]:
    for t in tokens:
        yield TokenResponse(access_token=t)''',
    # 21: retry around external validation
    '''import time

def validate_external_with_retry(token: str, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            return _validate_external(token)
        except OSError:
            time.sleep(2 ** attempt)
    return False''',
    # 22: localization for auth errors
    '''class Localizer:
    def error(self, key: str) -> str:
        return {"invalid_credentials": "Invalid credentials"}.get(key, key)''',
    # 23: LRU cache for token validation
    '''from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_validate(token: str) -> bool:
    return bool(token)''',
    # 24: plugin hook for custom backends
    '''_backends: list[Callable[[str, str], TokenResponse]] = []

def register_backend(fn: Callable[[str, str], TokenResponse]) -> None:
    _backends.append(fn)''',
    # 25: config validation that fails fast
    '''def validate_jwt_config(cfg: JWTConfig) -> None:
    if not cfg.secret:
        raise ValueError("JWT secret is required")''',
    # 26: group helpers (module unchanged)
    "",
    # 27: architecture diagram (module unchanged)
    "",
    # 28: final replay test (module unchanged)
    "",
    # 29: summarize feature (module unchanged)
    "",
]


FEATURE_LONG_INSTRUCTIONS = [
    "Add a `RefreshRequest` model for token refresh.",
    "Extract a `create_token` helper that mints a JWT for a given subject.",
    "Introduce a `JWTConfig` dataclass for the JWT settings.",
    "Refactor token creation into an `AuthService` with an injectable signer.",
    "Add a `RateLimiter` to the login endpoint without changing the response shape.",
    "Add a dependency-injection layer `get_auth_service` so the auth service can use a fake repository.",
    "Add a test suite for login success, invalid credentials, and expired tokens.",
    "Add async support around the token refresh path while preserving sync behavior.",
    "Refactor the package layout so auth logic lives in its own module.",
    "Add a Dockerfile that runs the API and mounts a config file.",
    "Add a CI workflow that runs lint, type checks, and auth tests.",
    "Add a lightweight benchmark `benchmark_login` for login throughput on a warm token cache.",
    "Add documentation for the auth API, config, and token lifecycle.",
    "Add a changelog entry for the authentication feature.",
    "Harden the endpoint against malformed payloads and oversized requests with `_check_request`.",
    "Add `AuthMetrics` for login success, failure, and rate-limit hits.",
    "Add observability hooks that propagate a trace id through auth responses via `bind_trace`.",
    "Add a migration path `legacy_session_login` from the old session cookie flow to bearer tokens.",
    "Add release notes that explain the new auth flow and breaking changes.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming token refresh endpoint `stream_refresh` that yields refresh events.",
    "Add retry logic `validate_external_with_retry` around external token validation with bounded backoff.",
    "Add localization support for user-facing auth errors via a `Localizer`.",
    "Add a performance optimization for token validation using an LRU cache `cached_validate`.",
    "Add a plugin hook `register_backend` for custom auth backends.",
    "Add config validation `validate_jwt_config` so bad JWT settings fail fast.",
    "Add a final refactor that groups related auth helpers into small modules.",
    "Add a short architecture diagram in text form and explain the auth flow.",
    "Add a final test that simulates the full feature conversation against the package.",
    "Finish by summarizing the feature, remaining risks, and next hardening step.",
]


BASE_DEFAULT_CODE = """from typing import Iterator

def fibonacci(n: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    values = [0, 1]
    for _ in range(2, n):
        values.append(values[-1] + values[-2])
    return values

def fibonacci_gen(n: int) -> Iterator[int]:
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
"""

# Coherent base modules for the *short* (non-long) scenarios, used as the
# read_file tool result so the agent payload stays coherent with the task the
# user actually pastes. The long scenarios paste their own cumulative module,
# so they do not use these. Each mirrors the code the corresponding short
# scenario's first user message shows.
SHORT_DEBUG_CODE = """def process_items(items):
    result = []
    for i in range(len(items)):
        result.append(items[i + 1])
    return result
"""

SHORT_REFACTOR_CODE = """def calculate_stats(data):
    total = 0
    count = 0
    for item in data:
        total += item
        count += 1
    avg = total / count

    variance = 0
    for item in data:
        variance += (item - avg) ** 2
    std = variance / count

    return avg, std
"""

# One code delta per DEFAULT_LONG_INSTRUCTIONS entry. Empty = artifact outside
# the module (tests, Dockerfile, CI, docs, changelog); module unchanged that turn.
DEFAULT_STEPS = [
    # 0: explain iterative + tradeoffs (docstring)
    '''def fibonacci(n: int) -> list[int]:
    """Iterative O(n) time, O(n) space Fibonacci."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    values = [0, 1]
    for _ in range(2, n):
        values.append(values[-1] + values[-2])
    return values''',
    # 1: refactor into a generator (already present; add note)
    '''def fibonacci_gen(n: int) -> Iterator[int]:
    """Yield Fibonacci numbers one at a time (O(1) space)."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b''',
    # 2: type hints + docstring (already applied above)
    "",
    # 3: config object
    '''@dataclass(slots=True)
class GeneratorConfig:
    count: int = 10
    format: str = "list"''',
    # 4: service class with DI
    '''class FibService:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def run(self) -> list[int]:
        return fibonacci(self.config.count)''',
    # 5: structured logging
    '''import logging

logger = logging.getLogger("fib")

def log_yield(value: int) -> None:
    logger.debug(json.dumps({"value": value}))''',
    # 6: CLI entry point
    '''import argparse

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fibonacci generator")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--format", default="list")
    return p''',
    # 7: pytest suite (separate file; module unchanged)
    "",
    # 8: async support
    '''import asyncio

async def afibonacci_gen(n: int) -> list[int]:
    return list(fibonacci_gen(n))''',
    # 9: package layout (structural; module unchanged)
    "",
    # 10: Dockerfile (separate; module unchanged)
    "",
    # 11: CI workflow (separate; module unchanged)
    "",
    # 12: benchmark throughput
    '''def benchmark_gen(n: int = 100_000) -> float:
    import time
    start = time.perf_counter()
    _ = list(fibonacci_gen(n))
    return time.perf_counter() - start''',
    # 13: documentation (module unchanged)
    "",
    # 14: changelog (module unchanged)
    "",
    # 15: harden CLI against invalid args
    '''def _parse_args(argv: list[str]) -> GeneratorConfig:
    args = build_cli().parse_args(argv)
    if args.count < 0:
        raise ValueError("count must be non-negative")
    return GeneratorConfig(count=args.count, format=args.format)''',
    # 16: metrics
    '''@dataclass
class GenMetrics:
    count: int = 0
    duration_ms: float = 0.0''',
    # 17: observability trace id
    '''import uuid

def bind_trace() -> str:
    return uuid.uuid4().hex''',
    # 18: migration from the old list API
    '''def legacy_fibonacci(n: int) -> list[int]:
    return fibonacci(n)''',
    # 19: release notes (module unchanged)
    "",
    # 20: streaming API
    '''def stream_fib(n: int) -> Iterator[int]:
    yield from fibonacci_gen(n)''',
    # 21: retry around config loading
    '''import time

def load_config_with_retry(path: str, retries: int = 3) -> GeneratorConfig:
    for attempt in range(retries):
        try:
            return _load_config(path)
        except OSError:
            time.sleep(2 ** attempt)
    raise OSError("config load failed")''',
    # 22: localization
    '''class Localizer:
    def error(self, key: str) -> str:
        return {"bad_count": "count must be non-negative"}.get(key, key)''',
    # 23: rolling-pair performance optimization
    '''def fibonacci(n: int) -> list[int]:
    """O(n) time, O(1) space using a rolling pair."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    a, b = 0, 1
    out = [a]
    for _ in range(1, n):
        a, b = b, a + b
        out.append(a)
    return out''',
    # 24: plugin hook for formatters
    '''_formatters: dict[str, Callable[[list[int]], str]] = {}

def register_formatter(name: str, fn: Callable[[list[int]], str]) -> None:
    _formatters[name] = fn''',
    # 25: config validation
    '''def validate_config(cfg: GeneratorConfig) -> None:
    if cfg.count < 0:
        raise ValueError("count must be non-negative")''',
    # 26: group helpers (module unchanged)
    "",
    # 27: architecture diagram (module unchanged)
    "",
    # 28: final replay test (module unchanged)
    "",
    # 29: summarize (module unchanged)
    "",
]


DEFAULT_LONG_INSTRUCTIONS = [
    "Document the iterative `fibonacci` with its O(n) time / O(n) space tradeoffs.",
    "Document the `fibonacci_gen` generator as O(1) space, yielding one value at a time.",
    "Add type hints and a small docstring to the helpers while preserving the public API.",
    "Introduce a `GeneratorConfig` that controls the sequence length and output format.",
    "Refactor the generator into a `FibService` with dependency injection for tests.",
    "Add structured logging around each yielded value via `log_yield` and keep the output stable.",
    "Add a CLI entry point that accepts `--count` and `--format`.",
    "Add a pytest suite for zero, one, many, and negative inputs.",
    "Add async support with a small `afibonacci_gen` wrapper around the generator.",
    "Refactor the code into a package layout with `src/`, `tests/`, and `pyproject.toml`.",
    "Add a Dockerfile that runs the CLI against a mounted config file.",
    "Add a GitHub Actions workflow that runs formatting, linting, and tests.",
    "Add a lightweight benchmark `benchmark_gen` that measures generator throughput on large n.",
    "Add documentation for usage, config, and the generator contract.",
    "Add a changelog entry for the generator refactor and CLI.",
    "Harden the CLI against invalid arguments and malformed config files via `_parse_args`.",
    "Add `GenMetrics` for count, duration, and yielded values.",
    "Add observability hooks that propagate a trace id through CLI output via `bind_trace`.",
    "Add a migration path `legacy_fibonacci` from the old list API to the new generator API.",
    "Add release notes that explain the new generator and CLI behavior.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming API variant `stream_fib` that yields formatted lines one at a time.",
    "Add retry logic `load_config_with_retry` around config loading with bounded backoff.",
    "Add localization support for user-facing CLI errors via a `Localizer`.",
    "Add a performance optimization for large n using a rolling pair in `fibonacci`.",
    "Add a plugin hook `register_formatter` for custom formatters.",
    "Add config validation `validate_config` so invalid settings fail fast.",
    "Add a final refactor that groups related helpers into small modules.",
    "Add a short architecture diagram in text form and explain the data flow.",
    "Finish by summarizing the refactor, remaining risks, and next hardening step.",
]


DEBUG_LONG_TASKS = _build_long_tasks(DEBUG_LONG_INSTRUCTIONS, BASE_DEBUG_CODE, DEBUG_STEPS)
FEATURE_LONG_TASKS = _build_long_tasks(FEATURE_LONG_INSTRUCTIONS, BASE_FEATURE_CODE, FEATURE_STEPS)
DEFAULT_LONG_TASKS = _build_long_tasks(DEFAULT_LONG_INSTRUCTIONS, BASE_DEFAULT_CODE, DEFAULT_STEPS)


# Cached fixture loader (loaded by file path so a missing fixture package can
# never break the benchmark module import). Reused by both the `opencode`/
# `fixtures` scenario builder and the synthetic scenario's realistic tool outputs.
_FIXTURE_LOADER: Any = None
_FIXTURE_LOADER_LOADED = False


def _get_fixture_loader() -> Any:
    """Return the fixture loader module, loading it once (cached). None on failure."""
    global _FIXTURE_LOADER, _FIXTURE_LOADER_LOADED
    if _FIXTURE_LOADER_LOADED:
        return _FIXTURE_LOADER
    _FIXTURE_LOADER_LOADED = True
    try:
        import importlib.util
        from pathlib import Path

        loader_path = Path(__file__).resolve().parent / "fixtures" / "loader.py"
        spec = importlib.util.spec_from_file_location("benchmark_fixtures_loader", loader_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FIXTURE_LOADER = module
    except Exception:
        _FIXTURE_LOADER = None
    return _FIXTURE_LOADER


def _build_opencode_scenario_tasks() -> list[list[dict]]:
    """Build the `opencode` (and `fixtures`) scenario: OpenCode-style replay.

    Each turn is a realistic agent payload — user task plus assistant tool_calls
    and the real tool outputs (file contents, test results) read from
    scripts/fixtures/. The `fixtures` scenario key is an alias of this builder.
    """
    loader = _get_fixture_loader()
    if loader is not None:
        try:
            return loader.build_fixture_agentic_tasks()
        except Exception:
            pass
    return [
        [
            {
                "role": "user",
                "content": "Build a small JSONL-backed user-analytics service. Start with the model.",
            }
        ]
    ]


# ---------------------------------------------------------------------------
# Real-life coding scenarios for benchmarking
# ---------------------------------------------------------------------------

# The OpenCode-harness replay is identical for both the `fixtures` and
# `opencode` scenario keys, so build it once at import instead of twice.
_OPENCODE_SCENARIO_TASKS = _build_opencode_scenario_tasks()

# Facts planted in Turn 1 of every scenario (drift measurement). The probe turn
# (last turn) asks the model to list them; fact recall measures how many
# survived the proxy's compaction by Turn N. Kept short, distinct, checkable.
_DRIFT_FACTS: list[str] = [
    "The project codename is ATLAS.",
    "We target Python 3.11.",
    "The database is Postgres.",
    "The max retry count is 3.",
    "The owning team is platform-infra.",
]

_DRIFT_PLANT = (
    "Context anchor — record these fixed project facts, we will refer back to "
    "them later:\n" + "\n".join(f"- {f}" for f in _DRIFT_FACTS)
)
_DRIFT_PROBE = (
    "Going back to the very first turn of this conversation: list the fixed facts "
    "we established about the project (codename, language, database, max retries, "
    "owning team). State each one explicitly."
)


def _inject_drift_probe(tasks: list, num_turns: int) -> list:
    """Inject the drift measurement into an existing scenario's task list.

    - Prepends the fact-planting anchor to the FIRST user turn (Turn 1), so the
      proxy must carry those facts forward through compaction. The anchor goes in
      Turn-1 *user* content (never the system prompt) to keep the cache-stable
      frozen prefix untouched.
    - Appends a recall probe as the FINAL turn, asking the model to list the
      planted facts. The probe lands on the last turn regardless of how many
      turns the runner cycles, because we size the returned list to ``num_turns``.

    Accepts either simple ("role", "content") tuples or OpenCode-style
    list[dict] exchanges and returns the same shape, extended to ``num_turns``.
    """
    if not tasks:
        return tasks

    # ── Prepend the anchor to the first user turn ────────────────────────
    first = tasks[0]
    if isinstance(first, tuple):
        role, content = first
        if role == "user":
            tasks = [("user", f"{_DRIFT_PLANT}\n\n{content}", *first[2:]), *tasks[1:]]
    elif isinstance(first, list):
        # OpenCode-style exchange: find the first user message and extend it.
        for msg in first:
            if msg.get("role") == "user":
                msg["content"] = f"{_DRIFT_PLANT}\n\n{msg['content']}"
                break

    # ── Append the recall probe as the final turn ───────────────────────
    probe_exchange: list[dict] = [{"role": "user", "content": _DRIFT_PROBE}]
    if len(tasks) >= num_turns:
        # Already long enough: replace the last turn with the probe so it always
        # lands on Turn N (the runner cycles by modulo, so the tail is what the
        # final turn sees).
        tasks = [*tasks[: num_turns - 1], probe_exchange]
    else:
        tasks = list(tasks) + [probe_exchange] * (num_turns - len(tasks) + 1)
    return tasks[:num_turns]


SCENARIOS = {
    "debug": {
        "description": "Debugging session with error analysis",
        "tasks": [
            ("user", "I have a Python function that's throwing an IndexError. Here's the code:\n\n```python\ndef process_items(items):\n    result = []\n    for i in range(len(items)):\n        result.append(items[i+1])\n    return result\n```\n\nWhat's wrong?"),
            ("user", "I fixed the index but now I'm getting a different error. The function returns None instead of the list. Why?"),
            ("user", "Now I need to add error handling for empty input. How should I do it?"),
        ],
    },
    "debug_long": {
        "description": "Long real-life debug conversation with 30 unique turns and code blocks",
        "tasks": [("user", task) for task in DEBUG_LONG_TASKS],
    },
    "refactor": {
        "description": "Code refactoring session",
        "tasks": [
            ("user", "Here's a function I want to refactor for better performance:\n\n```python\ndef calculate_stats(data):\n    total = 0\n    count = 0\n    for item in data:\n        total += item\n        count += 1\n    avg = total / count\n    \n    variance = 0\n    for item in data:\n        variance += (item - avg) ** 2\n    std = variance / count\n    \n    return avg, std\n```\n\nMake it more efficient."),
            ("user", "Can you add type hints and make it a class?"),
            ("user", "Add caching for repeated calls with the same data."),
        ],
    },
    "refactor_long": {
        "description": "Long real-life refactor conversation with 30 unique turns and code blocks",
        "tasks": [("user", task) for task in LONG_REFACTOR_TASKS],
    },
    "feature": {
        "description": "Feature implementation session",
        "tasks": [
            ("user", "I need to implement a REST API endpoint for user authentication. What's the best approach?"),
            ("user", "Write the FastAPI endpoint with JWT tokens."),
            ("user", "Add rate limiting to prevent brute force attacks."),
            ("user", "Add unit tests for the authentication endpoint."),
        ],
    },
    "feature_long": {
        "description": "Long real-life feature conversation with 30 unique turns and code blocks",
        "tasks": [("user", task) for task in FEATURE_LONG_TASKS],
    },
    "default": {
        "description": "General coding conversation",
        "tasks": [
            ("user", "Write a Python function to compute Fibonacci numbers iteratively and return them as a list."),
            ("user", "Now refactor it to use a generator instead of building a list."),
            ("user", "Add type hints and docstrings to the generator."),
            ("user", "Add a CLI entry point that accepts a --count argument and prints the sequence."),
        ],
    },
    "default_long": {
        "description": "Long general coding conversation with 30 unique turns and code blocks",
        "tasks": [("user", task) for task in DEFAULT_LONG_TASKS],
    },
    "fixtures": {
        "description": "OpenCode-harness replay from scripts/fixtures/ (alias of opencode: user task + tool calls + real tool outputs)",
        "tasks": _OPENCODE_SCENARIO_TASKS,
    },
    "opencode": {
        "description": "OpenCode-harness replay from scripts/fixtures/ (user task + tool calls + real tool outputs)",
        "tasks": _OPENCODE_SCENARIO_TASKS,
    },
}

# "all" scenario is handled specially - runs all individual scenarios

# ---------------------------------------------------------------------------
# Proxy management
# ---------------------------------------------------------------------------

_PROXY_PROCESS: subprocess.Popen | None = None
_HUMAN_OUTPUT_TO_STDERR = False


def _human_print(*parts: object) -> None:
    print(*parts, file=sys.stderr if _HUMAN_OUTPUT_TO_STDERR else sys.stdout)


def _proxy_is_running(port: int, timeout: float = 3.0) -> bool:
    """Check if the proxy is already listening on *port*."""
    try:
        import urllib.request

        url = f"http://127.0.0.1:{port}/v1/health"
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


def _start_proxy(port: int, wait: float = 60.0) -> subprocess.Popen | None:
    """Start the moeptimizer proxy as a background process and wait for it to be ready.

    Returns the Popen object on success, or *None* if the proxy was already running
    or failed to start.
    """
    global _PROXY_PROCESS

    # If already running, just verify and return None (we don't own it)
    if _proxy_is_running(port):
        _human_print(f"  Proxy already running on port {port}")
        return None

    _human_print(f"  Starting moeptimizer proxy on port {port} ...")
    env = os.environ.copy()
    # Pass through config env vars so the started process picks up the same settings.
    # If --port differs from MOEPT_PORT/default, force the child proxy to bind there.
    env["MOEPT_PORT"] = str(port)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "moeptimizer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        _PROXY_PROCESS = proc
    except OSError as e:
        _human_print(f"  ERROR: could not start proxy: {e}")
        return None

    # Wait for the health endpoint to become available
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _proxy_is_running(port):
            _human_print(f"  Proxy ready on port {port}")
            return proc
        time.sleep(0.5)

    # Give it a moment to flush startup logs
    stdout = ""
    try:
        stdout, _ = proc.communicate(timeout=2)
    except Exception:
        proc.kill()
        stdout, _ = proc.communicate()
    _human_print(f"  ERROR: proxy failed to start within {wait}s (exit={proc.returncode})")
    if stdout:
        for line in stdout.decode("utf-8", errors="replace").strip().splitlines()[-10:]:
            print(f"    | {line}")
    _PROXY_PROCESS = None
    return None


def _stop_proxy() -> None:
    """Stop the proxy if we started it."""
    global _PROXY_PROCESS
    proc = _PROXY_PROCESS
    _PROXY_PROCESS = None
    if proc is not None and proc.poll() is None:
        _human_print("  Stopping benchmark proxy ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _apply_profile_overrides(args: argparse.Namespace) -> None:
    """Apply benchmark-level context optimization profile overrides."""
    # Map benchmark profile names to the proxy's quality_profile presets so the
    # started proxy uses the matching preset (review03.md §10).
    profile_env = {
        "quality": {
            "MOEPT_AGENTIC__QUALITY_PROFILE": "quality",
            "MOEPT_AGENTIC__KEEP_FULL_STEPS": "6",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS": "24000",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS": "6000",
            "MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO": "0.7",
            "MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO": "0.9",
            "MOEPT_AGENTIC__HIERARCHICAL_SUMMARY_ENABLED": "false",
            "MOEPT_AGENTIC__RAG_ENABLED": "false",
            "MOEPT_AGENTIC__CODE_SKELETON_ENABLED": "false",
        },
        "aggressive": {
            "MOEPT_AGENTIC__QUALITY_PROFILE": "aggressive",
            "MOEPT_AGENTIC__KEEP_FULL_STEPS": "2",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS": "8000",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS": "2000",
            "MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO": "0.35",
            "MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO": "0.6",
        },
        "balanced": {
            "MOEPT_AGENTIC__QUALITY_PROFILE": "balanced",
            "MOEPT_AGENTIC__KEEP_FULL_STEPS": "3",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS": "12000",
            "MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS": "3000",
            "MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO": "0.45",
            "MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO": "0.75",
        },
    }
    overrides = profile_env.get(args.profile)
    if not overrides:
        return
    for key, value in overrides.items():
        os.environ.setdefault(key, value)

    _status(args, f"  Context profile: {args.profile}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request_body(
    messages: list[dict],
    max_tokens: int = 8192,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    session_id: str | None = None,
    stream: bool = False,
) -> dict:
    """Build the OpenAI-compatible chat/completions request body.

    Shared by the non-streaming and streaming paths so both send identical
    payloads (same messages/tools/session) — only ``stream`` differs.
    """
    body: dict = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if session_id:
        body["_session_id"] = session_id
    return body


def _request(url: str, body: dict, timeout: float = 180.0) -> tuple[dict, float, dict[str, str]]:
    """Send a POST request and return (response_json, elapsed_ms, headers)."""
    import requests

    t0 = time.monotonic()
    resp = requests.post(url, json=body, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = (resp.text or "")[:1000]
        raise requests.HTTPError(f"{e}: {detail}") from e
    elapsed_ms = (time.monotonic() - t0) * 1000
    return resp.json(), elapsed_ms, dict(resp.headers)


def _stream_request(
    url: str, body: dict, timeout: float = 180.0
) -> tuple[str, str, dict, float | None, float, dict[str, str], int, list[dict] | None]:
    """Streaming POST for TTFT measurement.

    Returns
        (content, reasoning_content, usage, ttft_ms, elapsed_ms, headers,
         prefix_cache_hit_tokens, tool_calls)

    Reconstructs the assistant message from SSE deltas. The proxy surfaces its
    authoritative prefix-cache hit count as an SSE comment
    (``: X-Prefix-Cache-Hit-Tokens: N``) and the optimized prompt token count as
    a response header; Lemonade/Direct surfaces ``usage`` (incl. cached_tokens)
    on the final usage chunk when ``stream_options.include_usage`` is set.
    """
    import requests

    stream_body = dict(body)
    stream_body["stream"] = True
    stream_opts = dict(stream_body.get("stream_options") or {})
    stream_opts["include_usage"] = True
    stream_body["stream_options"] = stream_opts

    t0 = time.monotonic()
    ttft_ms: float | None = None
    resp = requests.post(url, json=stream_body, timeout=timeout, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = (resp.text or "")[:1000]
        raise requests.HTTPError(f"{e}: {detail}") from e

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    usage: dict = {}
    prefix_cache_hit_tokens = 0
    headers = dict(resp.headers)

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith(":"):
            # SSE comment — proxy emits `: X-Prefix-Cache-Hit-Tokens: N`
            m = re.search(r"X-Prefix-Cache-Hit-Tokens:\s*(\d+)", raw_line)
            if m:
                prefix_cache_hit_tokens = int(m.group(1))
            continue
        if not raw_line.startswith("data:"):
            continue
        data = raw_line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        # Truthy check (not just `is not None`): some servers emit an opening
        # chunk with `content: ""` (e.g. the proxy's initial SSE chunk), which
        # would otherwise be mis-counted as the first generated token and make
        # TTFT collapse to ~0ms. Only non-empty content starts the TTFT clock.
        # For reasoning-heavy models (Qwen3.6-MTP) the first emitted token is
        # often a `reasoning_content` delta, so we also start the clock on the
        # first non-empty reasoning delta — otherwise proxy TTFT is never
        # recorded (the model emits only reasoning before any final `content`,
        # leaving ttft_ms=None and the dashboard showing an empty proxy series).
        if ttft_ms is None and (
            delta.get("content")
            or (delta.get("reasoning_content") is not None and delta.get("reasoning_content") != "")
        ):
            ttft_ms = (time.monotonic() - t0) * 1000
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content") is not None:
            reasoning_parts.append(delta["reasoning_content"])
        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0))
            acc = tool_calls_acc.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            if tc.get("id"):
                acc["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                acc["name"] = fn["name"]
            if fn.get("arguments"):
                acc["arguments"] += fn["arguments"]

    elapsed_ms = (time.monotonic() - t0) * 1000
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    tool_calls: list[dict] | None = None
    if tool_calls_acc:
        tool_calls = []
        for idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[idx]
            tool_calls.append(
                {
                    "id": acc["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
            )
    return content, reasoning, usage, ttft_ms, elapsed_ms, headers, prefix_cache_hit_tokens, tool_calls


def _message_text(content: Any) -> str:
    """Return message content as text for local token estimation."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content)


def _serialize_messages_text(messages: list[dict]) -> str:
    """Render a message list as plain text (role + content), newline-joined.

    Mirrors the proxy's X-MOEPT-Optimized-Prompt-Text serialization so the
    benchmark can compare the FULL pre-optimization prompt against the
    optimized prompt the proxy actually sent (prompt-faithfulness).
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = _message_text(content)
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts)


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate prompt tokens when the backend omits usage.prompt_tokens."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            content = _message_text(msg.get("content", ""))
            if content.strip():
                total += len(encoder.encode(content))
            total += 4
        return max(total, 1)
    except Exception:
        total = 0
        for msg in messages:
            content = _message_text(msg.get("content", ""))
            if content.strip():
                total += max(1, len(content.strip()) // 4)
            total += 4
        return max(total, 1)


def _context_size_summary(messages: list[dict]) -> dict[str, int]:
    """Return lightweight context-size metrics for benchmark progress logs."""
    chars = sum(len(_message_text(msg.get("content", ""))) for msg in messages)
    return {
        "messages": len(messages),
        "chars": chars,
        "estimated_tokens": _estimate_prompt_tokens(messages),
    }


def _looks_like_cached_response(usage: dict[str, Any]) -> bool:
    """Detect responses where usage is cache-driven or incomplete.

    A *cached* response is one the backend reports with ``cached_tokens > 0`` in
    ``prompt_tokens_details`` (the authoritative prefix-cache signal). The old
    heuristic ``prompt_tokens == 0 and completion_tokens > 0`` was a false-hit:
    many backends report a normal, non-cached response with ``prompt_tokens > 0``
    and ``completion_tokens > 0``, and the zero-prompt case is really "usage
    missing/zeroed" rather than "cached". We now require a positive
    ``cached_tokens`` signal, and only treat zero-prompt as cached when the
    backend clearly still produced output (completion_tokens > 0) — i.e. usage
    was omitted, not that the turn was served from cache.
    """
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(
        details.get("cached_tokens", 0)
        or usage.get("cached_tokens", 0)
        or usage.get("cache_hit_tokens", 0)
        or 0
    )
    if cached_tokens > 0:
        return True
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    # Usage omitted/zeroed but a completion was returned: treat as missing-usage,
    # not as a cache hit. This avoids mislabeling normal responses as cached.
    return prompt_tokens == 0 and completion_tokens > 0


def _resolve_prompt_tokens(
    raw_prompt_tokens: int,
    messages: list[dict],
    usage: dict[str, Any] | None = None,
    optimized_prompt_tokens: int = 0,
) -> tuple[int, str, bool]:
    """Return (effective_prompt_tokens, source, cached_response).

    Some Lemonade/cache responses can omit or zero out usage.prompt_tokens even
    when a completion was returned. Keep raw usage visible in the dataclass, but
    use an estimated prompt count for context-window and token-savings metrics so
    a cached/missing-usage response does not collapse final_prompt_tokens to 0.
    """
    if optimized_prompt_tokens > 0:
        return optimized_prompt_tokens, "optimized_header", _looks_like_cached_response(usage or {})
    if raw_prompt_tokens > 0:
        return raw_prompt_tokens, "usage", _looks_like_cached_response(usage or {})
    if messages:
        return _estimate_prompt_tokens(messages), "estimated_missing_usage", _looks_like_cached_response(usage or {})
    return 0, "missing_messages", False


def _calculate_timeout(turns: int, rounds: int) -> float:
    """Calculate the per-request timeout based on expected context growth.

    Each turn appends a user task plus (in agentic mode) assistant tool_calls
    and tool results, so the context -- and thus per-request latency -- grows
    roughly linearly with the number of turns. We scale the base timeout by
    ``1 + turns * 0.15`` (about +15% per turn) and cap it at 300s, so a
    10-turn run already reaches the cap and longer runs stay bounded.
    ``rounds`` is accepted for API symmetry but does not change the per-request
    timeout.
    """
    base_timeout = 120.0  # 2 minutes in seconds
    context_growth_factor = 1 + (turns * 0.15)  # ~15% more headroom per turn
    return min(300.0, base_timeout * context_growth_factor)


def _direct_request(messages: list[dict], max_tokens: int = 8192, timeout: float = 180.0, tools: list[dict] | None = None, temperature: float = 0.0) -> tuple[dict, float, dict[str, str]]:
    url = f"{LEMONADE_URL}/chat/completions"
    body = _build_request_body(messages, max_tokens=max_tokens, tools=tools, temperature=temperature)
    return _request(url, body, timeout)


def _proxy_request(
    messages: list[dict], session_id: str | None = None, max_tokens: int = 8192, timeout: float = 180.0, tools: list[dict] | None = None, temperature: float = 0.0
) -> tuple[dict, float, dict[str, str]]:
    url = f"http://127.0.0.1:{MOEPT_PORT}/v1/chat/completions"
    body = _build_request_body(messages, max_tokens=max_tokens, tools=tools, temperature=temperature, session_id=session_id)
    return _request(url, body, timeout)


def _direct_stream_request(messages: list[dict], max_tokens: int = 8192, timeout: float = 180.0, tools: list[dict] | None = None, temperature: float = 0.0):
    url = f"{LEMONADE_URL}/chat/completions"
    body = _build_request_body(messages, max_tokens=max_tokens, tools=tools, temperature=temperature)
    return _stream_request(url, body, timeout)


def _proxy_stream_request(
    messages: list[dict], session_id: str | None = None, max_tokens: int = 8192, timeout: float = 180.0, tools: list[dict] | None = None, temperature: float = 0.0,
):
    url = f"http://127.0.0.1:{MOEPT_PORT}/v1/chat/completions"
    body = _build_request_body(messages, max_tokens=max_tokens, tools=tools, temperature=temperature, session_id=session_id)
    return _stream_request(url, body, timeout)


def _reset_proxy_metrics(port: int) -> None:
    """Reset the proxy's process-wide metrics counters for per-round isolation."""
    import requests

    try:
        requests.post(f"http://127.0.0.1:{port}/v1/metrics/reset", timeout=10.0)
    except Exception:
        pass


def _fetch_proxy_metrics(port: int) -> dict | None:
    """Fetch the proxy's process-wide metrics snapshot, or None on failure."""
    import requests

    try:
        resp = requests.get(f"http://127.0.0.1:{port}/v1/metrics", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _warm_up_backend(timeout: float = 60.0) -> None:
    """Send one throwaway request to the backend so both benchmark conversations
    start against a warm model (loaded weights, warm KV cache) rather than the
    proxy conversation paying the full cold-start cost while the direct
    conversation rides a warm backend.

    This keeps each side a complete, contiguous, sorted conversation (we never
    interleave direct and proxified requests) while removing the systematic
    cold-start advantage the direct run would otherwise enjoy. Failures are
    ignored — warm-up is best-effort and the benchmark still runs without it.
    """
    import requests

    try:
        body = _build_request_body(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=8,
        )
        requests.post(f"{LEMONADE_URL}/chat/completions", json=body, timeout=timeout)
    except Exception:
        pass


def _check_foreign_markers(content: str) -> list[str]:
    """Return any internal markers that leaked into the response."""
    forbidden = ["[ARCHIVED", "[REASONING", "[PROGRESS", "[LOOP DETECTED"]
    return [m for m in forbidden if m in content]


def _embed_text(text: str, model: str | None = None, timeout: float = 30.0) -> list[float]:
    """Get embedding vector via the proxy's /v1/embeddings endpoint."""
    import requests

    embed_model = model or os.environ.get(
        "MOEPT_SERVER__EMBED_MODEL", "embed-gemma-300m-FLM"
    )
    resp = requests.post(
        f"http://127.0.0.1:{MOEPT_PORT}/v1/embeddings",
        json={"model": embed_model, "input": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = (sum(x * x for x in a) ** 0.5) or 1e-9
    norm_b = (sum(x * x for x in b) ** 0.5) or 1e-9
    return round(dot / (norm_a * norm_b), 6)


def _token_jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between token sets (word-level)."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return round(intersection / max(union, 1), 6)


def _lcs_len(words_a: list[str], words_b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Space-optimized DP (last two rows). Shared by the ROUGE-L helpers so the
    LCS table is computed in exactly one place.
    """
    m, n = len(words_a), len(words_b)
    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_a[i - 1] == words_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr[:], [0] * (n + 1)
    return prev[n]


def _rouge_l(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 score (longest common subsequence)."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    m, n = len(words_a), len(words_b)

    if m == 0 or n == 0:
        return 0.0

    lcs_len = _lcs_len(words_a, words_b)
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(f1, 6)


def _length_ratio(direct_content: str, proxy_content: str) -> float:
    """Ratio of proxy length to direct length. 1.0 = identical length. <1.0 = truncation, >1.0 = verbosity."""
    d_len = len(direct_content) or 1
    p_len = len(proxy_content) or 1
    return round(p_len / d_len, 4)


def _rouge_l_precision_recall(text_a: str, text_b: str) -> dict[str, float]:
    """Compute ROUGE-L precision and recall separately (longest common subsequence)."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    m, n = len(words_a), len(words_b)

    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0}

    lcs_len = _lcs_len(words_a, words_b)
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6)}


def _code_syntax_validity(content: str) -> float:
    """Fraction of fenced ```python code blocks in *content* that parse with ast.parse.

    Returns 1.0 when there are no python code blocks (nothing to validate). This
    is a hard correctness signal the embedding/lexical metrics cannot capture:
    the proxy must not emit syntactically broken code as a side effect of
    optimization (boundary compression, summarization, eviction).
    """
    import re

    blocks = re.findall(r"```(?:python|py)\n(.*?)```", content, re.DOTALL)
    if not blocks:
        return 1.0
    valid = 0
    for block in blocks:
        try:
            ast.parse(block)
            valid += 1
        except SyntaxError:
            pass
    return round(valid / len(blocks), 6)


# Tree-sitter based code-block comparison (language-agnostic).
#
# We use tree-sitter to *parse* each fenced code block with the grammar that
# matches its fence language tag (python, js, bash, json, go, rust, …). The
# grammar is loaded by name from tree-sitter-language-pack — no hardcoded
# per-language rules. From the parse tree we build a structural fingerprint
# that is invariant to whitespace/formatting but still distinguishes a renamed
# symbol or a dropped statement. For languages tree-sitter cannot parse (or
# parse errors / unknown fence tags) we fall back to a whitespace-normalized
# exact-text signature so the metric never crashes and still handles prose.
_TS_PARSER = None  # lazily set to a tree_sitter.Parser, or False if unavailable
_TS_MOD = None      # the single imported tree_sitter module instance
_TS_LP = None       # the single imported tree_sitter_language_pack module instance
_TS_LANG_CACHE: dict[str, object] = {}

# Curated fence-tag aliases -> tree-sitter language-pack names. These are
# overlaid on top of the grammars shipped by tree-sitter-language-pack so common
# shorthand tags (py, js, sh, tsx, …) resolve correctly. The base set of valid
# grammar ids is derived dynamically from the pack (see _build_fence_lang_map),
# so this list only needs to cover *aliases*, not every language.
_FENCE_LANG_ALIASES = {
    "py": "python", "py3": "python", "python": "python",
    "js": "javascript", "jsx": "javascript", "node": "javascript", "javascript": "javascript",
    "ts": "typescript", "tsx": "tsx", "typescript": "typescript",
    "sh": "bash", "shell": "bash", "zsh": "zsh", "bash": "bash",
    "yml": "yaml", "yaml": "yaml",
    "golang": "go", "go": "go",
    "cs": "csharp", "c#": "csharp", "csharp": "csharp",
    "kt": "kotlin", "kotlin": "kotlin",
    "rs": "rust", "rust": "rust",
    "rb": "ruby", "rake": "ruby", "ruby": "ruby",
    "pl": "perl", "pm": "perl",
    "hs": "haskell", "lisp": "commonlisp", "el": "elisp",
    "erl": "erlang", "ex": "elixir", "exs": "elixir",
    "ml": "ocaml", "sc": "scala", "scala": "scala",
    "md": "markdown", "docker": "dockerfile", "tf": "terraform",
    "sol": "solidity", "gql": "graphql", "graphql": "graphql",
    "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hxx": "cpp", "cpp": "cpp",
    "h": "c", "c": "c",
    "objc": "objc", "objectivec": "objc",
    "clj": "clojure", "jl": "julia", "nim": "nim", "dart": "dart", "swift": "swift",
    "groovy": "groovy", "r": "r", "lua": "lua", "sql": "sql", "zig": "zig",
    "vue": "vue", "svelte": "svelte", "proto": "proto",
    "make": "make", "cmake": "cmake", "toml": "toml", "ini": "ini",
    "json5": "json5", "jsonc": "json", "json": "json",
    "xml": "xml", "htm": "html", "html": "html", "css": "css",
    "sass": "scss", "less": "less",
    "php3": "php", "php4": "php", "php5": "php", "php7": "php", "php8": "php", "php": "php",
    "asm": "asm", "nasm": "nasm", "x86asm": "x86asm",
    "bat": "batch", "ps1": "powershell",
}

# Static fallback subset, used only if tree-sitter-language-pack is unavailable.
# Mirrors the proxy's fallback in src/moeptimizer/code_chunking.py.
_FENCE_LANG_FALLBACK = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "go": "go", "rust": "rust", "cpp": "cpp", "c": "c", "java": "java",
    "csharp": "csharp", "php": "php", "ruby": "ruby", "html": "html",
    "css": "css", "json": "json",
}


def _build_fence_lang_map() -> dict[str, str]:
    """Build the fence-tag -> grammar-id map, dynamically, like the proxy.

    The base is every grammar shipped by tree-sitter-language-pack (so the map
    can never reference a non-existent grammar), then curated aliases are
    layered on top. Falls back to a static subset if the pack is unavailable.
    This keeps the benchmark's language coverage in lock-step with the proxy
    (src/moeptimizer/code_chunking.py:_build_lang_map) — no hardcoded drift.
    """
    try:
        from tree_sitter_language_pack import manifest_languages

        base = {lang: lang for lang in manifest_languages()}
    except Exception:
        base = dict(_FENCE_LANG_FALLBACK)
    # Keep only aliases whose target grammar actually exists in the pack.
    base.update({k: v for k, v in _FENCE_LANG_ALIASES.items() if v in base})
    return base


_FENCE_LANG_MAP = _build_fence_lang_map()


def _ts_runtime():
    """Lazily import tree-sitter + language pack ONCE, reusing the same module
    instances.

    Returns ``(tree_sitter_module, language_pack_module, Parser)`` or ``None`` if
    tree-sitter is unavailable. Importing both modules a single time (rather than
    re-importing inside each call) is essential: a ``Parser`` and a ``Language``
    must come from the *same* ``tree_sitter`` C-extension instance, otherwise
    assigning ``parser.language = lang`` raises and the comparison silently
    falls back to text matching.
    """
    global _TS_PARSER, _TS_MOD, _TS_LP
    if _TS_PARSER is False:
        return None  # already known unavailable
    if _TS_PARSER is not None:
        return (_TS_MOD, _TS_LP, _TS_PARSER)
    try:
        import tree_sitter as _ts  # type: ignore
        import tree_sitter_language_pack as _lp  # type: ignore
        _TS_MOD = _ts
        _TS_LP = _lp
        _TS_PARSER = _ts.Parser()
        return (_ts, _lp, _TS_PARSER)
    except Exception:
        _TS_PARSER = False
        return None


def _ts_get_language(fence_tag: str):
    """Resolve a tree-sitter Language for a fence tag, or None if unsupported."""
    tag = (fence_tag or "").strip().lower()
    name = _FENCE_LANG_MAP.get(tag)
    if not name or name in _TS_LANG_CACHE and _TS_LANG_CACHE[name] is False:
        return None
    if name in _TS_LANG_CACHE:
        return _TS_LANG_CACHE[name]
    rt = _ts_runtime()
    if rt is None:
        _TS_LANG_CACHE[name] = False
        return None
    _ts, _lp, _parser = rt
    try:
        lang = _lp.get_language(name)
        _TS_LANG_CACHE[name] = lang
        return lang
    except Exception:
        _TS_LANG_CACHE[name] = False
        return None


def _code_block_fingerprint(block: str, fence_tag: str) -> str:
    """Structural fingerprint of a code block, language-agnostic.

    Walks the tree-sitter parse tree (grammar resolved by the fence tag, no
    hardcoded rules) and emits ``node_type`` tokens for the structure. Identifier
    handling is deliberately split:

    * **Local variable / reference / parameter names are anonymized** to ``_``
      — the model legitimately picks different local names across rounds, which
      is model phrasing variance, not optimizer damage, so it must not penalize
      the metric.
    * **Definition names** (function / class / method names exposed via the
      grammar's ``name`` field) are kept, so a *dropped* definition (e.g. direct
      emits ``def a`` and ``def b`` but the proxy drops ``def b``) is still
      detected instead of over-credited.

    Keywords, punctuation, operators, strings and numbers emit their node type
    only. The fingerprint is therefore invariant to whitespace/formatting and to
    local-variable renaming, while still catching dropped or structurally
    changed code.

    Falls back to a whitespace-normalized exact-text signature when tree-sitter
    cannot parse the block (unknown language, parse error, or no grammar).
    """
    lang = _ts_get_language(fence_tag)
    if lang is not None:
        rt = _ts_runtime()
        if rt is not None:
            _ts, _lp, _parser = rt
            try:
                _parser.language = lang
                tree = _parser.parse(block.encode("utf-8"))
                parts: list[str] = []

                def _walk(node):
                    # Skip comment nodes — they don't affect structure.
                    if node.type in ("comment",) or node.type.startswith("comment"):
                        return
                    if node.child_count == 0:
                        # Leaf node. Anonymize identifier names to "_" EXCEPT for
                        # declaration names (function/class/type names exposed via
                        # the grammar's ``name`` field). Rationale:
                        #   * Local variable / reference renames across rounds are
                        #     model phrasing variance, NOT optimizer damage — they
                        #     must NOT penalize the metric (name-invariant).
                        #   * Declaration names are stable per task; keeping them
                        #     lets us detect a *dropped* function/class (e.g. direct
                        #     emits ``def a`` and ``def b`` but proxy drops ``def b``)
                        #     instead of over-crediting it.
                        # Keywords, punctuation, operators, strings, numbers emit
                        # their node type only (literal text is irrelevant to
                        # structural preservation).
                        if node.type in ("identifier", "type_identifier", "property_identifier",
                                         "field_identifier", "shorthand_property_identifier"):
                            parent = node.parent
                            # tree-sitter nodes are not identity-stable across
                            # child_by_field_name() calls, so compare byte spans
                            # rather than using ``is``.
                            name_node = parent.child_by_field_name("name") if parent is not None else None
                            # Keep ONLY the names of definitions (functions,
                            # classes, methods) — these are stable per task, so a
                            # dropped definition is real loss we must detect.
                            # Local variable / parameter declarations (e.g.
                            # ``const x``, function params) are anonymized: the
                            # model freely renames locals across rounds, which is
                            # phrasing variance, not optimizer damage.
                            is_def_name = (
                                name_node is not None
                                and name_node.start_byte == node.start_byte
                                and name_node.end_byte == node.end_byte
                                and parent is not None
                                and (
                                    "definition" in parent.type
                                    or parent.type in (
                                        "function_declaration", "class_declaration",
                                        "method_definition", "generator_function_declaration",
                                        "generator_function_definition",
                                    )
                                )
                            )
                            if is_def_name:
                                text = block[node.start_byte:node.end_byte]
                                parts.append(f"{node.type}:{text}")
                            else:
                                parts.append(f"{node.type}:_")
                        else:
                            parts.append(node.type)
                    else:
                        parts.append(node.type)
                        for ch in node.children:
                            _walk(ch)

                _walk(tree.root_node)
                fp = " ".join(parts)
                if fp:
                    return fp
            except Exception:
                pass
    # Fallback: whitespace-normalized exact text. Coarser than the tree-sitter
    # fingerprint (it is rename-sensitive and formatting-normalized only), used
    # only when tree-sitter is unavailable or the fence tag is unknown.
    norm = re.sub(r"\s+", " ", block.strip())
    return norm


def _code_block_preserved(dblock: str, dtag: str, proxy_blocks: list[tuple[str, str]]) -> bool:
    """Decide whether a direct code block is preserved in the proxy response.

    Compares the structural, name-invariant :func:`_code_block_fingerprint` of
    the direct block (parsed with its own fence language) against each proxy
    block's fingerprint. Language-agnostic: the grammar is resolved by the fence
    tag via tree-sitter, with no hardcoded rules. Invariant to reformatting and
    to identifier renaming / literal changes (model phrasing variance, not
    optimizer damage); still catches dropped or structurally changed code.

    Falls back to an exact-substring check for trivially short blocks.
    """
    sig = _code_block_fingerprint(dblock, dtag)
    if not sig:
        return True  # empty / whitespace-only block: nothing to lose
    if any(_code_block_fingerprint(pb, ptag) == sig for pb, ptag in proxy_blocks):
        return True
    # Exact substring as a last resort for very short blocks.
    clean = re.sub(r"\s+", " ", dblock.strip()).strip()
    if len(clean) < 3:
        return True
    proxy_text = re.sub(r"\s+", " ", "".join(b for b, _ in proxy_blocks)).strip()
    return clean in proxy_text


def _code_block_preservation(direct_content: str, proxy_content: str) -> dict[str, float]:
    """Measure how many code blocks from the direct response are preserved in the proxy response.

    Returns dict with:
        block_ratio: fraction of direct's code blocks preserved in proxy
        has_code_direct: whether direct had any code blocks
        has_code_proxy: whether proxy had any code blocks

    Preservation is decided by :func:`_code_block_preserved`, which uses a
    tree-sitter structural fingerprint (language resolved from the fence tag,
    no hardcoded grammar) so it works for Python, JS, bash, JSON, etc. and is
    invariant to reformatting and to identifier renaming / literal changes
    (model phrasing variance, not optimizer damage) while still catching
    dropped or structurally changed code.
    """
    import re

    # Extract fenced code blocks WITH their language tag: ```lang ... ```
    fence_re = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    direct_blocks = [(m.group(2), m.group(1).strip()) for m in fence_re.finditer(direct_content)]
    proxy_blocks = [(m.group(2), m.group(1).strip()) for m in fence_re.finditer(proxy_content)]

    if not direct_blocks:
        return {"block_ratio": 1.0, "has_code_direct": False, "has_code_proxy": bool(proxy_blocks)}

    preserved = 0
    for dblock, dtag in direct_blocks:
        if _code_block_preserved(dblock, dtag, proxy_blocks):
            preserved += 1

    # If the proxy produced no extractable fenced blocks, fall back to a
    # content heuristic instead of assuming full preservation.
    if not proxy_blocks:
        proxy_has_code = any(
            kw in proxy_content
            for kw in ("def ", "class ", "import ", "return ", "if ", "for ", "while ")
        )
        if not proxy_has_code:
            # No fences and no code-like content: treat as loss.
            return {
                "block_ratio": 0.0,
                "has_code_direct": True,
                "has_code_proxy": False,
            }
        # Code-like content but no fences: score by structural fingerprint.
        preserved = 0
        for dblock, dtag in direct_blocks:
            sig = _code_block_fingerprint(dblock, dtag)
            if not sig or _code_block_fingerprint(proxy_content, dtag) == sig:
                preserved += 1

    return {
        "block_ratio": round(preserved / max(len(direct_blocks), 1), 6),
        "has_code_direct": True,
        "has_code_proxy": bool(proxy_blocks) or ("```" in proxy_content),
    }


def _markdown_structure_similarity(text_a: str, text_b: str) -> float:
    """Compare markdown structural elements between two texts.

    Counts headings (#), list markers (- / * / 1.), code fences (```), blockquotes (>),
    and returns Jaccard similarity of the structure signature vectors.
    """
    import re

    def _structure_sig(text: str) -> dict[str, int]:
        return {
            "headings": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
            "unordered_lists": len(re.findall(r"^\s*[-*]\s", text, re.MULTILINE)),
            "ordered_lists": len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE)),
            "code_fences": len(re.findall(r"^```", text, re.MULTILINE)),
            "blockquotes": len(re.findall(r"^\s*>", text, re.MULTILINE)),
        }

    sig_a = _structure_sig(text_a)
    sig_b = _structure_sig(text_b)

    all_keys = set(sig_a.keys()) | set(sig_b.keys())
    if not all_keys:
        return 1.0

    intersection = sum(min(sig_a.get(k, 0), sig_b.get(k, 0)) for k in all_keys)
    union = max(sum(max(sig_a.get(k, 0), sig_b.get(k, 0)) for k in all_keys), 1)
    return round(intersection / union, 6)


def _normalized_edit_similarity(text_a: str, text_b: str) -> float:
    """Compute normalized edit similarity using LCS ratio.

    Returns a value in [0, 1] where 1 means identical content.
    Uses the LCS length divided by max(len(a), len(b)).
    """
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    m, n = len(words_a), len(words_b)

    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_a[i - 1] == words_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr[:], [0] * (n + 1)

    lcs_len = prev[n]
    return round(lcs_len / max(m, n), 6)


def _vocabulary_richness_delta(text_a: str, text_b: str) -> float:
    """Difference in type-token ratio between two texts.

    TTR = unique_words / total_words. Measures vocabulary diversity.
    Returns absolute difference (0.0 = identical richness).
    """
    def _ttr(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    ttr_a = _ttr(text_a)
    ttr_b = _ttr(text_b)
    return round(abs(ttr_a - ttr_b), 6)


def _prompt_faithfulness(full_prompt: str, optimized_prompt: str) -> float | None:
    """Measure how much of the ORIGINAL context survived compaction.

    This is the DIRECT measure of the optimizer's one job (it compacts ONLY
    the input context). Unlike the response-vs-response overlap scores, it does
    not conflate optimizer quality with the model's phrasing variance.

    Returns token-set Jaccard between the full pre-optimization prompt and the
    optimized prompt the proxy actually sent. 1.0 = nothing lost; lower =
    more of the original context was dropped. None when either side is empty
    (e.g. turn 1, where there is nothing to compact yet).
    """
    if not full_prompt or not optimized_prompt:
        return None
    full_tokens = set(full_prompt.lower().split())
    opt_tokens = set(optimized_prompt.lower().split())
    if not full_tokens or not opt_tokens:
        return None
    intersection = len(full_tokens & opt_tokens)
    union = len(full_tokens | opt_tokens)
    return round(intersection / max(union, 1), 6)


def _evicted_content_recall(full_prompt: str, optimized_prompt: str) -> float | None:
    """Recall of content that lived ONLY in the evicted (early) part of the prompt.

    When the optimizer evicts early turns, the question is whether the
    SURVIVING optimized prompt still carries the key entities (file paths,
    function/identifier names, error strings) that originated in those evicted
    turns. We approximate "evicted content" as the tokens present in the full
    prompt but ABSENT from the most recent ~40% of it (i.e. the tail the
    optimizer always keeps), then measure what fraction of those tokens the
    optimized prompt retained.

    Returns a 0..1 recall score, or None when the prompt is too short to
    split meaningfully (nothing was evicted yet).
    """
    if not full_prompt or not optimized_prompt:
        return None
    full_tokens = full_prompt.lower().split()
    if len(full_tokens) < 40:
        return None
    # Tail = last 40% of the full prompt (the part the optimizer keeps verbatim).
    split = int(len(full_tokens) * 0.6)
    evicted_tokens = set(full_tokens[:split])
    if not evicted_tokens:
        return None
    opt_tokens = set(optimized_prompt.lower().split())
    retained = len(evicted_tokens & opt_tokens)
    return round(retained / max(len(evicted_tokens), 1), 6)


# ---------------------------------------------------------------------------
# Long-horizon / cross-turn quality metrics (drift, contradiction, wall)
# ---------------------------------------------------------------------------


def _grade_fact_recall(probe_response: str, facts: list[str]) -> float | None:
    """Grade how many planted facts the model still recalls at the probe turn.

    Used by the ``drift`` scenario: Turn 1 plants N explicit, checkable facts;
    the probe turn (last turn) asks the model to list them. Each fact is graded
    by embedding similarity between the fact and the response (the response need
    not quote the fact verbatim). Returns recall@probe in 0..1, or None when the
    response is empty.

    Embedding may be unavailable (proxy down); callers should treat None as
    "not measured" rather than a zero score.
    """
    if not probe_response or not facts:
        return None
    try:
        resp_emb = _embed_text(probe_response)
    except Exception:
        return None
    recalled = 0
    for fact in facts:
        try:
            fact_emb = _embed_text(fact)
        except Exception:
            continue
        # Threshold chosen so a clearly-relevant mention (not just shared stop
        # words) counts as recalled. Embeddings here are small (300d), so 0.35
        # is a conservative, low-false-positive cutoff.
        if _cosine_similarity(fact_emb, resp_emb) >= 0.35:
            recalled += 1
    return round(recalled / max(len(facts), 1), 4)


_NEGATION_MARKERS = ("not ", "never ", "no longer", "isn't", "is not", "aren't",
                     "are not", "don't", "do not", "doesn't", "does not", "won't",
                     "will not", "can't", "cannot", "shouldn't", "should not",
                     "instead of", "rather than", "contrary to")
_ASSERT_RE = re.compile(
    r"([A-Z][^.!?]*\b(?:is|are|was|were|must|should|will|always|never|means|equals|"
    r"requires|uses|runs on|codename|owner)\b[^.!?]*[.!?])",
    re.MULTILINE,
)


def _extract_assertions(text: str) -> list[str]:
    """Pull simple declarative assertions (sentences) from a response."""
    if not text:
        return []
    return [m.group(1).strip() for m in _ASSERT_RE.finditer(text)]


def _assertions_contradict(a: str, b: str) -> bool:
    """Heuristic: do two assertions contradict each other?

    Flags a contradiction when the same subject token appears in both but one
    carries a negation marker the other lacks. Deliberately conservative — it
    only fires on explicit negation flips, not on subtle semantic disagreement
    (which needs an LLM judge). Returns False on any parse failure.
    """
    try:
        a_low = a.lower()
        b_low = b.lower()
        a_neg = any(m in a_low for m in _NEGATION_MARKERS)
        b_neg = any(m in b_low for m in _NEGATION_MARKERS)
        if a_neg == b_neg:
            return False  # both negated or both affirmative -> not a flip
        # Find a shared content word (len >= 4) as a proxy for "same subject".
        a_words = {w for w in re.findall(r"[a-z]{4,}", a_low)}
        b_words = {w for w in re.findall(r"[a-z]{4,}", b_low)}
        shared = a_words & b_words
        # Require a meaningful shared subject, not just stop words.
        return len(shared) >= 2
    except Exception:
        return False


def _count_contradictions(contents: list[str]) -> int:
    """Count self-contradictions across a full conversation's responses.

    Compares each turn's assertions against the union of prior turns' assertions
    using a conservative negation-flip heuristic. This is a cheap, deterministic
    signal of context drift / memory loss (the model contradicts what it said
    earlier because the proxy dropped the earlier context). An LLM-judge path is
    intentionally out of scope here to keep the benchmark free of extra backend
    calls; the heuristic under-counts rather than over-counts, so it is a
    lower bound on the true contradiction rate.
    """
    prior_assertions: list[str] = []
    contradictions = 0
    for content in contents:
        cur = _extract_assertions(content)
        for assertion in cur:
            if any(_assertions_contradict(assertion, prev) for prev in prior_assertions):
                contradictions += 1
        prior_assertions.extend(cur)
    return contradictions


def _context_window_wall(turns: list["TurnComparison"]) -> dict[str, int | None]:
    """Find the first turn where quality collapses due to budget exhaustion.

    Derived purely from existing per-turn quality (no new requests). A "wall" is
    the first turn where the proxy's code preservation breaks down
    (code_block_ratio < 0.5) OR semantic fidelity collapses
    (semantic_similarity < 0.3). Returns the first such turn index per side, or
    None when no wall is reached within the run.
    """
    out: dict[str, int | None] = {"proxy": None, "direct": None}
    for side in ("proxy", "direct"):
        for t in turns:
            q = t.quality
            if not q:
                continue
            ratio = q.get("code_block_ratio")
            sim = q.get("semantic_similarity")
            hit = (ratio is not None and ratio < 0.5) or (sim is not None and sim < 0.3)
            if hit:
                out[side] = t.turn_index
                break
    return out


def _compute_quality_metrics(
    direct_content: str,
    proxy_content: str,
    full_prompt: str = "",
    optimized_prompt: str = "",
) -> dict[str, float]:
    """Compute quality comparison metrics between two responses."""
    metrics = {}

    # ── Content overlap (existing) ────────────────────────────────────
    metrics["token_jaccard"] = _token_jaccard(direct_content, proxy_content)
    rouge = _rouge_l_precision_recall(direct_content, proxy_content)
    metrics["rouge_l_f1"] = round(2 * rouge["precision"] * rouge["recall"] / (rouge["precision"] + rouge["recall"]) if (rouge["precision"] + rouge["recall"]) > 0 else 0.0, 6)
    metrics["rouge_l_precision"] = rouge["precision"]
    metrics["rouge_l_recall"] = rouge["recall"]

    # ── Character-level n-gram overlap ────────────────────────────────
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        text = text.lower().replace("\n", " ").replace("\r", "")
        return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()

    direct_bigrams = _char_ngrams(direct_content, 3)
    proxy_bigrams = _char_ngrams(proxy_content, 3)
    if direct_bigrams and proxy_bigrams:
        metrics["trigram_overlap"] = round(
            len(direct_bigrams & proxy_bigrams) / max(len(direct_bigrams | proxy_bigrams), 1), 6
        )

    # ── Length ratio (catches truncation / verbosity inflation) ───────
    metrics["length_ratio"] = _length_ratio(direct_content, proxy_content)

    # ── Edit similarity (word-level LCS ratio) ────────────────────────
    metrics["edit_similarity"] = _normalized_edit_similarity(direct_content, proxy_content)

    # ── Code block preservation ───────────────────────────────────────
    code = _code_block_preservation(direct_content, proxy_content)
    metrics["code_block_ratio"] = code["block_ratio"]
    metrics["has_code_direct"] = 1.0 if code["has_code_direct"] else 0.0
    metrics["has_code_proxy"] = 1.0 if code["has_code_proxy"] else 0.0

    # ── Syntactic validity of generated python code (hard correctness) ─
    # The proxy's emitted code must still parse; lexical/embedding metrics
    # cannot catch a broken `def`/`class` introduced by optimization.
    metrics["code_syntax_validity"] = _code_syntax_validity(proxy_content)

    # ── Markdown structure similarity ─────────────────────────────────
    metrics["markdown_structure_similarity"] = _markdown_structure_similarity(direct_content, proxy_content)

    # ── Vocabulary richness delta (higher = more divergent word usage) ─
    metrics["vocabulary_richness_delta"] = _vocabulary_richness_delta(direct_content, proxy_content)

    # ── Semantic similarity via embeddings (may fail if proxy not available) ─
    try:
        emb_direct = _embed_text(direct_content)
        emb_proxy = _embed_text(proxy_content)
        metrics["semantic_similarity"] = _cosine_similarity(emb_direct, emb_proxy)
    except Exception:
        metrics["semantic_similarity"] = None

    # ── Response structure / code-structure consistency ──────────────────
    # These are computed from the response content to assess how closely the
    # proxy's response preserves the direct response's structure (code-fence
    # counts, reasoning markers, and code keywords) — not MTP-specific.
    metrics["response_stability"] = _assess_response_stability(direct_content, proxy_content)
    metrics["code_structure_consistency"] = _assess_code_structure_consistency(direct_content, proxy_content)

    # ── Context faithfulness (the optimizer's ACTUAL job) ──────────
    # These measure how much of the ORIGINAL input context survived
    # compaction, NOT how similarly the model phrased its answer. They are
    # the primary quality signal for a context optimizer; the response-vs-
    # response overlap scores above are only informational for this use case.
    metrics["prompt_faithfulness"] = _prompt_faithfulness(full_prompt, optimized_prompt)
    metrics["evicted_content_recall"] = _evicted_content_recall(full_prompt, optimized_prompt)

    return metrics


def _assess_response_stability(direct_content: str, proxy_content: str) -> float:
    """Assess response-structure stability.

    Compares the structure and flow of responses to detect proxy-induced
    disruption. High similarity = stable structure. This looks at code-fence
    counts and reasoning markers; it is a generic structure signal, not an
    MTP-specific measurement.
    """
    # Check for consistent code block structure
    import re

    direct_code_blocks = len(re.findall(r"```", direct_content))
    proxy_code_blocks = len(re.findall(r"```", proxy_content))

    # Check for consistent reasoning patterns
    direct_thoughts = len(re.findall(r"<thought>|<\/thought>", direct_content, re.IGNORECASE))
    proxy_thoughts = len(re.findall(r"<thought>|<\/thought>", proxy_content, re.IGNORECASE))

    # Normalize to 0-1 score
    code_score = 1.0 if direct_code_blocks == 0 else min(1.0, proxy_code_blocks / direct_code_blocks)
    thought_score = 1.0 if direct_thoughts == 0 else min(1.0, proxy_thoughts / direct_thoughts)

    return round((code_score + thought_score) / 2, 4)


def _assess_code_structure_consistency(direct_content: str, proxy_content: str) -> float:
    """Assess code-structure consistency between responses.

    Checks if code structure and formatting are preserved (keyword sets).
    """
    import re

    # Extract code from both responses
    code_re = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
    direct_code = code_re.findall(direct_content)
    proxy_code = code_re.findall(proxy_content)

    if not direct_code:
        return 1.0

    # Check if code structure keywords are preserved
    keywords = ["def ", "class ", "import ", "return ", "if ", "for ", "while "]
    direct_keywords = set()
    proxy_keywords = set()

    for code in direct_code:
        for kw in keywords:
            if kw in code:
                direct_keywords.add(kw)

    for code in proxy_code:
        for kw in keywords:
            if kw in code:
                proxy_keywords.add(kw)

    if not direct_keywords:
        return 1.0

    # Jaccard similarity of keywords
    intersection = len(direct_keywords & proxy_keywords)
    union = len(direct_keywords | proxy_keywords)
    return round(intersection / max(union, 1), 4)


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


@dataclass
class TurnMetrics:
    """Per-turn metrics for one side (direct or proxy)."""

    turn_index: int = 0
    total_turns_at_request: int = 0
    prompt_tokens: int = 0
    raw_prompt_tokens: int = 0
    optimized_prompt_tokens: int = 0
    prompt_tokens_source: str = "usage"
    cached_response: bool = False
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_rate: float | None = None  # None when no real cached-token signal
    latency_ms: float = 0.0
    response_chars: int = 0
    finish_reason: str = ""
    foreign_markers: list[str] = field(default_factory=list)
    error: str | None = None
    content_preview: str = ""  # First 200 chars for dump
    chars_before_optimization: int = 0  # Total chars in messages before proxy optimization
    raw_input_tokens: int = 0  # Token count of the proxy's RAW input (messages before optimization). The true baseline for measuring compaction: savings = (raw_input_tokens - optimized_prompt_tokens) / raw_input_tokens.
    optimized_prompt_text: str = ""  # Plain-text optimized prompt the proxy sent to the backend (X-MOEPT-Optimized-Prompt-Text), for faithfulness measurement
    full_prompt_text: str = ""  # Plain-text FULL prompt BEFORE optimization (the optimizer's input), for faithfulness measurement
    prefix_cache_hit_tokens: int = 0  # Proxy's authoritative prefix-cache hit count (X-Prefix-Cache-Hit-Tokens)
    proxy_process_ms: float | None = None  # Proxy's own optimization/forwarding overhead (X-Proxy-Process-Ms), if emitted
    ttft_ms: float | None = None  # Time to first token (streaming / --measure-ttft path only)


@dataclass
class TurnComparison:
    """Side-by-side metrics for one turn."""

    turn_index: int = 0
    round_index: int = 0  # which benchmark round this turn belongs to
    direct: TurnMetrics = field(default_factory=TurnMetrics)
    proxy: TurnMetrics = field(default_factory=TurnMetrics)
    latency_delta_ms: float = 0.0  # proxy - direct (positive = slower)
    token_delta: int = 0  # proxy prompt - direct prompt
    quality: dict[str, float | None] = field(default_factory=dict)
    quality_computed: bool = True  # False when one side errored/empty and quality was skipped


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    config: dict = field(default_factory=dict)
    turns: list[TurnComparison] = field(default_factory=list)
    cache_reuse: list[dict] = field(default_factory=list)  # per-round proxy /v1/metrics snapshots
    # Long-horizon cross-turn signals (computed post-hoc, not per-turn).
    contradictions: dict[str, int] = field(default_factory=dict)  # {"proxy": int, "direct": int}
    fact_recall: dict[str, float | None] = field(default_factory=dict)  # {"proxy": float|None, "direct": float|None}
    context_window_wall: dict[str, int | None] = field(default_factory=dict)  # {"proxy": int|None, "direct": int|None}

    def summary(self) -> dict[str, Any]:
        """Return a flat summary dict for JSON output."""
        n = len(self.turns)
        if not self.turns:
            return {"error": "no data"}

        # Direct turns whose usage was omitted/zeroed (prompt_tokens_source ==
        # "estimated_missing_usage") were served from the backend's prefix cache
        # and the benchmark cannot time a real generation — their latency_ms is
        # just the ~4ms round-trip, not a comparable generation time. Including
        # them in the latency comparison understates the proxy's true penalty, so
        # we exclude them from latency stats/deltas and report the count.
        _artifact_turns = [
            t.turn_index
            for t in self.turns
            if t.direct.prompt_tokens_source == "estimated_missing_usage"
        ]
        _timed_turns = [
            t for t in self.turns if t.turn_index not in _artifact_turns
        ]

        direct_latencies = [t.direct.latency_ms for t in _timed_turns]
        proxy_latencies = [t.proxy.latency_ms for t in _timed_turns]
        latency_deltas = [t.latency_delta_ms for t in _timed_turns]

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            s = sorted(values)
            return {
                "mean": round(statistics.mean(s), 2),
                "median": round(statistics.median(s), 2),
                "p90": _percentile(s, 90),
                "p95": _percentile(s, 95),
                "p99": _percentile(s, 99),
                "min": round(min(s), 2),
                "max": round(max(s), 2),
            }

        direct_tokens = [t.direct.prompt_tokens for t in self.turns]
        proxy_tokens = [t.proxy.prompt_tokens for t in self.turns]
        raw_input_tokens = [t.proxy.raw_input_tokens for t in self.turns]
        cached = [t.proxy.cached_tokens for t in self.turns]

        total_direct_prompt = sum(direct_tokens)
        total_proxy_prompt = sum(proxy_tokens)
        total_raw_input = sum(raw_input_tokens)
        total_cached = sum(cached)
        tokens_saved_pct = (
            round((total_direct_prompt - total_proxy_prompt) / max(total_direct_prompt, 1) * 100, 2)
            if total_direct_prompt > 0
            else 0.0
        )
        # Savings measured against the proxy's RAW input (what it received), not
        # the direct path. This isolates the optimizer's own compaction from
        # differences in how the direct vs proxy paths are tokenized, and avoids
        # masking real compaction with role-tag overhead on short turns.
        tokens_saved_vs_raw_pct = (
            round((total_raw_input - total_proxy_prompt) / max(total_raw_input, 1) * 100, 2)
            if total_raw_input > 0
            else 0.0
        )

        # Context window growth: prompt_tokens at each turn vs theoretical full context
        final_turn = self.turns[-1]
        max_context_window = int(self.config.get("context_window", 262144))
        final_proxy_prompt_tokens = final_turn.proxy.prompt_tokens
        final_proxy_ctx_pct = round(final_proxy_prompt_tokens / max_context_window * 100, 2)

        # Completion-token sums (for cost modeling).
        direct_completion = [t.direct.completion_tokens for t in self.turns]
        proxy_completion = [t.proxy.completion_tokens for t in self.turns]
        total_direct_completion = sum(direct_completion)
        total_proxy_completion = sum(proxy_completion)

        # ── Quality metrics aggregation ───────────────────────────────
        # HEADLINE = the signals that actually validate a CONTEXT OPTIMIZER in
        #   agentic coding. The optimizer compacts ONLY the input context, so the
        #   primary question is "did the compressed prompt retain what the agent
        #   needed?" — answered by prompt_faithfulness / evicted_content_recall
        #   (how much of the original context survived) and code_syntax_validity
        #   (the one hard correctness check on emitted code).
        # SECONDARY = response-vs-response overlap scores. These measure how
        #   similarly the MODEL phrased its answer, which conflates optimizer
        #   quality with LLM phrasing variance and has near-zero discriminative
        #   power here (review: rouge/jaccard/edit are informational only, like
        #   semantic_similarity already is).
        # length_ratio is NOT a degradation signal: the proxy cannot control
        #   response verbosity, so its verbosity_count is mis-attributed and is
        #   reported only as an informational "model verbosity delta".
        headline_quality_metrics = [
            "prompt_faithfulness", "evicted_content_recall",
            "code_syntax_validity", "code_block_ratio",
        ]
        secondary_quality_metrics = [
            "rouge_l_f1", "token_jaccard", "edit_similarity",
            "trigram_overlap", "markdown_structure_similarity", "vocabulary_richness_delta",
            "rouge_l_precision", "rouge_l_recall", "response_stability",
            "code_structure_consistency", "has_code_direct", "has_code_proxy",
            "length_ratio",
        ]

        def _aggregate_quality(metric_names: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for qm in metric_names:
                values = [t.quality.get(qm) for t in self.turns if t.quality and t.quality.get(qm) is not None]
                if values:
                    s = sorted(values)
                    out[qm] = {
                        "mean": round(statistics.mean(s), 4),
                        "median": round(statistics.median(s), 4),
                        "min": round(min(s), 4),
                        "max": round(max(s), 4),
                    }
                else:
                    out[qm] = None
            return out

        quality_summary = _aggregate_quality(headline_quality_metrics)
        secondary_quality_summary = _aggregate_quality(secondary_quality_metrics)

        # semantic_similarity (informational only — weak on code)
        sem_values = [t.quality.get("semantic_similarity") for t in self.turns if t.quality and t.quality.get("semantic_similarity") is not None]
        semantic_summary = (
            {
                "mean": round(statistics.mean(sem_values), 4),
                "median": round(statistics.median(sem_values), 4),
                "min": round(min(sem_values), 4),
                "max": round(max(sem_values), 4),
            }
            if sem_values
            else None
        )

        # Count turns with low similarity (potential degradation)
        low_semantic_count = sum(
            1 for t in self.turns if t.quality and t.quality.get("semantic_similarity") is not None
            and t.quality["semantic_similarity"] < 0.75
        )
        low_jaccard_count = sum(
            1 for t in self.turns if t.quality and t.quality.get("token_jaccard") is not None
            and t.quality["token_jaccard"] < 0.40
        )

        # Turns where quality was NOT computed because one side errored or
        # returned empty. These are excluded from quality means, which can bias
        # the aggregate upward; we surface the count so the reader knows.
        quality_skipped_count = sum(1 for t in self.turns if not t.quality_computed)

        # ── Response length analysis ────────────────────────────────────
        # length_ratio = proxy_response_len / direct_response_len. The proxy
        # compacts ONLY the input context and CANNOT control response verbosity,
        # so a high ratio is a property of the MODEL under a different prompt,
        # not an optimizer defect. We therefore report it as an informational
        # "model verbosity delta" and do NOT count it as optimizer degradation.
        length_ratios = [t.quality.get("length_ratio") for t in self.turns if t.quality and t.quality.get("length_ratio") is not None]
        truncation_count = sum(1 for r in length_ratios if r < 0.5) if length_ratios else 0
        verbosity_count = sum(1 for r in length_ratios if r > 2.0) if length_ratios else 0

        # ── Code block preservation analysis ────────────────────────────
        code_block_ratios = [t.quality.get("code_block_ratio") for t in self.turns if t.quality and t.quality.get("code_block_ratio") is not None]
        code_loss_count = sum(1 for r in code_block_ratios if r < 1.0) if code_block_ratios else 0

        # ── Syntactic validity of generated python code ─────────────────
        invalid_code_turns = [
            t.turn_index
            for t in self.turns
            if t.quality and t.quality.get("code_syntax_validity") is not None
            and t.quality["code_syntax_validity"] < 1.0
        ]

        # ── ROUGE precision/recall gap (directionality of degradation) ──
        rouge_prec_values = [t.quality.get("rouge_l_precision") for t in self.turns if t.quality and t.quality.get("rouge_l_precision") is not None]
        rouge_rec_values = [t.quality.get("rouge_l_recall") for t in self.turns if t.quality and t.quality.get("rouge_l_recall") is not None]
        rouge_gap_mean = 0.0
        if rouge_prec_values and rouge_rec_values:
            gaps = [round(p - r, 4) for p, r in zip(rouge_prec_values, rouge_rec_values, strict=True)]
            rouge_gap_mean = round(statistics.mean(gaps), 4)

        # ── Quality trend correlation (quality vs context utilization) ──
        quality_trend: dict[str, Any] = {}
        if len(self.turns) >= 3:
            ctx_utils = []
            sem_sims = []
            for t in self.turns:
                sim = t.quality.get("semantic_similarity")
                prompt_tok = t.proxy.prompt_tokens if hasattr(t.proxy, "prompt_tokens") else 0
                if sim is not None and prompt_tok > 0:
                    ctx_utils.append(prompt_tok / max_context_window)
                    sem_sims.append(sim)

            if len(ctx_utils) >= 3:
                # Pearson correlation between context utilization and semantic similarity
                mean_ctx = statistics.mean(ctx_utils)
                mean_sim = statistics.mean(sem_sims)
                num = sum((c - mean_ctx) * (s - mean_sim) for c, s in zip(ctx_utils, sem_sims, strict=True))
                den = (statistics.stdev(ctx_utils) * statistics.stdev(sem_sims) * len(ctx_utils)) if statistics.stdev(ctx_utils) > 0 and statistics.stdev(sem_sims) > 0 else 1
                correlation = round(num / den, 4) if den != 0 else 0.0

                # Linear regression slope (quality change per 10% context increase)
                n_pts = len(ctx_utils)
                sum_x = sum(ctx_utils)
                sum_y = sum(sem_sims)
                sum_xy = sum(c * s for c, s in zip(ctx_utils, sem_sims, strict=True))
                sum_x2 = sum(c * c for c in ctx_utils)
                denom_reg = n_pts * sum_x2 - sum_x * sum_x
                slope = round((n_pts * sum_xy - sum_x * sum_y) / denom_reg * 10, 4) if denom_reg != 0 else 0.0

                quality_trend["context_correlation"] = correlation
                quality_trend["slope_per_10pct_ctx"] = slope
                quality_trend["turn_count"] = n_pts

        # ── Vocabulary richness trend ───────────────────────────────────
        vocab_deltas = [t.quality.get("vocabulary_richness_delta") for t in self.turns if t.quality and t.quality.get("vocabulary_richness_delta") is not None]

        # ── Eviction tracking ───────────────────────────────────────────
        # `char_budget` is the proxy's POST-optimization target
        # (MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS), so raw input exceeding it is
        # expected for long scenarios and is NOT an eviction/failure. We report
        # it as informational (`raw_exceeds_optimized_target`) and treat the
        # real eviction signal as `compaction_triggered` (proxy sent strictly
        # fewer tokens than direct this turn).
        budget = self.config.get("char_budget")
        chars_before = [t.proxy.chars_before_optimization for t in self.turns]
        total_chars_before = sum(chars_before)
        raw_exceeds_target_turns: list[int] = []
        compaction_turns: list[int] = []
        eviction_turns: list[int] = []
        if budget is not None and chars_before:
            raw_exceeds_target_turns = [
                t.turn_index
                for t in self.turns
                if t.proxy.chars_before_optimization > budget
            ]
            compaction_turns = [
                t.turn_index
                for t in self.turns
                if t.direct.prompt_tokens > t.proxy.prompt_tokens
            ]
            eviction_turns = [
                t.turn_index
                for t in self.turns
                if t.proxy.chars_before_optimization > budget
                and t.direct.prompt_tokens > t.proxy.prompt_tokens
            ]

        # ── Cache-reuse trend (proxy prefix-cache hits) ────────────────
        prefix_hits = [t.proxy.prefix_cache_hit_tokens for t in self.turns]
        cache_reuse_trend: dict[str, Any] = {
            "per_turn_prefix_cache_hit_tokens": _stats(prefix_hits) if prefix_hits else {},
            "total_prefix_cache_hit_tokens": sum(prefix_hits),
        }
        if self.cache_reuse:
            cache_reuse_trend["per_round_proxy_metrics"] = self.cache_reuse

        # ── TTFT aggregation (streaming / --measure-ttft path) ──────────
        direct_ttft = [t.direct.ttft_ms for t in self.turns if t.direct.ttft_ms is not None]
        proxy_ttft = [t.proxy.ttft_ms for t in self.turns if t.proxy.ttft_ms is not None]

        # ── Proxy overhead (X-Proxy-Process-Ms, when the proxy emits it) ─
        proxy_process = [t.proxy.proxy_process_ms for t in self.turns if t.proxy.proxy_process_ms is not None]

        # ── Latency delta confidence interval + sign test ───────────────
        # Bootstrap CI on the per-turn proxy-minus-direct latency delta, plus a
        # paired sign test, so we can tell a real slowdown from backend noise.
        delta_ci = _bootstrap_ci(latency_deltas) if len(latency_deltas) >= 2 else (0.0, 0.0)
        delta_pos, delta_neg = _paired_sign_test(latency_deltas)

        # ── Cost model (USD) ────────────────────────────────────────────
        price_in = float(self.config.get("price_in_usd_per_1m", 0.0) or 0.0)
        price_out = float(self.config.get("price_out_usd_per_1m", 0.0) or 0.0)
        direct_cost = (total_direct_prompt * price_in + total_direct_completion * price_out) / 1_000_000
        proxy_cost = (total_proxy_prompt * price_in + total_proxy_completion * price_out) / 1_000_000
        cost_savings_pct = (
            round((direct_cost - proxy_cost) / direct_cost * 100, 2) if direct_cost > 0 else 0.0
        )

        return {
            "config": self.config,
            "num_turns": n,
            "latency_ms": {
                "direct": _stats(direct_latencies),
                "proxy": _stats(proxy_latencies),
                "delta_proxy_minus_direct_ms": _stats(latency_deltas),
                "delta_ci95_ms": {"low": delta_ci[0], "high": delta_ci[1]},
                "delta_sign_test": {"proxy_slower_turns": delta_pos, "proxy_faster_turns": delta_neg},
                "excluded_cached_artifact_turns": _artifact_turns,
            },
            "ttft_ms": {
                "direct": _stats(direct_ttft),
                "proxy": _stats(proxy_ttft),
            },
            "proxy_overhead_ms": _stats(proxy_process) if proxy_process else {},
            "tokens": {
                "total_direct_prompt": total_direct_prompt,
                "total_proxy_prompt": total_proxy_prompt,
                "total_raw_input_prompt": total_raw_input,
                "total_cached_tokens": total_cached,
                "token_savings_pct": tokens_saved_pct,
                "token_savings_vs_raw_pct": tokens_saved_vs_raw_pct,
                "per_turn_direct": _stats(direct_tokens),
                "per_turn_proxy": _stats(proxy_tokens),
                "per_turn_raw_input": _stats(raw_input_tokens),
                "per_turn_cached": _stats(cached),
            },
            "cost_usd": {
                "price_in_per_1m": price_in,
                "price_out_per_1m": price_out,
                "direct_total": round(direct_cost, 6),
                "proxy_total": round(proxy_cost, 6),
                "savings_pct": cost_savings_pct,
                "savings_usd": round(direct_cost - proxy_cost, 6),
            },
            "context_window": {
                "final_prompt_tokens": final_proxy_prompt_tokens,
                "final_prompt_tokens_raw": final_turn.proxy.raw_prompt_tokens,
                "final_prompt_tokens_source": final_turn.proxy.prompt_tokens_source,
                "final_prompt_tokens_cached_response": final_turn.proxy.cached_response,
                "max_context_window": max_context_window,
                "utilization_pct": final_proxy_ctx_pct,
                "cached_or_missing_usage_turns": [
                    t.turn_index
                    for t in self.turns
                    if t.proxy.prompt_tokens_source.startswith("estimated")
                    or t.proxy.raw_prompt_tokens == 0
                    or t.proxy.cached_response
                ],
            },
            "correctness": {
                "total_foreign_markers": sum(
                    len(t.proxy.foreign_markers) for t in self.turns
                ),
                "turns_with_markers": [
                    t.turn_index
                    for t in self.turns
                    if t.proxy.foreign_markers
                ],
            },
            "quality": {
                **quality_summary,
                "low_semantic_similarity_turns": low_semantic_count,
                "low_token_jaccard_turns": low_jaccard_count,
                "truncation_count": truncation_count,
                # verbosity_count is informational only (model verbosity delta,
                # NOT optimizer-caused) — see length-ratio note above.
                "model_verbosity_delta_turns": verbosity_count,
                "code_block_loss_turns": code_loss_count,
                "code_syntax_invalid_turns": invalid_code_turns if invalid_code_turns else None,
                "rouge_precision_recall_gap_mean": rouge_gap_mean,
                "quality_skipped_turns": quality_skipped_count,
            },
            "secondary_quality": secondary_quality_summary,
            "semantic_similarity": semantic_summary,
            "quality_trend": quality_trend if quality_trend else {},
            "vocab_richness": {
                "mean_delta": round(statistics.mean(vocab_deltas), 4) if vocab_deltas else None,
                "max_delta": round(max(vocab_deltas), 4) if vocab_deltas else None,
                "turns_above_0.15": sum(1 for v in vocab_deltas if v > 0.15) if vocab_deltas else 0,
            },
            "eviction": {
                "char_budget": budget,
                "total_chars_before_optimization": total_chars_before,
                "turns_raw_exceeds_optimized_target": len(raw_exceeds_target_turns),
                "raw_exceeds_optimized_target_at_turns": raw_exceeds_target_turns if raw_exceeds_target_turns else None,
                "compaction_triggered_at_turns": compaction_turns if compaction_turns else None,
                "eviction_triggered_at_turns": eviction_turns if eviction_turns else None,
            },
            "cache_reuse": cache_reuse_trend,
            "per_round": _per_round_summary(self.turns, total_direct_prompt, total_proxy_prompt),
            # Long-horizon / cross-turn signals. These answer "does the proxy
            # still remember early context by the end of a 30-turn session?" —
            # the core risk of a context compressor. See the metric spec.
            "long_horizon": {
                "contradictions": self.contradictions or None,
                "fact_recall_turn30": self.fact_recall or None,
                "context_window_wall": self.context_window_wall or None,
            },
        }


def _per_round_summary(
    turns: list[TurnComparison],
    total_direct_prompt: int,
    total_proxy_prompt: int,
) -> dict[str, Any]:
    """Aggregate quality/token metrics per benchmark round.

    Rounds are isolated proxy sessions, so per-round stats expose run-to-run
    variance (e.g. from a flaky backend) that the pooled mean would hide. This
    is what the regression gate should reason about instead of the pooled mean.
    """
    by_round: dict[int, list[TurnComparison]] = {}
    for t in turns:
        by_round.setdefault(t.round_index, []).append(t)

    rounds_out: dict[str, Any] = {}
    for rnd, rturns in sorted(by_round.items()):
        sims = [
            t.quality["semantic_similarity"]
            for t in rturns
            if t.quality and t.quality.get("semantic_similarity") is not None
        ]
        d_tok = sum(t.direct.prompt_tokens for t in rturns)
        p_tok = sum(t.proxy.prompt_tokens for t in rturns)
        raw_tok = sum(t.proxy.raw_input_tokens for t in rturns)
        savings = (
            round((d_tok - p_tok) / max(d_tok, 1) * 100, 2) if d_tok > 0 else 0.0
        )
        savings_vs_raw = (
            round((raw_tok - p_tok) / max(raw_tok, 1) * 100, 2) if raw_tok > 0 else 0.0
        )
        rounds_out[str(rnd)] = {
            "turns": len(rturns),
            "semantic_similarity": {
                "mean": round(statistics.mean(sims), 4) if sims else None,
                "min": round(min(sims), 4) if sims else None,
                "max": round(max(sims), 4) if sims else None,
            },
            "token_savings_pct": savings,
            "token_savings_vs_raw_pct": savings_vs_raw,
        }

    # Pooled per-round mean/min of semantic similarity, used by the gate so a
    # single noisy round cannot swing the decision.
    if rounds_out:
        round_means = [v["semantic_similarity"]["mean"] for v in rounds_out.values() if v["semantic_similarity"]["mean"] is not None]
        round_mins = [v["semantic_similarity"]["min"] for v in rounds_out.values() if v["semantic_similarity"]["min"] is not None]
        rounds_out["_pooled"] = {
            "round_mean_of_means": round(statistics.mean(round_means), 4) if round_means else None,
            "round_min_of_means": round(min(round_means), 4) if round_means else None,
            "round_min_of_mins": round(min(round_mins), 4) if round_mins else None,
        }
    return rounds_out


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile from already-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return round(d0 + d1, 2)


def _bootstrap_ci(values: list[float], pct: float = 95.0, n: int = 2000) -> tuple[float, float]:
    """Bootstrap confidence interval (percentile method) for the mean.

    Returns (low, high) at the requested confidence level. Used to report
    uncertainty on latency/quality means so a single noisy run cannot be
    over-interpreted. Falls back to (0, 0) on insufficient data.
    """
    if len(values) < 2:
        return (0.0, 0.0)
    rng = random.Random(0xC0FFEE)
    estimates: list[float] = []
    size = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    alpha = (100.0 - pct) / 2.0
    low = _percentile(estimates, alpha)
    high = _percentile(estimates, 100.0 - alpha)
    return (low, high)


def _paired_sign_test(diffs: list[float]) -> tuple[int, int]:
    """Paired sign test over per-turn differences (proxy - direct).

    Returns (pos, neg) counts of strictly-positive / strictly-negative diffs.
    A large imbalance indicates the latency delta is systematic, not noise.
    Ties (zero diffs) are excluded.
    """
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return (pos, neg)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _collect_direct_conversation(
    messages: list[dict],
    num_turns: int,
    user_tasks: list[str],
    fallback_user_task: str,
    request_timeout: float,
    max_tokens: int,
    turn_offset: int,
    turn_exchanges: list[list[dict]] | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    stream: bool = False,
) -> tuple[list[TurnMetrics], list[str]]:
    """Run a full conversation against direct Lemonade.

    When ``turn_exchanges`` is provided (OpenCode-harness / agentic mode), each
    turn appends a full agent payload (user task + assistant tool_calls + tool
    results) instead of a single user message, so the backend sees the same kind
    of messages a real coding agent sends.
    """
    direct_contents: list[str] = []
    direct_metrics: list[TurnMetrics] = []

    for local_turn in range(num_turns):
        turn_index = turn_offset + local_turn + 1
        # Select scenario content by the WITHIN-ROUND position (local_turn), not
        # the global turn_index, so every round replays the SAME num_turns-long
        # slice of exchanges/tasks. Indexing by the global turn_index made rounds
        # replay different slices whenever len(turn_exchanges) did not divide
        # evenly into rounds*num_turns (e.g. 30 exchanges over 5×10=50 turns made
        # R1/R4 use the small exchanges 0-9 while R2/R3/R5 used the large 10-29),
        # which broke round-to-round comparability. turn_index is still used for
        # labels/metrics below.
        if turn_exchanges:
            exchange = turn_exchanges[local_turn % len(turn_exchanges)]
            messages.extend(exchange)
        else:
            user_content = (
                user_tasks[local_turn % len(user_tasks)]
                if user_tasks
                else fallback_user_task.format(turn_index=turn_index)
            )
            messages.append({"role": "user", "content": user_content})

        direct_context = _context_size_summary(messages)
        _human_print(
            f"  Direct turn {local_turn + 1:02d}/{num_turns:02d}: "
            f"backend-facing ~{direct_context['estimated_tokens']:,} tok "
            f"(raw {direct_context['messages']} msgs/{direct_context['chars']:,} chars, no proxy)"
        )

        try:
            if stream:
                d_content, d_reasoning, d_usage, d_ttft, d_latency, _, _, d_tool_calls = _direct_stream_request(
                    messages, max_tokens=max_tokens, timeout=request_timeout, tools=tools, temperature=temperature
                )
                d_usage = d_usage or {}
                d_msg = {"role": "assistant", "content": d_content}
                if d_reasoning:
                    d_msg["reasoning_content"] = d_reasoning
                if d_tool_calls:
                    d_msg["tool_calls"] = d_tool_calls
                d_finish_reason = ""
                # Mirror the non-streaming path (line ~2887): the quality/preview
                # text includes the reasoning trace, so quality is computed even
                # when the model emits only `reasoning_content` and no final
                # `content` (typical for reasoning-heavy models at small
                # max_tokens). Without this, streaming runs report quality_sem=n/a
                # while non-streaming runs compute it -- an inconsistency.
                d_content = (d_content or "") + (d_reasoning or "")
            else:
                direct_resp, d_latency, _ = _direct_request(
                    messages, max_tokens=max_tokens, timeout=request_timeout, tools=tools, temperature=temperature
                )
                d_usage = direct_resp.get("usage", {}) or {}
                d_msg = direct_resp["choices"][0]["message"]
                d_content = (d_msg.get("content") or "") + (d_msg.get("reasoning_content") or "")
                d_ttft = None
                d_finish_reason = direct_resp["choices"][0].get("finish_reason", "")

            _d_prompt_raw = int(d_usage.get("prompt_tokens", 0) or 0)
            _d_cached = int((d_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
            _d_prompt, _d_prompt_source, _d_cached_response = _resolve_prompt_tokens(
                _d_prompt_raw, messages, d_usage
            )
            metrics = TurnMetrics(
                turn_index=turn_index,
                total_turns_at_request=len(messages) - 1,  # exclude system
                prompt_tokens=_d_prompt,
                raw_prompt_tokens=_d_prompt_raw,
                prompt_tokens_source=_d_prompt_source,
                cached_response=_d_cached_response,
                completion_tokens=int(d_usage.get("completion_tokens", 0) or 0),
                total_tokens=int(d_usage.get("total_tokens", 0) or 0),
                cached_tokens=_d_cached,
                cache_hit_rate=round(_d_cached / max(_d_prompt, 1), 2) if _d_cached > 0 else None,
                latency_ms=round(d_latency, 2),
                ttft_ms=round(d_ttft, 2) if d_ttft is not None else None,
                response_chars=len(d_content),
                finish_reason=d_finish_reason,
                content_preview=d_content[:200],
            )
            _human_print(
                f"    → backend-facing: {metrics.prompt_tokens:,} tok "
                f"(source={metrics.prompt_tokens_source}, cached={metrics.cached_tokens:,})"
            )
        except Exception as e:
            metrics = TurnMetrics(
                turn_index=turn_index,
                total_turns_at_request=len(messages) - 1,
                prompt_tokens=_estimate_prompt_tokens(messages) if messages else 0,
                prompt_tokens_source="estimated_after_error",
                latency_ms=0.0,
                error=str(e)[:200],
            )
            d_content = ""
            d_msg = {}

        if d_content or d_msg.get("tool_calls"):
            _append_assistant_message(messages, d_msg)
            direct_contents.append(d_content)
        else:
            direct_contents.append("")

        direct_metrics.append(metrics)

    return direct_metrics, direct_contents


def _collect_proxy_conversation(
    messages: list[dict],
    session_id: str,
    num_turns: int,
    user_tasks: list[str],
    fallback_user_task: str,
    request_timeout: float,
    max_tokens: int,
    turn_offset: int,
    turn_exchanges: list[list[dict]] | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    stream: bool = False,
) -> tuple[list[TurnMetrics], list[str]]:
    """Run a full conversation through the moeptimizer proxy.

    Mirrors :func:`_collect_direct_conversation`; when ``turn_exchanges`` is set
    each turn appends a full OpenCode-style agent payload.
    """
    proxy_contents: list[str] = []
    proxy_metrics: list[TurnMetrics] = []

    for local_turn in range(num_turns):
        turn_index = turn_offset + local_turn + 1
        # Select scenario content by the WITHIN-ROUND position (local_turn) so
        # every round replays the SAME slice; see _collect_direct_conversation
        # for the rationale (global-index selection broke round comparability).
        if turn_exchanges:
            exchange = turn_exchanges[local_turn % len(turn_exchanges)]
            messages.extend(exchange)
        else:
            user_content = (
                user_tasks[local_turn % len(user_tasks)]
                if user_tasks
                else fallback_user_task.format(turn_index=turn_index)
            )
            messages.append({"role": "user", "content": user_content})

        proxy_context = _context_size_summary(messages)
        # Capture the FULL (pre-optimization) prompt text so we can later
        # measure how much of the original context survived compaction
        # (prompt-faithfulness). This is the direct input to the optimizer.
        _full_prompt_text = _serialize_messages_text(messages)
        _human_print(
            f"  Proxy turn {local_turn + 1:02d}/{num_turns:02d}: "
            f"raw {proxy_context['messages']} msgs/{proxy_context['chars']:,} chars/"
            f"~{proxy_context['estimated_tokens']:,} tok (pre-optimization)"
        )

        try:
            if stream:
                p_content, p_reasoning, p_usage, p_ttft, p_latency, p_headers, p_prefix_hit, p_tool_calls = _proxy_stream_request(
                    messages, session_id=session_id, max_tokens=max_tokens, timeout=request_timeout, tools=tools, temperature=temperature
                )
                p_usage = p_usage or {}
                p_msg = {"role": "assistant", "content": p_content}
                if p_reasoning:
                    p_msg["reasoning_content"] = p_reasoning
                if p_tool_calls:
                    p_msg["tool_calls"] = p_tool_calls
                p_finish_reason = ""
                # Mirror the non-streaming path (line ~3000): include the reasoning
                # trace in the quality/preview text so quality is computed even
                # when the model emits only `reasoning_content` (no final
                # `content`). Keeps streaming and non-streaming consistent.
                p_content = (p_content or "") + (p_reasoning or "")
            else:
                proxy_resp, p_latency, p_headers = _proxy_request(
                    messages, session_id=session_id, max_tokens=max_tokens, timeout=request_timeout, tools=tools, temperature=temperature
                )
                p_usage = proxy_resp.get("usage", {}) or {}
                p_msg = proxy_resp["choices"][0]["message"]
                p_content = (p_msg.get("content") or "") + (p_msg.get("reasoning_content") or "")
                p_ttft = None
                # Non-streaming: the proxy surfaces its authoritative prefix-cache
                # hit count as a response header (app.py: X-Prefix-Cache-Hit-Tokens).
                p_prefix_hit = int(
                    p_headers.get("X-Prefix-Cache-Hit-Tokens")
                    or p_headers.get("x-prefix-cache-hit-tokens")
                    or 0
                )
                p_finish_reason = proxy_resp["choices"][0].get("finish_reason", "")

            # Proxy's own optimization/forwarding overhead, surfaced as a header
            # when the proxy emits it. Lets us separate proxy cost from backend
            # latency. None when the proxy does not report it.
            _p_process_ms = p_headers.get("X-Proxy-Process-Ms") or p_headers.get("x-proxy-process-ms")
            _p_process_ms = float(_p_process_ms) if _p_process_ms else None

            _p_prompt_raw = int(p_usage.get("prompt_tokens", 0) or 0)
            _p_cached = int((p_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
            _p_optimized_prompt_tokens = int(
                p_headers.get("X-Optimized-Prompt-Tokens")
                or p_headers.get("x-optimized-prompt-tokens")
                or 0
            )
            # The proxy can expose the exact optimized prompt TEXT it sent to the
            # backend (X-MOEPT-Optimized-Prompt-Text). This is the direct
            # measure of the optimizer's one job (it compacts ONLY the input
            # context), so we capture it to compute prompt-faithfulness.
            _p_optimized_text = (
                p_headers.get("X-MOEPT-Optimized-Prompt-Text")
                or p_headers.get("x-moept-optimized-prompt-text")
                or ""
            )
            # Header value has newlines escaped as literal "\n"; restore them.
            if _p_optimized_text:
                _p_optimized_text = _p_optimized_text.replace("\\n", "\n")
            _p_prompt, _p_prompt_source, _p_cached_response = _resolve_prompt_tokens(
                _p_prompt_raw,
                messages,
                p_usage,
                optimized_prompt_tokens=_p_optimized_prompt_tokens,
            )
            # The proxy rewrites the prompt, so the backend's KV-cache hit count
            # for the proxy path is (correctly) ~0 — the optimized prompt is a
            # novel token sequence to the model, so its prefix cache cannot match.
            # The proxy's *own* cache reuse is surfaced via X-Prefix-Cache-Hit-Tokens.
            # Report that as the proxy's cached_tokens so the benchmark reflects the
            # proxy's real cache efficiency; otherwise total_cached_tokens collapses
            # to 0 and the proxy looks cache-less even though it reused ~129k tokens.
            # Fall back to the backend signal only when the proxy reports no hits.
            _p_cached_tokens = p_prefix_hit if p_prefix_hit > 0 else _p_cached
            _p_cached_response = _p_cached_response or (p_prefix_hit > 0)
            # Measure total chars before proxy optimization (for eviction tracking)
            _chars_before = sum(len(_message_text(m.get("content", ""))) for m in messages)
            metrics = TurnMetrics(
                turn_index=turn_index,
                total_turns_at_request=len(messages) - 1,
                prompt_tokens=_p_prompt,
                raw_prompt_tokens=_p_prompt_raw,
                optimized_prompt_tokens=_p_optimized_prompt_tokens,
                optimized_prompt_text=_p_optimized_text,
                full_prompt_text=_full_prompt_text,
                raw_input_tokens=_estimate_prompt_tokens(messages) if messages else 0,
                prompt_tokens_source=_p_prompt_source,
                cached_response=_p_cached_response,
                completion_tokens=int(p_usage.get("completion_tokens", 0) or 0),
                total_tokens=int(p_usage.get("total_tokens", 0) or 0),
                cached_tokens=_p_cached_tokens,
                cache_hit_rate=round(_p_cached_tokens / max(_p_prompt, 1), 2) if _p_cached_tokens > 0 else None,
                latency_ms=round(p_latency, 2),
                ttft_ms=round(p_ttft, 2) if p_ttft is not None else None,
                prefix_cache_hit_tokens=p_prefix_hit,
                proxy_process_ms=round(_p_process_ms, 2) if _p_process_ms is not None else None,
                response_chars=len(p_content),
                finish_reason=p_finish_reason,
                content_preview=p_content[:200],
                chars_before_optimization=_chars_before,
            )

            # Check for leaked internal markers
            metrics.foreign_markers = _check_foreign_markers(p_content)
            _human_print(
                f"    → backend-facing: {metrics.prompt_tokens:,} tok "
                f"(source={metrics.prompt_tokens_source}, cached={metrics.cached_tokens:,}, "
                f"prefix_hits={metrics.prefix_cache_hit_tokens:,}, "
                f"raw={metrics.chars_before_optimization:,} chars)"
            )
        except Exception as e:
            metrics = TurnMetrics(
                turn_index=turn_index,
                total_turns_at_request=len(messages) - 1,
                prompt_tokens=_estimate_prompt_tokens(messages) if messages else 0,
                prompt_tokens_source="estimated_after_error",
                latency_ms=0.0,
                error=str(e)[:200],
            )
            # Try to extract optimization error from response headers if available
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                opt_error = e.response.headers.get("X-Optimization-Error")
                if opt_error:
                    metrics.error = f"{metrics.error} | optimization: {opt_error}"
            # Surface the failure immediately so a broken proxy turn is never
            # silent in the log (it would otherwise just lack the backend-facing
            # line and look like a missing metric).
            _human_print(
                f"    ⚠ proxy turn {turn_index:02d} FAILED: {metrics.error}"
            )
            p_content = ""
            p_msg = {}

        if p_content or (isinstance(p_msg, dict) and p_msg.get("tool_calls")):
            _append_assistant_message(messages, p_msg)
            proxy_contents.append(p_content)
        else:
            proxy_contents.append("")

        proxy_metrics.append(metrics)

    return proxy_metrics, proxy_contents


def _build_turn_comparisons(
    direct_metrics: list[TurnMetrics],
    proxy_metrics: list[TurnMetrics],
    direct_contents: list[str],
    proxy_contents: list[str],
) -> list[TurnComparison]:
    """Build per-turn comparisons after both full conversations complete."""
    comparisons: list[TurnComparison] = []
    for direct, proxy, d_content, p_content in zip(
        direct_metrics,
        proxy_metrics,
        direct_contents,
        proxy_contents,
        strict=True,
    ):
        quality: dict[str, float | None] = {}
        if d_content and p_content:
            quality.update(_compute_quality_metrics(d_content, p_content, proxy.full_prompt_text, proxy.optimized_prompt_text))

        comparison = TurnComparison(
            turn_index=direct.turn_index,
            direct=direct,
            proxy=proxy,
            latency_delta_ms=round(proxy.latency_ms - direct.latency_ms, 2),
            token_delta=proxy.prompt_tokens - direct.prompt_tokens,
            quality=quality,
            quality_computed=bool(d_content and p_content),
        )
        comparisons.append(comparison)
    return comparisons


def _validate_conversation_invariant(
    proxy_metrics: list[TurnMetrics],
    direct_metrics: list[TurnMetrics],
    num_turns: int,
    round_num: int,
) -> None:
    """Enforce the benchmark's core invariant: each side is a complete, sorted,
    non-interleaved multi-turn conversation.

    - Both conversations must contain exactly ``num_turns`` turns (complete).
    - Each must be sorted by ``turn_index`` with no gaps or duplicates (sorted).
    - The proxy conversation is collected fully before the direct one by call
      order in :func:`run_benchmark`; this guard catches any future regression
      that would interleave or truncate either side.
    """
    for label, metrics in (("proxy", proxy_metrics), ("direct", direct_metrics)):
        if len(metrics) != num_turns:
            raise AssertionError(
                f"Round {round_num}: {label} conversation incomplete "
                f"({len(metrics)}/{num_turns} turns) -- non-interleaving "
                f"invariant violated."
            )
        indices = [m.turn_index for m in metrics]
        if indices != sorted(indices):
            raise AssertionError(
                f"Round {round_num}: {label} turns not sorted by turn_index "
                f"({indices}) -- non-interleaving invariant violated."
            )
        if len(set(indices)) != len(indices):
            raise AssertionError(
                f"Round {round_num}: {label} has duplicate turn indices "
                f"({indices}) -- non-interleaving invariant violated."
            )
    # Proxy and direct must describe the same turn set (1:1, no interleaving).
    if {m.turn_index for m in proxy_metrics} != {m.turn_index for m in direct_metrics}:
        raise AssertionError(
            f"Round {round_num}: proxy/direct turn sets differ "
            f"({[m.turn_index for m in proxy_metrics]} vs "
            f"{[m.turn_index for m in direct_metrics]}) -- non-interleaving "
            f"invariant violated."
        )


def run_benchmark(
    num_turns: int,
    rounds: int,
    max_tokens: int,
    proxy_port: int,
    budget: int | None = None,
    scenario: str = "default",
    agentic: bool = True,
    temperature: float = 0.0,
    measure_ttft: bool = True,
    context_window: int = 262144,
    price_in: float = 0.0,
    price_out: float = 0.0,
    max_wall_seconds: float | None = None,
) -> BenchmarkReport:
    """Run the multi-turn benchmark and collect metrics.

    The proxy conversation runs first against a cold backend, then the direct
    Lemonade conversation runs as its own full, continuous session. Each stays a
    contiguous multi-turn conversation (we do NOT interleave direct/proxy
    requests). Running the proxy first means its prefix-cache hits (turns 2..N)
    reflect the proxy's own stable prefix rather than a warm cache left by a
    prior direct run, so the cache/latency comparison is not confounded.

    When ``agentic`` is True (or the scenario already ships OpenCode-style
    exchanges), each turn appends a full agent payload -- user task plus assistant
    ``tool_calls`` and the corresponding ``tool`` results -- and the OpenAI ``tools``
    schema is forwarded to the backend, exactly like a real coding client.
    """

    # Update module-level port so _proxy_request uses it
    global MOEPT_PORT
    MOEPT_PORT = proxy_port

    # Calculate dynamic timeout based on turns and rounds
    request_timeout = _calculate_timeout(num_turns, rounds)
    char_budget = budget or int(os.environ.get("MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS", "12000"))

    config = {
        "lemonade_url": LEMONADE_URL,
        "model": MODEL_ID,
        "num_turns": num_turns,
        "rounds": rounds,
        "max_tokens": max_tokens,
        "proxy_port": proxy_port,
        "char_budget": char_budget,
        "scenario": scenario,
        "request_timeout": request_timeout,
        "execution_order": "direct_full_conversation_then_proxy_full_conversation",
        "measure_ttft": measure_ttft,
        "context_window": context_window,
        "price_in_usd_per_1m": price_in,
        "price_out_usd_per_1m": price_out,
        "max_wall_seconds": max_wall_seconds,
    }

    report = BenchmarkReport(config=config)

    # Get scenario tasks. A scenario's tasks may be a mix of:
    #   - ("role", "content") tuples  -> simple messages (backward compatible)
    #   - list[dict]                  -> a full OpenCode-style agentic turn exchange
    # The latter are collected into `turn_exchanges` and appended per turn.
    scenario_data = SCENARIOS.get(scenario, SCENARIOS["default"])
    # The `drift` scenario builds its task list as a function of num_turns so the
    # recall probe lands exactly on the final turn; other scenarios ship a static
    # list. Support both shapes here.
    _raw_tasks = scenario_data["tasks"]
    base_tasks = _raw_tasks(num_turns) if callable(_raw_tasks) else _raw_tasks
    # Inject the long-horizon drift probe (plant facts in Turn 1, recall probe
    # on the final turn) into every scenario so drift is measured on the real
    # benchmark conversation, not a synthetic one.
    base_tasks = _inject_drift_probe(base_tasks, num_turns)

    system_prompt = SYSTEM_PROMPT
    base_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    turn_exchanges: list[list[dict]] = []
    user_tasks: list[str] = []
    # Simple (tuple) scenarios get wrapped into agentic exchanges below. When that
    # happens we must NOT also seed base_messages with the plain user messages,
    # otherwise every task would be sent twice (once plain, once as an
    # agentic user+tool_calls+tool payload) and the baseline context would be
    # inflated. We only seed base_messages for the non-agentic path.
    will_wrap_agentic = agentic and not any(isinstance(t, list) for t in base_tasks)
    for item in base_tasks:
        if isinstance(item, list):
            # An agentic turn exchange (user + assistant tool_calls + tool results)
            turn_exchanges.append(item)
        else:
            role, content = item
            if role == "user":
                user_tasks.append(content)
            if not will_wrap_agentic:
                base_messages.append({"role": role, "content": content})

    # --agentic wraps simple scenarios into OpenCode-style exchanges with
    # synthesized tool outputs, so even the synthetic scenarios emit a realistic
    # agent payload (user task + tool calls + tool results). The read_file tool
    # returns the scenario's own current module so the tool output is coherent
    # with the task being worked on. For the short scenarios the tasks carry no
    # code block, so we pass the scenario's BASE module as the read content
    # instead of letting it fall back to a placeholder string.
    if agentic and not turn_exchanges:
        scenario_read_paths = {
            "debug": "app/items.py",
            "debug_long": "app/items.py",
            "refactor": "users/repository.py",
            "refactor_long": "users/repository.py",
            "feature": "auth/service.py",
            "feature_long": "auth/service.py",
            "default": "fib.py",
            "default_long": "fib.py",
        }
        # Base module per short scenario, used as the coherent read_file result
        # when the task carries no code block of its own.
        scenario_base_code = {
            "debug": SHORT_DEBUG_CODE,
            "refactor": SHORT_REFACTOR_CODE,
            "feature": BASE_FEATURE_CODE,
            "default": BASE_DEFAULT_CODE,
        }
        # For the long scenarios the user task pastes the *cumulative* module
        # (base + every prior step), so read_file must return that same
        # cumulative code -- not the static base -- to stay coherent with the
        # task. Map scenario -> (base, steps) and recompute it per turn.
        scenario_long_data = {
            "refactor_long": (BASE_REFACTOR_CODE, REFACTOR_STEPS),
            "debug_long": (BASE_DEBUG_CODE, DEBUG_STEPS),
            "feature_long": (BASE_FEATURE_CODE, FEATURE_STEPS),
            "default_long": (BASE_DEFAULT_CODE, DEFAULT_STEPS),
        }
        long_pair = scenario_long_data.get(scenario)
        read_path = scenario_read_paths.get(scenario)
        turn_exchanges = [
            _agentic_exchange(
                content,
                i + 1,
                read_path=read_path,
                read_content_override=(
                    _cumulative_code(long_pair[0], long_pair[1], i)
                    if long_pair
                    else scenario_base_code.get(scenario)
                ),
            )
            for i, content in enumerate(user_tasks)
        ]
        user_tasks = []

    # Forward the OpenAI tool schemas whenever we are in agentic mode, so the
    # backend accepts the tool_calls / tool messages we send (OpenAI requires
    # `tools` to be present alongside them).
    tools = OPENCODE_TOOLS if (turn_exchanges or agentic) else None

    fallback_user_task = (
        "Turn {turn_index}: Remember the fibonacci generator we discussed? "
        "Now write a test suite for it using pytest."
    )

    # One-time backend warm-up so both conversations start warm (see
    # _warm_up_backend). Keeps each side complete and sorted; only removes the
    # cold-start bias that would otherwise advantage the direct run.
    _warm_up_backend(request_timeout)

    _wall_start = time.monotonic()
    # Long-horizon signal accumulators (summed across rounds; rounds are
    # isolated sessions, so contradictions/fact-recall are additive).
    _contradiction_proxy = 0
    _contradiction_direct = 0
    _fact_recall_proxy: float | None = None
    _fact_recall_direct: float | None = None
    _drift_probe_turn = num_turns  # 1-based; the probe is the final turn
    for round_num in range(rounds):
        # Wall-clock guard: abort remaining rounds if we exceed the budget so a
        # long --scenario all run cannot hang indefinitely.
        if max_wall_seconds is not None and (time.monotonic() - _wall_start) > max_wall_seconds:
            _human_print(
                f"  Wall-clock budget {max_wall_seconds:.0f}s reached; "
                f"stopping after round {round_num}/{rounds}."
            )
            break

        # Each round gets an isolated proxy session so prior-round state cannot
        # leak into the next benchmark round.
        session_id = f"benchmark-{int(time.time())}-{round_num}-{uuid.uuid4().hex[:8]}"

        # Per-round KV-cache bust. The backend's prefix cache is GLOBAL (keyed by
        # token prefix, not by session), so a later round's "fresh" turn-1 prompt
        # can reuse a stale cached prefix left by an EARLIER round's proxy run.
        # That contamination makes later rounds' responses diverge in SHAPE from
        # earlier ones (and from the direct run), even though the conversation is
        # identical — breaking round-to-round comparability.
        #
        # The marker MUST be PREPENDED (not appended): the backend caches by the
        # LEADING token sequence, so a unique token at the very start makes the
        # entire prefix distinct and prevents any cross-round cache hit. Appending
        # it at the end leaves the shared leading prefix (system prompt + tools +
        # proxy scaffolding, ~764 tokens) identical across rounds, so the cache
        # still hits on that prefix — which is exactly the 764 we kept seeing.
        #
        # The PROXY and DIRECT paths each get their OWN unique marker. This
        # isolates the two paths: the direct conversation must NOT reuse the
        # proxy's cached prefix (the proxy runs first in each round, so without
        # separate markers the direct turn-1 would inherit the proxy's end-of-
        # round cache and report a spurious cache hit). Separate markers mean
        # each path builds its own natural prefix cache from a cold start, so the
        # proxy-vs-direct comparison is fair AND cross-ROUND leakage is prevented.
        #
        # CRITICAL: the markers must diverge at the VERY FIRST token. The backend
        # caches by the leading token sequence, so two markers that share a long
        # common prefix (e.g. "<!-- benchmark round N proxy" vs "...direct")
        # still collide on the backend's prefix-cache key and leak cache between
        # paths. Prefixing with a distinct single character ('P' vs 'D') makes
        # the first token unique, so neither path can reuse the other's cache.
        proxy_marker = f"P{{benchmark round {round_num} proxy session {session_id}}}\n"
        direct_marker = f"D{{benchmark round {round_num} direct session {session_id}}}\n"
        proxy_system_prompt = proxy_marker + system_prompt
        direct_system_prompt = direct_marker + system_prompt

        # Run the PROXY conversation first against a cold backend so its
        # prefix-cache hits (turns 2..N) reflect the proxy's own stable
        # prefix rather than a warm cache left by a prior direct run. The
        # direct conversation follows as its own full, continuous session; any
        # cache hits it sees are its own natural prefix reuse, not borrowed
        # from the proxy. (We deliberately do NOT interleave the two, so each
        # stays a contiguous multi-turn conversation.)
        if measure_ttft:
            # Isolate this round's proxy metrics so the /v1/metrics snapshot
            # reflects only this round's prefix-cache reuse.
            _reset_proxy_metrics(proxy_port)

        _human_print(f"  Round {round_num + 1}/{rounds}: proxy conversation")
        proxy_messages: list[dict] = [dict(msg) for msg in base_messages]
        proxy_messages[0] = {"role": "system", "content": proxy_system_prompt}
        proxy_metrics, proxy_contents = _collect_proxy_conversation(
            proxy_messages,
            session_id,
            num_turns,
            user_tasks,
            fallback_user_task,
            request_timeout,
            max_tokens,
            turn_offset=round_num * num_turns,
            turn_exchanges=turn_exchanges or None,
            tools=tools,
            temperature=temperature,
            stream=measure_ttft,
        )

        if measure_ttft:
            snap = _fetch_proxy_metrics(proxy_port)
            if snap is not None:
                report.cache_reuse.append({"round": round_num, **snap})

        _human_print(f"  Round {round_num + 1}/{rounds}: direct conversation")
        direct_messages: list[dict] = [dict(msg) for msg in base_messages]
        direct_messages[0] = {"role": "system", "content": direct_system_prompt}
        direct_metrics, direct_contents = _collect_direct_conversation(
            direct_messages,
            num_turns,
            user_tasks,
            fallback_user_task,
            request_timeout,
            max_tokens,
            turn_offset=round_num * num_turns,
            turn_exchanges=turn_exchanges or None,
            tools=tools,
            temperature=temperature,
            stream=measure_ttft,
        )

        # ── Long-horizon cross-turn signals ──────────────────────────────
        # Contradiction rate: how often the model contradicts its own earlier
        # statements within this round's conversation. Summed across rounds.
        _contradiction_proxy += _count_contradictions(proxy_contents)
        _contradiction_direct += _count_contradictions(direct_contents)
        # Fact recall (drift): grade the final turn's response against the
        # planted facts. The probe is always the last turn, so the tail of
        # contents is the probe answer for both proxy and direct.
        if proxy_contents:
            _fact_recall_proxy = _grade_fact_recall(proxy_contents[-1], _DRIFT_FACTS)
            _fact_recall_direct = _grade_fact_recall(direct_contents[-1], _DRIFT_FACTS)

        # Enforce the non-interleaving invariant: each side is a complete,
        # sorted, non-interleaved multi-turn conversation (proxy fully before
        # direct). Catches any future regression that would interleave/truncate.
        _validate_conversation_invariant(
            proxy_metrics, direct_metrics, num_turns, round_num
        )

        comparisons = _build_turn_comparisons(
            direct_metrics,
            proxy_metrics,
            direct_contents,
            proxy_contents,
        )
        for _c in comparisons:
            _c.round_index = round_num
        report.turns.extend(comparisons)

        for comparison in comparisons:
            q_sem = comparison.quality.get("semantic_similarity") if comparison.quality else None
            q_jaccard = comparison.quality.get("token_jaccard") if comparison.quality else None
            q_rouge = comparison.quality.get("rouge_l_f1") if comparison.quality else None
            q_faith = comparison.quality.get("prompt_faithfulness") if comparison.quality else None
            q_evict = comparison.quality.get("evicted_content_recall") if comparison.quality else None
            direct_error = f" direct_error={comparison.direct.error[:80]!r}" if comparison.direct.error else ""
            proxy_error = f" proxy_error={comparison.proxy.error[:80]!r}" if comparison.proxy.error else ""
            quality_parts = [
                f"quality_sem={q_sem:.3f}" if q_sem is not None else "quality_sem=n/a",
                f"jaccard={q_jaccard:.3f}" if q_jaccard is not None else "jaccard=n/a",
                f"rouge={q_rouge:.3f}" if q_rouge is not None else "rouge=n/a",
                f"faith={q_faith:.3f}" if q_faith is not None else "faith=n/a",
                f"evict={q_evict:.3f}" if q_evict is not None else "evict=n/a",
            ]
            _human_print(
                f"  Turn {comparison.turn_index:02d}: "
                f"direct={comparison.direct.latency_ms:.0f}ms/{comparison.direct.prompt_tokens:,}tok/{comparison.direct.response_chars:,}chars"
                f" proxy={comparison.proxy.latency_ms:.0f}ms/{comparison.proxy.prompt_tokens:,}tok/"
                f"{comparison.proxy.chars_before_optimization:,}chars_raw/{comparison.proxy.response_chars:,}chars"
                f" raw_input={comparison.proxy.raw_input_tokens:,}tok"
                f" ttft(d/p)={_fmt_ttft(comparison.direct.ttft_ms)}/{_fmt_ttft(comparison.proxy.ttft_ms)}"
                f" delta={comparison.latency_delta_ms:+.0f}ms/{comparison.token_delta:+,}tok"
                f" cache={comparison.proxy.cached_tokens:,}/{comparison.proxy.cache_hit_rate if comparison.proxy.cache_hit_rate is not None else 'n/a'} "
                f"{' '.join(quality_parts)}"
                f"{direct_error}{proxy_error}"
            )

    # Finalize long-horizon signals on the report (derived from all rounds).
    report.contradictions = {"proxy": _contradiction_proxy, "direct": _contradiction_direct}
    report.fact_recall = {"proxy": _fact_recall_proxy, "direct": _fact_recall_direct}
    report.context_window_wall = _context_window_wall(report.turns)

    return report


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_ttft(value: float | None) -> str:
    """Format a TTFT value (ms) for the per-turn log, or 'n/a' when missing."""
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple ASCII table."""
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    pad = 2
    header_line = "".join(
        h.ljust(w + pad) for h, w in zip(headers, widths, strict=True)
    )
    sep = "-" * len(header_line)

    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("  ".join(c.ljust(widths[i] + pad) for i, c in enumerate(row)))
    lines.append(sep)
    return "\n".join(lines)


def _status(args: argparse.Namespace, *parts: object) -> None:
    """Print human status output without polluting JSON stdout."""
    print(*parts, file=sys.stderr if getattr(args, "json_output", False) else sys.stdout)


def print_report(report: BenchmarkReport) -> None:
    """Print a human-readable benchmark report."""
    summary = report.summary()

    # ── Config ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MOEPTIMIZER MULTI-TURN BENCHMARK REPORT")
    print("=" * 72)
    cfg = summary["config"]
    print(f"\n  Model:          {cfg['model']}")
    print(f"  Lemonade URL:   {cfg['lemonade_url']}")
    print(f"  Proxy port:     {cfg['proxy_port']}")
    print(f"  Turns per round:{cfg['num_turns']}")
    print(f"  Rounds:         {cfg['rounds']}")

    # ── Latency comparison ────────────────────────────────────────────
    lat = summary["latency_ms"]
    d_stats = lat["direct"]
    p_stats = lat["proxy"]
    delta_stats = lat["delta_proxy_minus_direct_ms"]

    print("\n" + "-" * 72)
    print("  LATENCY (milliseconds)")
    print("-" * 72)
    headers = ["Metric", "Direct", "Proxy", "Delta (+/-)", "Speed change"]
    rows: list[list[str]] = []

    for stat_name in ["mean", "median", "p95"]:
        d_val = f"{d_stats.get(stat_name, 'N/A')}"
        p_val = f"{p_stats.get(stat_name, 'N/A')}"
        delta_val = f"{delta_stats.get(stat_name, 'N/A')}"

        if stat_name in ("mean", "median") and d_stats.get(stat_name) and delta_stats.get(stat_name):
            pct_change = (delta_stats[stat_name] / d_stats[stat_name]) * 100
            speed_label = f"{pct_change:+.1f}%"
        else:
            speed_label = ""

        rows.append([stat_name.capitalize(), d_val, p_val, delta_val, speed_label])

    print(_fmt_table(headers, rows))

    # Latency delta confidence interval + sign test (is the slowdown real?)
    delta_ci = lat.get("delta_ci95_ms", {})
    sign = lat.get("delta_sign_test", {})
    if delta_ci:
        print(f"  Latency delta 95% CI: [{delta_ci.get('low', 0):.0f}, {delta_ci.get('high', 0):.0f}] ms")
    if sign:
        print(
            f"  Paired sign test: proxy slower on {sign.get('proxy_slower_turns', 0)} turns, "
            f"faster on {sign.get('proxy_faster_turns', 0)} turns"
        )

    # TTFT (time-to-first-token) — only present when --measure-ttft (default)
    ttft = summary.get("ttft_ms", {})
    if ttft.get("direct") or ttft.get("proxy"):
        print("\n  TIME TO FIRST TOKEN (ms)")
        ttft_headers = ["Metric", "Direct", "Proxy"]
        ttft_rows: list[list[str]] = []
        for stat in ["mean", "median", "p95"]:
            d_val = f"{ttft['direct'].get(stat, 'N/A'):,.0f}" if ttft.get("direct") else "N/A"
            p_val = f"{ttft['proxy'].get(stat, 'N/A'):,.0f}" if ttft.get("proxy") else "N/A"
            ttft_rows.append([stat.capitalize(), d_val, p_val])
        print(_fmt_table(ttft_headers, ttft_rows))

    # Proxy overhead (X-Proxy-Process-Ms) when the proxy reports it
    overhead = summary.get("proxy_overhead_ms")
    if overhead:
        print(f"  Proxy overhead (own processing): mean {overhead.get('mean', 0):.0f} ms, "
              f"p95 {overhead.get('p95', 0):.0f} ms")

    # ── Token usage ───────────────────────────────────────────────────
    tok = summary["tokens"]
    print("\n" + "-" * 72)
    print("  TOKEN USAGE")
    print("-" * 72)

    token_headers = ["Metric", "Value"]
    token_rows: list[list[str]] = [
        ["Total direct prompt tokens", f"{tok['total_direct_prompt']:,}"],
        ["Total proxy prompt tokens", f"{tok['total_proxy_prompt']:,}"],
        ["Total cached tokens (proxy)", f"{tok['total_cached_tokens']:,}"],
        ["Token savings vs direct", f"{tok['token_savings_pct']}%"],
    ]

    print(_fmt_table(token_headers, token_rows))

    # ── Cost model ─────────────────────────────────────────────────────
    cost = summary.get("cost_usd", {})
    if cost.get("price_in_per_1m") or cost.get("price_out_per_1m"):
        print("\n" + "-" * 72)
        print("  ESTIMATED COST (USD)")
        print("-" * 72)
        cost_rows: list[list[str]] = [
            ["Price in ($/1M tok)", f"{cost.get('price_in_per_1m', 0):.2f}"],
            ["Price out ($/1M tok)", f"{cost.get('price_out_per_1m', 0):.2f}"],
            ["Direct total", f"${cost.get('direct_total', 0):.4f}"],
            ["Proxy total", f"${cost.get('proxy_total', 0):.4f}"],
            ["Savings", f"${cost.get('savings_usd', 0):.4f} ({cost.get('savings_pct', 0)}%)"],
        ]
        print(_fmt_table(["Metric", "Value"], cost_rows))

    # Per-turn breakdown
    if tok.get("per_turn_direct"):
        pt_headers = ["Metric", "Direct", "Proxy"]
        pt_rows: list[list[str]] = []
        for stat in ["mean", "p95"]:
            d_val = f"{tok['per_turn_direct'].get(stat, 'N/A'):,.0f}"
            p_val = f"{tok['per_turn_proxy'].get(stat, 'N/A'):,.0f}"
            pt_rows.append([f"prompt_tokens ({stat})", d_val, p_val])

        cached_stats = tok.get("per_turn_cached", {})
        if cached_stats:
            c_mean = f"{cached_stats.get('mean', 0):,.0f}"
            pt_rows.append(["cached_tokens (mean)", "N/A", c_mean])

        print(_fmt_table(pt_headers, pt_rows))

    # ── Context window ────────────────────────────────────────────────
    cw = summary["context_window"]
    print("\n" + "-" * 72)
    print("  CONTEXT WINDOW UTILIZATION")
    print("-" * 72)
    cw_headers = ["Metric", "Value"]
    cw_rows: list[list[str]] = [
        ["Final proxy prompt tokens", f"{cw['final_prompt_tokens']:,}"],
        ["Final proxy prompt tokens (raw usage)", f"{cw.get('final_prompt_tokens_raw', 0):,}"],
        ["Final prompt token source", str(cw.get('final_prompt_tokens_source', 'usage'))],
        ["Final cached/missing usage response", str(cw.get('final_prompt_tokens_cached_response', False))],
        ["Max context window", f"{cw['max_context_window']:,}"],
        ["Utilization at last turn", f"{cw['utilization_pct']}%"],
    ]
    print(_fmt_table(cw_headers, cw_rows))

    # ── Eviction tracking ─────────────────────────────────────────────
    ev = summary.get("eviction", {})
    budget = ev.get("char_budget")
    if budget is not None:
        print("\n" + "-" * 72)
        print(f"  EVICTION TRACKING (budget={budget:,} chars)")
        print("-" * 72)
        ev_rows: list[list[str]] = [
            ["Total chars before optimization", f"{ev.get('total_chars_before_optimization', 0):,}"],
            ["Turns raw exceeds optimized target", str(ev.get("turns_raw_exceeds_optimized_target", 0))],
            ["Raw exceeds target at turns", ", ".join(str(t) for t in ev.get("raw_exceeds_optimized_target_at_turns") or []) or "never"],
            ["Compaction triggered at turns", ", ".join(str(t) for t in ev.get("compaction_triggered_at_turns") or []) or "never"],
            ["Budget eviction triggered at turns", ", ".join(str(t) for t in ev.get("eviction_triggered_at_turns") or []) or "never"],
        ]
        print(_fmt_table(["Metric", "Value"], ev_rows))
    # ── Per-round variance ────────────────────────────────────────────
    per_round = summary.get("per_round", {})
    pooled = per_round.get("_pooled") if per_round else None
    if per_round and pooled is not None:
        print("\n" + "-" * 72)
        print("  PER-ROUND VARIANCE (isolated proxy sessions)")
        print("-" * 72)
        pr_headers = ["Round", "Turns", "Sim mean", "Sim min", "Sim max", "Token savings %"]
        pr_rows: list[list[str]] = []
        for rnd in sorted(per_round.keys()):
            if rnd == "_pooled":
                continue
            rv = per_round[rnd]
            sim = rv.get("semantic_similarity", {})
            pr_rows.append([
                rnd,
                str(rv.get("turns", "N/A")),
                f"{sim.get('mean', 'N/A')}",
                f"{sim.get('min', 'N/A')}",
                f"{sim.get('max', 'N/A')}",
                f"{rv.get('token_savings_pct', 'N/A')}",
            ])
        pr_rows.append([
            "pooled",
            "-",
            f"{pooled.get('round_mean_of_means', 'N/A')}",
            f"{pooled.get('round_min_of_means', 'N/A')}",
            f"{pooled.get('round_min_of_mins', 'N/A')}",
            "-",
        ])
        print(_fmt_table(pr_headers, pr_rows))
    # ── Response quality ──────────────────────────────────────────────
    qual = summary.get("quality", {})
    print("\n" + "-" * 72)
    print("  RESPONSE QUALITY (direct vs proxy)")
    print("-" * 72)

    q_headers = ["Metric", "Mean", "Median", "Min", "Max"]
    qual_rows: list[list[str]] = []

    quality_metric_keys = [
        ("semantic_similarity", "Semantic similarity"),
        ("token_jaccard", "Token Jaccard"),
        ("rouge_l_f1", "ROUGE-L F1"),
        ("trigram_overlap", "Trigram overlap"),
        ("edit_similarity", "Edit similarity"),
        ("code_block_ratio", "Code block ratio"),
        ("markdown_structure_similarity", "Markdown structure"),
        ("length_ratio", "Length ratio"),
        ("vocabulary_richness_delta", "Vocab richness delta"),
        ("response_stability", "Response stability"),
        ("code_structure_consistency", "Code structure"),
    ]

    for key, label in quality_metric_keys:
        if qual.get(key):
            qs = qual[key]
            qual_rows.append([
                label,
                f"{qs.get('mean', 'N/A')}",
                f"{qs.get('median', 'N/A')}",
                f"{qs.get('min', 'N/A')}",
                f"{qs.get('max', 'N/A')}",
            ])

    # ROUGE precision/recall as separate rows
    for suffix in ["_precision", "_recall"]:
        key = f"rouge_l{suffix}"
        if qual.get(key):
            rl = qual[key]
            qual_rows.append([
                f"ROUGE-L{suffix.title()}",
                f"{rl.get('mean', 'N/A')}",
                f"{rl.get('median', 'N/A')}",
                f"{rl.get('min', 'N/A')}",
                f"{rl.get('max', 'N/A')}",
            ])

    if qual_rows:
        print(_fmt_table(q_headers, qual_rows))

    # ── Degradation flags ──────────────────────────────────────────
    low_semantic = qual.get("low_semantic_similarity_turns", 0)
    low_jaccard = qual.get("low_token_jaccard_turns", 0)
    truncation_count = qual.get("truncation_count", 0)
    verbosity_count = qual.get("model_verbosity_delta_turns", 0)
    code_loss = qual.get("code_block_loss_turns", 0)
    rouge_gap = qual.get("rouge_precision_recall_gap_mean", 0.0)
    quality_skipped = qual.get("quality_skipped_turns", 0)
    faith = qual.get("prompt_faithfulness")
    evict_recall = qual.get("evicted_content_recall")

    degradation_notes: list[str] = []
    if quality_skipped > 0:
        degradation_notes.append(
            f"{quality_skipped} turn(s) excluded from quality (one side errored/empty) — means may be optimistic"
        )
    # prompt_faithfulness / evicted_content_recall are the PRIMARY optimizer
    # signals: low values mean the compaction dropped needed context.
    if faith is not None and faith < 0.5:
        degradation_notes.append(f"LOW context faithfulness ({faith:.3f}) — optimizer dropped >50% of original prompt tokens")
    if evict_recall is not None and evict_recall < 0.3:
        degradation_notes.append(f"LOW evicted-content recall ({evict_recall:.3f}) — early-turn context not carried forward")
    if code_loss > 0:
        degradation_notes.append(f"{code_loss} turn(s) with lost code blocks")
    if truncation_count > 0:
        degradation_notes.append(f"{truncation_count} turn(s) severely truncated (length_ratio <0.5)")
    # verbosity_count is INFORMATIONAL only: the proxy cannot control response
    # length, so a verbose proxy response is model behavior, not optimizer
    # degradation. Reported as a note, not a degradation flag.
    if verbosity_count > 0:
        degradation_notes.append(f"{verbosity_count} turn(s) verbose inflation (length_ratio >2.0) — MODEL verbosity, not optimizer-caused")
    if rouge_gap and abs(rouge_gap) > 0.05:
        direction = "proxy loses recall" if rouge_gap < 0 else "proxy adds content"
        degradation_notes.append(f"ROUGE gap {rouge_gap:+.4f} → proxy {direction}")

    # ── Quality trend analysis ────────────────────────────────────────
    trend = summary.get("quality_trend", {})
    vocab = summary.get("vocab_richness", {})
    if trend:
        corr = trend.get("context_correlation")
        slope = trend.get("slope_per_10pct_ctx")
        if corr is not None and abs(corr) > 0.1:
            direction = "negative" if corr < 0 else "positive"
            degradation_notes.append(f"context-quality correlation {direction} (r={corr:.4f})")
        if slope is not None and abs(slope) > 0.01:
            degradation_notes.append(f"quality slope {slope:+.4f} per 10% context increase")

    vocab_mean = vocab.get("mean_delta")
    if vocab_mean is not None and vocab_mean > 0.1:
        degradation_notes.append(f"vocab richness delta mean={vocab_mean:.4f}")

    if degradation_notes:
        print("\n  Degradation indicators:")
        for note in degradation_notes:
            print(f"    ⚠ {note}")
        print()
    else:
        print("\n  All turns show strong response quality alignment.\n")

    # ── Correctness / integrity ───────────────────────────────────────
    correctness = summary["correctness"]
    print("-" * 72)
    print("  RESPONSE INTEGRITY")
    print("-" * 72)
    if correctness["total_foreign_markers"] == 0:
        print("\n  All proxy responses passed integrity check.")
        print("  No internal markers ([ARCHIVED], [REASONING], etc.) leaked.\n")
    else:
        print(f"\n  WARNING: {correctness['total_foreign_markers']} foreign marker(s) detected")
        if correctness["turns_with_markers"]:
            print(f"  In turns: {correctness['turns_with_markers']}\n")

    # ── Per-turn detail table (last round only, truncated to first 10 + last 3) ─
    turns = report.turns
    if len(turns) > 15:
        show_turns = list(range(10)) + list(range(len(turns) - 3, len(turns)))
    else:
        show_turns = range(len(turns))

    print("-" * 72)
    print("  PER-TURN DETAIL (selected turns)")
    print("-" * 72)

    detail_headers = [
        "Turn",
        "Ctx Turns",
        "Direct Toks",
        "Proxy Toks",
        "Cached",
        "Delta Tok",
        "Direct Lat",
        "Proxy Lat",
        "Lat Delta",
        "Chars In",
        "Eviction",
        "Length Ratio",
        "Semantic Sim",
        "Code Block",
        "Error",
    ]
    detail_rows: list[list[str]] = []
    for idx in show_turns:
        t = turns[idx]
        d_tok = f"{t.direct.prompt_tokens:,}" if hasattr(t.direct, 'prompt_tokens') else "-"
        p_tok = f"{t.proxy.prompt_tokens:,}" if hasattr(t.proxy, 'prompt_tokens') else "-"
        cached = f"{t.proxy.cached_tokens:,}" if hasattr(t.proxy, 'cached_tokens') else "-"
        tok_delta = t.token_delta
        d_lat = f"{t.direct.latency_ms:,.0f}ms" if hasattr(t.direct, 'latency_ms') and t.direct.latency_ms > 0 else "-"
        p_lat = f"{t.proxy.latency_ms:,.0f}ms" if hasattr(t.proxy, 'latency_ms') and t.proxy.latency_ms > 0 else "-"
        lat_delta = f"{t.latency_delta_ms:+,.0f}ms"

        # Length ratio
        lr = "N/A"
        if t.quality and t.quality.get("length_ratio") is not None:
            val = t.quality["length_ratio"]
            marker = " ⚠️" if val < 0.5 or val > 2.0 else ""
            lr = f"{val:.3f}{marker}"

        # Semantic similarity
        sim = "N/A"
        if t.quality and t.quality.get("semantic_similarity") is not None:
            val = t.quality["semantic_similarity"]
            marker = " ⚠️" if val < 0.75 else ""
            sim = f"{val:.3f}{marker}"

        # Code block ratio
        cb = "N/A"
        if t.quality and t.quality.get("code_block_ratio") is not None:
            val = t.quality["code_block_ratio"]
            marker = " ⚠️" if val < 1.0 else ""
            cb = f"{val:.2f}{marker}"

        ctx_turns = t.direct.total_turns_at_request if hasattr(t.direct, 'total_turns_at_request') else "?"
        chars_in = f"{t.proxy.chars_before_optimization:,}" if hasattr(t.proxy, 'chars_before_optimization') and t.proxy.chars_before_optimization > 0 else "-"
        # Eviction = the proxy actually sent fewer tokens than direct this turn
        # (real compaction/optimization), not merely raw input exceeding the
        # post-optimization target (which is expected for long scenarios).
        eviction_flag = ""
        if hasattr(t.direct, 'prompt_tokens') and hasattr(t.proxy, 'prompt_tokens'):
            eviction_flag = "YES ⚠️" if t.proxy.prompt_tokens < t.direct.prompt_tokens else "no"
        detail_rows.append([
            str(t.turn_index),
            str(ctx_turns),
            d_tok, p_tok, cached, f"{tok_delta:+}",
            d_lat, p_lat, lat_delta, chars_in, eviction_flag, lr, sim, cb,
            t.proxy.error or "",
        ])

    print(_fmt_table(detail_headers, detail_rows))
    print()


def run_all_scenarios(args) -> int:
    """Run all scenarios and produce aggregated metrics. Returns the regression
    gate exit code (0 = pass, 2 = fail, review03.md §10)."""
    all_reports: dict[str, BenchmarkReport] = {}

    global _HUMAN_OUTPUT_TO_STDERR
    _HUMAN_OUTPUT_TO_STDERR = args.json_output

    _status(args, f"\n  Running all scenarios: {args.turns} turns x {args.rounds} round(s)")
    _status(args, f"  Model: {MODEL_ID}")
    _status(args, f"  Lemonade: {LEMONADE_URL}")
    _status(args, f"  Proxy: http://127.0.0.1:{args.port}/v1")

    if args.budget is not None:
        os.environ["MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS"] = str(args.budget)
        _status(args, f"  Context char budget: {args.budget}")

    _apply_profile_overrides(args)

    _start_proxy(args.port)

    try:
        # `opencode` is an alias of `fixtures` (both bind the same
        # `_OPENCODE_SCENARIO_TASKS`), so running it in the "all" loop would
        # execute the identical scenario twice and double-weight it in the
        # aggregated mean / regression gate. Skip the alias here; `--scenario
        # opencode` still works as a standalone invocation.
        for scenario_name in SCENARIOS:
            if scenario_name == "opencode":
                continue
            print(f"\n  Running scenario: {scenario_name}")
            report = run_benchmark(
                num_turns=args.turns,
                rounds=args.rounds,
                max_tokens=args.max_tokens,
                proxy_port=args.port,
                budget=args.budget,
                scenario=scenario_name,
                agentic=args.agentic,
                temperature=args.temperature,
                measure_ttft=args.measure_ttft,
                context_window=args.context_window,
                price_in=args.price_in,
                price_out=args.price_out,
                max_wall_seconds=args.max_wall_seconds,
            )
            all_reports[scenario_name] = report

        # Aggregate all reports
        aggregated = _aggregate_reports(all_reports)

        if args.json_output:
            json.dump(aggregated, sys.stdout, indent=2)
            print()
        else:
            print("\n" + "=" * 72)
            print("  AGGREGATED BENCHMARK RESULTS (all scenarios)")
            print("=" * 72)
            _print_aggregated(aggregated)

        # Regression gate across the aggregated mean (review03.md §10).
        agg = aggregated.get("aggregated", {})
        args._agg_snapshot = agg  # let the gate read semantic_similarity floor
        gate_value, gate_metric = _select_aggregated_gate(agg)
        gate = _check_similarity_gate(args, gate_value, gate_metric)
        if gate:
            return gate

        # Baseline regression gate (optional, --baseline): diff the aggregated
        # means against a prior --json report. The aggregated dict carries
        # token_savings_pct and the headline quality means in the same shape
        # the gate expects, so we adapt it into a synthetic summary.
        if args.baseline:
            baseline = _load_baseline(args.baseline)
            if baseline is not None:
                current = {
                    "tokens": {
                        "token_savings_pct": agg.get("token_savings_pct", {}).get("mean", 0.0)
                    },
                    "cost_usd": {
                        "savings_pct": agg.get("cost_savings_pct", {}).get("mean", 0.0)
                    },
                    "quality": {
                        qm: agg.get(qm, {}).get("mean", 0.0)
                        for qm in ("rouge_l_f1", "token_jaccard", "code_block_ratio", "edit_similarity")
                    },
                }
                _print_baseline_diff(current, baseline)
                bgate = _check_baseline_gate(args, current, baseline)
                if bgate:
                    return bgate

    finally:
        _stop_proxy()


def _aggregate_reports(reports: dict[str, BenchmarkReport]) -> dict[str, Any]:
    """Aggregate metrics from all scenario reports."""
    aggregated: dict[str, Any] = {
        "scenarios": list(reports.keys()),
        "config": {
            "model": MODEL_ID,
            "lemonade_url": LEMONADE_URL,
        },
        "per_scenario": {},
        "aggregated": {},
    }

    # Collect all metrics
    all_latencies: list[float] = []
    all_semantic: list[float] = []
    all_rouge: list[float] = []
    all_jaccard: list[float] = []
    all_token_savings: list[float] = []
    all_ttft: list[float] = []
    all_proxy_overhead: list[float] = []
    all_cost_savings: list[float] = []
    all_quality_skipped: list[int] = []
    all_code_block: list[float] = []
    all_edit_sim: list[float] = []

    for name, report in reports.items():
        summary = report.summary()
        aggregated["per_scenario"][name] = {
            "num_turns": summary.get("num_turns", 0),
            "latency_mean_ms": summary.get("latency_ms", {}).get("proxy", {}).get("mean", 0),
            "semantic_similarity_mean": summary.get("semantic_similarity", {}).get("mean", 0),
            "rouge_l_f1_mean": summary.get("quality", {}).get("rouge_l_f1", {}).get("mean", 0),
            "token_jaccard_mean": summary.get("quality", {}).get("token_jaccard", {}).get("mean", 0),
            "code_block_ratio_mean": summary.get("quality", {}).get("code_block_ratio", {}).get("mean", 0),
            "edit_similarity_mean": summary.get("quality", {}).get("edit_similarity", {}).get("mean", 0),
            "token_savings_pct": summary.get("tokens", {}).get("token_savings_pct", 0),
            "ttft_proxy_mean_ms": summary.get("ttft_ms", {}).get("proxy", {}).get("mean", 0),
            "proxy_overhead_mean_ms": summary.get("proxy_overhead_ms", {}).get("mean", 0),
            "cost_savings_pct": summary.get("cost_usd", {}).get("savings_pct", 0),
            "quality_skipped_turns": summary.get("quality", {}).get("quality_skipped_turns", 0),
        }

        # Collect for aggregation
        lat = summary.get("latency_ms", {}).get("proxy", {}).get("mean", 0)
        if lat:
            all_latencies.append(lat)

        sem = summary.get("quality", {}).get("semantic_similarity", {}).get("mean", 0)
        if sem:
            all_semantic.append(sem)

        rg = summary.get("quality", {}).get("rouge_l_f1", {}).get("mean", 0)
        if rg:
            all_rouge.append(rg)

        jc = summary.get("quality", {}).get("token_jaccard", {}).get("mean", 0)
        if jc:
            all_jaccard.append(jc)

        cb = summary.get("quality", {}).get("code_block_ratio", {}).get("mean", 0)
        if cb:
            all_code_block.append(cb)

        es = summary.get("quality", {}).get("edit_similarity", {}).get("mean", 0)
        if es:
            all_edit_sim.append(es)

        ts = summary.get("tokens", {}).get("token_savings_pct", 0)
        all_token_savings.append(ts)

        ttft = summary.get("ttft_ms", {}).get("proxy", {}).get("mean", 0)
        if ttft:
            all_ttft.append(ttft)

        pov = summary.get("proxy_overhead_ms", {}).get("mean", 0)
        if pov:
            all_proxy_overhead.append(pov)

        cs = summary.get("cost_usd", {}).get("savings_pct", 0)
        if cs:
            all_cost_savings.append(cs)

        qs = summary.get("quality", {}).get("quality_skipped_turns", 0)
        all_quality_skipped.append(qs)

    # Compute aggregated stats
    if all_latencies:
        aggregated["aggregated"]["latency_ms"] = {
            "mean": round(statistics.mean(all_latencies), 2),
            "min": round(min(all_latencies), 2),
            "max": round(max(all_latencies), 2),
        }

    if all_semantic:
        aggregated["aggregated"]["semantic_similarity"] = {
            "mean": round(statistics.mean(all_semantic), 4),
            "min": round(min(all_semantic), 4),
            "max": round(max(all_semantic), 4),
        }

    if all_rouge:
        aggregated["aggregated"]["rouge_l_f1"] = {
            "mean": round(statistics.mean(all_rouge), 4),
            "min": round(min(all_rouge), 4),
            "max": round(max(all_rouge), 4),
        }

    if all_jaccard:
        aggregated["aggregated"]["token_jaccard"] = {
            "mean": round(statistics.mean(all_jaccard), 4),
            "min": round(min(all_jaccard), 4),
            "max": round(max(all_jaccard), 4),
        }

    if all_code_block:
        aggregated["aggregated"]["code_block_ratio"] = {
            "mean": round(statistics.mean(all_code_block), 4),
            "min": round(min(all_code_block), 4),
            "max": round(max(all_code_block), 4),
        }

    if all_edit_sim:
        aggregated["aggregated"]["edit_similarity"] = {
            "mean": round(statistics.mean(all_edit_sim), 4),
            "min": round(min(all_edit_sim), 4),
            "max": round(max(all_edit_sim), 4),
        }

    aggregated["aggregated"]["token_savings_pct"] = {
        "mean": round(statistics.mean(all_token_savings), 2),
        "min": round(min(all_token_savings), 2),
        "max": round(max(all_token_savings), 2),
    }

    if all_ttft:
        aggregated["aggregated"]["ttft_ms"] = {
            "mean": round(statistics.mean(all_ttft), 2),
            "min": round(min(all_ttft), 2),
            "max": round(max(all_ttft), 2),
        }

    if all_proxy_overhead:
        aggregated["aggregated"]["proxy_overhead_ms"] = {
            "mean": round(statistics.mean(all_proxy_overhead), 2),
            "min": round(min(all_proxy_overhead), 2),
            "max": round(max(all_proxy_overhead), 2),
        }

    if all_cost_savings:
        aggregated["aggregated"]["cost_savings_pct"] = {
            "mean": round(statistics.mean(all_cost_savings), 2),
            "min": round(min(all_cost_savings), 2),
            "max": round(max(all_cost_savings), 2),
        }

    aggregated["aggregated"]["quality_skipped_turns"] = {
        "total": sum(all_quality_skipped),
        "mean": round(statistics.mean(all_quality_skipped), 2),
    }

    return aggregated


def _print_aggregated(aggregated: dict[str, Any]) -> None:
    """Print aggregated results in human-readable format."""
    print("\n  Per-Scenario Summary:")
    for name, data in aggregated.get("per_scenario", {}).items():
        print(f"    {name}:")
        print(f"      Latency: {data.get('latency_mean_ms', 0):.0f}ms")
        print(f"      Semantic similarity: {data.get('semantic_similarity_mean', 0):.4f}")
        print(f"      Token savings: {data.get('token_savings_pct', 0):.1f}%")
        if data.get("ttft_proxy_mean_ms"):
            print(f"      TTFT (proxy): {data.get('ttft_proxy_mean_ms', 0):.0f}ms")
        if data.get("proxy_overhead_mean_ms"):
            print(f"      Proxy overhead: {data.get('proxy_overhead_mean_ms', 0):.0f}ms")
        if data.get("cost_savings_pct"):
            print(f"      Cost savings: {data.get('cost_savings_pct', 0):.1f}%")
        if data.get("quality_skipped_turns"):
            print(f"      Quality-skipped turns: {data.get('quality_skipped_turns', 0)}")

    print("\n  Aggregated Metrics:")
    agg = aggregated.get("aggregated", {})
    if "latency_ms" in agg:
        print(f"    Latency (mean): {agg['latency_ms']['mean']:.0f}ms")
    if "semantic_similarity" in agg:
        print(f"    Semantic similarity (mean): {agg['semantic_similarity']['mean']:.4f}")
    if "token_savings_pct" in agg:
        print(f"    Token savings (mean): {agg['token_savings_pct']['mean']:.1f}%")
    if "ttft_ms" in agg:
        print(f"    TTFT (proxy, mean): {agg['ttft_ms']['mean']:.0f}ms")
    if "proxy_overhead_ms" in agg:
        print(f"    Proxy overhead (mean): {agg['proxy_overhead_ms']['mean']:.0f}ms")
    if "cost_savings_pct" in agg:
        print(f"    Cost savings (mean): {agg['cost_savings_pct']['mean']:.1f}%")
    if "quality_skipped_turns" in agg:
        print(f"    Quality-skipped turns (total): {agg['quality_skipped_turns']['total']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn benchmark: direct Lemonade vs moeptimizer proxy"
    )
    parser.add_argument("--turns", type=int, default=10, help="Number of conversation turns")
    parser.add_argument("--rounds", type=int, default=5, help="Number of full conversation rounds (default 5; run >=5 so per-round variance and bootstrap CIs are stable)")
    parser.add_argument("--context-window", type=int, default=262144, help="Model context-window size in tokens (used for utilization %%). Defaults to 262144 (Qwen3.6-35B-MTP).")
    parser.add_argument("--price-in", type=float, default=0.0, dest="price_in", help="Input token price in USD per 1M tokens. Enables the estimated-cost section.")
    parser.add_argument("--price-out", type=float, default=0.0, dest="price_out", help="Output token price in USD per 1M tokens. Enables the estimated-cost section.")
    parser.add_argument("--max-wall-seconds", type=float, default=None, dest="max_wall_seconds", help="Abort remaining rounds (or scenarios, with --scenario all) once this wall-clock budget is exceeded.")
    parser.add_argument("--baseline", type=str, default=None, help="Path to a prior JSON report (from --json) to diff against. Prints per-metric deltas and fails the regression gate if token-savings or quality regress beyond --baseline-tolerance.")
    parser.add_argument("--baseline-tolerance", type=float, default=0.05, dest="baseline_tolerance", help="Absolute tolerance for baseline diffs (e.g. 0.05 = quality mean may drop by 0.05 before the gate fails).")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per response. Default 8192: an agentic coding turn may rewrite a whole source file inside a single tool-call argument (the largest fixture, loader.py, is ~21KB ~= 6.2K tokens; JSON-string escaping + a short reasoning preamble push the worst case to ~7.6-8.3K). A smaller value (e.g. 1024) truncates those tool-call arguments mid-string, so llama.cpp fails to parse the unterminated JSON and returns HTTP 500.")
    parser.add_argument("--port", type=int, default=MOEPT_PORT, help="Proxy server port")
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output report as JSON to stdout"
    )
    parser.add_argument(
        "--dump-responses", action="store_true", dest="dump_responses",
        help="Print direct vs proxy response pairs for quality inspection"
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Override max_optimized_chars (char budget). Eviction triggers when context exceeds this.",
    )
    parser.add_argument(
        "--profile", type=str, default="balanced",
        choices=["quality", "balanced", "aggressive"],
        help="Context optimization profile for the proxy. 'quality' maximizes fidelity "
             "(no summarization/RAG, only boundary compression); 'aggressive' favors token "
             "savings with top-only eviction.",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="Regression gate (review03.md §10): exit non-zero if the composite "
             "lexical quality (mean of code_block_ratio, rouge_l_f1, edit_similarity) "
             "falls below this threshold. Use in CI to block quality regressions.",
    )
    parser.add_argument(
        "--min-semantic", type=float, default=None,
        help="Hard floor on embedding semantic_similarity for the regression gate "
             "(review P0.6). Catches the 'proxy dropped all task context' collapse "
             "where code_block_ratio is high but semantic_similarity ~0. Defaults "
             "to 0.5 * --min-similarity when unset.",
    )
    parser.add_argument(
        "--scenario", type=str, default="default",
        choices=[*SCENARIOS.keys(), "all"],
        help="Real-life coding scenario: debug, refactor, feature, default, fixtures, opencode, or all",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for both direct and proxy runs. Defaults to 0 "
             "(deterministic) so quality metrics are reproducible and the "
             "regression gate is not confounded by model sampling variance. "
             "Raise only to stress-test nondeterminism.",
    )
    parser.add_argument(
        "--no-agentic", dest="agentic", action="store_false",
        help="Disable OpenCode-style agent payloads (user task + tool calls + tool "
             "outputs); send plain user messages instead. Agentic mode is the default "
             "for every scenario, since real coding clients send tool traffic.",
    )
    parser.add_argument(
        "--no-measure-ttft", dest="measure_ttft", action="store_false",
        help="Disable TTFT measurement and per-turn prefix-cache hit capture. By "
             "default the benchmark streams responses to measure time-to-first-token "
             "(TTFT) and capture the proxy's per-turn prefix-cache hit count; "
             "conversations stay contiguous (full multi-turn sessions), only the "
             "transport switches to SSE. Pass this to fall back to non-streaming.",
    )
    args = parser.parse_args()

    # Handle "all" scenario - run all individual scenarios
    if args.scenario == "all":
        return run_all_scenarios(args)

    global _HUMAN_OUTPUT_TO_STDERR
    _HUMAN_OUTPUT_TO_STDERR = args.json_output

    # Get scenario tasks
    scenario = SCENARIOS.get(args.scenario, SCENARIOS["default"])
    _status(args, f"\n  Starting benchmark: {args.turns} turns x {args.rounds} round(s)")
    _status(args, f"  Scenario: {args.scenario} - {scenario['description']}")
    _status(args, f"  Model: {MODEL_ID}")
    _status(args, f"  Lemonade: {LEMONADE_URL}")
    _status(args, f"  Proxy: http://127.0.0.1:{args.port}/v1")

    # Inject budget/profile overrides so the started proxy picks them up
    if args.budget is not None:
        os.environ["MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS"] = str(args.budget)
        _status(args, f"  Context char budget: {args.budget} (eviction will trigger when exceeded)")

    _apply_profile_overrides(args)

    # Auto-start proxy if not already running
    _start_proxy(args.port)

    try:
        report = run_benchmark(
            num_turns=args.turns,
            rounds=args.rounds,
            max_tokens=args.max_tokens,
            proxy_port=args.port,
            budget=args.budget,
            scenario=args.scenario,
            agentic=args.agentic,
            temperature=args.temperature,
            measure_ttft=args.measure_ttft,
            context_window=args.context_window,
            price_in=args.price_in,
            price_out=args.price_out,
            max_wall_seconds=args.max_wall_seconds,
        )

        if args.json_output or not args.dump_responses:
            # Print report (or JSON only)
            if args.json_output:
                json.dump(report.summary(), sys.stdout, indent=2)
                print()
            else:
                print_report(report)

        # Regression gate (review03.md §10): fail the run if semantic
        # similarity drops below the requested threshold. Intended for CI.
        # Gate on the mean-of-round-means (per-round means averaged) so a single
        # noisy round cannot swing the decision, while a systematic regression
        # across rounds still fails. Falls back to the pooled mean for 1 round.
        _summary = report.summary()
        args._agg_snapshot = {"semantic_similarity": _summary.get("semantic_similarity", {})}
        gate_value, gate_metric = _select_quality_gate(_summary)
        gate = _check_similarity_gate(args, gate_value, gate_metric)
        if gate:
            return gate

        # Baseline regression gate (optional, --baseline): diff against a prior
        # --json report and fail if token/cost savings or quality regress.
        if args.baseline:
            baseline = _load_baseline(args.baseline)
            if baseline is not None:
                _print_baseline_diff(_summary, baseline)
                bgate = _check_baseline_gate(args, _summary, baseline)
                if bgate:
                    return bgate

        if args.dump_responses:
            print("\n" + "=" * 72)
            print("  RESPONSE PAIRS (direct vs proxy)")
            print("=" * 72)
            for t in report.turns:
                d_preview = t.direct.content_preview or "(error/no response)"
                p_preview = t.proxy.content_preview or "(error/no response)"

                # Find the user prompt for this turn from messages_copy context
                ctx_turns = t.direct.total_turns_at_request if hasattr(t.direct, 'total_turns_at_request') else "?"

                print(f"\n  Turn {t.turn_index} (context: {ctx_turns} turns)")
                print(f"    Direct ({t.direct.response_chars} chars):")
                for line in d_preview.split("\n"):
                    print(f"      | {line}")
                print(f"    Proxy  ({t.proxy.response_chars} chars):")
                for line in p_preview.split("\n"):
                    print(f"      | {line}")

                if t.quality:
                    q = t.quality
                    parts = []
                    for key in ["prompt_faithfulness", "evicted_content_recall", "semantic_similarity", "token_jaccard", "rouge_l_f1", "edit_similarity", "code_block_ratio", "length_ratio", "response_stability", "code_structure_consistency"]:
                        val = q.get(key)
                        if val is not None:
                            parts.append(f"{key}={val:.4f}")
                    print(f"    Quality: {', '.join(parts)}")

                # Show degradation markers
                if t.quality and isinstance(t.quality.get("prompt_faithfulness"), float) and t.quality["prompt_faithfulness"] < 0.5:
                    print(f"    ⚠️  LOW CONTEXT FAITHFULNESS ({t.quality['prompt_faithfulness']:.3f}) — optimizer dropped >50% of original prompt")
                if t.quality and isinstance(t.quality.get("evicted_content_recall"), float) and t.quality["evicted_content_recall"] < 0.3:
                    print(f"    ⚠️  LOW EVICTED-CONTENT RECALL ({t.quality['evicted_content_recall']:.3f}) — early-turn context not carried forward")
                if t.quality and isinstance(t.quality.get("semantic_similarity"), float) and t.quality["semantic_similarity"] < 0.75:
                    print(f"    ⚠️  LOW SEMANTIC SIMILARITY ({t.quality['semantic_similarity']:.3f})")
                if t.quality and isinstance(t.quality.get("length_ratio"), float):
                    lr = t.quality["length_ratio"]
                    if lr < 0.5:
                        print(f"    ⚠️  SEVERE TRUNCATION (length_ratio={lr:.3f})")
                    elif lr > 2.0:
                        print(f"    ⚠️  VERBOSE INFLATION (length_ratio={lr:.3f}) — MODEL verbosity, not optimizer-caused")
                if t.quality and isinstance(t.quality.get("code_block_ratio"), float) and t.quality["code_block_ratio"] < 1.0:
                    print(f"    ⚠️  CODE BLOCK LOSS ({t.quality['code_block_ratio']:.2f})")
                    # Show full content for debugging
                    import re as _re
                    d_code_blocks = _re.findall(r"```(?:\w*)\n(.*?)\`\`\`", t.direct.content_preview or "", _re.DOTALL)
                    p_code_blocks = _re.findall(r"```(?:\w*)\n(.*?)\`\`\`", t.proxy.content_preview or "", _re.DOTALL)
                    print(f"    Direct code blocks: {len(d_code_blocks)}, Proxy code blocks: {len(p_code_blocks)}")

    finally:
        _stop_proxy()


def _select_quality_gate(summary: dict[str, Any]) -> tuple[float, str]:
    """Pick the best available quality metric for the regression gate.

    Prefers the PRIMARY optimizer signal — ``prompt_faithfulness`` (how much
    of the original context survived compaction) and ``evicted_content_recall``
    — because the proxy compacts ONLY the input context, so context retention
    is the question that actually validates it. Falls back to robust *lexical*
    signals (code-block preservation, ROUGE-L F1, token Jaccard, edit
    similarity) over embedding cosine similarity. Embeddings are weak on code —
    two equivalent code blocks with renamed variables score low cosine while two
    verbose-but-wrong answers score high — so ``semantic_similarity`` is
    treated as informational and only used as a last resort.
    Returns ``(value, metric_name)``.
    """
    qual = summary.get("quality", {})

    # Primary optimizer signal: context retention (1.0 = nothing lost).
    for metric in ("prompt_faithfulness", "evicted_content_recall"):
        val = (qual.get(metric) or {}).get("mean")
        if val is not None and val > 0:
            return val, metric

    # Composite lexical battery (review P0.6): a single lexical metric can pass
    # while the proxy dropped all task context, so gate on the mean of the
    # lexical signals that best capture "did the proxy preserve the task".
    lexical = ("code_block_ratio", "rouge_l_f1", "edit_similarity")
    vals = [(qual.get(m) or {}).get("mean") for m in lexical]
    present = [v for v in vals if isinstance(v, (int, float)) and v > 0]
    if present:
        return sum(present) / len(present), "composite_lexical"

    # Fallback to semantic similarity (informational only), using the robust
    # round-mean-of-means when multiple rounds were run.
    per_round = summary.get("per_round", {})
    pooled = per_round.get("_pooled", {}) if per_round else {}
    sem_robust = pooled.get("round_mean_of_means") if pooled else None
    if sem_robust is not None and sem_robust > 0:
        return sem_robust, "semantic_similarity"
    sem_pooled = (summary.get("semantic_similarity") or {}).get("mean")
    if sem_pooled is not None and sem_pooled > 0:
        return sem_pooled, "semantic_similarity"

    return 0.0, "none"


def _select_aggregated_gate(agg: dict[str, Any]) -> tuple[float, str]:
    """Compute the composite regression-gate value from lexical quality signals.

    A single-metric gate hides collapse: ``code_block_ratio`` (0.733) can pass a
    0.85 floor while ``semantic_similarity`` (0.248) is catastrophic, because the
    proxy removed the code the model needed. We therefore gate on a **composite**
    of the lexical signals that best capture "did the proxy preserve the task":

      composite = mean(code_block_ratio, rouge_l_f1, edit_similarity)

    and additionally require ``semantic_similarity`` to clear a hard floor
    (review P0.6). The returned metric name documents which statistic drove the
    gate. Embedding cosine is intentionally NOT the primary gate (weak on code).
    """
    lexical = ("code_block_ratio", "rouge_l_f1", "edit_similarity")
    vals = [(agg.get(m) or {}).get("mean") for m in lexical]
    present = [v for v in vals if isinstance(v, (int, float)) and v > 0]
    if present:
        composite = sum(present) / len(present)
        return composite, "composite_lexical"
    sem = (agg.get("semantic_similarity") or {}).get("mean")
    if sem is not None and sem > 0:
        return sem, "semantic_similarity"
    return 0.0, "none"


def _check_similarity_gate(
    args: argparse.Namespace, gate_value: float, gate_metric: str = "composite_lexical"
) -> int:
    """Return 0 if the regression gate passes, 2 if it fails (review03.md §10).

    ``gate_value`` is the composite lexical statistic. In addition to the
    composite floor (``--min-similarity``), a hard floor is enforced on
    ``semantic_similarity`` so a proxy that drops all task context (code_block_ratio
    high but semantic_similarity ~0) cannot pass (review P0.6). The semantic floor
    defaults to 0.5 of ``--min-similarity`` when not set explicitly.
    """
    if args.min_similarity is None:
        return 0
    failed: list[str] = []
    if gate_value < args.min_similarity:
        failed.append(
            f"{gate_metric}={gate_value:.4f} < --min-similarity={args.min_similarity:.4f}"
        )
    # Hard floor on embedding similarity (catches "no code / no context" collapse).
    sem = (args._agg_snapshot or {}).get("semantic_similarity", {}).get("mean") if hasattr(args, "_agg_snapshot") else None
    sem_floor = getattr(args, "min_semantic", None)
    if sem_floor is None:
        sem_floor = args.min_similarity * 0.5
    if sem is not None and sem > 0 and sem < sem_floor:
        failed.append(f"semantic_similarity={sem:.4f} < floor={sem_floor:.4f}")
    if failed:
        _status(args, "\n  ❌ REGRESSION GATE FAILED: " + "; ".join(failed))
        return 2
    _status(
        args,
        f"\n  ✅ Regression gate passed: {gate_metric}={gate_value:.4f} "
        f">= --min-similarity={args.min_similarity:.4f}",
    )
    return 0


def _load_baseline(path: str) -> dict[str, Any] | None:
    """Load a prior JSON report produced by ``--json``. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ⚠️  Could not read baseline report {path!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"  ⚠️  Baseline report {path!r} is not a JSON object", file=sys.stderr)
        return None
    return data


def _baseline_get(d: dict[str, Any], *keys) -> float | None:
    """Drill into a nested summary dict and return a numeric value, or None."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if isinstance(cur, dict):
        cur = cur.get("mean")
    return cur if isinstance(cur, (int, float)) else None


def _print_baseline_diff(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Print a concise delta table of key metrics vs the baseline."""
    rows = [
        ("token_savings_pct", "tokens", "token_savings_pct", "{:.2f}pp", 1.0),
        ("cost_savings_pct", "cost_usd", "savings_pct", "{:.2f}pp", 1.0),
        ("rouge_l_f1", "quality", "rouge_l_f1", "{:.4f}", 1.0),
        ("token_jaccard", "quality", "token_jaccard", "{:.4f}", 1.0),
        ("code_block_ratio", "quality", "code_block_ratio", "{:.4f}", 1.0),
        ("edit_similarity", "quality", "edit_similarity", "{:.4f}", 1.0),
        ("prompt_faithfulness", "quality", "prompt_faithfulness", "{:.4f}", 1.0),
        ("evicted_content_recall", "quality", "evicted_content_recall", "{:.4f}", 1.0),
        ("semantic_similarity", "semantic_similarity", "mean", "{:.4f}", 1.0),
        ("proxy_latency_mean_ms", "latency_ms", "proxy", "mean", "{:.1f}ms", 1.0),
        ("ttft_proxy_mean_ms", "ttft_ms", "proxy", "mean", "{:.1f}ms", 1.0),
    ]
    print("\n  Baseline diff (current vs baseline):")
    print(f"    {'metric':<24}{'baseline':>14}{'current':>14}{'delta':>14}")
    print("    " + "-" * 64)
    for label, *keys, fmt, _scale in rows:
        base = _baseline_get(baseline, *keys)
        cur = _baseline_get(current, *keys)
        if base is None and cur is None:
            continue
        b_str = fmt.format(base) if base is not None else "n/a"
        c_str = fmt.format(cur) if cur is not None else "n/a"
        if base is not None and cur is not None:
            delta = cur - base
            d_str = (("+" if delta >= 0 else "") + fmt.format(delta))
        else:
            d_str = "n/a"
        print(f"    {label:<24}{b_str:>14}{c_str:>14}{d_str:>14}")


def _check_baseline_gate(
    args: argparse.Namespace,
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> int:
    """Diff *current* against *baseline* and fail the regression gate on
    unacceptable regressions (token/cost savings or headline quality).

    ``--baseline-tolerance`` is expressed on the 0..1 quality scale (e.g. 0.05 =
    a quality mean may drop by 0.05). Token/cost savings are in percentage
    points, so the same tolerance is applied as ``tol * 100`` pp. Returns 0 if
    within tolerance, 2 if a regression exceeds it.
    """
    tol = args.baseline_tolerance
    failures: list[str] = []

    # Token & cost savings (higher is better; a drop is a regression).
    for key, *bkeys in (
        ("token_savings_pct", "tokens", "token_savings_pct"),
        ("cost savings_pct", "cost_usd", "savings_pct"),
    ):
        cur = _baseline_get(current, *bkeys) or 0.0
        base = _baseline_get(baseline, *bkeys) or 0.0
        drop = base - cur
        if base > 0 and drop > tol * 100:
            failures.append(
                f"{key} regressed {drop:.2f}pp (baseline {base:.2f} -> {cur:.2f}, tol {tol*100:.2f}pp)"
            )

    # Headline quality metrics (lower is a regression). prompt_faithfulness
    # and evicted_content_recall are the PRIMARY optimizer signals (context
    # retention); the lexical battery is secondary for this use case.
    for qm in ("prompt_faithfulness", "evicted_content_recall", "code_block_ratio", "rouge_l_f1", "token_jaccard", "edit_similarity"):
        cur = _baseline_get(current, "quality", qm) or 0.0
        base = _baseline_get(baseline, "quality", qm) or 0.0
        drop = base - cur
        if drop > tol:
            failures.append(
                f"{qm} regressed {drop:.4f} (baseline {base:.4f} -> {cur:.4f}, tol {tol:.4f})"
            )

    if failures:
        _status(args, "\n  ❌ BASELINE REGRESSION GATE FAILED:")
        for f in failures:
            _status(args, f"    - {f}")
        return 2

    _status(args, f"\n  ✅ Baseline regression gate passed (tolerance {tol:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
