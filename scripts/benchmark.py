#!/usr/bin/env python3
"""Multi-turn benchmark: direct Lemonade vs moeptimizer proxy.

Compares latency, token usage, context-window efficiency, and response quality
across realistic multi-turn conversations that grow the context window.

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

    # Real-life coding scenarios
    python scripts/benchmark.py --scenario debug --turns 15
    python scripts/benchmark.py --scenario debug_long --turns 30
    python scripts/benchmark.py --scenario refactor_long --turns 30
    python scripts/benchmark.py --scenario feature_long --turns 30
    python scripts/benchmark.py --scenario default_long --turns 30

    # Run all scenarios
    python scripts/benchmark.py --scenario all --turns 10

    # Stress test with large context
    python scripts/benchmark.py --turns 50 --budget 8000
"""

from __future__ import annotations

import argparse
import json
import os
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

# ---------------------------------------------------------------------------
# Long benchmark scenarios
# ---------------------------------------------------------------------------

def _long_refactor_code_snippet(turn: int) -> str:
    """Return a realistic code snippet for a long refactor benchmark."""
    extra = ""
    if turn % 3 == 0:
        extra = """

@dataclass(slots=True)
class Config:
    input_path: Path = Path("users.jsonl")
    dry_run: bool = False
"""
    elif turn % 3 == 1:
        extra = """

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def active_count(self) -> int:
        return sum(1 for user in self.repository.load() if user.active)
"""
    else:
        extra = """

def main() -> None:
    repository = UserRepository(Path("users.jsonl"))
    users = repository.load()
    print(summarize(users))
"""
    return f"""from dataclasses import dataclass
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
    return {{"count": len(users), "active": sum(1 for user in users if user.active)}}
{extra}
"""


LONG_REFACTOR_INSTRUCTIONS = [
    "Start by turning this into a typed, testable module with dataclasses and a clear separation between IO and business logic.",
    "Now add a `UserRepository` class that reads from a JSONL file, validates the schema, and returns `User` objects instead of raw dicts.",
    "Next, introduce a `Config` dataclass that loads from environment variables and supports a `dry_run` flag.",
    "Refactor the summarizer into a service class with dependency injection so tests can swap in a fake repository.",
    "Add structured logging to the service so each step emits a compact event object.",
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

LONG_REFACTOR_TASKS = [
    f"""{instruction}

Conversation constraints:
- Preserve the existing public API unless the request explicitly asks to change it.
- Prefer small, incremental patches over broad rewrites.
- Keep code blocks complete enough to compile or explain exactly what changed.
- Mention any tradeoff that affects latency, cache stability, or testability.

Current code:

```python
{_long_refactor_code_snippet(index)}
```

Please apply the requested change and keep the response concise."""
    for index, instruction in enumerate(LONG_REFACTOR_INSTRUCTIONS)
]


def _build_long_tasks(instructions: list[str], code_snippet_fn) -> list[str]:
    """Build 30-turn long benchmark tasks from instructions and a code snippet generator."""
    return [
        f"""{instruction}

Conversation constraints:
- Preserve the existing public API unless the request explicitly asks to change it.
- Prefer small, incremental patches over broad rewrites.
- Keep code blocks complete enough to compile or explain exactly what changed.
- Mention any tradeoff that affects latency, cache stability, or testability.

Current code:

```python
{code_snippet_fn(index)}
```

Current tests and config:

```python
def test_placeholder() -> None:
    assert True
```

```toml
[project]
name = "benchmark-package"
version = "0.1.0"
requires-python = ">=3.11"
```

Please apply the requested change and keep the response concise."""
        for index, instruction in enumerate(instructions)
    ]


def _long_debug_code_snippet(turn: int) -> str:
    """Return a realistic code snippet for a long debug benchmark."""
    extra = ""
    if turn % 3 == 0:
        extra = """

class Item(BaseModel):
    name: str
    quantity: int
"""
    elif turn % 3 == 1:
        extra = """

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
"""
    else:
        extra = """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    return f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

{extra}

@app.post("/items")
def create_item(item: Item):
    if item.quantity < 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    return {{"name": item.name, "total": item.quantity * 10}}
"""


DEBUG_LONG_INSTRUCTIONS = [
    "Start by diagnosing the IndexError in the helper and explain the failing path.",
    "Now fix the off-by-one bug and add a regression test for empty input.",
    "Add clearer validation for malformed records and return structured error details.",
    "Introduce logging around the failing section so the next incident is easier to trace.",
    "Add a retry wrapper around the file read path and keep the public API stable.",
    "Refactor the error handling into a small helper that can be reused across endpoints.",
    "Add a unit test for the retry path using a fake file object.",
    "Make the API return a consistent error shape for validation failures.",
    "Add a CLI smoke test that exercises the failing path end to end.",
    "Refactor the module so the pure logic is separated from FastAPI plumbing.",
    "Add a timeout around the file read path and document the tradeoff.",
    "Introduce a small metrics object that counts successful and failed parses.",
    "Add structured logging for every request and include the trace id in the response.",
    "Harden the endpoint against oversized payloads without changing the happy path.",
    "Add a compatibility shim for clients that still send the old dict format.",
    "Refactor the retry logic to use exponential backoff with jitter.",
    "Add a test that verifies the endpoint rejects negative quantities.",
    "Add a small benchmark fixture that measures the debug path on 10k rows.",
    "Add documentation for the new error contract and retry behavior.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming variant that yields parsed items one at a time.",
    "Add localization support for user-facing error messages.",
    "Add a plugin hook for custom validators that can be registered at startup.",
    "Add config validation so bad environment variables fail fast.",
    "Add a final refactor that groups related helpers into small modules.",
    "Add a short architecture diagram in text form and explain the data flow.",
    "Add a final test that replays the debug session against the package.",
    "Finish by summarizing the fix, remaining risks, and next hardening step.",
    "Add a small observability hook that exports request duration and parse errors.",
    "Add a final release note that explains the bug fix and the new safeguards.",
]


def _long_feature_code_snippet(turn: int) -> str:
    """Return a realistic code snippet for a long feature benchmark."""
    extra = ""
    if turn % 3 == 0:
        extra = """

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
"""
    elif turn % 3 == 1:
        extra = """

def _now() -> datetime:
    return datetime.utcnow()
"""
    else:
        extra = """

def _fake_dependency() -> str:
    return "test-user"
"""
    return f"""from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt

app = FastAPI()
SECRET_KEY = "dev-secret"

{extra}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(payload: LoginRequest) -> TokenResponse:
    if payload.password != "secret":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({{"sub": payload.username, "exp": datetime.utcnow() + timedelta(hours=1)}}, SECRET_KEY, algorithm="HS256")
    return TokenResponse(access_token=token)
"""


FEATURE_LONG_INSTRUCTIONS = [
    "Start by sketching the authentication API shape and the data models it needs.",
    "Add a login endpoint that validates credentials and returns a bearer token.",
    "Introduce a config object for JWT settings and keep the public API stable.",
    "Refactor token creation into a small service so tests can swap the signer.",
    "Add rate limiting to the login endpoint without changing the response shape.",
    "Add a dependency injection layer so the auth service can use a fake repository.",
    "Add a test suite for login success, invalid credentials, and expired tokens.",
    "Add async support around the token refresh path while preserving sync behavior.",
    "Refactor the package layout so auth logic lives in its own module.",
    "Add a Dockerfile that runs the API and mounts a config file.",
    "Add a CI workflow that runs lint, type checks, and auth tests.",
    "Add a lightweight benchmark for login throughput on a warm token cache.",
    "Add documentation for the auth API, config, and token lifecycle.",
    "Add a changelog entry for the authentication feature.",
    "Harden the endpoint against malformed payloads and oversized requests.",
    "Add metrics for login success, failure, and rate-limit hits.",
    "Add observability hooks that propagate a trace id through auth responses.",
    "Add a migration path from the old session cookie flow to bearer tokens.",
    "Add release notes that explain the new auth flow and breaking changes.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming token refresh endpoint that yields refresh events.",
    "Add retry logic around external token validation with bounded backoff.",
    "Add localization support for user-facing auth errors.",
    "Add a performance optimization for token validation using an LRU cache.",
    "Add a plugin hook for custom auth backends.",
    "Add config validation so bad JWT settings fail fast.",
    "Add a final refactor that groups related auth helpers into small modules.",
    "Add a short architecture diagram in text form and explain the auth flow.",
    "Add a final test that simulates the full feature conversation against the package.",
    "Finish by summarizing the feature, remaining risks, and next hardening step.",
]


def _long_default_code_snippet(turn: int) -> str:
    """Return a realistic code snippet for a long default benchmark."""
    extra = ""
    if turn % 3 == 0:
        extra = """

from dataclasses import dataclass

@dataclass(slots=True)
class Result:
    value: int
"""
    elif turn % 3 == 1:
        extra = """

def _validate_n(n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
"""
    else:
        extra = """

def main() -> None:
    print(list(fibonacci_gen(10)))
"""
    return f"""from typing import Iterator

{extra}

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


DEFAULT_LONG_INSTRUCTIONS = [
    "Start by explaining the simplest iterative Fibonacci implementation and its tradeoffs.",
    "Refactor the helper into a generator that yields one value at a time.",
    "Add type hints and a small docstring while preserving the public API.",
    "Introduce a config object that controls the sequence length and output format.",
    "Refactor the generator into a service class with dependency injection for tests.",
    "Add structured logging around each yielded value and keep the output stable.",
    "Add a CLI entry point that accepts `--count` and `--format`.",
    "Add a pytest suite for zero, one, many, and negative inputs.",
    "Add async support with a small wrapper around the generator.",
    "Refactor the code into a package layout with `src/`, `tests/`, and `pyproject.toml`.",
    "Add a Dockerfile that runs the CLI against a mounted config file.",
    "Add a GitHub Actions workflow that runs formatting, linting, and tests.",
    "Add a lightweight benchmark that measures generator throughput on large n.",
    "Add documentation for usage, config, and the generator contract.",
    "Add a changelog entry for the generator refactor and CLI.",
    "Harden the CLI against invalid arguments and malformed config files.",
    "Add metrics for count, duration, and yielded values.",
    "Add observability hooks that propagate a trace id through CLI output.",
    "Add a migration path from the old list API to the new generator API.",
    "Add release notes that explain the new generator and CLI behavior.",
    "Do a final cleanup pass to remove dead code and tighten type hints.",
    "Add a streaming API variant that yields formatted lines one at a time.",
    "Add retry logic around config loading with bounded backoff.",
    "Add localization support for user-facing CLI errors.",
    "Add a performance optimization for large n using a rolling pair.",
    "Add a plugin hook for custom formatters.",
    "Add config validation so invalid settings fail fast.",
    "Add a final refactor that groups related helpers into small modules.",
    "Add a short architecture diagram in text form and explain the data flow.",
    "Finish by summarizing the refactor, remaining risks, and next hardening step.",
]


DEBUG_LONG_TASKS = _build_long_tasks(DEBUG_LONG_INSTRUCTIONS, _long_debug_code_snippet)
FEATURE_LONG_TASKS = _build_long_tasks(FEATURE_LONG_INSTRUCTIONS, _long_feature_code_snippet)
DEFAULT_LONG_TASKS = _build_long_tasks(DEFAULT_LONG_INSTRUCTIONS, _long_default_code_snippet)

# ---------------------------------------------------------------------------
# Real-life coding scenarios for benchmarking
# ---------------------------------------------------------------------------

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
            ("user", "What is 2+2? Answer with just the number."),
            ("user", "Now write a Python function to compute Fibonacci numbers iteratively."),
            ("user", "Great. Now refactor it to use a generator instead of building a list."),
            ("user", "Add type hints and docstrings to the generator."),
        ],
    },
    "default_long": {
        "description": "Long general coding conversation with 30 unique turns and code blocks",
        "tasks": [("user", task) for task in DEFAULT_LONG_TASKS],
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
    if args.profile != "aggressive":
        return

    overrides = {
        "MOEPT_AGENTIC__KEEP_FULL_STEPS": "3",
        "MOEPT_AGENTIC__MAX_OPTIMIZED_CHARS": "12000",
        "MOEPT_AGENTIC__MAX_OPTIMIZED_TOKENS": "3000",
        "MOEPT_AGENTIC__PROACTIVE_TRIM_RATIO": "0.45",
        "MOEPT_AGENTIC__COMPACTION_TRIGGER_RATIO": "0.75",
    }
    for key, value in overrides.items():
        os.environ.setdefault(key, value)

    _status(args, "  Context profile: aggressive (top-only eviction, 3000-token cap)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Detect responses where usage is cache-driven or incomplete."""
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(
        details.get("cached_tokens", 0)
        or usage.get("cached_tokens", 0)
        or usage.get("cache_hit_tokens", 0)
        or 0
    )
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens == 0 and (cached_tokens > 0 or completion_tokens > 0)


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
    """Calculate timeout based on turns and rounds.

    Context-size dependent logic:
    - Short contexts (<1000 tokens): 60-120s
    - Medium contexts (1000-3000 tokens): 120-180s
    - Long contexts (>3000 tokens): 180-300s

    This scales timeout with expected context growth per turn.
    """
    # Base timeout per request
    base_timeout = 120.0  # 2 minutes in seconds

    # Scale with turns: later turns have more context
    # Each turn adds ~100-150 tokens of context
    # Context grows: 200 → 500+ tokens over 10-15 turns
    # Timeout should scale: 120s → 300s
    context_growth_factor = 1 + (turns * 0.15)  # 15% increase per turn

    return min(300.0, base_timeout * context_growth_factor)


def _direct_request(messages: list[dict], max_tokens: int = 256, timeout: float = 180.0) -> tuple[dict, float, dict[str, str]]:
    url = f"{LEMONADE_URL}/chat/completions"
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
        "max_tokens": max_tokens,
    }
    return _request(url, body, timeout)


def _proxy_request(
    messages: list[dict], session_id: str | None = None, max_tokens: int = 256, timeout: float = 180.0
) -> tuple[dict, float, dict[str, str]]:
    url = f"http://127.0.0.1:{MOEPT_PORT}/v1/chat/completions"
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if session_id:
        body["_session_id"] = session_id
    return _request(url, body, timeout)


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


def _rouge_l(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 score (longest common subsequence)."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    m, n = len(words_a), len(words_b)

    if m == 0 or n == 0:
        return 0.0

    # LCS table (space-optimized to last two rows)
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

    # LCS via space-optimized DP
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
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6)}


def _code_block_preservation(direct_content: str, proxy_content: str) -> dict[str, float]:
    """Measure how many code blocks from the direct response are preserved in the proxy response.

    Returns dict with:
        block_ratio: fraction of direct's code blocks whose content appears (≥50%) in proxy
        has_code_direct: whether direct had any code blocks
        has_code_proxy: whether proxy had any code blocks
    """
    import re

    # Extract fenced code blocks (```lang ... ```)
    code_block_re = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
    direct_blocks = code_block_re.findall(direct_content)
    proxy_blocks = code_block_re.findall(proxy_content)

    if not direct_blocks:
        return {"block_ratio": 1.0, "has_code_direct": False, "has_code_proxy": bool(proxy_blocks)}

    preserved = 0
    for dblock in direct_blocks:
        # Check if at least half of the block content appears anywhere in proxy
        clean = re.sub(r"\s+", " ", dblock.strip()).strip()
        if len(clean) < 3:
            preserved += 1
            continue
        # For short blocks require exact match; for longer blocks use character overlap
        proxy_text = re.sub(r"\s+", " ", "".join(proxy_blocks)).strip()
        if clean in proxy_text:
            preserved += 1
        elif len(clean) > 20:
            # Fuzzy: check if the majority of unique words appear
            direct_words = set(clean.split())
            proxy_words = set(proxy_text.split())
            overlap = len(direct_words & proxy_words) / max(len(direct_words), 1)
            if overlap >= 0.3:  # Lowered threshold for more lenient matching
                preserved += 1
        else:
            # For short blocks, check if key code elements are present
            # (handles cases where code is reformatted but semantically same)
            key_elements = ["def ", "class ", "import ", "return ", "if ", "for ", "while "]
            direct_has_code = any(kw in clean for kw in key_elements)
            proxy_has_code = any(kw in proxy_text for kw in key_elements)
            if direct_has_code and proxy_has_code:
                preserved += 1

    # Also check if proxy has code fences (structure preservation)
    proxy_has_fences = "```" in proxy_content
    if not proxy_blocks and proxy_has_fences:
        # Proxy has code fences but we couldn't extract - might be different format
        # Check if there's any code-like content
        if re.search(r"def |class |import |return |for |while ", proxy_content):
            preserved = len(direct_blocks)  # Assume preserved
    elif not proxy_blocks:
        # No code blocks in proxy - check if code-like content exists
        key_elements = ["def ", "class ", "import ", "return ", "if ", "for ", "while "]
        proxy_has_code = any(kw in proxy_content for kw in key_elements)
        if proxy_has_code:
            # Proxy has code-like content but no fences - check word overlap
            for dblock in direct_blocks:
                clean = re.sub(r"\s+", " ", dblock.strip()).strip()
                if len(clean) < 3:
                    preserved += 1
                    continue
                direct_words = set(clean.split())
                proxy_words = set(proxy_content.split())
                overlap = len(direct_words & proxy_words) / max(len(direct_words), 1)
                if overlap >= 0.5:
                    preserved += 1

    return {
        "block_ratio": round(preserved / max(len(direct_blocks), 1), 6),
        "has_code_direct": True,
        "has_code_proxy": bool(proxy_blocks) or proxy_has_fences,
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


def _compute_quality_metrics(direct_content: str, proxy_content: str) -> dict[str, float]:
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

    # ── MTP-specific metrics ───────────────────────────────────────────────
    # These are computed from the response content to assess MTP performance
    metrics["mtp_stability"] = _assess_mtp_stability(direct_content, proxy_content)
    metrics["syntax_consistency"] = _assess_syntax_consistency(direct_content, proxy_content)

    return metrics


def _assess_mtp_stability(direct_content: str, proxy_content: str) -> float:
    """Assess MTP prediction stability.

    Compares the structure and flow of responses to detect MTP-related issues.
    High similarity = stable MTP predictions.
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


def _assess_syntax_consistency(direct_content: str, proxy_content: str) -> float:
    """Assess syntax consistency between responses.

    Checks if code structure and formatting are preserved.
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
    cache_hit_rate: float = 0.0
    latency_ms: float = 0.0
    response_chars: int = 0
    finish_reason: str = ""
    foreign_markers: list[str] = field(default_factory=list)
    error: str | None = None
    content_preview: str = ""  # First 200 chars for dump
    chars_before_optimization: int = 0  # Total chars in messages before proxy optimization


@dataclass
class TurnComparison:
    """Side-by-side metrics for one turn."""

    turn_index: int = 0
    direct: TurnMetrics = field(default_factory=TurnMetrics)
    proxy: TurnMetrics = field(default_factory=TurnMetrics)
    latency_delta_ms: float = 0.0  # proxy - direct (positive = slower)
    token_delta: int = 0  # proxy prompt - direct prompt
    quality: dict[str, float | None] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    config: dict = field(default_factory=dict)
    turns: list[TurnComparison] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a flat summary dict for JSON output."""
        n = len(self.turns)
        if not self.turns:
            return {"error": "no data"}

        direct_latencies = [t.direct.latency_ms for t in self.turns]
        proxy_latencies = [t.proxy.latency_ms for t in self.turns]
        latency_deltas = [t.latency_delta_ms for t in self.turns]

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
        cached = [t.proxy.cached_tokens for t in self.turns]

        total_direct_prompt = sum(direct_tokens)
        total_proxy_prompt = sum(proxy_tokens)
        total_cached = sum(cached)
        tokens_saved_pct = (
            round((total_direct_prompt - total_proxy_prompt) / max(total_direct_prompt, 1) * 100, 2)
            if total_direct_prompt > 0
            else 0.0
        )

        # Context window growth: prompt_tokens at each turn vs theoretical full context
        final_turn = self.turns[-1]
        max_context_window = 262144  # model default (Qwen3.6-35B-MTP)
        final_proxy_prompt_tokens = final_turn.proxy.prompt_tokens
        final_proxy_ctx_pct = round(final_proxy_prompt_tokens / max_context_window * 100, 2)

        # ── Quality metrics aggregation ───────────────────────────────
        quality_metrics = [
            "semantic_similarity", "token_jaccard", "rouge_l_f1", "trigram_overlap",
            "length_ratio", "edit_similarity", "code_block_ratio",
            "markdown_structure_similarity", "vocabulary_richness_delta",
            "rouge_l_precision", "rouge_l_recall",
        ]
        quality_summary: dict[str, Any] = {}
        for qm in quality_metrics:
            values = [t.quality.get(qm) for t in self.turns if t.quality and t.quality.get(qm) is not None]
            if values:
                s = sorted(values)
                quality_summary[qm] = {
                    "mean": round(statistics.mean(s), 4),
                    "median": round(statistics.median(s), 4),
                    "min": round(min(s), 4),
                    "max": round(max(s), 4),
                }
            else:
                quality_summary[qm] = None

        # Count turns with low similarity (potential degradation)
        low_semantic_count = sum(
            1 for t in self.turns if t.quality and t.quality.get("semantic_similarity") is not None
            and t.quality["semantic_similarity"] < 0.75
        )
        low_jaccard_count = sum(
            1 for t in self.turns if t.quality and t.quality.get("token_jaccard") is not None
            and t.quality["token_jaccard"] < 0.40
        )

        # ── Response length analysis ────────────────────────────────────
        length_ratios = [t.quality.get("length_ratio") for t in self.turns if t.quality and t.quality.get("length_ratio") is not None]
        truncation_count = sum(1 for r in length_ratios if r < 0.5) if length_ratios else 0
        verbosity_count = sum(1 for r in length_ratios if r > 2.0) if length_ratios else 0

        # ── Code block preservation analysis ────────────────────────────
        code_block_ratios = [t.quality.get("code_block_ratio") for t in self.turns if t.quality and t.quality.get("code_block_ratio") is not None]
        code_loss_count = sum(1 for r in code_block_ratios if r < 1.0) if code_block_ratios else 0

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
        budget = self.config.get("char_budget")
        chars_before = [t.proxy.chars_before_optimization for t in self.turns]
        total_chars_before = sum(chars_before)
        budget_exceeded_turns: list[int] = []
        compaction_turns: list[int] = []
        eviction_turns: list[int] = []
        if budget is not None and chars_before:
            budget_exceeded_turns = [
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

        return {
            "config": self.config,
            "num_turns": n,
            "latency_ms": {
                "direct": _stats(direct_latencies),
                "proxy": _stats(proxy_latencies),
                "delta_proxy_minus_direct_ms": _stats(latency_deltas),
            },
            "tokens": {
                "total_direct_prompt": total_direct_prompt,
                "total_proxy_prompt": total_proxy_prompt,
                "total_cached_tokens": total_cached,
                "token_savings_pct": tokens_saved_pct,
                "per_turn_direct": _stats(direct_tokens),
                "per_turn_proxy": _stats(proxy_tokens),
                "per_turn_cached": _stats(cached),
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
                "verbosity_count": verbosity_count,
                "code_block_loss_turns": code_loss_count,
                "rouge_precision_recall_gap_mean": rouge_gap_mean,
            },
            "quality_trend": quality_trend if quality_trend else {},
            "vocab_richness": {
                "mean_delta": round(statistics.mean(vocab_deltas), 4) if vocab_deltas else None,
                "max_delta": round(max(vocab_deltas), 4) if vocab_deltas else None,
                "turns_above_0.15": sum(1 for v in vocab_deltas if v > 0.15) if vocab_deltas else 0,
            },
            "eviction": {
                "char_budget": budget,
                "total_chars_before_optimization": total_chars_before,
                "turns_exceeding_budget": len(budget_exceeded_turns),
                "budget_exceeded_at_turns": budget_exceeded_turns if budget_exceeded_turns else None,
                "compaction_triggered_at_turns": compaction_turns if compaction_turns else None,
                "eviction_triggered_at_turns": eviction_turns if eviction_turns else None,
            },
        }


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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _build_conversation_turns(num_turns: int) -> list[dict]:
    """Build a multi-turn conversation with growing context."""
    system_prompt = (
        "You are a helpful coding assistant. You reason carefully before answering. "
        "Keep your reasoning concise and focus on the user's actual question."
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Multi-turn prompts that build context progressively
    tasks = [
        ("user", "What is 2+2? Answer with just the number."),
        ("assistant", "4"),
        ("user", "Now write a Python function to compute Fibonacci numbers iteratively."),
        ("assistant", (
            "Here's an iterative Fibonacci implementation:\n\n"
            "```python\n"
            "def fibonacci(n: int) -> list[int]:\n"
            "    if n <= 0:\n"
            "        return []\n"
            "    elif n == 1:\n"
            "        return [0]\n"
            "    fib = [0, 1]\n"
            "    for i in range(2, n):\n"
            "        fib.append(fib[i-1] + fib[i-2])\n"
            "    return fib\n"
            "```\n\n"
            "This runs in O(n) time and O(n) space."
        )),
        ("user", "Great. Now refactor it to use a generator instead of building a list."),
        ("assistant", (
            "Here's the generator version:\n\n"
            "```python\n"
            "def fibonacci_gen(n: int):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        yield a\n"
            "        a, b = b, a + b\n"
            "```\n\n"
            "This is O(1) space since it yields values one at a time."
        )),
        ("user", "Add type hints and docstrings to the generator."),
        ("assistant", (
            "Here's the fully typed version:\n\n"
            "```python\n"
            "from typing import Generator\n\n"
            "\n"
            "def fibonacci_gen(n: int) -> Generator[int, None, None]:\n"
            "    '''Generate the first n Fibonacci numbers.\n\n"
            "    Args:\n"
            "        n: Number of Fibonacci numbers to generate.\n\n"
            "    Yields:\n"
            "        int: The next Fibonacci number in the sequence.\n"
            "    '''\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        yield a\n"
            "        a, b = b, a + b\n"
            "```\n\n"
            "This provides full type safety and documentation."
        )),
    ]

    # Add the base conversation (6 turns: system + 5 pairs)
    for role, content in tasks:
        messages.append({"role": role, "content": content})

    # Pad with additional turns to reach num_turns if needed
    turn_count = len(messages) - 1  # exclude system
    i = 0
    while turn_count < num_turns * 2 + 1:  # each "turn" = user+assistant pair
        messages.append({"role": "user", "content": f"Turn {i}: Remember the fibonacci generator we discussed? Now write a test suite for it using pytest."})
        messages.append({"role": "assistant", "content": (
            f"Here's a comprehensive test suite for turn {i}:\n\n"
            f"```python\n"
            f"import pytest\n"
            f"from fib import fibonacci_gen\n\n"
            f"@pytest.mark.parametrize('n,expected', [\n"
            f"    (0, []),\n"
            f"    (1, [0]),\n"
            f"    (5, [0, 1, 1, 2, 3]),\n"
            f"    (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),\n"
            f"])\n"
            f"def test_fibonacci_gen(n, expected):\n"
            f"    assert list(fibonacci_gen(n)) == expected\n"
            f"```\n\n"
            f"This covers edge cases and standard sequences."
        )})
        turn_count += 2
        i += 1

    return messages


def run_benchmark(
    num_turns: int,
    rounds: int,
    max_tokens: int,
    proxy_port: int,
    budget: int | None = None,
    scenario: str = "default",
) -> BenchmarkReport:
    """Run the multi-turn benchmark and collect metrics."""

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
    }

    report = BenchmarkReport(config=config)

    # Get scenario tasks
    scenario_data = SCENARIOS.get(scenario, SCENARIOS["default"])
    base_tasks = scenario_data["tasks"]

    system_prompt = (
        "You are a helpful coding assistant. You reason carefully before answering. "
        "Keep your reasoning concise and focus on the user's actual question."
    )
    base_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for role, content in base_tasks:
        base_messages.append({"role": role, "content": content})

    turn_index = 0
    user_tasks = [content for role, content in base_tasks if role == "user"]
    fallback_user_task = (
        "Turn {turn_index}: Remember the fibonacci generator we discussed? "
        "Now write a test suite for it using pytest."
    )

    for round_num in range(rounds):
        # Each round gets an isolated proxy session so prior-round state cannot
        # leak into the next benchmark round.
        session_id = f"benchmark-{int(time.time())}-{round_num}-{uuid.uuid4().hex[:8]}"
        direct_messages: list[dict] = [dict(msg) for msg in base_messages]
        proxy_messages: list[dict] = [dict(msg) for msg in base_messages]

        for _ in range(num_turns):
            turn_index += 1
            user_content = (
                user_tasks[(turn_index - 1) % len(user_tasks)]
                if user_tasks
                else fallback_user_task.format(turn_index=turn_index)
            )

            # Add the current user turn to both contexts before either request.
            direct_messages.append({"role": "user", "content": user_content})
            proxy_messages.append({"role": "user", "content": user_content})

            direct_context = _context_size_summary(direct_messages)
            proxy_context = _context_size_summary(proxy_messages)
            _human_print(
                f"  Turn {turn_index:02d}/{num_turns:02d}: "
                f"direct_ctx={direct_context['messages']} msgs/{direct_context['chars']:,} chars/"
                f"~{direct_context['estimated_tokens']:,} tokens; "
                f"proxy_ctx={proxy_context['messages']} msgs/{proxy_context['chars']:,} chars/"
                f"~{proxy_context['estimated_tokens']:,} tokens"
            )

            # --- Direct request ---
            direct_resp: dict | None = None
            proxy_resp: dict | None = None
            d_content = ""
            p_content = ""

            try:
                direct_resp, direct_latency, _direct_headers = _direct_request(
                    direct_messages, max_tokens=max_tokens, timeout=request_timeout
                )
                d_usage = direct_resp.get("usage", {}) or {}
                d_msg = direct_resp["choices"][0]["message"]
                d_content = (d_msg.get("content") or "") + (d_msg.get("reasoning_content") or "")

                _d_prompt_raw = int(d_usage.get("prompt_tokens", 0) or 0)
                _d_cached = int((d_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                _d_prompt, _d_prompt_source, _d_cached_response = _resolve_prompt_tokens(
                    _d_prompt_raw, direct_messages, d_usage
                )
                direct_metrics = TurnMetrics(
                    turn_index=turn_index,
                    total_turns_at_request=len(direct_messages) - 1,  # exclude system
                    prompt_tokens=_d_prompt,
                    raw_prompt_tokens=_d_prompt_raw,
                    prompt_tokens_source=_d_prompt_source,
                    cached_response=_d_cached_response,
                    completion_tokens=int(d_usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(d_usage.get("total_tokens", 0) or 0),
                    cached_tokens=_d_cached,
                    cache_hit_rate=round(_d_cached / max(_d_prompt, 1), 2),
                    latency_ms=round(direct_latency, 2),
                    response_chars=len(d_content),
                    finish_reason=direct_resp["choices"][0].get("finish_reason", ""),
                    content_preview=d_content[:200],
                )
            except Exception as e:
                direct_metrics = TurnMetrics(
                    turn_index=turn_index,
                    total_turns_at_request=len(direct_messages) - 1,
                    prompt_tokens=_estimate_prompt_tokens(direct_messages) if direct_messages else 0,
                    prompt_tokens_source="estimated_after_error",
                    latency_ms=0.0,
                    error=str(e)[:200],
                )

            # --- Proxy request ---
            try:
                proxy_resp, proxy_latency, proxy_headers = _proxy_request(
                    proxy_messages, session_id=session_id, max_tokens=max_tokens, timeout=request_timeout
                )
                p_usage = proxy_resp.get("usage", {}) or {}
                p_msg = proxy_resp["choices"][0]["message"]
                p_content = (p_msg.get("content") or "") + (p_msg.get("reasoning_content") or "")

                _p_prompt_raw = int(p_usage.get("prompt_tokens", 0) or 0)
                _p_cached = int((p_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                _p_optimized_prompt_tokens = int(
                    proxy_headers.get("X-Optimized-Prompt-Tokens")
                    or proxy_headers.get("x-optimized-prompt-tokens")
                    or 0
                )
                _p_prompt, _p_prompt_source, _p_cached_response = _resolve_prompt_tokens(
                    _p_prompt_raw,
                    proxy_messages,
                    p_usage,
                    optimized_prompt_tokens=_p_optimized_prompt_tokens,
                )
                # Measure total chars before proxy optimization (for eviction tracking)
                _chars_before = sum(len(_message_text(m.get("content", ""))) for m in proxy_messages)
                proxy_metrics = TurnMetrics(
                    turn_index=turn_index,
                    total_turns_at_request=len(proxy_messages) - 1,
                    prompt_tokens=_p_prompt,
                    raw_prompt_tokens=_p_prompt_raw,
                    optimized_prompt_tokens=_p_optimized_prompt_tokens,
                    prompt_tokens_source=_p_prompt_source,
                    cached_response=_p_cached_response,
                    completion_tokens=int(p_usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(p_usage.get("total_tokens", 0) or 0),
                    cached_tokens=_p_cached,
                    cache_hit_rate=round(_p_cached / max(_p_prompt, 1), 2),
                    latency_ms=round(proxy_latency, 2),
                    response_chars=len(p_content),
                    finish_reason=proxy_resp["choices"][0].get("finish_reason", ""),
                    content_preview=p_content[:200],
                    chars_before_optimization=_chars_before,
                )

                # Check for leaked internal markers
                proxy_metrics.foreign_markers = _check_foreign_markers(p_content)

            except Exception as e:
                proxy_metrics = TurnMetrics(
                    turn_index=turn_index,
                    total_turns_at_request=len(proxy_messages) - 1,
                    prompt_tokens=_estimate_prompt_tokens(proxy_messages) if proxy_messages else 0,
                    prompt_tokens_source="estimated_after_error",
                    latency_ms=0.0,
                    error=str(e)[:200],
                )
                # Try to extract optimization error from response headers if available
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    opt_error = e.response.headers.get("X-Optimization-Error")
                    if opt_error:
                        proxy_metrics.error = f"{proxy_metrics.error} | optimization: {opt_error}"

            # Append each side's assistant response to its own context for the next turn.
            if d_content:
                direct_messages.append({"role": "assistant", "content": d_content})
            if p_content:
                proxy_messages.append({"role": "assistant", "content": p_content})

            # Compute quality metrics (only if both responses are valid)
            quality: dict[str, float | None] = {}
            if d_content and p_content:
                quality = _compute_quality_metrics(d_content, p_content)

            # Compute deltas
            comparison = TurnComparison(
                turn_index=turn_index,
                direct=direct_metrics,
                proxy=proxy_metrics,
                latency_delta_ms=round(proxy_metrics.latency_ms - direct_metrics.latency_ms, 2),
                token_delta=proxy_metrics.prompt_tokens - direct_metrics.prompt_tokens,
                quality=quality,
            )

            q_sem = quality.get("semantic_similarity") if quality else None
            q_jaccard = quality.get("token_jaccard") if quality else None
            q_rouge = quality.get("rouge_l_f1") if quality else None
            direct_error = f" direct_error={direct_metrics.error[:80]!r}" if direct_metrics.error else ""
            proxy_error = f" proxy_error={proxy_metrics.error[:80]!r}" if proxy_metrics.error else ""
            quality_parts = [
                f"quality_sem={q_sem:.3f}" if q_sem is not None else "quality_sem=n/a",
                f"jaccard={q_jaccard:.3f}" if q_jaccard is not None else "jaccard=n/a",
                f"rouge={q_rouge:.3f}" if q_rouge is not None else "rouge=n/a",
            ]
            _human_print(
                f"  Turn {turn_index:02d}: "
                f"direct={direct_metrics.latency_ms:.0f}ms/{direct_metrics.prompt_tokens:,}tok/{direct_metrics.response_chars:,}chars"
                f" proxy={proxy_metrics.latency_ms:.0f}ms/{proxy_metrics.prompt_tokens:,}tok/"
                f"{proxy_metrics.chars_before_optimization:,}chars_raw/{proxy_metrics.response_chars:,}chars"
                f" delta={comparison.latency_delta_ms:+.0f}ms/{comparison.token_delta:+,}tok"
                f" cache={proxy_metrics.cached_tokens:,}/{proxy_metrics.cache_hit_rate:.2f} "
                f"{' '.join(quality_parts)}"
                f"{direct_error}{proxy_error}"
            )

            report.turns.append(comparison)

    return report


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple ASCII table."""
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    pad = 2
    header_line = "  ".join(_fmt_table.__code__.co_consts[1:3]) if False else "".join(
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
            ["Turns exceeding budget", str(ev.get("turns_exceeding_budget", 0))],
            ["Budget exceeded at turns", ", ".join(str(t) for t in ev.get("budget_exceeded_at_turns") or []) or "never"],
            ["Compaction triggered at turns", ", ".join(str(t) for t in ev.get("compaction_triggered_at_turns") or []) or "never"],
            ["Budget eviction triggered at turns", ", ".join(str(t) for t in ev.get("eviction_triggered_at_turns") or []) or "never"],
        ]
        print(_fmt_table(["Metric", "Value"], ev_rows))
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
        ("mtp_stability", "MTP stability"),
        ("syntax_consistency", "Syntax consistency"),
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

    # ── Degradation flags ─────────────────────────────────────────────
    low_semantic = qual.get("low_semantic_similarity_turns", 0)
    low_jaccard = qual.get("low_token_jaccard_turns", 0)
    truncation_count = qual.get("truncation_count", 0)
    verbosity_count = qual.get("verbosity_count", 0)
    code_loss = qual.get("code_block_loss_turns", 0)
    rouge_gap = qual.get("rouge_precision_recall_gap_mean", 0.0)

    degradation_notes: list[str] = []
    if low_semantic > 0:
        degradation_notes.append(f"{low_semantic} turn(s) low semantic similarity (<0.75)")
    if low_jaccard > 0:
        degradation_notes.append(f"{low_jaccard} turn(s) low token overlap (<0.40)")
    if truncation_count > 0:
        degradation_notes.append(f"{truncation_count} turn(s) severely truncated (length_ratio <0.5)")
    if verbosity_count > 0:
        degradation_notes.append(f"{verbosity_count} turn(s) verbose inflation (length_ratio >2.0)")
    if code_loss > 0:
        degradation_notes.append(f"{code_loss} turn(s) with lost code blocks")
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
        budget_val = report.config.get("char_budget")
        eviction_flag = ""
        if budget_val is not None and hasattr(t.proxy, 'chars_before_optimization'):
            eviction_flag = "YES ⚠️" if t.proxy.chars_before_optimization > budget_val else "no"
        detail_rows.append([
            str(t.turn_index),
            str(ctx_turns),
            d_tok, p_tok, cached, f"{tok_delta:+}",
            d_lat, p_lat, lat_delta, chars_in, eviction_flag, lr, sim, cb,
            t.proxy.error or "",
        ])

    print(_fmt_table(detail_headers, detail_rows))
    print()


def run_all_scenarios(args) -> None:
    """Run all scenarios and produce aggregated metrics."""
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
        for scenario_name in SCENARIOS:
            print(f"\n  Running scenario: {scenario_name}")
            report = run_benchmark(
                num_turns=args.turns,
                rounds=args.rounds,
                max_tokens=args.max_tokens,
                proxy_port=args.port,
                budget=args.budget,
                scenario=scenario_name,
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
    all_token_savings: list[float] = []

    for name, report in reports.items():
        summary = report.summary()
        aggregated["per_scenario"][name] = {
            "num_turns": summary.get("num_turns", 0),
            "latency_mean_ms": summary.get("latency_ms", {}).get("proxy", {}).get("mean", 0),
            "semantic_similarity_mean": summary.get("quality", {}).get("semantic_similarity", {}).get("mean", 0),
            "token_savings_pct": summary.get("tokens", {}).get("token_savings_pct", 0),
        }

        # Collect for aggregation
        lat = summary.get("latency_ms", {}).get("proxy", {}).get("mean", 0)
        if lat:
            all_latencies.append(lat)

        sem = summary.get("quality", {}).get("semantic_similarity", {}).get("mean", 0)
        if sem:
            all_semantic.append(sem)

        ts = summary.get("tokens", {}).get("token_savings_pct", 0)
        all_token_savings.append(ts)

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

    aggregated["aggregated"]["token_savings_pct"] = {
        "mean": round(statistics.mean(all_token_savings), 2),
        "min": round(min(all_token_savings), 2),
        "max": round(max(all_token_savings), 2),
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

    print("\n  Aggregated Metrics:")
    agg = aggregated.get("aggregated", {})
    if "latency_ms" in agg:
        print(f"    Latency (mean): {agg['latency_ms']['mean']:.0f}ms")
    if "semantic_similarity" in agg:
        print(f"    Semantic similarity (mean): {agg['semantic_similarity']['mean']:.4f}")
    if "token_savings_pct" in agg:
        print(f"    Token savings (mean): {agg['token_savings_pct']['mean']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn benchmark: direct Lemonade vs moeptimizer proxy"
    )
    parser.add_argument("--turns", type=int, default=10, help="Number of conversation turns")
    parser.add_argument("--rounds", type=int, default=1, help="Number of full conversation rounds")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
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
        choices=["balanced", "aggressive"],
        help="Context optimization profile for the proxy. 'aggressive' favors token savings with top-only eviction.",
    )
    parser.add_argument(
        "--scenario", type=str, default="default",
        choices=[*SCENARIOS.keys(), "all"],
        help="Real-life coding scenario: debug, refactor, feature, default, or all",
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
        )

        if args.json_output or not args.dump_responses:
            # Print report (or JSON only)
            if args.json_output:
                json.dump(report.summary(), sys.stdout, indent=2)
                print()
            else:
                print_report(report)

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
                    for key in ["semantic_similarity", "token_jaccard", "rouge_l_f1", "edit_similarity", "code_block_ratio", "length_ratio", "mtp_stability", "syntax_consistency"]:
                        val = q.get(key)
                        if val is not None:
                            parts.append(f"{key}={val:.4f}")
                    print(f"    Quality: {', '.join(parts)}")

                # Show degradation markers
                if t.quality and isinstance(t.quality.get("semantic_similarity"), float) and t.quality["semantic_similarity"] < 0.75:
                    print(f"    ⚠️  LOW SEMANTIC SIMILARITY ({t.quality['semantic_similarity']:.3f})")
                if t.quality and isinstance(t.quality.get("length_ratio"), float):
                    lr = t.quality["length_ratio"]
                    if lr < 0.5:
                        print(f"    ⚠️  SEVERE TRUNCATION (length_ratio={lr:.3f})")
                    elif lr > 2.0:
                        print(f"    ⚠️  VERBOSE INFLATION (length_ratio={lr:.3f})")
                if t.quality and isinstance(t.quality.get("code_block_ratio"), float) and t.quality["code_block_ratio"] < 1.0:
                    print(f"    ⚠️  CODE BLOCK LOSS ({t.quality['code_block_ratio']:.2f})")
                    # Show full content for debugging
                    import re as _re
                    d_code_blocks = _re.findall(r"```(?:\w*)\n(.*?)\`\`\`", t.direct.content_preview or "", _re.DOTALL)
                    p_code_blocks = _re.findall(r"```(?:\w*)\n(.*?)\`\`\`", t.proxy.content_preview or "", _re.DOTALL)
                    print(f"    Direct code blocks: {len(d_code_blocks)}, Proxy code blocks: {len(p_code_blocks)}")

    finally:
        _stop_proxy()


if __name__ == "__main__":
    main()
