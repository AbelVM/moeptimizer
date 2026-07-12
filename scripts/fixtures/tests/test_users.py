"""Tests for the users package.

Covers the happy path, missing file, invalid JSONL (strict mode), and dry-run
behavior — the same surface the benchmark's refactor scenario asks the agent to
produce, so the fixtures double as a reference implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from users.config import Config
from users.models import User, UserSchemaError
from users.repository import UserRepository
from users.service import SummarizerService


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "users.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_happy_path(tmp_path: Path) -> None:
    p = _write(tmp_path, [{"id": 1, "name": "ada"}, {"id": 2, "name": "bob", "active": False}])
    users = UserRepository(p).load()
    assert [u.name for u in users] == ["ada", "bob"]
    assert users[1].active is False


def test_missing_file_raises(tmp_path: Path) -> None:
    repo = UserRepository(tmp_path / "nope.jsonl")
    try:
        repo.load()
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError for missing file")


def test_invalid_jsonl_strict_mode(tmp_path: Path) -> None:
    p = _write(tmp_path, [{"id": 1, "name": "ada"}, {"id": 2}])
    try:
        UserRepository(p).load_strict()
    except UserSchemaError as exc:
        assert "bad rows" in str(exc)
    else:
        raise AssertionError("expected UserSchemaError")


def test_service_summarize(tmp_path: Path) -> None:
    p = _write(tmp_path, [{"id": 1, "name": "a"}, {"id": 2, "name": "b", "active": False}])
    summary = SummarizerService(UserRepository(p)).summarize()
    assert summary == {"count": 2, "active": 1}


def test_config_from_env(monkeypatch: object) -> None:
    import os

    monkeypatch.setenv("USERS_INPUT", "custom.jsonl")
    monkeypatch.setenv("USERS_DRY_RUN", "1")
    cfg = Config.from_env()
    assert cfg.input_path == Path("custom.jsonl")
    assert cfg.dry_run is True
