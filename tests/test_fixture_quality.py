"""Executable quality gate for the multi-file replay fixture."""

import os
import subprocess
import sys
from pathlib import Path

import benchmark as bm


def test_fixture_project_quality_gate() -> None:
    fixture_root = Path(__file__).parents[1] / "scripts" / "fixtures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fixture_root)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=fixture_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixture_replay_uses_file_language_fences() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    task = loader.build_fixture_tasks(max_turns=14)[-1][1]
    assert "# users.jsonl\n```json" in task
    assert "# pyproject.toml\n```toml" in task
    assert "# Dockerfile\n```dockerfile" in task


def test_fixture_replay_preserves_task_contract() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    tasks = [content for _, content in loader.build_fixture_tasks()]

    required_files = ("users/models.py", "users/repository.py", "users/service.py", "tests/test_users.py")
    assert all(any(f"# {path}\n```" in task for task in tasks) for path in required_files)
    assert all(
        constraint in task
        for task in tasks
        for constraint in (
            "Preserve the existing public API",
            "Prefer small, incremental patches",
            "Mention any tradeoff that affects latency, cache stability, or testability",
        )
    )


def test_fixture_cli_smoke_gate(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parents[1] / "scripts" / "fixtures"
    input_path = tmp_path / "users.jsonl"
    input_path.write_text("{\"id\": 1, \"name\": \"ada\"}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fixture_root)
    result = subprocess.run(
        [sys.executable, "-m", "users.cli", "--input", str(input_path), "--dry-run"],
        cwd=fixture_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"count": 1' in result.stdout
