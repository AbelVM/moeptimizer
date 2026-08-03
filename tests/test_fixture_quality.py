"""Executable quality gate for the multi-file replay fixture."""

import os
import subprocess
import sys
from pathlib import Path

from benchmark import benchmark as bm


def test_fixture_project_quality_gate() -> None:
    fixture_root = Path(__file__).parents[1] / "benchmark" / "fixtures"
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


def test_fixture_manifest_references_existing_files_in_order() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    known: set[str] = set()

    for entry in loader._MANIFEST:
        path = entry.get("add", entry.get("refine"))
        if path is None:
            assert "instruction" in entry
            continue
        assert isinstance(path, str)
        assert (loader.fixture_root() / path).is_file()
        if "refine" in entry:
            assert path in known
        else:
            known.add(path)


def test_fixture_cli_smoke_gate(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parents[1] / "benchmark" / "fixtures"
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


def test_fixture_acceptance_cli_materializes_requested_turns(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.fixtures.loader", "--turns", "10", "--seed", "1"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(__import__("json").loads(result.stdout)) == 10
    assert "RuntimeWarning" not in result.stderr


def test_fixture_refinements_carry_state_forward() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    tasks = loader.build_fixture_agentic_tasks(max_turns=9)
    first_read = next(message for message in tasks[1] if message.get("name") == "read_file")
    second_read = next(message for message in tasks[7] if message.get("name") == "read_file")
    assert first_read["content"] != second_read["content"]
    assert "benchmark refinement 3" in second_read["content"]


def test_fixture_seeded_variants_are_stable_and_distinct() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None

    first = loader.build_fixture_tasks(max_turns=3, seed=7)
    repeat = loader.build_fixture_tasks(max_turns=3, seed=7)
    other = loader.build_fixture_tasks(max_turns=3, seed=8)

    assert first == repeat
    assert first != other
    assert "Scenario variant 7" in first[0][1]


def test_fixture_seed_places_reproducible_failure_turn() -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None

    first = loader.build_fixture_agentic_tasks(max_turns=11, seed=4)
    repeat = loader.build_fixture_agentic_tasks(max_turns=11, seed=4)

    assert first == repeat
    failures = [
        turn
        for turn, messages in enumerate(first, 1)
        if any("1 failed, 4 passed" in str(message.get("content", "")) for message in messages)
    ]
    assert failures == [11]


def test_benchmark_uses_seeded_fixture_builder() -> None:
    from benchmark import benchmark as benchmark_module

    tasks = benchmark_module._build_opencode_scenario_tasks(seed=12)

    assert "Scenario variant 12" in tasks[0][0]["content"]


def test_materialized_fixture_workspace_passes_acceptance_suite(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    workspace = loader.materialize_fixture_workspace(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=workspace,
        env={**os.environ, "PYTHONPATH": str(workspace)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_materialized_seed_changes_independent_workspace_state(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    first = loader.materialize_fixture_workspace(tmp_path / "first", max_turns=9, seed=1)
    second = loader.materialize_fixture_workspace(tmp_path / "second", max_turns=9, seed=2)

    assert first.joinpath("users/repository.py").read_text() != second.joinpath(
        "users/repository.py"
    ).read_text()


def test_fixture_runner_executes_real_acceptance_steps(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None

    records = loader.run_fixture_acceptance(tmp_path, max_turns=12)

    assert len(records) == 12
    assert records[0]["action"] == "edit"
    verified = [record for record in records if record["verified"]]
    assert verified
    assert all(record["returncode"] == 0 for record in verified)
    assert all(len(str(record["output_tail"])) <= 2000 for record in verified)


def test_fixture_runner_applies_generated_unified_patch(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    workspace = loader.materialize_fixture_workspace(tmp_path, max_turns=1)
    patch = (
        "--- users/models.py\n"
        "+++ users/models.py\n"
        "@@ -24,2 +24,3 @@\n"
        " def summarize(users: list[User]) -> dict[str, int | bool]:\n"
        "     return {\"count\": len(users), \"active\": sum(1 for user in users if user.active)}\n"
        "+\n"
    )

    assert loader.apply_fixture_patch(workspace, patch)
    assert workspace.joinpath("users/models.py").read_text().endswith("\n\n")
    assert not loader.apply_fixture_patch(workspace, patch)


def test_fixture_runner_applies_generated_patch_before_acceptance(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    patch = (
        "--- users/models.py\n"
        "+++ users/models.py\n"
        "@@ -24,2 +24,3 @@\n"
        " def summarize(users: list[User]) -> dict[str, int | bool]:\n"
        "     return {\"count\": len(users), \"active\": sum(1 for user in users if user.active)}\n"
        "+\n"
    )

    records = loader.run_fixture_acceptance(
        tmp_path,
        max_turns=11,
        generated_patches={11: patch},
    )

    assert records[-1]["generated_patch_applied"] is True
    assert records[-1]["verified"] is True


def test_fixture_loader_reads_per_turn_generated_patches(tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    (tmp_path / "turn-3.patch").write_text("patch-three", encoding="utf-8")
    (tmp_path / "turn-12.patch").write_text("patch-twelve", encoding="utf-8")

    assert loader.load_generated_patches(tmp_path, max_turns=3) == {3: "patch-three"}


def test_fixture_runner_retries_failed_acceptance(monkeypatch, tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    calls = 0

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def run_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Result(1 if calls == 1 else 0)

    monkeypatch.setattr(loader.subprocess, "run", run_once)
    records = loader.run_fixture_acceptance(tmp_path, max_turns=11, max_retries=1)

    verified = [record for record in records if record["verified"]]
    assert verified[-1]["attempts"] == 2
    assert "output_tail" in verified[-1]


def test_fixture_runner_repairs_before_retry(monkeypatch, tmp_path: Path) -> None:
    loader = bm._get_fixture_loader()
    assert loader is not None
    pytest_calls = 0
    patch_calls = 0

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def run_once(command, *args, **kwargs):
        nonlocal patch_calls, pytest_calls
        if command[0] == "patch":
            patch_calls += 1
            return Result(0)
        pytest_calls += 1
        return Result(1 if pytest_calls == 1 else 0)

    monkeypatch.setattr(loader.subprocess, "run", run_once)
    records = loader.run_fixture_acceptance(
        tmp_path,
        max_turns=11,
        max_retries=1,
        repair_patches={11: "generated patch"},
    )

    assert records[-1]["verified"] is True
    assert records[-1]["repair_applied"] is True
    assert patch_calls == 2
