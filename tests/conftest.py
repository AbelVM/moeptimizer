"""Pytest configuration for moeptimizer tests."""

import sys
from pathlib import Path

# Make the repo root importable so the `benchmark` and `scripts` folders are
# importable as namespace packages (e.g. `benchmark.benchmark`), and add
# scripts/ itself so bare `import <script_module>` still works. This lets the
# benchmark helpers be unit-tested without running the full harness.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SCRIPTS_DIR = str(Path(_REPO_ROOT) / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run e2e tests against real Lemonade server (default: dry run with mocks)",
    )
    parser.addoption(
        "--rounds",
        action="store",
        type=int,
        default=1,
        help="Number of benchmark rounds for live e2e tests",
    )
