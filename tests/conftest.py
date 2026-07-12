"""Pytest configuration for moeptimizer tests."""

import sys
from pathlib import Path

# Make the scripts/ directory importable as a package (e.g. `scripts.benchmark`)
# so benchmark helpers can be unit-tested without running the full harness.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
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
