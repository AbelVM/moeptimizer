"""Real-use-case fixtures for the multi-turn benchmark.

See README.md and loader.py. Importing this package must never raise, so the
benchmark module stays importable even if a fixture file is missing.
"""

from __future__ import annotations

from .loader import available_files, build_fixture_tasks, fixture_root

__all__ = ["available_files", "build_fixture_tasks", "fixture_root"]
