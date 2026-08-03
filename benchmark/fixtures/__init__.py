"""Real-use-case fixtures for the multi-turn benchmark.

See README.md and loader.py. Importing this package must never raise, so the
benchmark module stays importable even if a fixture file is missing.
"""

from __future__ import annotations

__all__ = ["available_files", "build_fixture_tasks", "fixture_root"]


def __getattr__(name: str):
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(name)
