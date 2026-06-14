"""Tests for pattern injector."""

import pytest

from moeptimizer.pattern_injector import (
    PatternInjector,
    get_pattern_injector,
)


class TestPatternInjector:
    def test_empty_injector(self) -> None:
        """Empty injector has no state."""
        injector = PatternInjector()
        assert injector is not None

    def test_inject_markers(self) -> None:
        """Inject markers into context."""
        injector = PatternInjector()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        result = injector.inject_markers(messages)
        # Should have markers added
        assert "<!-- STATIC_LAYER -->" in result[0].get("content", "")
        assert "<!-- CONTEXT_LAYER -->" in result[1].get("content", "")

    def test_remove_markers(self) -> None:
        """Remove markers from context."""
        injector = PatternInjector()
        messages = [
            {"role": "system", "content": "<!-- STATIC_LAYER -->System"},
        ]
        result = injector.remove_markers(messages)
        assert "<!-- STATIC_LAYER -->" not in result[0].get("content", "")

    def test_singleton(self) -> None:
        """Get pattern injector returns new instance each time."""
        i1 = get_pattern_injector()
        i2 = get_pattern_injector()
        # Function returns new instances
        assert isinstance(i1, PatternInjector)
        assert isinstance(i2, PatternInjector)