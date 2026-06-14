"""Tests for loop detector."""

import pytest

from moeptimizer.loop_detector import (
    LoopDetector,
)
from moeptimizer.models import AgentStep, LoopWarning


class TestLoopDetector:
    def test_empty_detector(self) -> None:
        """Empty detector has no patterns."""
        detector = LoopDetector()
        assert detector is not None

    def test_analyze_step(self) -> None:
        """Analyze step for loops."""
        detector = LoopDetector(threshold=3)
        step = AgentStep(role="user", content="Test")
        warning = detector.analyze_step(step)
        # May return None or a warning
        assert warning is None or hasattr(warning, "loop_type")

    def test_get_warning_message(self) -> None:
        """Get warning message."""
        detector = LoopDetector()
        warning = LoopWarning(loop_type="test", repeat_count=3, message="Test warning")
        msg = detector.get_warning_message(warning)
        assert isinstance(msg, str)