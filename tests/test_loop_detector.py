"""Tests for LoopDetector."""


from moeptimizer.loop_detector import LoopDetector
from moeptimizer.models import AgentStep, LoopWarning


class TestLoopDetector:
    def test_no_loop_few_steps(self) -> None:
        detector = LoopDetector(threshold=3)
        for i in range(2):
            warning = detector.analyze_step(AgentStep(role="assistant", content="do something"))
            assert warning is None

    def test_tool_repeat_loop(self) -> None:
        detector = LoopDetector(threshold=3)
        for i in range(3):
            warning = detector.analyze_step(AgentStep(role="assistant", tool_name="search"))
        assert warning is not None
        assert warning.loop_type == "tool_repeat"
        assert warning.tool_name == "search"
        assert warning.repeat_count == 3

    def test_action_repeat_loop(self) -> None:
        detector = LoopDetector(threshold=3)
        content = "call search("
        for i in range(3):
            warning = detector.analyze_step(AgentStep(role="assistant", content=content))
        assert warning is not None
        assert warning.loop_type == "action_repeat"

    def test_thinking_loop(self) -> None:
        detector = LoopDetector(threshold=3)
        for i in range(3):
            warning = detector.analyze_step(AgentStep(role="thinking", content="thinking..."))
        assert warning is not None
        assert warning.loop_type == "thinking_loop"

    def test_reset_thinking_on_action(self) -> None:
        detector = LoopDetector(threshold=3)
        detector.analyze_step(AgentStep(role="thinking", content="thinking 1"))
        detector.analyze_step(AgentStep(role="thinking", content="thinking 2"))
        detector.analyze_step(AgentStep(role="assistant", content="call search("))
        # After action, thinking counter resets
        warning = detector.analyze_step(AgentStep(role="thinking", content="thinking 3"))
        # Should not trigger thinking_loop since counter was reset
        assert warning is None or warning.loop_type != "thinking_loop"

    def test_get_warning_message(self) -> None:
        detector = LoopDetector(threshold=3)
        warning = LoopWarning(loop_type="tool_repeat", tool_name="search", repeat_count=3)
        msg = detector.get_warning_message(warning)
        assert "[LOOP DETECTED: tool_repeat]" in msg

    def test_reset_warnings(self) -> None:
        detector = LoopDetector(threshold=3)
        detector._recent_warnings.append(LoopWarning(loop_type="test"))
        detector.reset_warnings()
        assert detector.get_recent_warnings() == []
