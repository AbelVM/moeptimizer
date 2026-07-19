"""Tests for quality_guard.py — Adaptive Context Quality Guard."""


from moeptimizer.quality_guard import (
    AdaptiveQualityGuard,
    ContentProtection,
    QualityIndicators,
    get_quality_guard,
    reset_quality_guard,
)


class TestQualityIndicators:
    """QualityIndicators.from_response tests."""

    def test_healthy_response(self) -> None:
        content = (
            "Here's the fix for the bug in `process_data`:\n\n"
            "```python\n"
            "def process_data(items):\n"
            "    return [item.strip() for item in items if item]\n"
            "```\n\n"
            "This uses a list comprehension with a filter for empty items."
        )
        ind = QualityIndicators.from_response(content)
        assert ind.has_code_block is True
        assert ind.code_line_count >= 2
        assert ind.is_stub is False
        assert ind.has_hallucination_markers is False
        assert ind.repetition_score < 0.3
        assert ind.score() > 0.8

    def test_stub_response(self) -> None:
        content = "Let me check that for you and get back with the results."
        ind = QualityIndicators.from_response(content)
        assert ind.is_stub is True
        assert ind.score() < 0.6

    def test_hallucination_markers(self) -> None:
        content = "I don't have access to the file system in this context."
        ind = QualityIndicators.from_response(content)
        assert ind.has_hallucination_markers is True
        assert ind.score() < 0.7

    def test_refusal_patterns(self) -> None:
        content = "As an AI, I cannot execute code directly."
        ind = QualityIndicators.from_response(content)
        assert ind.has_hallucination_markers is True
        assert ind.score() < 0.7

    def test_repetitive_response(self) -> None:
        # Build a response with heavy repetition
        part = "the quick brown fox jumps over the lazy dog "
        content = part * 20
        ind = QualityIndicators.from_response(content)
        assert ind.repetition_score > 0.45  # High repetition
        assert ind.score() < 0.8

    def test_empty_response(self) -> None:
        content = ""
        ind = QualityIndicators.from_response(content)
        assert ind.is_stub is True
        assert ind.score() < 0.4

    def test_truncation_detection(self) -> None:
        # Response close to max_tokens limit
        content = "word " * 800  # ~4000 chars ≈ 1000 tokens
        ind = QualityIndicators.from_response(content, max_tokens_hint=1100)
        assert ind.truncated is True

    def test_not_truncated(self) -> None:
        content = "short response"
        ind = QualityIndicators.from_response(content, max_tokens_hint=700)
        assert ind.truncated is False

    def test_code_response_no_hallucination(self) -> None:
        content = (
            "Here's the implementation:\n"
            "```python\n"
            "def solve():\n"
            "    return 42\n"
            "```\n"
            "This solves the problem by returning the answer directly."
        )
        ind = QualityIndicators.from_response(content)
        assert ind.has_code_block is True
        assert ind.code_line_count >= 2
        assert ind.has_hallucination_markers is False
        assert ind.score() > 0.7


class TestContentProtection:
    """ContentProtection tests."""

    def test_protect_and_expire(self) -> None:
        cp = ContentProtection()
        cp.protect("src/main.py", turns=3)
        assert cp.is_protected("src/main.py") is True

        cp.tick()
        assert cp.is_protected("src/main.py") is True

        cp.tick()
        assert cp.is_protected("src/main.py") is True

        cp.tick()
        assert cp.is_protected("src/main.py") is False

    def test_protect_one_turn(self) -> None:
        cp = ContentProtection()
        cp.protect("README.md", turns=1)
        assert cp.is_protected("README.md") is True
        cp.tick()
        assert cp.is_protected("README.md") is False

    def test_multiple_paths(self) -> None:
        cp = ContentProtection()
        cp.protect("a.py", turns=2)
        cp.protect("b.py", turns=1)
        cp.tick()
        assert cp.is_protected("a.py") is True
        assert cp.is_protected("b.py") is False

    def test_protected_paths_set(self) -> None:
        cp = ContentProtection()
        cp.protect("x.py", turns=2)
        assert "x.py" in cp.protected_paths()
        cp.tick()
        assert "x.py" in cp.protected_paths()
        cp.tick()
        assert "x.py" not in cp.protected_paths()

    def test_reset(self) -> None:
        cp = ContentProtection()
        cp.protect("keep.py", turns=10)
        assert cp.is_protected("keep.py") is True
        cp.reset()
        assert cp.is_protected("keep.py") is False
        assert len(cp.protected_paths()) == 0

    def test_state_snapshot(self) -> None:
        cp = ContentProtection()
        cp.protect("a.py", turns=5)
        state = cp.state()
        assert "a.py" in state["protected_paths"]
        assert isinstance(state["turn_counters"], dict)


class TestAdaptiveQualityGuard:
    """AdaptiveQualityGuard tests."""

    def test_initial_quality(self) -> None:
        guard = AdaptiveQualityGuard()
        assert guard.quality_score == 1.0
        assert guard.is_collapsed is False
        assert guard.is_degraded is False

    def test_healthy_response_keeps_high_score(self) -> None:
        guard = AdaptiveQualityGuard()
        content = "```python\ndef f(): pass\n```\nHere's the implementation."
        guard.record_response(content)
        assert guard.quality_score > 0.8
        assert guard.is_collapsed is False

    def test_stub_lowers_score(self) -> None:
        guard = AdaptiveQualityGuard()
        guard.record_response("Let me check that.")  # stub
        assert guard.quality_score < 0.8
        assert guard.is_collapsed is False  # one stub shouldn't trigger critical

    def test_critical_collapse(self) -> None:
        guard = AdaptiveQualityGuard(
            quality_ema_alpha=1.0,  # Instant update
            quality_critical=0.35,  # Slightly above the 0.3 score of empty
        )
        for _ in range(5):
            guard.record_response("")  # empty = severe degradation
        assert guard.is_collapsed is True
        assert guard.consecutive_collapsed >= 2

    def test_compression_multiplier_healthy(self) -> None:
        guard = AdaptiveQualityGuard()
        content = "Here's the code:\n```python\ndef solve(): pass\n```"
        guard.record_response(content)
        assert guard.get_compression_multiplier() == 1.0

    def test_compression_multiplier_critical(self) -> None:
        guard = AdaptiveQualityGuard(quality_ema_alpha=1.0)  # Instant update
        guard.record_response("I don't know how to do that.")
        assert guard.get_compression_multiplier() == 0.0

    def test_compression_multiplier_degraded(self) -> None:
        guard = AdaptiveQualityGuard(
            quality_ema_alpha=1.0,  # Instant update
            quality_critical=0.2,
            quality_degraded=0.6,
            quality_healthy=0.8,
        )
        # Score around 0.5 (degraded but not critical)
        guard.record_response("Let me look into this and get back to you.")
        mult = guard.get_compression_multiplier()
        assert 0.0 < mult < 1.0

    def test_should_skip_compression_collapsed(self) -> None:
        guard = AdaptiveQualityGuard(quality_ema_alpha=1.0)
        guard.record_response("I don't know.")
        assert guard.should_skip_compression() is True

    def test_should_not_skip_compression_healthy(self) -> None:
        guard = AdaptiveQualityGuard()
        guard.record_response("```python\nx = 1\n```\nDone.")
        assert guard.should_skip_compression() is False

    def test_disabled_guard(self) -> None:
        guard = AdaptiveQualityGuard(enabled=False)
        guard.record_response("")  # would collapse if enabled
        assert guard.get_compression_multiplier() == 1.0
        assert guard.should_skip_compression() is False

    def test_consecutive_collapsed_tracking(self) -> None:
        guard = AdaptiveQualityGuard(quality_ema_alpha=1.0)
        guard.record_response("I cannot do that.")  # bad
        assert guard.consecutive_collapsed == 1
        guard.record_response("I don't know.")  # bad
        assert guard.consecutive_collapsed == 2
        guard.record_response("```python\npass\n```")  # good
        assert guard.consecutive_collapsed == 0

    def test_state_snapshot(self) -> None:
        guard = AdaptiveQualityGuard()
        guard.record_response("some response")
        state = guard.state()
        assert "quality_ema" in state
        assert "is_collapsed" in state
        assert "compression_multiplier" in state

    def test_reset(self) -> None:
        guard = AdaptiveQualityGuard(quality_ema_alpha=1.0)
        guard.record_response("I don't know.")
        assert guard.is_collapsed is True
        guard.reset()
        assert guard.quality_score == 1.0
        assert guard.is_collapsed is False
        assert guard.consecutive_collapsed == 0

    def test_total_responses_counted(self) -> None:
        guard = AdaptiveQualityGuard()
        for i in range(5):
            guard.record_response(f"response {i}")
        assert guard._total_responses == 5


class TestIntegration:
    """Integration scenarios with the global singleton."""

    def test_singleton_get(self) -> None:
        reset_quality_guard()
        g1 = get_quality_guard()
        g2 = get_quality_guard()
        assert g1 is g2

    def test_singleton_reset(self) -> None:
        reset_quality_guard()
        g = get_quality_guard()
        g.record_response("bad response")
        assert g.quality_score < 1.0
        reset_quality_guard()
        g2 = get_quality_guard()
        assert g2.quality_score == 1.0
        assert g2 is not g

    def test_scenario_quality_collapse_and_recovery(self) -> None:
        """Simulate a real agentic session with quality collapse."""
        guard = AdaptiveQualityGuard()

        # Turn 1: Good response with code
        guard.record_response(
            "Here's the implementation:\n"
            "```python\n"
            "def solve():\n"
            "    return 42\n"
            "```\n"
        )
        assert guard.quality_score > 0.7
        assert guard.get_compression_multiplier() == 1.0

        # Turn 2: Stub
        guard.record_response("Let me check that.")
        assert guard.quality_score < 0.8

        # Turn 3: Collapsed
        guard.record_response("I don't know how to access that file.")
        assert guard.quality_score < 0.7
        mult = guard.get_compression_multiplier()
        assert mult < 1.0  # Compression should back off

        # Turn 4: Recovery with good response
        guard.record_response(
            "Found the issue:\n"
            "```python\n"
            "def solve():\n"
            "    return 42\n"
            "```\n"
        )
        # Quality should start recovering
        assert guard.quality_score > 0.5
