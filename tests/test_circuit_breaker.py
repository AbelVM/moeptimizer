"""Tests for the CircuitBreaker fault-tolerance utility."""

from __future__ import annotations

from moeptimizer.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_closed_state_passes_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1, name="t")
        assert cb.state == "CLOSED"
        assert cb.call(lambda: 42) == 42

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1, name="t")
        for _ in range(3):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")), fallback="fb")
        assert cb.state == "OPEN"
        # While open, calls fast-fail with the fallback.
        assert cb.call(lambda: 1, fallback="fb") == "fb"

    def test_half_open_recovers_after_cooldown(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05, name="t")
        # Open it.
        for _ in range(2):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), fallback=0)
        assert cb.state == "OPEN"
        # Wait past cooldown, then a successful probe closes it.
        import time

        time.sleep(0.08)
        assert cb.call(lambda: "ok", fallback="fb") == "ok"
        assert cb.state == "CLOSED"

    def test_half_open_reopens_on_probe_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05, name="t")
        cb.call(lambda: (_ for _ in ()).throw(Exception()), fallback=0)
        assert cb.state == "OPEN"
        import time

        time.sleep(0.08)
        # Probe fails -> re-open.
        assert cb.call(lambda: (_ for _ in ()).throw(Exception()), fallback=7) == 7
        assert cb.state == "OPEN"

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1, name="t")
        cb.call(lambda: (_ for _ in ()).throw(ValueError()), fallback=0)
        cb.call(lambda: (_ for _ in ()).throw(ValueError()), fallback=0)
        # A success resets the consecutive counter.
        assert cb.call(lambda: "ok", fallback=0) == "ok"
        assert cb.state == "CLOSED"
        # Two more failures should not open it (counter was reset).
        cb.call(lambda: (_ for _ in ()).throw(ValueError()), fallback=0)
        cb.call(lambda: (_ for _ in ()).throw(ValueError()), fallback=0)
        assert cb.state == "CLOSED"

    def test_stats_reports_state(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1.0, name="emb")
        cb.call(lambda: 1)
        stats = cb.stats()
        assert stats["name"] == "emb"
        assert stats["state"] == "CLOSED"
        assert stats["failure_threshold"] == 2
        assert stats["total_successes"] == 1
