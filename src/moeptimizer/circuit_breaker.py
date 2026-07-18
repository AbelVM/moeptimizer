"""Circuit breaker for fault-tolerant external service calls.

Prevents a failing downstream dependency (e.g. the embedding server) from
throttling the optimization pipeline. After ``failure_threshold`` consecutive
failures the breaker *opens* and immediately returns the fallback value for
``cooldown_seconds`` without contacting the dependency. It then moves to
*half-open* for one probe call; success closes it, failure re-opens it.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Bounded-failure circuit breaker with three states.

    States:
    - CLOSED: normal operation, calls pass through.
    - OPEN: failing fast, calls return the fallback immediately.
    - HALF_OPEN: a single probe call is allowed to test recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        name: str = "circuit",
    ) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._name = name
        self._lock = threading.RLock()
        self._consecutive_failures = 0
        self._state = "CLOSED"
        self._opened_at = 0.0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == "OPEN" and (time.time() - self._opened_at) >= self._cooldown_seconds:
            self._state = "HALF_OPEN"
            logger.info("[%s] Circuit half-open: probing recovery", self._name)

    def call(self, func, *args, fallback=None, **kwargs):
        """Run *func* under circuit-breaker protection.

        On open/half-open failure the breaker records the failure and returns
        *fallback*. Exceptions from *func* are swallowed (logged) and treated
        as failures so the pipeline never crashes on a dependency outage.
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == "OPEN":
                logger.debug("[%s] Circuit open: fast-fail with fallback", self._name)
                return fallback

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            self._on_failure()
            logger.warning("[%s] Call failed, using fallback: %s", self._name, e)
            return fallback

        self._on_success()
        return result

    def _on_failure(self) -> None:
        with self._lock:
            self._total_failures += 1
            if self._state == "HALF_OPEN":
                # Probe failed: re-open for another cooldown.
                self._state = "OPEN"
                self._opened_at = time.time()
                self._consecutive_failures = self._failure_threshold
                logger.warning("[%s] Half-open probe failed: re-opening", self._name)
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.time()
                logger.warning(
                    "[%s] Circuit opened after %d consecutive failures",
                    self._name,
                    self._consecutive_failures,
                )

    def _on_success(self) -> None:
        with self._lock:
            self._total_successes += 1
            if self._state == "HALF_OPEN":
                logger.info("[%s] Half-open probe succeeded: circuit closed", self._name)
            self._state = "CLOSED"
            self._consecutive_failures = 0

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._opened_at = 0.0

    def stats(self) -> dict[str, object]:
        """Return a snapshot of breaker state for diagnostics/dashboards."""
        with self._lock:
            return {
                "name": self._name,
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_threshold": self._failure_threshold,
                "cooldown_seconds": self._cooldown_seconds,
            }
