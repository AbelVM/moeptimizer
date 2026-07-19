"""Adaptive Context Quality Guard (ACQG) — closed-loop response quality regulation.

The core insight: context compression is currently an open-loop system — the
optimizer squeezes context *forward* based on size alone, with no feedback from
whether the *last* response was actually useful. When quality collapses, the
model starts emitting short stubs, missing code blocks, hallucinations, and
repetitive text. Without a backward signal, compression continues on the same
aggressive course, deepening the collapse.

ACQG closes the loop:

  1. **Response-quality indicators** — Lightweight, cheap-to-compute signals
     extracted from the last assistant response *after* it arrives:
     - Response length ratio (proxy response vs direct baseline)
     - Code block density (does the response contain code when it should?)
     - Repetition score (n-gram overlap within the response)
     - Hallucination markers (unexpected tokens, self-contradictions)
     - Truncation flag (did the response hit max_tokens?)

  2. **Adaptive compression multiplier** — Each indicator feeds into a running
     quality score (exponential moving average). When quality degrades, the
     compression multiplier is *raised* (less aggressive compression), and
     when quality is healthy, it stays at the configured baseline.

  3. **Protected content zones** — Content that was *referenced* in a good
     response (files read, code touched, symbols mentioned) is marked as
     protected and exempted from front-eviction for the next N turns.

  4. **Graceful degradation fallback** — If the EMA quality score drops below
     a critical threshold, compression is paused entirely for that turn and
     the full context is forwarded with only the immutable-prefix guard active.

The guard is transparent to the backend prefix cache: it only adjusts the
compression *aggressiveness* parameters passed to existing pipeline stages,
never mutating the frozen prefix itself.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, deque
from typing import Any

logger = logging.getLogger(__name__)

# ─── Quality indicators ──────────────────────────────────────────────────────

_MIN_RESPONSE_CHARS = 50          # Below this → likely a stub
_MIN_CODE_LINES = 3               # Expected code lines in a coding response
_REPETITION_NGRAM = 4             # N-gram size for repetition detection
_REPETITION_THRESHOLD = 0.45      # Fraction of repeated n-grams → flagged
_HALLUCINATION_PATTERNS = re.compile(
    r"(?i)\b(I (don't|do not) know|I cannot|I'm not able|"
    r"I am not able|I don't have access|unable to (determine|find|access)|"
    r"as an AI|as a language model|I was not (provided|given)|"
    r"no information (about|on)|not specified in the|"
    r"the provided (context|code) does not)\b"
)
_STUB_PATTERNS = re.compile(
    r"(?i)^(let me |i(?:'ll)? (check|look|try|see|investigate)|"
    r"i need to |i should |one moment|give me a moment|"
    r"based on the |looking at the |analyzing|examining|"
    r"i (?:will|can) (?:help|assist|explain|provide))\b"
)

# ─── State machine ──────────────────────────────────────────────────────────

_QUALITY_EMA_ALPHA = 0.3          # EMA smoothing factor
_QUALITY_CRITICAL = 0.3           # Below this → pause compression
_QUALITY_DEGRADED = 0.55          # Below this → reduce aggressiveness
_QUALITY_HEALTHY = 0.75           # Above this → use configured aggressiveness
_PROTECTED_TURNS = 3              # Turns a referenced item stays protected


class QualityIndicators:
    """Per-response quality indicators, cheap to compute."""

    __slots__ = (
        "code_line_count",
        "has_code_block",
        "has_hallucination_markers",
        "is_stub",
        "repetition_score",
        "response_chars",
        "response_role",
        "truncated",
    )

    def __init__(self) -> None:
        self.response_chars: int = 0
        self.has_code_block: bool = False
        self.code_line_count: int = 0
        self.repetition_score: float = 0.0
        self.has_hallucination_markers: bool = False
        self.is_stub: bool = False
        self.truncated: bool = False
        self.response_role: str = ""

    @classmethod
    def from_response(
        cls,
        content: str,
        role: str = "assistant",
        max_tokens_hint: int | None = None,
    ) -> QualityIndicators:
        """Compute indicators from an assistant response."""
        ind = cls()
        ind.response_role = role
        ind.response_chars = len(content)

        # Code block detection
        code_blocks = re.findall(r"```[\s\S]*?```", content)
        ind.has_code_block = bool(code_blocks)
        ind.code_line_count = sum(
            len(b.split("\n")) for b in code_blocks
        )

        # Repetition score (n-gram overlap within response)
        tokens = re.findall(r"\S+", content.lower())
        if len(tokens) >= _REPETITION_NGRAM * 2:
            ngrams = [
                " ".join(tokens[i : i + _REPETITION_NGRAM])
                for i in range(len(tokens) - _REPETITION_NGRAM + 1)
            ]
            if ngrams:
                counts = Counter(ngrams)
                total = len(ngrams)
                repeated = sum(c for c in counts.values() if c > 1)
                ind.repetition_score = repeated / total

        # Hallucination / refusal markers
        ind.has_hallucination_markers = bool(
            _HALLUCINATION_PATTERNS.search(content)
        )

        # Stub detection (short, formulaic openings)
        ind.is_stub = (
            ind.response_chars < _MIN_RESPONSE_CHARS
            or bool(_STUB_PATTERNS.match(content.strip()))
        )

        # Truncation hint
        if max_tokens_hint is not None and max_tokens_hint > 0:
            # Rough estimate: if response is suspiciously close to max_tokens
            token_estimate = ind.response_chars / 4  # rough chars→tokens
            ind.truncated = token_estimate >= max_tokens_hint * 0.9

        return ind

    def score(self) -> float:
        """Aggregate indicators into a single [0, 1] quality score.

        1.0 = perfect response. 0.0 = completely collapsed.
        """
        penalties = 0.0

        # Stub → severe penalty
        if self.is_stub:
            penalties += 0.5

        # Hallucination markers → severe penalty
        if self.has_hallucination_markers:
            penalties += 0.4

        # Very short response (even if not stub)
        if self.response_chars < _MIN_RESPONSE_CHARS:
            penalties += 0.2
        elif self.response_chars < _MIN_RESPONSE_CHARS * 3:
            penalties += 0.05

        # Repetition penalty
        if self.repetition_score > _REPETITION_THRESHOLD:
            excess = self.repetition_score - _REPETITION_THRESHOLD
            penalties += min(0.3, excess)

        # Missing code when expected (heuristic: if user message contains code)
        # This is context-dependent; we apply a small penalty for missing code
        # only when the indicator is explicitly set. Caller can override.
        if self.code_line_count == 0 and self.has_code_block:
            # Contradiction: has_code_block=True but code_line_count=0
            penalties += 0.1

        return max(0.0, 1.0 - penalties)


class ContentProtection:
    """Tracks protected content zones that should survive eviction."""

    __slots__ = ("_protected_paths", "_turn_counters")

    def __init__(self) -> None:
        self._protected_paths: dict[str, int] = {}
        self._turn_counters: dict[str, int] = {}

    def protect(self, path: str, turns: int = _PROTECTED_TURNS) -> None:
        """Mark a file path as protected for N turns."""
        self._protected_paths[path] = turns
        self._turn_counters[path] = 0

    def tick(self) -> None:
        """Decrement all protection counters; remove expired entries."""
        expired: list[str] = []
        for path in list(self._protected_paths):
            self._turn_counters[path] = self._turn_counters.get(path, 0) + 1
            if self._turn_counters[path] >= self._protected_paths[path]:
                expired.append(path)
        for path in expired:
            del self._protected_paths[path]
            del self._turn_counters[path]

    def is_protected(self, path: str) -> bool:
        """Check if a path is currently protected."""
        return path in self._protected_paths

    def protected_paths(self) -> set[str]:
        """Return the set of currently protected paths."""
        return set(self._protected_paths.keys())

    def reset(self) -> None:
        """Clear all protection. Used on session reset."""
        self._protected_paths.clear()
        self._turn_counters.clear()

    def state(self) -> dict[str, object]:
        """Return serializable state snapshot."""
        return {
            "protected_paths": list(self._protected_paths.keys()),
            "turn_counters": dict(self._turn_counters),
        }


class AdaptiveQualityGuard:
    """Closed-loop quality guard that adapts compression aggressiveness.

    Usage:
      1. After each backend response, call ``record_response(content)``.
      2. Before the next ``optimize_messages`` call, call
         ``get_compression_multiplier()`` to get a [0.0, 1.0] factor where
         1.0 = full configured compression, 0.0 = pause compression.
      3. When content is protected (files read), call ``content_protection.protect(path)``.
      4. At the start of each turn, call ``content_protection.tick()``.
    """

    def __init__(
        self,
        enabled: bool = True,
        quality_ema_alpha: float = _QUALITY_EMA_ALPHA,
        quality_critical: float = _QUALITY_CRITICAL,
        quality_degraded: float = _QUALITY_DEGRADED,
        quality_healthy: float = _QUALITY_HEALTHY,
    ) -> None:
        self._enabled = enabled
        self._alpha = quality_ema_alpha
        self._critical_threshold = quality_critical
        self._degraded_threshold = quality_degraded
        self._healthy_threshold = quality_healthy

        self._quality_ema: float = 1.0  # Start optimistic
        self._indicators_history: deque[QualityIndicators] = deque(maxlen=10)
        self._consecutive_collapsed: int = 0
        self._total_responses: int = 0

        self.content_protection = ContentProtection()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def quality_score(self) -> float:
        """Current EMA quality score, [0, 1]."""
        return self._quality_ema

    @property
    def is_collapsed(self) -> bool:
        """True when quality is critically low."""
        return self._quality_ema < self._critical_threshold

    @property
    def is_degraded(self) -> bool:
        """True when quality is below healthy."""
        return self._quality_ema < self._degraded_threshold

    @property
    def consecutive_collapsed(self) -> int:
        """How many consecutive turns had critically low quality."""
        return self._consecutive_collapsed

    def record_response(
        self,
        content: str,
        role: str = "assistant",
        max_tokens_hint: int | None = None,
    ) -> float:
        """Record a backend response and return its quality score.

        Updates the internal EMA and counters. Call this once per turn
        after receiving the full assistant response.
        """
        indicators = QualityIndicators.from_response(
            content, role=role, max_tokens_hint=max_tokens_hint
        )
        score = indicators.score()

        self._indicators_history.append(indicators)
        self._total_responses += 1

        # Update EMA
        self._quality_ema = (
            self._alpha * score + (1 - self._alpha) * self._quality_ema
        )

        # Track consecutive collapsed turns
        if score < self._critical_threshold:
            self._consecutive_collapsed += 1
        else:
            self._consecutive_collapsed = 0

        logger.debug(
            "[QualityGuard] score=%.3f ema=%.3f collapsed=%d indicator=%s",
            score, self._quality_ema, self._consecutive_collapsed,
            {
                "chars": indicators.response_chars,
                "code": indicators.has_code_block,
                "code_lines": indicators.code_line_count,
                "repetition": round(indicators.repetition_score, 3),
                "hallucination": indicators.has_hallucination_markers,
                "stub": indicators.is_stub,
                "truncated": indicators.truncated,
            },
        )

        return score

    def get_compression_multiplier(self) -> float:
        """Return a [0.0, 1.0] compression aggressiveness multiplier.

        1.0 = use configured compression as-is (healthy quality).
        0.0 = pause all compression (critically degraded).
        Between thresholds = linear interpolation.
        """
        if not self._enabled:
            return 1.0

        if self._quality_ema >= self._healthy_threshold:
            return 1.0

        if self._quality_ema < self._critical_threshold:
            logger.info(
                "[QualityGuard] CRITICAL quality ema=%.3f — pausing compression",
                self._quality_ema,
            )
            return 0.0

        # Linear interpolation between critical and degraded
        # At degraded threshold → 1.0, at critical → 0.0
        t = (
            (self._quality_ema - self._critical_threshold)
            / (self._degraded_threshold - self._critical_threshold)
        )
        return max(0.0, min(1.0, t))

    def should_skip_compression(self) -> bool:
        """True when quality is so degraded that all compression should pause.

        This is a stronger signal than get_compression_multiplier returning 0:
        it means we skip not just aggressive stages but also moderate ones.
        """
        return self._enabled and (
            self._quality_ema < self._critical_threshold
            or self._consecutive_collapsed >= 2
        )

    def get_protected_content_guard(self) -> dict[str, Any]:
        """Return a protection guard dict for the optimizer.

        The optimizer uses this to exempt protected content from eviction.
        """
        return {
            "active": self._enabled,
            "quality_score": self._quality_ema,
            "is_collapsed": self.is_collapsed,
            "protected_paths": list(self.content_protection.protected_paths()),
            "compression_multiplier": self.get_compression_multiplier(),
        }

    def reset(self) -> None:
        """Reset all state. Used on session reset."""
        self._quality_ema = 1.0
        self._indicators_history.clear()
        self._consecutive_collapsed = 0
        self._total_responses = 0
        self.content_protection.reset()

    def state(self) -> dict[str, object]:
        """Return serializable state for diagnostics."""
        return {
            "enabled": self._enabled,
            "quality_ema": self._quality_ema,
            "is_collapsed": self.is_collapsed,
            "consecutive_collapsed": self._consecutive_collapsed,
            "total_responses": self._total_responses,
            "compression_multiplier": self.get_compression_multiplier(),
            "content_protection": self.content_protection.state(),
        }


# ── Global singleton ─────────────────────────────────────────────────────────

_quality_guard: AdaptiveQualityGuard | None = None


def get_quality_guard() -> AdaptiveQualityGuard:
    """Get or create the global quality guard."""
    global _quality_guard
    if _quality_guard is None:
        _quality_guard = AdaptiveQualityGuard()
    return _quality_guard


def reset_quality_guard() -> None:
    """Reset the global quality guard (used in tests)."""
    global _quality_guard
    _quality_guard = None
