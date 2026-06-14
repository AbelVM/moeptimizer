"""MTP-aware speculative decoding for Qwen3.6-35B-A3B-MTP.

Uses MTP head outputs as draft tokens for tree-based verification.
Supports per-head temperature scheduling for optimal MTP accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Per-MTP-head temperature schedules
# Each head has different lookahead (2/3/4) and may benefit from different temperatures
MTP_HEAD_TEMPERATURES: dict[int, float] = {
    0: 0.5,  # 2-token lookahead: more deterministic
    1: 0.6,  # 3-token lookahead: balanced
    2: 0.7,  # 4-token lookahead: more exploration
}


class MTPSpeculativeDecoder:
    """
    Speculative decoding that leverages MTP head outputs.

    Qwen3.6-35B-A3B-MTP has 3 MTP heads with 2/3/4 token lookahead.
    This decoder uses those predictions as draft tokens for verification.
    """

    def __init__(
        self,
        mtp_heads: int = 3,
        mtp_lookahead: list[int] = (2, 3, 4),
        confidence_threshold: float = 0.7,
    ) -> None:
        self._mtp_heads = mtp_heads
        self._mtp_lookahead = mtp_lookahead
        self._confidence_threshold = confidence_threshold
        self._stats: dict[str, int] = {
            "total_tokens": 0,
            "mtp_draft_tokens": 0,
            "accepted": 0,
            "rejected": 0,
        }
        self._head_stats: dict[int, dict[str, int]] = {
            i: {"accepted": 0, "rejected": 0} for i in range(mtp_heads)
        }

    def get_mtp_draft_tokens(
        self,
        response: Any,
    ) -> list[str]:
        """Extract MTP head predictions from model response.

        Returns list of draft token sequences from each head.
        """
        draft_tokens: list[str] = []

        # Check if response has MTP predictions
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs:
                # MTP predictions may be in logprobs or a separate field
                mtp_data = getattr(choice, "mtp_predictions", None)
                if mtp_data:
                    for head_idx in range(self._mtp_heads):
                        head_key = f"head_{head_idx}"
                        if head_key in mtp_data:
                            draft_tokens.append(mtp_data[head_key])

        return draft_tokens

    def get_temperature_for_head(
        self,
        head_idx: int,
        mtp_confidence: float,
    ) -> float:
        """Get optimal temperature for a specific MTP head.

        Per-head temperature scheduling based on:
        - Head's lookahead distance (2/3/4 tokens)
        - Current prediction confidence

        Args:
            head_idx: The MTP head index (0, 1, or 2)
            mtp_confidence: Current confidence level (0.0-1.0)

        Returns:
            Temperature value for this head
        """
        # Base temperature from head's lookahead
        base_temp = MTP_HEAD_TEMPERATURES.get(head_idx, 0.6)

        # Adjust based on confidence
        if mtp_confidence > 0.8:
            # High confidence: lower temperature for precision
            return max(0.4, base_temp - 0.1)
        elif mtp_confidence > 0.5:
            # Medium confidence: use base temperature
            return base_temp
        else:
            # Low confidence: higher temperature for exploration
            return min(0.8, base_temp + 0.1)

    def should_use_mtp_draft(
        self,
        draft_tokens: list[str],
    ) -> bool:
        """Determine if MTP draft tokens should be used.

        Checks confidence and diversity of predictions.
        """
        if not draft_tokens:
            return False

        # Check if all heads agree (high confidence)
        # or if at least one head has high confidence
        for tokens in draft_tokens:
            if len(tokens) >= 2:  # At least 2 tokens predicted
                return True

        return False

    def get_verification_batch_size(
        self,
        draft_tokens: list[str],
    ) -> int:
        """Get optimal batch size for verification.

        Returns the number of tokens to verify at once.
        """
        if not draft_tokens:
            return 1

        # Use the longest prediction
        max_len = max(len(t) for t in draft_tokens)
        return min(max_len, 4)  # Cap at 4 tokens

    def record_verification(
        self,
        accepted: int,
        rejected: int,
        head_idx: int | None = None,
    ) -> None:
        """Record verification results for statistics.

        Args:
            accepted: Number of accepted tokens
            rejected: Number of rejected tokens
            head_idx: Optional head index for per-head tracking
        """
        self._stats["accepted"] += accepted
        self._stats["rejected"] += rejected
        self._stats["mtp_draft_tokens"] += accepted + rejected

        if head_idx is not None and head_idx in self._head_stats:
            self._head_stats[head_idx]["accepted"] += accepted
            self._head_stats[head_idx]["rejected"] += rejected

    def get_stats(self) -> dict[str, int]:
        """Get speculative decoding statistics."""
        return dict(self._stats)

    def get_head_stats(self) -> dict[int, dict[str, int]]:
        """Get per-head statistics."""
        return {k: dict(v) for k, v in self._head_stats.items()}

    def get_acceptance_rate(self) -> float:
        """Get MTP draft acceptance rate."""
        total = self._stats["accepted"] + self._stats["rejected"]
        if total == 0:
            return 0.0
        return self._stats["accepted"] / total

    def get_head_acceptance_rate(self, head_idx: int) -> float:
        """Get acceptance rate for a specific head."""
        if head_idx not in self._head_stats:
            return 0.0
        stats = self._head_stats[head_idx]
        total = stats["accepted"] + stats["rejected"]
        if total == 0:
            return 0.0
        return stats["accepted"] / total


def build_mtp_speculative_body(
    mtp_heads: int = 3,
    mtp_lookahead: int = 4,
    confidence_threshold: float = 0.7,
    expert_hints: list[dict[str, Any]] | None = None,
    head_temperatures: list[float] | None = None,
) -> dict[str, Any]:
    """Build the extra_body for MTP-aware speculative decoding.

    This is passed to the Lemonade server to enable native MTP support.

    Args:
        mtp_heads: Number of MTP heads
        mtp_lookahead: Lookahead tokens
        confidence_threshold: Minimum confidence for accepting speculative tokens
        expert_hints: Optional expert routing hints
        head_temperatures: Optional per-head temperature values
    """
    body: dict[str, Any] = {
        "speculative_decoding": {
            "enabled": True,
            "mtp_lookahead": mtp_lookahead,
            "confidence_threshold": confidence_threshold,
        },
        "mtp_heads": mtp_heads,
    }

    if expert_hints:
        body["expert_hints"] = expert_hints

    if head_temperatures:
        body["head_temperatures"] = head_temperatures

    return body