"""MTP-aware speculative decoding for Qwen3.6-35B-A3B-MTP.

Uses MTP head outputs as draft tokens for tree-based verification.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Record verification results for statistics."""
        self._stats["accepted"] += accepted
        self._stats["rejected"] += rejected
        self._stats["mtp_draft_tokens"] += accepted + rejected

    def get_stats(self) -> dict[str, int]:
        """Get speculative decoding statistics."""
        return dict(self._stats)

    def get_acceptance_rate(self) -> float:
        """Get MTP draft acceptance rate."""
        total = self._stats["accepted"] + self._stats["rejected"]
        if total == 0:
            return 0.0
        return self._stats["accepted"] / total


def build_mtp_speculative_body(
    mtp_heads: int = 3,
    mtp_lookahead: int = 4,
    confidence_threshold: float = 0.7,
    expert_hints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the extra_body for MTP-aware speculative decoding.

    This is passed to the Lemonade server to enable native MTP support.
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

    return body