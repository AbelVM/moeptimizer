"""Tests for MTP-aware speculative decoding."""

import pytest

from moeptimizer.mtp_speculative import (
    MTPSpeculativeDecoder,
    build_mtp_speculative_body,
)


class TestMTPSpeculativeDecoder:
    def test_empty_decoder(self) -> None:
        """Empty decoder has no predictions."""
        decoder = MTPSpeculativeDecoder()
        assert decoder.get_stats()["accepted"] == 0
        assert decoder.get_stats()["rejected"] == 0

    def test_get_mtp_draft_tokens_empty(self) -> None:
        """Get MTP draft tokens with empty response."""
        decoder = MTPSpeculativeDecoder()
        tokens = decoder.get_mtp_draft_tokens(None)
        assert tokens == []

    def test_should_use_mtp_draft(self) -> None:
        """Should use MTP draft when tokens available."""
        decoder = MTPSpeculativeDecoder()
        # With enough tokens (len >= 2), should use
        assert decoder.should_use_mtp_draft(["ab", "cd", "ef"])
        # With too few tokens, should not
        assert not decoder.should_use_mtp_draft(["a"])

    def test_get_verification_batch_size(self) -> None:
        """Get verification batch size."""
        decoder = MTPSpeculativeDecoder()
        # len() of strings, so "ab" has len 2
        assert decoder.get_verification_batch_size(["ab", "cd", "ef", "gh"]) == 2
        assert decoder.get_verification_batch_size(["ab", "cd"]) == 2
        assert decoder.get_verification_batch_size([]) == 1

    def test_record_verification(self) -> None:
        """Record verification updates stats."""
        decoder = MTPSpeculativeDecoder()
        decoder.record_verification(accepted=5, rejected=2)
        stats = decoder.get_stats()
        assert stats["accepted"] == 5
        assert stats["rejected"] == 2

    def test_get_acceptance_rate(self) -> None:
        """Get acceptance rate calculation."""
        decoder = MTPSpeculativeDecoder()
        assert decoder.get_acceptance_rate() == 0.0
        decoder.record_verification(accepted=8, rejected=2)
        assert decoder.get_acceptance_rate() == 0.8

    def test_build_mtp_speculative_body(self) -> None:
        """Build MTP speculative body for Lemonade server."""
        body = build_mtp_speculative_body(
            mtp_heads=3,
            mtp_lookahead=4,
            confidence_threshold=0.7,
        )
        assert body["speculative_decoding"]["enabled"] is True
        assert body["mtp_heads"] == 3
        assert body["speculative_decoding"]["mtp_lookahead"] == 4