"""Tests for backend client MTP integration."""

import pytest

from moeptimizer.backend_client import (
    LemonadeClient,
    SpeculativeDecoder,
)


class TestLemonadeClient:
    def test_client_creation(self) -> None:
        """Client can be created with base URL."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client is not None

    def test_speculative_decoder_disabled(self) -> None:
        """Speculative decoder is disabled by default."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        assert client.speculative_decoder is None

    def test_enable_speculative_decoding(self) -> None:
        """Speculative decoding can be enabled."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        client.enable_speculative_decoding()
        assert client.speculative_decoder is not None


class TestSpeculativeDecoder:
    def test_get_temperature_for_mtp_confidence(self) -> None:
        """Temperature is adjusted based on MTP confidence.

        For precise coding tasks, target ~0.6 for best results.
        """
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        decoder = SpeculativeDecoder(client)

        # High confidence → precise coding temperature
        assert decoder.get_temperature_for_mtp_confidence(0.9) == 0.5
        # Medium confidence → recommended for coding
        assert decoder.get_temperature_for_mtp_confidence(0.6) == 0.6
        # Low confidence → allow exploration
        assert decoder.get_temperature_for_mtp_confidence(0.3) == 0.7

    def test_get_stats(self) -> None:
        """Stats are tracked correctly."""
        client = LemonadeClient(base_url="http://localhost:13305/api/v1")
        decoder = SpeculativeDecoder(client)
        stats = decoder.get_stats()
        assert "accepted" in stats
        assert "rejected" in stats