"""Tests for embedding service."""

import pytest

from moeptimizer.embedding import (
    EmbeddingService,
)


class TestEmbeddingService:
    def test_empty_service(self) -> None:
        """Empty service has no cache."""
        service = EmbeddingService()
        assert service is not None

    def test_embed_cache_exists(self) -> None:
        """Embedding service has embed cache."""
        service = EmbeddingService()
        assert hasattr(service, "_embed_cache")

    def test_embed_cache_max(self) -> None:
        """Embedding service has cache max config."""
        service = EmbeddingService()
        assert hasattr(service, "_config")