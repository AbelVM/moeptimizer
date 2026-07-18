"""Tests for embedding service."""


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

    def test_rag_table_name_is_stable(self) -> None:
        """Index and search must target the same LanceDB table (review §9).

        The old sharded scheme wrote to ``agent_turns_{turn_id[:4]}`` but read
        from a non-existent ``agent_turns`` table, so RAG silently returned [].
        Both paths must now use the single stable ``_TABLE_NAME``.
        """
        assert EmbeddingService._TABLE_NAME == "agent_turns"
        # The table name must not depend on the turn id (no per-turn sharding).
        assert "{" not in EmbeddingService._TABLE_NAME
