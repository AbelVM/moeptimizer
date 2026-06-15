"""Tests for parallel_embedding_lookup module."""

import pytest

from moeptimizer.parallel_embedding_lookup import (
    ParallelEmbeddingLookup,
    get_parallel_embedding_lookup,
)


class TestParallelEmbeddingLookup:
    def setup_method(self) -> None:
        self.lookup = ParallelEmbeddingLookup(max_workers=2)

    def test_embed_batch_sync(self) -> None:
        results = []

        def embed_fn(text):
            results.append(text)
            return [0.1, 0.2]

        texts = ["hello", "world", "foo"]
        embeddings = self.lookup.embed_batch(texts, embed_fn)
        assert len(embeddings) == 3
        assert all(e == [0.1, 0.2] for e in embeddings)
        assert len(results) == 3

    def test_embed_batch_empty(self) -> None:
        embeddings = self.lookup.embed_batch([], lambda t: [0.0])
        assert embeddings == []

    def test_get_stats(self) -> None:
        self.lookup.embed_batch(["a", "b"], lambda t: [0.0])
        stats = self.lookup.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_items"] == 2
        assert stats["batches_processed"] == 1

    def test_reset_stats(self) -> None:
        self.lookup.embed_batch(["a"], lambda t: [0.0])
        self.lookup.reset_stats()
        stats = self.lookup.get_stats()
        assert stats["total_requests"] == 0

    def test_global_instance(self) -> None:
        lookup = get_parallel_embedding_lookup()
        assert isinstance(lookup, ParallelEmbeddingLookup)
