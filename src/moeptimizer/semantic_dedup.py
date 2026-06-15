"""Semantic deduplication for near-duplicate context.

Uses embedding similarity to detect and merge near-duplicate context
that would otherwise waste KV-cache slots.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Detects and removes near-duplicate context using embedding similarity.

    Unlike exact deduplication (which removes identical code blocks),
    this finds semantically similar content that may have different
    wording but conveys the same information.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_messages: int = 100,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._max_messages = max_messages
        self._stats: dict[str, int] = {
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "similar_pairs_checked": 0,
        }

    def deduplicate(
        self,
        messages: list[dict[str, Any]],
        embedding_service: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Remove near-duplicate messages based on semantic similarity.

        Args:
            messages: The message list to deduplicate
            embedding_service: Optional embedding service for similarity computation

        Returns:
            Deduplicated message list
        """
        if not messages or len(messages) < 2:
            return messages

        if embedding_service is None:
            return messages

        # Get embeddings for all messages
        embeddings = self._get_message_embeddings(messages, embedding_service)
        if not embeddings:
            return messages

        # Find and remove duplicates
        to_remove: set[int] = set()
        n = len(messages)

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                self._stats["similar_pairs_checked"] += 1
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= self._similarity_threshold:
                    # Remove the older message (higher index)
                    to_remove.add(j)
                    self._stats["duplicates_found"] += 1

        # Build result without duplicates
        result = [
            msg for i, msg in enumerate(messages)
            if i not in to_remove
        ]
        self._stats["duplicates_removed"] = len(to_remove)

        return result

    def _get_message_embeddings(
        self,
        messages: list[dict[str, Any]],
        embedding_service: Any,
    ) -> list[np.ndarray] | None:
        """Get embeddings for all messages."""
        try:
            embeddings = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    # Use sync embedding method to avoid async issues in sync pipeline
                    emb = embedding_service._sync_get_embedding(content)
                    if emb is not None:
                        embeddings.append(emb)
            return embeddings if len(embeddings) == len(messages) else None
        except Exception as e:
            logger.debug("Failed to get embeddings for deduplication: %s", e)
            return None

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> dict[str, int]:
        """Get deduplication statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "similar_pairs_checked": 0,
        }


def get_semantic_deduplicator(
    similarity_threshold: float = 0.95,
) -> SemanticDeduplicator:
    """Get a semantic deduplicator instance."""
    return SemanticDeduplicator(similarity_threshold=similarity_threshold)