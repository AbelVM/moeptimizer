"""NPU embedding with LanceDB semantic index."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from moeptimizer.cache import cache_get, cache_key, cache_put
from moeptimizer.config import get_config

logger = logging.getLogger(__name__)

# Shared event loop for synchronous embedding calls (avoids creating new loops)
_sync_loop: asyncio.AbstractEventLoop | None = None
_sync_loop_lock: asyncio.Lock | None = None  # Created lazily to avoid import-time event loop requirement


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """Get or create a reusable event loop for sync embedding calls."""
    global _sync_loop, _sync_loop_lock
    if _sync_loop is None or _sync_loop.is_closed():
        _sync_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_sync_loop)
    # Create lock lazily inside the loop's context
    if _sync_loop_lock is None:
        _sync_loop_lock = asyncio.Lock()
    return _sync_loop


class EmbeddingService:
    """
    Provides embeddings via the Lemonade NPU with local caching.

    Integrates with LanceDB for persistent semantic search over agent turns.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._embed_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._http_client: Any | None = None
        self._lancedb_db: Any = None

    async def initialize(self) -> None:
        """Initialize HTTP client and LanceDB connection."""
        import httpx2  # type: ignore[import-untyped]

        limits = httpx2.Limits(
            max_keepalive_connections=30,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        self._http_client = httpx2.AsyncClient(
            base_url=self._config.server.url,
            limits=limits,
            timeout=httpx2.Timeout(30.0, connect=10.0),
            transport=httpx2.AsyncHTTPTransport(retries=2),
        )

        try:
            import lancedb  # type: ignore[import-untyped]

            db_path = self._config.cache.lancedb_path
            # lancedb.connect() is synchronous (not async) in v0.17+
            self._lancedb_db = lancedb.connect(db_path)
        except Exception as e:
            logger.warning("LanceDB not available, using memory-only cache: %s", e)
            self._lancedb_db = None

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    async def get_embedding(self, text: str) -> NDArray[np.float32]:
        """Get an embedding for the given text, using cache when possible."""
        cache_key_str = cache_key(text)
        cached = cache_get(self._embed_cache, cache_key_str)
        if cached is not None:
            return cached

        assert self._http_client is not None, "HTTP client not initialized"
        try:
            result = await self._http_client.post(
                "/embeddings",
                json={"input": text, "model": self._config.server.embed_model},
            )
            if result.status_code != 200:
                embedding = np.zeros(self._config.code_chunking.embedding_dim, dtype=np.float32)
            else:
                data = result.json()
                embedding_data = data["data"]["embedding"]
                embedding = np.array(
                    embedding_data[: self._config.code_chunking.embedding_dim],
                    dtype=np.float32,
                )

            cache_put(
                self._embed_cache,
                cache_key_str,
                embedding,
                self._config.cache.embed_cache_max,
            )
            return embedding
        except Exception:
            zero_vec = np.zeros(
                self._config.code_chunking.embedding_dim,
                dtype=np.float32,
            )
            cache_put(
                self._embed_cache,
                cache_key_str,
                zero_vec,
                self._config.cache.embed_cache_max,
            )
            return zero_vec

    def _sync_get_embedding(self, text: str) -> NDArray[np.float32]:
        """Synchronous embedding (for use in thread pools)."""
        loop = _get_sync_loop()
        return loop.run_until_complete(self.get_embedding(text))

    def embed_batch_sync(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Embed a batch of texts synchronously via thread pool."""
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [pool.submit(self._sync_get_embedding, t) for t in texts]
            return [f.result() for f in futures]

    async def index_turn(
        self,
        turn_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Index a single agent turn in LanceDB for semantic search."""
        if self._lancedb_db is None:
            return

        embedding = await self.get_embedding(content)
        row = {
            "turn_id": turn_id,
            "content": content,
            "embedding": embedding.tolist(),
        }
        if metadata:
            row.update(metadata)

        try:
            table_name = f"agent_turns_{turn_id[:4]}"
            try:
                table = self._lancedb_db.open_table(table_name)
                table.add([row])
            except Exception:
                table = self._lancedb_db.create_table(
                    table_name,
                    [row],
                    mode="overwrite",
                )
        except Exception as e:
            logger.warning("Failed to index turn %s: %s", turn_id, e)

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search for similar agent turns by semantic similarity."""
        if self._lancedb_db is None:
            return []

        try:
            query_embedding = await self.get_embedding(query)
            table = self._lancedb_db.open_table("agent_turns")
            results = (
                table.search(query_embedding.tolist())
                .limit(limit)
                .to_list()
            )
            return results
        except Exception as e:
            logger.warning("LanceDB search failed: %s", e)
            return []
