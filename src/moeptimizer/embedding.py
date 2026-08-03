"""NPU embedding with LanceDB semantic index."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from moeptimizer.cache import cache_get, cache_key, cache_put
from moeptimizer.circuit_breaker import CircuitBreaker
from moeptimizer.config import get_config

logger = logging.getLogger(__name__)

# Bound on how long a sync caller waits for an embedding scheduled onto the
# dedicated loop. A hang here would otherwise stall the optimizer executor.
_SYNC_FETCH_TIMEOUT = 30.0


class EmbeddingService:
    """
    Provides embeddings via the Lemonade NPU with local caching.

    Integrates with LanceDB for persistent semantic search over agent turns.

    All embedding HTTP I/O runs on ONE dedicated background event loop started in
    ``initialize()`` (review §4.8.1). The httpx async client is only ever used from
    that loop, and synchronous callers (the optimizer executor) schedule
    ``get_embedding`` onto it via ``run_coroutine_threadsafe``. The previous design
    also meant to use a shared loop but never started it, so
    ``run_coroutine_threadsafe(...).result()`` deadlocked; combined with the
    per-session service never being initialized, every embedding came back a zero
    vector and RAG ranking silently did nothing.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._embed_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._http_client: Any | None = None
        self._lancedb_db: Any = None
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None
        # Circuit breaker: an embedding-server outage must not block the
        # optimization pipeline. After repeated failures we fast-fail with a
        # zero vector for a cooldown window instead of hammering the server.
        self._breaker = CircuitBreaker(
            failure_threshold=5,
            cooldown_seconds=30.0,
            name="embedding",
        )

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        """Start (once) and return the dedicated background event loop."""
        if self._sync_loop is None or not self._sync_loop.is_running():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="moept-embedding", daemon=True
            )
            thread.start()
            self._sync_loop = loop
            self._sync_thread = thread
        return self._sync_loop

    async def initialize(self) -> None:
        """Initialize the dedicated loop, HTTP client, and LanceDB connection."""
        import httpx2  # type: ignore[import-untyped]

        self._ensure_sync_loop()

        limits = httpx2.Limits(
            max_keepalive_connections=30,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        self._http_client = httpx2.AsyncClient(
            base_url=self._config.server.embed_url,
            headers={"Authorization": f"Bearer {self._config.server.embed_api_key}"},
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
        """Close the HTTP client (on its owning loop) and stop the loop."""
        loop = self._sync_loop
        if self._http_client and loop is not None and loop.is_running():
            # aclose() must run on the loop that owns the client's connection pool.
            try:
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(self._http_client.aclose(), loop)
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Error closing embedding client: %s", e)
        elif self._http_client:
            await self._http_client.aclose()
        self._http_client = None
        if self._lancedb_db is not None and hasattr(self._lancedb_db, "close"):
            self._lancedb_db.close()
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if self._sync_thread is not None:
                self._sync_thread.join(timeout=5.0)
            self._sync_loop = None
            self._sync_thread = None

    async def get_embedding(self, text: str) -> NDArray[np.float32]:
        """Get an embedding for the given text, using cache when possible.

        The external embedding call is protected by a circuit breaker so a
        server outage fast-fails to a zero vector instead of blocking the
        optimization pipeline (review §10).

        Genuinely async (review §4.8.1): the HTTP call is awaited directly on the
        caller's event loop. The previous implementation scheduled the coroutine
        onto a separate loop via ``run_coroutine_threadsafe(...).result()`` — but
        that loop was never run, so the call deadlocked the event loop (masked only
        because the per-session service was never initialized, which in turn made
        every embedding a zero vector and silently disabled RAG ranking).
        """
        cache_key_str = cache_key(text)
        cached = cache_get(self._embed_cache, cache_key_str)
        if cached is not None:
            return cached

        zero_vec = np.zeros(self._config.code_chunking.embedding_dim, dtype=np.float32)

        async def _fetch() -> NDArray[np.float32]:
            assert self._http_client is not None, "HTTP client not initialized"
            result = await self._http_client.post(
                "/embeddings",
                json={"input": text, "model": self._config.server.embed_model},
            )
            if result.status_code != 200:
                return zero_vec
            data = result.json()
            embedding_data = data["data"]["embedding"]
            return np.array(
                embedding_data[: self._config.code_chunking.embedding_dim],
                dtype=np.float32,
            )

        embedding = await self._breaker.call_async(_fetch, fallback=zero_vec)
        cache_put(
            self._embed_cache,
            cache_key_str,
            embedding,
            self._config.cache.embed_cache_max,
        )
        return embedding

    def breaker_stats(self) -> dict[str, object]:
        """Return circuit-breaker state for diagnostics/dashboards."""
        return self._breaker.stats()

    def _sync_get_embedding(self, text: str) -> NDArray[np.float32]:
        """Synchronous embedding helper for the (sync) optimizer executor path.

        Schedules ``get_embedding`` onto the dedicated background loop and waits for
        the result (review §4.8.1). The httpx client is only ever used from that one
        loop, so this is safe from any caller thread. When the service was never
        initialized (direct construction in tests) there is no loop/client, so fall
        back to a one-shot ``asyncio.run`` which trips the breaker's assert and
        returns a zero vector — the legacy behavior.
        """
        loop = self._sync_loop
        if loop is None or not loop.is_running() or self._http_client is None:
            return asyncio.run(self.get_embedding(text))
        return asyncio.run_coroutine_threadsafe(
            self.get_embedding(text), loop
        ).result(timeout=_SYNC_FETCH_TIMEOUT)

    def embed_batch_sync(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Embed a batch of texts synchronously.

        Schedules every ``get_embedding`` onto the single dedicated loop (concurrent
        on that loop) and collects the results — no nested thread pools and no
        per-call event loops (review §4.8.1 / §4.4.5).
        """
        if not texts:
            return []
        loop = self._sync_loop
        if loop is not None and loop.is_running() and self._http_client is not None:
            scheduled = [
                asyncio.run_coroutine_threadsafe(self.get_embedding(t), loop)
                for t in texts
            ]
            return [f.result(timeout=_SYNC_FETCH_TIMEOUT) for f in scheduled]

        def _embed_one(text: str) -> NDArray[np.float32]:
            return asyncio.run(self.get_embedding(text))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(texts), 8)
        ) as pool:
            return [f.result() for f in [pool.submit(_embed_one, t) for t in texts]]

    # Single stable table for all agent turns. Sharding by turn-id prefix
    # (the old `agent_turns_{turn_id[:4]}` scheme) made `search_similar` read
    # from a table that was never written, so RAG silently returned [] (review
    # §9). One table keeps index and search consistent.
    _TABLE_NAME = "agent_turns"

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
            try:
                table = self._lancedb_db.open_table(self._TABLE_NAME)
                table.add([row])
            except Exception:
                table = self._lancedb_db.create_table(
                    self._TABLE_NAME,
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
            table = self._lancedb_db.open_table(self._TABLE_NAME)
            results = (
                table.search(query_embedding.tolist())
                .limit(limit)
                .to_list()
            )
            return results
        except Exception as e:
            logger.warning("LanceDB search failed: %s", e)
            return []
