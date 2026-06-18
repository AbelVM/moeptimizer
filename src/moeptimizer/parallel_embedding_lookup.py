"""Parallel Embedding Lookup using thread pools.

Executes embedding fetches in a thread-pool, overlapping I/O with model
inference to reduce end-to-end latency.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ParallelEmbeddingLookup:
    """
    Executes embedding fetches in a thread pool.

    Overlaps I/O (embedding requests) with model inference to reduce
    end-to-end latency. Uses a shared thread pool to avoid thread
    creation overhead on each request.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._stats: dict[str, int] = {
            "total_requests": 0,
            "batches_processed": 0,
            "total_items": 0,
        }

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create the shared thread pool executor."""
        if self._executor is None or self._executor._shutdown:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="embed_lookup",
            )
        return self._executor

    def embed_batch(
        self,
        texts: list[str],
        embed_fn,
    ) -> list[Any]:
        """Embed a batch of texts in parallel using a thread pool.

        Args:
            texts: List of text strings to embed
            embed_fn: Async or sync function to embed a single text

        Returns:
            List of embeddings in the same order as texts
        """
        if not texts:
            return []

        self._stats["total_requests"] += 1
        self._stats["total_items"] += len(texts)

        # Check if embed_fn is async
        import inspect

        is_async = inspect.iscoroutinefunction(embed_fn)

        if is_async:
            # For async functions, run them in the event loop via thread pool
            executor = self._get_executor()

            futures = [
                executor.submit(self._run_async_in_sync, embed_fn, text)
                for text in texts
            ]
            results = [f.result() for f in futures]
        else:
            # For sync functions, submit directly to thread pool
            executor = self._get_executor()
            futures = [executor.submit(embed_fn, text) for text in texts]
            results = [f.result() for f in futures]

        self._stats["batches_processed"] += 1
        return results

    def embed_batch_async(
        self,
        texts: list[str],
        embed_fn,
    ) -> list[Any]:
        """Async version: embed a batch of texts concurrently."""
        if not texts:
            return []

        import asyncio

        self._stats["total_requests"] += 1
        self._stats["total_items"] += len(texts)

        async def _gather() -> list[Any]:
            return list(await asyncio.gather(*(embed_fn(t) for t in texts)))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            results = asyncio.run(_gather())
        else:
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="async_runner",
                ) as pool:
                    results = pool.submit(
                        lambda: asyncio.new_event_loop().run_until_complete(_gather())
                    ).result()
            else:
                results = loop.run_until_complete(_gather())

        self._stats["batches_processed"] += 1
        return results


    def _run_async_in_sync(self, coro_fn, *args) -> Any:
        """Run an async function from a sync context."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_fn(*args))

        if not loop.is_running():
            return loop.run_until_complete(coro_fn(*args))

        def _runner() -> Any:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro_fn(*args))
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async_runner",
        ) as pool:
            return pool.submit(_runner).result()

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def get_stats(self) -> dict[str, int]:
        """Get lookup statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_requests": 0,
            "batches_processed": 0,
            "total_items": 0,
        }


# Global instance
_parallel_lookup: ParallelEmbeddingLookup | None = None


def get_parallel_embedding_lookup(max_workers: int = 8) -> ParallelEmbeddingLookup:
    """Get or create the global parallel embedding lookup."""
    global _parallel_lookup
    if _parallel_lookup is None:
        _parallel_lookup = ParallelEmbeddingLookup(max_workers=max_workers)
    return _parallel_lookup
