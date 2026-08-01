"""Async I/O for Heavy Stages.

Moves AST parsing, embedding retrieval, and compression to async workers
to keep the request thread responsive.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class AsyncIOStage:
    """
    Manages async execution of heavy pipeline stages.

    Offloads CPU-bound work (AST parsing, compression) to a thread pool
    and I/O-bound work (embedding retrieval) to async tasks, keeping
    the request thread responsive.
    """

    def __init__(
        self,
        max_thread_workers: int = 4,
        max_async_concurrency: int = 16,
    ) -> None:
        self._max_thread_workers = max_thread_workers
        self._max_async_concurrency = max_async_concurrency
        self._thread_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._queue: queue.Queue[tuple[concurrent.futures.Future[Any], Any, tuple[Any, ...], dict[str, Any], str, float]] | None = None
        self._consumer_loop: threading.Event | None = None
        self._consumer_thread: threading.Thread | None = None
        self._stats: dict[str, int] = {
            "sync_stages_completed": 0,
            "async_stages_completed": 0,
            "thread_offloads": 0,
            "total_sync_ms": 0,
            "total_async_ms": 0,
            "queue_submitted": 0,
            "queue_processed": 0,
            "queue_depth": 0,
            "max_queue_depth": 0,
            "total_queue_wait_ms": 0,
            "queue_wait_samples": 0,
            "cancellation_attempts": 0,
        }

    def _get_thread_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create the thread pool for CPU-bound work."""
        if self._thread_executor is None or self._thread_executor._shutdown:
            self._thread_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_thread_workers,
                thread_name_prefix="heavy_stage",
            )
        return self._thread_executor

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the async semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_async_concurrency)
        return self._semaphore

    async def run_async_stage(
        self,
        coro: Coroutine[Any, Any, Any],
        stage_name: str = "unknown",
    ) -> Any:
        """Run an async coroutine stage with concurrency control.

        Args:
            coro: The coroutine to run
            stage_name: Name for logging

        Returns:
            Result of the coroutine
        """
        sem = self._get_semaphore()
        start = time.monotonic()

        async with sem:
            try:
                result = await coro
                elapsed = (time.monotonic() - start) * 1000
                self._stats["async_stages_completed"] += 1
                self._stats["total_async_ms"] += int(elapsed)
                logger.debug("[AsyncIO] %s completed in %.1fms", stage_name, elapsed)
                return result
            except Exception as e:
                logger.warning("[AsyncIO] %s failed: %s", stage_name, e)
                raise

    def run_sync_stage(
        self,
        fn,
        *args,
        stage_name: str = "unknown",
        **kwargs,
    ) -> Any:
        """Run a CPU-bound function in the thread pool.

        Args:
            fn: The function to run
            *args: Positional arguments
            stage_name: Name for logging
            **kwargs: Keyword arguments

        Returns:
            Result of the function
        """
        executor = self._get_thread_executor()
        start = time.monotonic()

        try:
            future = executor.submit(fn, *args, **kwargs)
            result = future.result(timeout=30.0)
            elapsed = (time.monotonic() - start) * 1000
            self._stats["sync_stages_completed"] += 1
            self._stats["thread_offloads"] += 1
            self._stats["total_sync_ms"] += int(elapsed)
            logger.debug("[AsyncIO] %s completed in %.1fms", stage_name, elapsed)
            return result
        except concurrent.futures.TimeoutError:
            self._stats["cancellation_attempts"] += 1
            future.cancel()
            logger.warning("[AsyncIO] %s timed out; cancellation requested", stage_name)
            raise
        except Exception as e:
            logger.warning("[AsyncIO] %s failed: %s", stage_name, e)
            raise

    def _ensure_queue_consumer(self) -> queue.Queue[tuple[concurrent.futures.Future[Any], Any, tuple[Any, ...], dict[str, Any], str, float]]:
        if self._queue is not None and self._consumer_thread is not None and self._consumer_thread.is_alive():
            return self._queue
        self._queue = queue.Queue()
        self._consumer_loop = threading.Event()
        self._consumer_thread = threading.Thread(
            target=self._consume_sync_queue,
            name="heavy_stage_queue",
            daemon=True,
        )
        self._consumer_thread.start()
        return self._queue

    def _consume_sync_queue(self) -> None:
        assert self._queue is not None
        assert self._consumer_loop is not None
        while not self._consumer_loop.is_set():
            try:
                future, fn, args, kwargs, stage_name, submitted_at = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._stats["queue_depth"] = self._queue.qsize()
            self._stats["total_queue_wait_ms"] += int((time.monotonic() - submitted_at) * 1000)
            self._stats["queue_wait_samples"] += 1
            try:
                future.set_result(self.run_sync_stage(fn, *args, stage_name=stage_name, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                self._stats["queue_processed"] += 1
                self._queue.task_done()

    def submit_sync(self, fn: Any, *args: Any, stage_name: str = "unknown", **kwargs: Any) -> Any:
        """Submit synchronous work through the background queue consumer."""
        work_queue = self._ensure_queue_consumer()
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        work_queue.put((future, fn, args, kwargs, stage_name, time.monotonic()))
        self._stats["queue_submitted"] += 1
        depth = work_queue.qsize()
        self._stats["queue_depth"] = depth
        self._stats["max_queue_depth"] = max(self._stats["max_queue_depth"], depth)
        try:
            return future.result(timeout=30.0)
        except concurrent.futures.TimeoutError:
            self._stats["cancellation_attempts"] += 1
            future.cancel()
            logger.warning("[AsyncIO] %s timed out; cancellation requested", stage_name)
            raise

    async def run_ast_parsing(
        self,
        code: str,
        lang_id: str,
    ) -> Any:
        """Run AST parsing asynchronously.

        Args:
            code: The code to parse
            lang_id: The language identifier

        Returns:
            Parsed AST tree or None
        """
        async def _parse() -> Any:
            from moeptimizer.code_chunking import _get_cached_parser
            parser = _get_cached_parser(lang_id)
            if parser is None:
                return None
            try:
                return parser.parse(code)
            except Exception:
                return None

        return await self.run_async_stage(_parse(), stage_name=f"ast_parse:{lang_id}")

    async def run_embedding_retrieval(
        self,
        texts: list[str],
        embedding_service: Any,
    ) -> list[Any]:
        """Run embedding retrieval asynchronously.

        Args:
            texts: List of texts to embed
            embedding_service: The embedding service

        Returns:
            List of embeddings
        """
        async def _embed() -> list[Any]:
            sem = self._get_semaphore()
            async with sem:
                results = []
                for text in texts:
                    emb = await embedding_service.get_embedding(text)
                    results.append(emb)
                return results

        return await self.run_async_stage(
            _embed(),
            stage_name=f"embedding:{len(texts)}",
        )

    async def run_compression(
        self,
        messages: list[dict[str, Any]],
        compressor: Any,
    ) -> list[dict[str, Any]]:
        """Run context compression in a thread pool.

        Args:
            messages: The messages to compress
            compressor: The context compressor

        Returns:
            Compressed messages
        """
        def _compress() -> list[dict[str, Any]]:
            return compressor.compress(messages)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._get_thread_executor(),
            _compress,
        )
        self._stats["sync_stages_completed"] += 1
        self._stats["thread_offloads"] += 1
        return result

    async def run_batch_embeddings(
        self,
        texts: list[str],
        embedding_service: Any,
        batch_size: int = 32,
    ) -> list[Any]:
        """Run batch embedding retrieval with controlled concurrency.

        Args:
            texts: List of texts to embed
            embedding_service: The embedding service
            batch_size: Number of texts per batch

        Returns:
            List of embeddings
        """
        results: list[Any] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = await self.run_embedding_retrieval(batch, embedding_service)
            results.extend(batch_results)

        return results

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        if self._consumer_loop is not None:
            self._consumer_loop.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=1.0)
        self._consumer_thread = None
        self._consumer_loop = None
        self._queue = None
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=False)
            self._thread_executor = None

    def get_stats(self) -> dict[str, int]:
        """Get async I/O statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "sync_stages_completed": 0,
            "async_stages_completed": 0,
            "thread_offloads": 0,
            "total_sync_ms": 0,
            "total_async_ms": 0,
            "queue_submitted": 0,
            "queue_processed": 0,
            "queue_depth": 0,
            "max_queue_depth": 0,
            "total_queue_wait_ms": 0,
            "queue_wait_samples": 0,
            "cancellation_attempts": 0,
        }


# Global instance
_async_io: AsyncIOStage | None = None


def get_async_io_stage(
    max_thread_workers: int = 4,
    max_async_concurrency: int = 16,
) -> AsyncIOStage:
    """Get or create the global async I/O stage manager."""
    global _async_io
    if _async_io is None:
        _async_io = AsyncIOStage(
            max_thread_workers=max_thread_workers,
            max_async_concurrency=max_async_concurrency,
        )
    return _async_io
