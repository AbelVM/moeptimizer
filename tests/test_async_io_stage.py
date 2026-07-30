"""Tests for AsyncIOStage queue consumer (submit_to_queue / submit_sync)."""

from __future__ import annotations

import pytest

from moeptimizer.async_io_stage import AsyncIOStage


class TestAsyncIOStageQueue:
    """Tests for the background queue consumer added for sync callers."""

    def test_submit_sync_returns_result(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            result = stage.submit_sync(lambda: "hello", stage_name="test")
            assert result == "hello"
        finally:
            stage.shutdown()

    def test_submit_sync_passes_args(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            result = stage.submit_sync(lambda x, y: x + y, 2, 3, stage_name="add")
            assert result == 5
        finally:
            stage.shutdown()

    def test_submit_sync_passes_kwargs(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            result = stage.submit_sync(
                lambda a, b=10: a + b, 5, b=3, stage_name="kw"
            )
            assert result == 8
        finally:
            stage.shutdown()

    def test_submit_sync_propagates_exception(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            with pytest.raises(ValueError, match="boom"):
                stage.submit_sync(lambda: (_ for _ in ()).throw(ValueError("boom")), stage_name="fail")
        finally:
            stage.shutdown()

    def test_submit_sync_multiple_sequential(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            results = [
                stage.submit_sync(lambda i=i: i * 2, stage_name=f"double{i}")
                for i in range(5)
            ]
            assert results == [0, 2, 4, 6, 8]
        finally:
            stage.shutdown()

    def test_submit_sync_updates_stats(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        try:
            stage.submit_sync(lambda: 1, stage_name="s1")
            stage.submit_sync(lambda: 2, stage_name="s2")
            stats = stage.get_stats()
            assert stats["queue_submitted"] == 2
            assert stats["queue_processed"] == 2
            assert stats["thread_offloads"] == 2
        finally:
            stage.shutdown()

    def test_consumer_thread_is_daemon(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        stage.submit_sync(lambda: 1, stage_name="start")
        thread = stage._consumer_thread
        assert thread is not None
        assert thread.daemon is True
        stage.shutdown()

    def test_shutdown_cleans_up(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        stage.submit_sync(lambda: 1, stage_name="start")
        stage.shutdown()
        assert stage._consumer_thread is None
        assert stage._consumer_loop is None
        assert stage._queue is None
        assert stage._thread_executor is None

    def test_submit_sync_after_shutdown_restarts(self) -> None:
        stage = AsyncIOStage(max_thread_workers=2)
        stage.submit_sync(lambda: 1, stage_name="first")
        stage.shutdown()
        # After shutdown, submit_sync should restart the consumer.
        result = stage.submit_sync(lambda: 2, stage_name="second")
        assert result == 2
        stage.shutdown()
