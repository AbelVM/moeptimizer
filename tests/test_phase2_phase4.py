"""Tests for Release 0.7.27 — Phase 2 (Context Savings) & Phase 4 (Throughput Optimization)."""

from __future__ import annotations

from unittest.mock import MagicMock

from moeptimizer.async_io_stage import AsyncIOStage
from moeptimizer.tool_output_compressor import ToolOutputCompressor
from moeptimizer.tool_output_filter import ToolOutputFilter


def test_semantic_stack_frame_pruning() -> None:
    compressor = ToolOutputCompressor(max_chars=400)
    lines = [
        'Traceback (most recent call last):',
        '  File "/workspace/src/main.py", line 10, in main',
        '  File "/usr/lib/python3.11/site-packages/flask/app.py", line 100, in handle',
        '  File "/usr/lib/python3.11/site-packages/werkzeug/serving.py", line 200, in run',
        '  File "/usr/lib/python3.11/dist-packages/urllib3/poolmanager.py", line 50, in get',
        '  File "/usr/lib/python3.11/site-packages/requests/api.py", line 30, in request',
        '  File "/usr/lib/python3.11/dist-packages/pip/internal.py", line 12, in do_work',
        '  File "/workspace/src/utils.py", line 45, in parse',
    ]
    traceback_text = "\n".join(lines)
    compressed = compressor.compress(traceback_text)

    assert "omitted" in compressed
    assert 'File "/workspace/src/main.py"' in compressed
    assert 'File "/workspace/src/utils.py"' in compressed


def test_tool_output_filter_fast_path_and_large_payload() -> None:
    tf = ToolOutputFilter()

    # Fast path: already has filter marker
    marker = "[go test result]"
    assert tf.filter(marker) == marker

    # Large payload guard: 600K chars
    large_payload = "ok  pkg/foo  0.5s\n" + ("a" * 600000)
    filtered = tf.filter(large_payload)
    assert filtered == "[go test result]"


def test_async_io_offloading_integration() -> None:
    async_stage = AsyncIOStage(max_thread_workers=2)
    mock_fn = MagicMock(return_value="optimized_result")

    res = async_stage.run_sync_stage(mock_fn, stage_name="test_stage")
    assert res == "optimized_result"
    mock_fn.assert_called_once()

    stats = async_stage.get_stats()
    assert stats["sync_stages_completed"] == 1
    assert stats["thread_offloads"] == 1
