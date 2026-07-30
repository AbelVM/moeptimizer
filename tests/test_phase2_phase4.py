"""Tests for Release 0.7.27 — Phase 2 (Context Savings) & Phase 4 (Throughput Optimization)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from moeptimizer.async_io_stage import AsyncIOStage
from moeptimizer.thinking_preserver import ThinkingPreserver
from moeptimizer.tool_output_compressor import ToolOutputCompressor
from moeptimizer.tool_output_filter import ToolOutputFilter


def test_json_minification_in_compressor() -> None:
    compressor = ToolOutputCompressor(max_chars=100)
    json_data = {
        "status": "success",
        "items": [1, 2, 3, 4, 5],
        "metadata": {"nested": "value", "count": 42},
    }
    pretty_json = json.dumps(json_data, indent=4)
    assert len(pretty_json) > 100

    compressed = compressor.compress(pretty_json)
    minified_expected = json.dumps(json_data, separators=(",", ":"))
    assert compressed == minified_expected
    assert len(compressed) < len(pretty_json)


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


def test_distill_old_reasoning() -> None:
    preserver = ThinkingPreserver()
    long_reasoning = "Step 1: analyze.\n" + ("X" * 400) + "\nConclusion: root cause found in utils."
    messages = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": f"<think>\n{long_reasoning}\n</think>\nResponse 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": f"<think>\n{long_reasoning}\n</think>\nResponse 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": f"<think>\n{long_reasoning}\n</think>\nResponse 3"},
    ]

    distilled = preserver.distill_old_reasoning(messages, protect_recent=2)

    # Turn 1 (index 1) should be distilled
    assert "[Distilled reasoning:" in distilled[1]["content"]
    assert "Conclusion: root cause found in utils." in distilled[1]["content"]

    # Turn 2 and 3 should be preserved (recent 2 assistant turns)
    assert long_reasoning in distilled[3]["content"]
    assert long_reasoning in distilled[5]["content"]


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
