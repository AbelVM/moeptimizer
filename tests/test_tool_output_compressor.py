"""Tests for ToolOutputCompressor (review §3/§5.1 boundary compression)."""

from __future__ import annotations

from moeptimizer.tool_output_compressor import (
    ToolOutputCompressor,
    compress_tool_messages,
)


def test_small_output_unchanged() -> None:
    c = ToolOutputCompressor(max_chars=4000)
    text = "short output\nwith two lines"
    assert c.compress(text) == text
    assert not c.should_compress(text)


def test_truncates_oversized_output() -> None:
    c = ToolOutputCompressor(max_chars=4000)
    # Force a clearly oversized input.
    huge = "x" * 10000
    out = c.compress(huge)
    assert "truncated" in out
    assert len(out) < len(huge)


def test_collapses_repeated_lines() -> None:
    # max_chars large enough that the collapsed output is NOT re-truncated,
    # but the input itself is over the threshold so compression runs.
    c = ToolOutputCompressor(max_chars=200)
    text = "\n".join(["SAME LINE"] * 100)  # ~900 chars -> over 200
    out = c.compress(text)
    assert "repeated 100 times" in out
    # The repeated line appears once, not 100 times.
    assert out.count("SAME LINE") == 1


def test_strips_ansi() -> None:
    c = ToolOutputCompressor(max_chars=200)
    text = "\x1b[31mred text\x1b[0m\nmore\n" + "y" * 500  # over 200
    out = c.compress(text)
    assert "\x1b[" not in out


def test_collapses_repeated_stack_frames() -> None:
    c = ToolOutputCompressor(max_chars=200)
    # A block of *distinct* frames (so repeated-line collapse can't fold it),
    # appearing verbatim twice, separated by a non-frame line. This mirrors a
    # real traceback recurring across a retry loop.
    block = "\n".join(
        f'  File "app.py", line {n}, in handler_{n}' for n in range(10)
    )
    text = block + "\nDuring handling of the above exception:\n" + block
    assert len(text) > 200  # genuinely oversized so compression runs
    out = c.compress(text)
    assert "stack frame block repeated" in out


def test_compress_tool_messages_only_large_tool_outputs() -> None:
    c = ToolOutputCompressor(max_chars=20)
    msgs = [
        {"role": "user", "content": "x" * 100},
        {"role": "tool", "content": "y" * 100},
        {"role": "assistant", "content": "z" * 100},
        {"role": "tool", "content": "small"},
    ]
    out = compress_tool_messages(msgs, c)
    # user message untouched (not in roles)
    assert out[0]["content"] == "x" * 100
    # tool + assistant large outputs compressed
    assert out[1]["content"] != "y" * 100
    assert out[2]["content"] != "z" * 100
    # small tool output unchanged
    assert out[3]["content"] == "small"
    # original list not mutated
    assert msgs[1]["content"] == "y" * 100


def test_idempotent_on_compressed_output() -> None:
    c = ToolOutputCompressor(max_chars=20)
    big = "a" * 200
    once = c.compress(big)
    twice = c.compress(once)
    assert twice == once
