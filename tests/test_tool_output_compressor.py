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


# ---------------------------------------------------------------------------
# Error-aware extraction (review §4.1.1)
# ---------------------------------------------------------------------------


def _failing_log(n_passing: int = 40) -> str:
    passing = "\n".join(f"tests/test_ok_{i}.py::test_case PASSED" for i in range(n_passing))
    return (
        "============================= test session starts ==============================\n"
        "collected 42 items\n"
        f"{passing}\n"
        "tests/test_auth.py::test_login FAILED\n"
        "    def test_login():\n"
        ">       assert client.login('x') == 200\n"
        "E       AssertionError: assert 401 == 200\n"
        "tests/test_auth.py:18: AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: assert 401 == 200\n"
        "========================= 1 failed, 41 passed in 2.34s =========================\n"
    )


def test_error_aware_keeps_failure_diagnostics() -> None:
    c = ToolOutputCompressor(max_chars=800)
    log = _failing_log()
    out = c.compress(log)
    # The diagnostic signal survives...
    assert "AssertionError: assert 401 == 200" in out
    assert "FAILED tests/test_auth.py::test_login" in out
    assert "1 failed, 41 passed" in out
    # ...while the bulk of the passing-test noise is dropped.
    assert len(out) < len(log)
    assert out.count("PASSED") < 40


def test_error_aware_collapses_pure_success() -> None:
    c = ToolOutputCompressor(max_chars=400)
    log = (
        "\n".join(f"tests/test_ok_{i}.py::test_case PASSED" for i in range(60))
        + "\n========================= 60 passed in 1.23s =========================\n"
    )
    out = c.compress(log)
    assert "60 passed in 1.23s" in out
    # Collapsed to the verdict: passing lines are gone.
    assert "PASSED" not in out


def test_error_aware_skips_raw_stack_trace_without_summary() -> None:
    # No verdict/summary line -> not a structured result -> falls through to the
    # head/tail truncator (no error-aware marker), preserving prior behavior.
    c = ToolOutputCompressor(max_chars=200)
    trace = "\n".join(f'  File "app.py", line {n}, in f{n}' for n in range(30))
    out = c.compress(trace)
    assert "error-aware compressed" not in out


def test_error_aware_idempotent() -> None:
    c = ToolOutputCompressor(max_chars=800)
    once = c.compress(_failing_log())
    twice = c.compress(once)
    assert twice == once


def test_has_failure_signal() -> None:
    from moeptimizer.tool_output_compressor import has_failure_signal

    assert has_failure_signal("E   AssertionError: assert 1 == 2")
    assert has_failure_signal('  File "x.py", line 3, in foo')
    assert has_failure_signal("error: could not compile `crate`")
    assert not has_failure_signal("60 passed in 1.23s")
    assert not has_failure_signal("Build succeeded")
