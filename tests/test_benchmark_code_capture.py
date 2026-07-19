"""Regression tests for code-preservation capture (REVIEW.md P2 / v0.7.20).

The proxy's `has_code_proxy` metric was a false zero because the model emits
code *inside tool-call arguments* (e.g. a str_replace/bash tool call rewriting a
source file) rather than in the message `content` text. The benchmark only
graded `content`+`reasoning`, so tool-emitted code was invisible. These tests
guard the fix: `_tool_calls_text` surfaces tool-call code, `_has_code_content`
detects unfenced code, and `_code_block_preservation` now sees tool-emitted code.
"""

import json

import benchmark as bm


def test_tool_calls_text_surfaces_code_in_arguments():
    tcs = [
        {
            "function": {
                "name": "str_replace",
                "arguments": json.dumps(
                    {"old_string": "x = 1", "new_string": "def f():\n    return 2"}
                ),
            }
        }
    ]
    text = bm._tool_calls_text(tcs)
    assert "def f():" in text
    assert bm._has_code_content(text) is True


def test_tool_calls_text_empty_is_safe():
    assert bm._tool_calls_text(None) == ""
    assert bm._tool_calls_text([]) == ""
    assert bm._tool_calls_text([{"function": {"name": "x", "arguments": ""}}]) == ""


def test_has_code_content_detects_unfenced_indented_code():
    # No fences, but a 4-space-indented def line -> still code.
    text = "Here is the helper:\n\n    def add(a, b):\n        return a + b\n"
    assert bm._has_code_content(text) is True


def test_has_code_content_detects_inline_backtick_code():
    text = "Use the `sum()` builtin to combine the values."
    assert bm._has_code_content(text) is True


def test_has_code_content_false_for_pure_prose():
    text = "The repository should be hardened against invalid input and fail fast."
    assert bm._has_code_content(text) is False


def test_code_block_preservation_sees_tool_emitted_code():
    # Direct response: fenced block. Proxy response: code only inside a tool call.
    direct = "```python\ndef f():\n    return 2\n```"
    proxy_tool_only = bm._tool_calls_text(
        [
            {
                "function": {
                    "name": "str_replace",
                    "arguments": json.dumps({"new_string": "def f():\n    return 2"}),
                }
            }
        ]
    )
    res = bm._code_block_preservation(direct, proxy_tool_only)
    # Before the fix, has_code_proxy would be False (tool code invisible).
    assert res["has_code_proxy"] is True
    assert res["has_code_direct"] is True
