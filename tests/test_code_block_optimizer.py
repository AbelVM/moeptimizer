"""Tests for code block detection and query-aware syntactic slicing."""

from moeptimizer.code_block_optimizer import (
    extract_code_blocks,
    has_code_blocks,
    slice_code_to_query,
)

_PY_CODE = """import os
import sys

def alpha():
    return 1

def beta():
    return 2

class Gamma:
    def method(self):
        return 3

def delta():
    return 4
"""


class TestCodeBlockExtraction:
    def test_has_code_blocks(self) -> None:
        assert has_code_blocks("text\n```python\ndef f(): pass\n```\n")
        assert not has_code_blocks("plain text, no fences")

    def test_extract_code_blocks(self) -> None:
        text = "before\n```python\ndef f():\n    return 1\n```\nafter"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        lang, code, start, end = blocks[0]
        assert lang == "python"
        assert "def f():" in code
        assert text[start:end].startswith("```python")


class TestSliceCodeToQuery:
    def test_slices_to_referenced_definition(self) -> None:
        sliced = slice_code_to_query(_PY_CODE, "python", "please refactor the beta function")
        assert "def beta():" in sliced
        assert "return 2" in sliced
        # Imports header is preserved.
        assert "import os" in sliced
        assert "import sys" in sliced
        # Unreferenced siblings are collapsed.
        assert "def alpha" not in sliced
        assert "class Gamma" not in sliced
        assert "def delta" not in sliced
        assert "collapsed" in sliced
        assert len(sliced) < len(_PY_CODE)

    def test_multiple_matches_kept_in_source_order(self) -> None:
        sliced = slice_code_to_query(_PY_CODE, "python", "update beta and Gamma")
        assert "def beta():" in sliced
        assert "class Gamma:" in sliced
        assert "def alpha" not in sliced
        assert "def delta" not in sliced
        assert sliced.index("def beta") < sliced.index("class Gamma")

    def test_keeps_direct_dependencies_of_queried_definition(self) -> None:
        code = """def helper():
    return 1

def target():
    return helper()

def unrelated():
    return [
""" + "        3,\n" * 100 + "    ]\n"
        sliced = slice_code_to_query(code, "python", "update target")
        assert "def target():" in sliced
        assert "def helper():" in sliced
        assert "def unrelated():" not in sliced

    def test_no_match_fails_open(self) -> None:
        assert slice_code_to_query(_PY_CODE, "python", "xyzzy") == _PY_CODE

    def test_all_match_fails_open(self) -> None:
        assert slice_code_to_query(_PY_CODE, "python", "alpha beta Gamma delta") == _PY_CODE

    def test_generic_language_fails_open(self) -> None:
        assert slice_code_to_query(_PY_CODE, "generic", "beta") == _PY_CODE

    def test_empty_query_fails_open(self) -> None:
        assert slice_code_to_query(_PY_CODE, "python", "") == _PY_CODE

    def test_no_definitions_fails_open(self) -> None:
        code = "import os\nimport sys\n"
        assert slice_code_to_query(code, "python", "os") == code

    def test_never_expands(self) -> None:
        # A tiny block where the collapse marker would outweigh the savings.
        code = "def a(): pass\ndef beta(): pass\n"
        sliced = slice_code_to_query(code, "python", "beta")
        assert len(sliced) <= len(code)

    def test_non_ascii_byte_offsets(self) -> None:
        # Multi-byte chars before the target shift byte offsets past code-point
        # offsets; correct byte slicing must still extract beta cleanly. The
        # alpha body is padded so collapsing it genuinely saves space.
        code = (
            'def alpha():\n'
            '    """café résumé naïve."""\n'
            '    x = "éééééééééééééééééééé"\n'
            '    y = "àààààààààààààààààààà"\n'
            '    return x + y\n'
            '\n'
            'def beta():\n'
            '    return 2\n'
        )
        sliced = slice_code_to_query(code, "python", "beta")
        assert "def beta():" in sliced
        assert "return 2" in sliced
        assert "def alpha" not in sliced
        assert len(sliced) < len(code)

    def test_unavailable_grammar_reports_reason_code(self) -> None:
        """REVIEW_luna P1: when the grammar is missing, slicing fails open (full
        file retained) AND reports the language via the ``unavailable`` out-param,
        so callers can surface a ``parser_unavailable`` reason code rather than
        silently implying a delta was produced."""
        unavailable: set[str] = set()
        sliced = slice_code_to_query(_PY_CODE, "not_a_real_language_xyz", "beta", unavailable)
        # Fail-open: the full file is retained unchanged.
        assert sliced == _PY_CODE
        # And the missing grammar is reported.
        assert unavailable == {"not_a_real_language_xyz"}

    def test_available_grammar_reports_nothing(self) -> None:
        """A working grammar leaves the ``unavailable`` set untouched."""
        unavailable: set[str] = set()
        slice_code_to_query(_PY_CODE, "python", "beta", unavailable)
        assert unavailable == set()
