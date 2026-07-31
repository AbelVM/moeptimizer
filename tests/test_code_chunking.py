"""Tests for code chunking."""


from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    chunk_text_fallback,
    deduplicate_chunks,
    detect_language_and_id,
)


class TestCodeChunking:
    def test_detect_language(self) -> None:
        """Detect language from code."""
        code = "def foo():\n    pass\n"
        lang = detect_language_and_id(code)
        assert lang in ("python", "generic")

    def test_chunk_code(self) -> None:
        """Chunk code into pieces."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = chunk_code_with_treesitter(code, "python", 1000)
        assert isinstance(chunks, list)

    def test_deduplicate_chunks(self) -> None:
        """Deduplicate chunks."""
        chunks = ["def foo(): pass", "def foo(): pass", "def bar(): pass"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_lang_map(self) -> None:
        """Language map has expected entries."""
        assert "python" in LANG_MAP
        assert "javascript" in LANG_MAP

    def test_ast_path_actually_runs(self) -> None:
        """Regression: the tree-sitter AST path must run, not silently fall back.

        Under tree-sitter 0.25 the API uses properties (``root_node``, ``type``,
        ``child_count``, ``byte_range``) and ``parse`` takes bytes; calling them as
        methods raised inside the ``except`` guard and the function always degraded
        to ``chunk_text_fallback``. The AST path's signature behavior is prepending
        the imports header to *every* chunk — the line fallback never does that, so
        asserting the imports land in a non-first chunk proves the AST path ran.
        """
        lines = ["import os", "import sys", ""]
        for i in range(20):
            lines += [f"def func_{i}():", f"    return {i}", ""]
        code = "\n".join(lines)

        chunks = chunk_code_with_treesitter(code, "python", 80)
        assert len(chunks) >= 2
        # AST signature: imports header prepended to a non-first chunk.
        assert "import os" in chunks[1]
        assert "import sys" in chunks[1]
        # And the result must not be identical to the line-based fallback.
        assert chunks != chunk_text_fallback(code, 80)

    def test_chunk_zero_does_not_duplicate_imports(self) -> None:
        """Regression (REVIEW.md §4.5 #5): the header nodes are re-emitted via the
        per-chunk header prefix, so the body loop must skip them — otherwise chunk 0
        carries the imports twice (once in the prefix, once as leading body nodes).
        """
        lines = ["import os", "import sys", ""]
        for i in range(20):
            lines += [f"def func_{i}():", f"    return {i}", ""]
        code = "\n".join(lines)

        chunks = chunk_code_with_treesitter(code, "python", 80)
        assert chunks[0].count("import os") == 1
        assert chunks[0].count("import sys") == 1
        # No code is lost by skipping the header nodes in the body.
        joined = "\n".join(chunks)
        assert all(f"def func_{i}()" in joined for i in range(20))
