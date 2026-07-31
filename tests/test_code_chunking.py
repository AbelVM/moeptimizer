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
