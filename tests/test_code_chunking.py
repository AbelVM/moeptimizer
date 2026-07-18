"""Tests for code chunking."""


from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
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
