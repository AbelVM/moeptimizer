"""Tests for cache-aware chunking."""

import pytest

from moeptimizer.cache_aware_chunker import (
    CacheAwareChunker,
    get_cache_aware_chunker,
)


class TestCacheAwareChunker:
    def test_empty_chunker(self) -> None:
        """Empty chunker has default block size."""
        chunker = CacheAwareChunker()
        assert chunker.get_block_size() == 128

    def test_chunk_context_small(self) -> None:
        """Small context is not chunked."""
        chunker = CacheAwareChunker()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = chunker.chunk_context(messages)
        assert len(result) == len(messages)

    def test_chunk_context_large(self) -> None:
        """Large context with code blocks is chunked."""
        chunker = CacheAwareChunker(block_size=100)
        # Create large context with multiple code blocks
        large_code = "```python\n" + "x" * 200 + "\n```\n```python\n" + "y" * 200 + "\n```"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": large_code},
        ]
        result = chunker.chunk_context(messages)
        # Should be chunked into multiple messages
        assert len(result) > len(messages)

    def test_preserve_ast_structure(self) -> None:
        """Preserve AST structure returns code unchanged."""
        chunker = CacheAwareChunker()
        code = "def foo():\n    pass\n"
        result = chunker.preserve_ast_structure(code)
        assert result == code

    def test_singleton(self) -> None:
        """Get cache aware chunker returns new instance each time."""
        chunker1 = get_cache_aware_chunker()
        chunker2 = get_cache_aware_chunker()
        # Function returns new instances
        assert isinstance(chunker1, CacheAwareChunker)
        assert isinstance(chunker2, CacheAwareChunker)