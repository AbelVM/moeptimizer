"""Tests for cache module."""

import pytest

from moeptimizer.cache import (
    align_to_block_boundary,
    canonicalize_code_for_cache,
    get_block_aligned_cache_key,
    get_block_size,
    set_block_size,
)


class TestCache:
    def test_get_block_size(self) -> None:
        """Get default block size."""
        size = get_block_size()
        assert size == 128

    def test_set_block_size(self) -> None:
        """Set block size changes the value."""
        original = get_block_size()
        set_block_size(256)
        assert get_block_size() == 256
        # Reset to original
        set_block_size(original)

    def test_align_to_block_boundary(self) -> None:
        """Align text to block boundary."""
        # Already aligned
        result = align_to_block_boundary("x" * 128)
        assert len(result) == 128

        # Needs padding
        result = align_to_block_boundary("x" * 100)
        assert len(result) == 128

    def test_canonicalize_code_for_cache(self) -> None:
        """Canonicalize code for cache key generation."""
        code = "def foo():\n    pass\n"
        result = canonicalize_code_for_cache(code)
        assert isinstance(result, str)

    def test_get_block_aligned_cache_key(self) -> None:
        """Get cache key for messages."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        key = get_block_aligned_cache_key(messages)
        assert isinstance(key, str)
        assert len(key) > 0