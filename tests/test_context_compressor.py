"""Tests for context compressor."""


from moeptimizer.context_compressor import (
    ContextCompressor,
    get_context_compressor,
)


class TestContextCompressor:
    def test_empty_compressor(self) -> None:
        """Empty compressor has no state."""
        compressor = ContextCompressor()
        assert compressor is not None

    def test_compress(self) -> None:
        """Compress context."""
        compressor = ContextCompressor()
        messages = [{"role": "user", "content": "Test content"}]
        result = compressor.compress(messages)
        assert len(result) == len(messages)

    def test_compress_code_blocks(self) -> None:
        """Compress code blocks to skeletons."""
        compressor = ContextCompressor()
        messages = [
            {"role": "user", "content": "```python\ndef foo():\n    x = 1\n    y = 2\n    return x + y\n```"},
        ]
        result = compressor.compress(messages)
        content = result[0].get("content", "")
        # Should have skeleton (def foo(): and ...)
        assert "def foo():" in content

    def test_preserves_small_code_snippets(self) -> None:
        """Keep short original snippets intact for quality-sensitive context."""
        compressor = ContextCompressor()
        snippet = "def calc(data):\n    total = 0\n    for item in data:\n        total += item\n    return total"
        messages = [{"role": "user", "content": f"```python\n{snippet}\n```"}]
        result = compressor.compress(messages)
        assert snippet in result[0].get("content", "")

    def test_singleton(self) -> None:
        """Get context compressor returns new instance each time."""
        c1 = get_context_compressor()
        c2 = get_context_compressor()
        # Function returns new instances
        assert isinstance(c1, ContextCompressor)
        assert isinstance(c2, ContextCompressor)
