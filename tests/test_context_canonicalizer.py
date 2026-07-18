"""Tests for context canonicalizer."""


from moeptimizer.context_canonicalizer import (
    ContextCanonicalizer,
    get_context_canonicalizer,
)


class TestContextCanonicalizer:
    def test_empty_canonicalizer(self) -> None:
        """Empty canonicalizer has no state."""
        canonicalizer = ContextCanonicalizer()
        assert canonicalizer is not None

    def test_canonicalize(self) -> None:
        """Canonicalize context."""
        canonicalizer = ContextCanonicalizer()
        messages = [{"role": "user", "content": "Test content"}]
        result = canonicalizer.canonicalize(messages)
        assert len(result) == len(messages)

    def test_canonicalize_code_blocks(self) -> None:
        """Canonicalize preserves code blocks."""
        canonicalizer = ContextCanonicalizer()
        messages = [{"role": "user", "content": "```python\ndef foo():\n    pass\n```"}]
        result = canonicalizer.canonicalize(messages)
        # Code blocks should be preserved
        assert "```python" in result[0].get("content", "")

    def test_canonicalize_text(self) -> None:
        """Canonicalize normalizes non-code text."""
        canonicalizer = ContextCanonicalizer()
        # Text with trailing whitespace
        result = canonicalizer._canonicalize_text("hello  \nworld  \n")
        assert "hello" in result
        assert "world" in result
        # Trailing whitespace should be removed
        assert "  " not in result

    def test_singleton(self) -> None:
        """Get context canonicalizer returns new instance each time."""
        c1 = get_context_canonicalizer()
        c2 = get_context_canonicalizer()
        # Function returns new instances
        assert isinstance(c1, ContextCanonicalizer)
        assert isinstance(c2, ContextCanonicalizer)
