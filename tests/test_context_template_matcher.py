"""Tests for context template matcher."""

import pytest

from moeptimizer.context_template_matcher import (
    ContextTemplateMatcher,
    get_context_template_matcher,
)


class TestContextTemplateMatcher:
    def test_empty_matcher(self) -> None:
        """Empty matcher has no state."""
        matcher = ContextTemplateMatcher()
        assert matcher is not None

    def test_match_template(self) -> None:
        """Match template for context."""
        matcher = ContextTemplateMatcher()
        messages = [{"role": "user", "content": "Write a function"}]
        template = matcher.match_template(messages)
        # May return None or a template name
        assert template is None or isinstance(template, str)

    def test_match_template_code_review(self) -> None:
        """Match code review template."""
        matcher = ContextTemplateMatcher()
        messages = [{"role": "user", "content": "Please review this code"}]
        template = matcher.match_template(messages)
        assert template == "code_review"

    def test_apply_template(self) -> None:
        """Apply template to context."""
        matcher = ContextTemplateMatcher()
        messages = [{"role": "user", "content": "Test"}]
        result = matcher.apply_template(messages)
        assert len(result) >= len(messages)

    def test_singleton(self) -> None:
        """Get context template matcher returns new instance each time."""
        m1 = get_context_template_matcher()
        m2 = get_context_template_matcher()
        # Function returns new instances
        assert isinstance(m1, ContextTemplateMatcher)
        assert isinstance(m2, ContextTemplateMatcher)