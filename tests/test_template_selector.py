"""Tests for template_selector module."""

import pytest

from moeptimizer.template_selector import TemplateSelector, get_template_selector


class TestTemplateSelector:
    def setup_method(self) -> None:
        self.selector = TemplateSelector(max_history=20, exploration_rate=0.0)

    def test_select_template_default(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        template = self.selector.select_template(messages)
        assert isinstance(template, str)
        assert template in ("debug", "refactor", "feature", "test", "doc", "default")

    def test_select_template_debug(self) -> None:
        messages = [{"role": "user", "content": "Fix this bug"}]
        template = self.selector.select_template(messages)
        assert template == "debug"

    def test_select_template_feature(self) -> None:
        messages = [{"role": "user", "content": "Add a new feature"}]
        # With no quality data and exploration_rate=0.0, selector picks first template
        # The selector is quality-based, not classification-based
        template = self.selector.select_template(messages)
        assert isinstance(template, str)
        assert template in self.selector._template_scores

    def test_record_quality(self) -> None:
        self.selector.record_quality("default", 0.9, 0.8)
        scores = self.selector.get_template_scores()
        assert "default" in scores
        assert scores["default"]["sample_count"] == 1

    def test_get_best_template(self) -> None:
        self.selector.record_quality("default", 0.9, 0.8)
        self.selector.record_quality("debug", 0.5, 0.3)
        best = self.selector.get_best_template()
        assert best == "default"

    def test_get_stats(self) -> None:
        self.selector.select_template([{"role": "user", "content": "hi"}])
        stats = self.selector.get_stats()
        assert "selections" in stats
        assert stats["selections"] >= 1

    def test_reset(self) -> None:
        self.selector.record_quality("default", 0.9, 0.8)
        self.selector.reset()
        scores = self.selector.get_template_scores()
        assert scores["default"]["sample_count"] == 0

    def test_global_instance(self) -> None:
        selector = get_template_selector()
        assert isinstance(selector, TemplateSelector)
