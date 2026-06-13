"""Tests for prompt template versioning."""

import pytest

from moeptimizer.prompt_templates import (
    PromptTemplateManager,
    classify_and_template,
    get_template_manager,
)


class TestPromptTemplateManager:
    def test_classify_debug_task(self) -> None:
        """Classify debug task from prompt."""
        manager = PromptTemplateManager()
        task_type = manager.classify_task("Fix this error in the code")
        assert task_type == "debug"

    def test_classify_refactor_task(self) -> None:
        """Classify refactor task from prompt."""
        manager = PromptTemplateManager()
        task_type = manager.classify_task("Refactor this function to be cleaner")
        assert task_type == "refactor"

    def test_classify_feature_task(self) -> None:
        """Classify feature task from prompt."""
        manager = PromptTemplateManager()
        task_type = manager.classify_task("Add a new feature to the API")
        assert task_type == "feature"

    def test_classify_test_task(self) -> None:
        """Classify test task from prompt."""
        manager = PromptTemplateManager()
        task_type = manager.classify_task("Write unit tests for this function")
        assert task_type == "test"

    def test_classify_default_task(self) -> None:
        """Classify default task for generic prompt."""
        manager = PromptTemplateManager()
        task_type = manager.classify_task("Hello, how are you?")
        assert task_type == "default"

    def test_apply_template_preserves_messages(self) -> None:
        """Apply template preserves message structure."""
        manager = PromptTemplateManager()
        messages = [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "Task"},
        ]
        result = manager.apply_template(messages, "debug")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_get_cache_partition_key(self) -> None:
        """Get cache partition key for task type."""
        manager = PromptTemplateManager()
        key = manager.get_cache_partition_key("debug", "static content")
        assert "debug:" in key

    def test_get_template_manager_singleton(self) -> None:
        """Get template manager returns instance."""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        assert manager1 is manager2

    def test_classify_and_template(self) -> None:
        """Test the classify_and_template function."""
        messages = [
            {"role": "user", "content": "Fix this bug"},
        ]
        result, task_type = classify_and_template(messages)
        assert task_type == "debug"
        assert len(result) == 1