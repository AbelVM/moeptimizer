"""Prompt template versioning for cache partitioning.

Different task types benefit from different prompt structures.
This module provides versioned templates to improve cache hit rates.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Task type classification patterns
TASK_PATTERNS = {
    "debug": [
        "error",
        "bug",
        "fix",
        "crash",
        "exception",
        "traceback",
        "fail",
        "broken",
    ],
    "refactor": [
        "refactor",
        "clean",
        "simplify",
        "restructure",
        "optimize",
        "improve",
    ],
    "feature": [
        "add",
        "implement",
        "create",
        "build",
        "new",
        "feature",
    ],
    "test": [
        "test",
        "unit",
        "spec",
        "coverage",
        "assert",
    ],
    "doc": [
        "document",
        "docstring",
        "comment",
        "explain",
        "readme",
    ],
}

# Template versions for each task type
TEMPLATE_VERSIONS = {
    "debug": {
        "system": "You are a debugging assistant. Focus on error analysis and root cause identification.",
        "format": "error_first: Show the error, then analyze, then fix.",
    },
    "refactor": {
        "system": "You are a refactoring assistant. Focus on code structure and maintainability.",
        "format": "structure_first: Show current structure, then improvements, then code.",
    },
    "feature": {
        "system": "You are a feature implementation assistant. Focus on clean, working code.",
        "format": "plan_first: Show plan, then implementation, then verification.",
    },
    "test": {
        "system": "You are a testing assistant. Focus on test coverage and edge cases.",
        "format": "test_first: Show test cases, then implementation, then results.",
    },
    "doc": {
        "system": "You are a documentation assistant. Focus on clarity and completeness.",
        "format": "doc_first: Show structure, then documentation, then examples.",
    },
    "default": {
        "system": "You are a helpful coding assistant.",
        "format": "standard: Show context, then analysis, then code.",
    },
}


class PromptTemplateManager:
    """
    Manages prompt templates for different task types.

    Uses task classification to select appropriate template,
    improving cache hit rates through template partitioning.
    """

    def __init__(self) -> None:
        self._template_cache: dict[str, str] = {}
        self._version_cache: dict[str, int] = {}

    def classify_task(self, prompt: str) -> str:
        """Classify task type from prompt content."""
        prompt_lower = prompt.lower()
        scores: dict[str, int] = {}

        for task_type, patterns in TASK_PATTERNS.items():
            score = sum(1 for p in patterns if p in prompt_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "default"

        return max(scores, key=scores.get)

    def get_template(
        self,
        task_type: str,
        static_content: str,
    ) -> str:
        """Get versioned template for task type."""
        version = self._get_template_version(task_type)
        cache_key = f"{task_type}:{version}"

        if cache_key not in self._template_cache:
            template = TEMPLATE_VERSIONS.get(task_type, TEMPLATE_VERSIONS["default"])
            self._template_cache[cache_key] = template["system"]

        return self._template_cache[cache_key]

    def _get_template_version(self, task_type: str) -> int:
        """Get current template version for task type."""
        if task_type not in self._version_cache:
            # Hash-based versioning for stability
            version = int(
                hashlib.md5(task_type.encode()).hexdigest()[:8], 16
            ) % 1000
            self._version_cache[task_type] = version
        return self._version_cache[task_type]

    def apply_template(
        self,
        messages: list[dict[str, Any]],
        task_type: str,
    ) -> list[dict[str, Any]]:
        """Apply template to message list.

        Returns new message list with template-appropriate structure.
        Does NOT modify system message content to preserve cache integrity.
        """
        # For cache partitioning, we track task type but don't modify content
        # The task type is used for cache key generation instead
        return [dict(msg) for msg in messages]

    def get_cache_partition_key(
        self,
        task_type: str,
        static_content: str,
    ) -> str:
        """Generate cache partition key for task type."""
        version = self._get_template_version(task_type)
        return f"{task_type}:{version}:{hashlib.md5(static_content.encode()).hexdigest()[:8]}"


# Global template manager instance
_template_manager: PromptTemplateManager | None = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def classify_and_template(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Classify task and apply template.

    Returns (templated_messages, task_type).
    """
    manager = get_template_manager()

    # Find user prompt for classification
    user_prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    task_type = manager.classify_task(user_prompt)
    templated = manager.apply_template(messages, task_type)

    return templated, task_type