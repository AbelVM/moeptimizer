"""Context template matcher for cache optimization.

Matches context to known cached templates.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, ClassVar


class ContextTemplateMatcher:
    """
    Matches context to known cached templates.

    - Match context to known cached templates
    - Use task-specific templates
    - Minimize deviation from cached patterns
    """

    TEMPLATES: ClassVar[dict[str, Any]] = {
        "code_review": {
            "pattern": r"(review|analyze|check).*code",
            "static": "You are a code reviewer. Analyze the following code:",
        },
        "bug_fix": {
            "pattern": r"(fix|bug|error|issue)",
            "static": "You are a debugging assistant. Fix the following issue:",
        },
        "refactor": {
            "pattern": r"(refactor|improve|optimize)",
            "static": "You are a refactoring assistant. Improve the following code:",
        },
        "test": {
            "pattern": r"(test|spec|unit)",
            "static": "You are a testing assistant. Write tests for the following code:",
        },
    }

    def __init__(self) -> None:
        self._template_cache: dict[str, str] = {}

    def match_template(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Match context to a known template."""
        if not messages:
            return None

        # Get user query
        user_query = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            return None

        # Match against templates
        for name, template in self.TEMPLATES.items():
            if re.search(template["pattern"], user_query, re.IGNORECASE):
                return name

        return None

    def get_template_static(
        self,
        template_name: str,
    ) -> str | None:
        """Get static layer for a template."""
        template = self.TEMPLATES.get(template_name)
        return template["static"] if template else None

    def apply_template(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return messages unchanged.

        Inserting or changing a system prompt mid-session changes the first
        tokens sent to llama.cpp and forces a full static-prefix re-prefill.
        """
        return [dict(m) for m in messages]

    def get_template_key(
        self,
        template_name: str,
    ) -> str:
        """Get cache key for a template."""
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        static = self.get_template_static(template_name)
        if static:
            key = hashlib.md5(static.encode()).hexdigest()[:16]
            self._template_cache[template_name] = key
            return key

        return ""


def get_context_template_matcher() -> ContextTemplateMatcher:
    """Get a context template matcher instance."""
    return ContextTemplateMatcher()
