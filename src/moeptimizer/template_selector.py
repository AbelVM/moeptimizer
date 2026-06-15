"""Template Selector for prompt template optimization.

Chooses the most suitable prompt template based on recent quality metrics
(semantic similarity, token savings) to improve cache hit rates.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class TemplateSelector:
    """
    Selects the best prompt template based on recent quality metrics.

    Tracks performance of each template variant and selects the one
    with the best recent quality metrics (semantic similarity, token savings).
    This improves cache hit rates by consistently using the best-performing
    template structure.
    """

    def __init__(
        self,
        max_history: int = 100,
        exploration_rate: float = 0.1,
    ) -> None:
        self._max_history = max_history
        self._exploration_rate = exploration_rate
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._template_scores: dict[str, dict[str, float]] = {}
        self._current_template: str = "default"
        self._stats: dict[str, int] = {
            "selections": 0,
            "explorations": 0,
            "exploits": 0,
        }

        # Initialize scores for all templates
        from moeptimizer.prompt_templates import TEMPLATE_VERSIONS

        for template_name in TEMPLATE_VERSIONS:
            self._template_scores[template_name] = {
                "semantic_similarity": 0.0,
                "token_savings": 0.0,
                "combined_score": 0.5,
                "sample_count": 0,
            }

    def select_template(
        self,
        messages: list[dict[str, Any]],
        force_explore: bool = False,
    ) -> str:
        """Select the best template based on recent quality metrics.

        Uses epsilon-greedy: explore new templates with probability
        exploration_rate, otherwise exploit the best-known template.

        Args:
            messages: The message list for context
            force_explore: Force exploration regardless of rate

        Returns:
            Selected template name
        """
        self._stats["selections"] += 1

        # Epsilon-greedy selection
        if force_explore or (
            self._should_explore() and len(self._template_scores) > 1
        ):
            return self._explore_template()

        return self._exploit_best_template()

    def _should_explore(self) -> bool:
        """Determine if we should explore a new template."""
        import random

        return random.random() < self._exploration_rate

    def _explore_template(self) -> str:
        """Select a template for exploration."""
        import random

        templates = list(self._template_scores.keys())
        # Prefer templates with fewer samples for exploration
        weights = [
            1.0 / max(s["sample_count"], 1) for s in self._template_scores.values()
        ]
        template = random.choices(templates, weights=weights, k=1)[0]
        self._stats["explorations"] += 1
        return template

    def _exploit_best_template(self) -> str:
        """Select the best-performing template."""
        best_template = "default"
        best_score = -1.0

        for template_name, scores in self._template_scores.items():
            # Untested templates get a neutral-low score; tested templates use their real score
            score = (
                scores["combined_score"]
                if scores["sample_count"] > 0
                else 0.0  # Untested templates start at 0.0
            )

            if score > best_score:
                best_score = score
                best_template = template_name

        self._stats["exploits"] += 1
        self._current_template = best_template
        return best_template

    def record_quality(
        self,
        template_name: str,
        semantic_similarity: float,
        token_savings: float,
    ) -> None:
        """Record quality metrics for a template.

        Args:
            template_name: The template that was used
            semantic_similarity: Semantic similarity score (0.0-1.0)
            token_savings: Token savings ratio (0.0-1.0)
        """
        combined = (semantic_similarity * 0.6 + token_savings * 0.4)

        if template_name not in self._template_scores:
            self._template_scores[template_name] = {
                "semantic_similarity": 0.0,
                "token_savings": 0.0,
                "combined_score": 0.5,
                "sample_count": 0,
            }

        scores = self._template_scores[template_name]
        n = scores["sample_count"]

        # Exponential moving average
        alpha = 0.3
        scores["semantic_similarity"] = (
            scores["semantic_similarity"] * (1 - alpha) + semantic_similarity * alpha
        )
        scores["token_savings"] = (
            scores["token_savings"] * (1 - alpha) + token_savings * alpha
        )
        scores["combined_score"] = (
            scores["semantic_similarity"] * 0.6 + scores["token_savings"] * 0.4
        )
        scores["sample_count"] = n + 1

        self._history.append({
            "template": template_name,
            "semantic_similarity": semantic_similarity,
            "token_savings": token_savings,
            "combined_score": combined,
            "timestamp": time.time(),
        })

    def get_best_template(self) -> str:
        """Get the current best-performing template."""
        return self._exploit_best_template()

    def get_template_scores(self) -> dict[str, dict[str, float]]:
        """Get scores for all templates."""
        return {
            name: dict(scores)
            for name, scores in self._template_scores.items()
        }

    def get_stats(self) -> dict[str, int]:
        """Get selector statistics."""
        return dict(self._stats)

    def reset(self) -> None:
        """Reset all scores and history."""
        self._history.clear()
        self._template_scores.clear()
        from moeptimizer.prompt_templates import TEMPLATE_VERSIONS

        for template_name in TEMPLATE_VERSIONS:
            self._template_scores[template_name] = {
                "semantic_similarity": 0.0,
                "token_savings": 0.0,
                "combined_score": 0.5,
                "sample_count": 0,
            }
        self._current_template = "default"
        self._stats = {
            "selections": 0,
            "explorations": 0,
            "exploits": 0,
        }


# Global instance
_template_selector: TemplateSelector | None = None


def get_template_selector() -> TemplateSelector:
    """Get or create the global template selector."""
    global _template_selector
    if _template_selector is None:
        _template_selector = TemplateSelector()
    return _template_selector
