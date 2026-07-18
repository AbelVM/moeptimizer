"""Goal-relevance scoring for task-aware context pruning.

Ranks agent steps by their relevance to the current session goal so that
low-relevance history can be evicted before token budgeting. Uses cheap
lexical overlap + subtask matching + recency decay, matching the project's
preference for structural heuristics over heavy semantic RAG.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from moeptimizer.models import AgentStep, GoalNode

if TYPE_CHECKING:
    from moeptimizer.config import AgenticConfig


class GoalRelevanceScorer:
    """Score AgentStep objects for relevance to the current goal.

    Weights (tuned for agentic loops):
    - Subtask exact match: +10.0
    - Tool name match: +5.0
    - Role match with current step: +1.0
    - Keyword overlap with goal/subtasks: scaled by Jaccard index * 5.0
    - Recency decay: 1.0 / (1.0 + distance from newest step)
    """

    def __init__(self, config: AgenticConfig) -> None:
        self._config = config
        self._goal_keywords: list[str] = []
        self._subtask_keywords: list[str] = []

    def set_goal(self, goal: GoalNode | None) -> None:
        """Refresh keyword sets from the current goal and its subtasks."""
        self._goal_keywords = self._tokenize(goal.original_prompt) if goal else []
        self._subtask_keywords = []
        if goal and goal.subtasks:
            for subtask in goal.subtasks:
                self._subtask_keywords.extend(self._tokenize(subtask))

    def score_step(self, step: AgentStep, newest_index: int) -> float:
        """Return a relevance score for *step*.

        *newest_index* is the step_index of the most recent step in the
        session, used for recency decay.
        """
        score = 0.0

        # Subtask match (strongest signal).
        step_subtask = step.metadata.get("subtask")
        if step_subtask and any(
            self._fuzzy_match(step_subtask, kw) for kw in self._subtask_keywords
        ):
            score += 10.0

        # Tool name match.
        if step.tool_name and any(
            self._fuzzy_match(step.tool_name, kw) for kw in self._goal_keywords
        ):
            score += 5.0

        # Keyword overlap with goal + subtasks (fuzzy, morphology-tolerant).
        step_tokens = self._tokenize(step.content)
        goal_tokens = self._goal_keywords + self._subtask_keywords
        if step_tokens and goal_tokens:
            matched = sum(
                1 for st in step_tokens
                if any(self._fuzzy_match(st, gt) for gt in goal_tokens)
            )
            score += (matched / len(step_tokens)) * 5.0

        # Recency decay — newer steps are more likely to be relevant.
        distance = max(0, newest_index - step.step_index)
        score += 1.0 / (1.0 + distance)

        return score

    def score_steps(
        self, steps: list[AgentStep], goal: GoalNode | None = None
    ) -> list[tuple[AgentStep, float]]:
        """Score every step against *goal* and return (step, score) pairs."""
        if goal is not None:
            self.set_goal(goal)
        newest_index = steps[-1].step_index if steps else 0
        return [(step, self.score_step(step, newest_index)) for step in steps]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokens, stop-word filtered."""
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "up", "it",
            "its", "this", "that", "these", "those", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she",
            "her", "they", "them", "their", "what", "which", "who",
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    @staticmethod
    def _fuzzy_match(a: str, b: str) -> bool:
        """Return True when *a* and *b* share a significant substring."""
        a_lower = a.lower()
        b_lower = b.lower()
        if a_lower in b_lower or b_lower in a_lower:
            return True
        # Shared 4-char substring as a cheap proxy for semantic overlap.
        return any(a_lower[i : i + 4] in b_lower for i in range(len(a_lower) - 3))
