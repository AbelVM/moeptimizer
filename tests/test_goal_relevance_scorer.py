"""Tests for task-aware goal-relevance pruning (P3)."""

from __future__ import annotations

from moeptimizer.config import AppConfig
from moeptimizer.goal_relevance_scorer import GoalRelevanceScorer
from moeptimizer.models import AgentStep, GoalNode
from moeptimizer.optimizer import AgentContextOptimizer
from moeptimizer.state_store import AgentStateStore


def _make_step(role: str, content: str, step_index: int, **kwargs: object) -> AgentStep:
    return AgentStep(role=role, content=content, step_index=step_index, **kwargs)


class TestGoalRelevanceScorer:
    def setup_method(self) -> None:
        self.config = AppConfig()
        self.scorer = GoalRelevanceScorer(self.config.agentic)

    def test_set_goal_builds_keywords(self) -> None:
        goal = GoalNode(original_prompt="Refactor the authentication module to use JWT tokens")
        self.scorer.set_goal(goal)
        assert "refactor" in self.scorer._goal_keywords
        assert "authentication" in self.scorer._goal_keywords
        assert "jwt" in self.scorer._goal_keywords

    def test_subtask_match_scores_high(self) -> None:
        goal = GoalNode(original_prompt="Build a REST API for user management",
                        subtasks=["user management"])
        self.scorer.set_goal(goal)
        step = _make_step("assistant", "Implemented the user creation endpoint", 0,
                          metadata={"subtask": "user management"})
        score = self.scorer.score_step(step, newest_index=0)
        assert score >= 10.0

    def test_irrelevant_step_scores_low(self) -> None:
        goal = GoalNode(original_prompt="Refactor the payment processing pipeline")
        self.scorer.set_goal(goal)
        step = _make_step("user", "What is the weather in Paris today?", 0)
        score = self.scorer.score_step(step, newest_index=0)
        # Only recency decay contributes (1.0); no goal overlap.
        assert score < 2.0

    def test_recency_decay_rewards_newer_steps(self) -> None:
        goal = GoalNode(original_prompt="generic task")
        self.scorer.set_goal(goal)
        old = _make_step("assistant", "old step", 0)
        new = _make_step("assistant", "new step", 10)
        assert self.scorer.score_step(new, 10) > self.scorer.score_step(old, 10)

    def test_score_steps_returns_pairs(self) -> None:
        goal = GoalNode(original_prompt="Fix the login bug")
        steps = [
            _make_step("user", "Fix the login bug", 0),
            _make_step("assistant", "Investigating the login flow", 1),
            _make_step("user", "The weather is nice today", 2),
        ]
        scored = self.scorer.score_steps(steps, goal)
        assert len(scored) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in scored)


class TestGoalRelevancePruning:
    def _build_store(self, goal_text: str, steps: list[AgentStep]) -> AgentStateStore:
        store = AgentStateStore()
        store.set_goal(goal_text)
        for step in steps:
            store.add_step(step)
        return store

    def test_prune_removes_irrelevant_old_steps(self) -> None:
        config = AppConfig()
        config.agentic.goal_relevance_threshold = 2.0
        config.agentic.keep_full_steps = 2

        store = self._build_store(
            "Refactor the authentication module",
            [
                _make_step("user", "Refactor the authentication module", 0),
                _make_step("assistant", "Started refactoring auth", 1),
                _make_step("user", "What is the capital of France?", 2),
                _make_step("assistant", "Paris is the capital of France", 3),
                _make_step("user", "Continue the auth refactor please", 4),
                _make_step("assistant", "Finished the auth refactor", 5),
            ],
        )
        # 6 steps, keep_recent=2 -> evictable body is steps 0..3.
        removed = store.prune_by_relevance(
            threshold=config.agentic.goal_relevance_threshold,
            goal=store.get_goal(),
            keep_recent=config.agentic.keep_full_steps,
        )
        # The two off-topic steps (2,3) should be evicted.
        assert removed == 2
        # Recent tail (4,5) preserved (content-based; step_index is renumbered
        # by _rebuild_indices after pruning).
        assert len(store.steps) == 4
        assert store.steps[-1].content == "Finished the auth refactor"
        assert store.steps[-2].content == "Continue the auth refactor please"

    def test_prune_disabled_when_threshold_zero(self) -> None:
        config = AppConfig()
        config.agentic.goal_relevance_threshold = 0.0
        store = self._build_store(
            "Task",
            [_make_step("user", f"message {i}", i) for i in range(6)],
        )
        removed = store.prune_by_relevance(0.0, store.get_goal(), keep_recent=2)
        assert removed == 0
        assert len(store.steps) == 6

    def test_prune_preserves_recent_tail(self) -> None:
        config = AppConfig()
        config.agentic.goal_relevance_threshold = 100.0  # everything is "irrelevant"
        config.agentic.keep_full_steps = 3
        store = self._build_store(
            "Task",
            [_make_step("user", f"message {i}", i) for i in range(8)],
        )
        removed = store.prune_by_relevance(100.0, store.get_goal(), keep_recent=3)
        # Even with a huge threshold, the 3 most recent steps survive.
        assert removed == 5
        assert len(store.steps) == 3
        # Protected tail is the 3 most recent original steps (content-based check,
        # since step_index is renumbered by _rebuild_indices after pruning).
        assert [s.content for s in store.steps] == [f"message {i}" for i in (5, 6, 7)]


class TestGoalRelevanceIntegration:
    def test_optimizer_runs_with_pruning_enabled(self) -> None:
        config = AppConfig()
        config.agentic.goal_relevance_threshold = 2.0
        config.agentic.fast_path_enabled = False
        config.v050.hit_prediction_enabled = False
        config.v050.cache_stable_mode = True
        config.v050.frozen_prefix_turns = 1
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Refactor the authentication module to use JWT"},
            {"role": "assistant", "content": "I'll start by examining the auth code."},
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "The weather in Paris is sunny."},
            {"role": "user", "content": "Continue the JWT refactor"},
            {"role": "assistant", "content": "Done with the refactor."},
        ]
        result = optimizer.optimize_messages(messages)
        # Pipeline completes without error and returns a valid message list.
        assert len(result) >= 2
        assert result[0]["role"] == "system"
