"""AgentStateStore — KV graph: Goal -> Subtask -> Tool -> Outcome.

Indexes context by structural relationships rather than flat embeddings,
replacing semantic RAG with graph traversal for agentic workflows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

from moeptimizer.config import get_config
from moeptimizer.models import AgentStep, GoalNode


class AgentStateStore:
    """
    Key-Value graph state store for agentic workflows.

    Indexes context by Goal -> Subtask -> Tool Used -> Outcome.
    Replaces flat embedding RAG with structural graph traversal.
    """

    def __init__(self) -> None:
        self.steps: list[AgentStep] = []
        self.goals: dict[str, GoalNode] = {}
        self.subtask_index: dict[str, list[str]] = defaultdict(list)
        self.tool_index: dict[str, list[str]] = defaultdict(list)
        self._step_index: dict[str, int] = {}
        self._step_hashes: dict[int, str] = {}
        self._step_hash_set: set[str] = set()
        self._goal_id: str | None = None
        self._config = get_config().agentic
        self._max_steps_override: int | None = None

    def set_max_steps(self, max_steps: int) -> None:
        """Override the configured ``max_state_steps`` cap (e.g. with a value
        derived from the live backend window). ``None`` restores the config floor.
        """
        self._max_steps_override = max_steps if max_steps and max_steps > 0 else None

    def add_step(self, step: AgentStep) -> str:
        """Register a step and index it by role, tool, and subtask."""
        idx = len(self.steps)
        step.step_index = idx
        self.steps.append(step)
        self._step_index[step.step_id] = idx
        self._step_hashes[idx] = self.step_fingerprint(step)
        self._step_hash_set.add(self._step_hashes[idx])

        if step.tool_name:
            self.tool_index[step.tool_name].append(step.step_id)

        subtask = step.metadata.get("subtask") or self._infer_subtask(step)
        if subtask:
            self.subtask_index[subtask].append(step.step_id)

        self._prune_if_needed()
        return step.step_id

    def _prune_if_needed(self) -> None:
        """Bound memory by dropping the oldest archived steps beyond ``max_state_steps``.

        The store appends a step per ingested message and never otherwise prunes,
        so a long agentic session would grow without bound (review §10). When the
        step count exceeds the configured cap we drop the oldest steps from the
        front (never the recent/protected tail) and rebuild the derived indices.
        """
        max_steps = self._max_steps_override if self._max_steps_override is not None else self._config.max_state_steps
        if max_steps <= 0 or len(self.steps) <= max_steps:
            return
        excess = len(self.steps) - max_steps
        del self.steps[:excess]
        self._rebuild_indices()

    def prune_by_relevance(
        self,
        threshold: float,
        goal: GoalNode | None = None,
        keep_recent: int = 3,
    ) -> int:
        """Evict low-relevance steps from the evictable body.

        Operates on the *archived* (oldest) portion of the step list so the
        recent/protected tail and the frozen prefix are never mutated. Returns
        the number of steps removed.

        Args:
            threshold: Minimum relevance score; steps below this are dropped.
            goal: Current goal node for scoring. Falls back to the stored goal.
            keep_recent: Minimum number of recent steps to preserve regardless
                of score (matches ``keep_full_steps`` semantics).
        """
        from moeptimizer.goal_relevance_scorer import GoalRelevanceScorer

        if threshold <= 0 or len(self.steps) <= keep_recent:
            return 0

        scorer = GoalRelevanceScorer(self._config)
        active_goal = goal or self.get_goal()
        if active_goal is not None:
            scorer.set_goal(active_goal)

        # Only score the evictable body (everything except the recent tail).
        evictable = self.steps[:-keep_recent] if len(self.steps) > keep_recent else []
        protected_tail = self.steps[-keep_recent:] if len(self.steps) > keep_recent else []

        if not evictable:
            return 0

        scored = scorer.score_steps(evictable, active_goal)
        kept = [step for step, score in scored if score >= threshold]
        removed = len(evictable) - len(kept)

        if removed > 0:
            self.steps = kept + protected_tail
            self._rebuild_indices()

        return removed

    def _rebuild_indices(self) -> None:
        """Recompute all derived indices after steps are pruned/shifted."""
        self._step_index.clear()
        self._step_hashes.clear()
        self._step_hash_set.clear()
        self.tool_index.clear()
        self.subtask_index.clear()
        for idx, step in enumerate(self.steps):
            step.step_index = idx
            self._step_index[step.step_id] = idx
            fp = self.step_fingerprint(step)
            self._step_hashes[idx] = fp
            self._step_hash_set.add(fp)
            if step.tool_name:
                self.tool_index[step.tool_name].append(step.step_id)
            subtask = step.metadata.get("subtask") or self._infer_subtask(step)
            if subtask:
                self.subtask_index[subtask].append(step.step_id)

    @staticmethod
    def step_fingerprint(step: AgentStep) -> str:
        """Return a stable fingerprint for a message-position pair."""
        tool_call_id = step.tool_call_id or ""
        metadata = json.dumps(step.metadata, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(
            f"{step.role}\0{step.step_index}\0{step.content}\0{tool_call_id}\0{metadata}".encode()
        ).hexdigest()

    def has_step_fingerprint(self, fingerprint: str) -> bool:
        """Return True if this exact message-position fingerprint was ingested."""
        return fingerprint in self._step_hash_set

    def set_goal(self, original_prompt: str) -> str:
        """Set the root goal for this agent session."""
        goal = GoalNode(original_prompt=original_prompt)
        self.goals[goal.goal_id] = goal
        self._goal_id = goal.goal_id
        return goal.goal_id

    def get_goal(self) -> GoalNode | None:
        """Get the root goal node."""
        if self._goal_id:
            return self.goals.get(self._goal_id)
        return None

    def get_recent_steps(self, n: int | None = None) -> list[AgentStep]:
        """Get the last N steps in full detail."""
        n = n or self._config.keep_full_steps
        return self.steps[-n:] if len(self.steps) >= n else list(self.steps)

    def get_archived_steps(self) -> list[AgentStep]:
        """Get steps older than the keep threshold."""
        threshold = len(self.steps) - self._config.archive_threshold
        return self.steps[:threshold] if threshold > 0 else []

    def get_related_context(self, current_step: AgentStep) -> list[AgentStep]:
        """
        Retrieve structurally related steps based on:
        1. Same subtask
        2. Same tool used
        3. Same role type (for pattern matching)
        """
        related: set[str] = set()

        subtask = current_step.metadata.get("subtask", "")
        if subtask and subtask in self.subtask_index:
            related.update(self.subtask_index[subtask])

        if current_step.tool_name and current_step.tool_name in self.tool_index:
            related.update(self.tool_index[current_step.tool_name])

        result = [
            s for s in self.steps
            if s.step_id in related and s.step_id != current_step.step_id
        ]
        return result[: self._config.keep_full_steps * 2]

    def get_compacted_history(self) -> list[dict[str, Any]]:
        """
        Return the full history in compacted form:
        - Recent steps: full content
        - Archived steps: summary only
        """
        recent = self.get_recent_steps()
        archived = self.get_archived_steps()

        result: list[dict[str, Any]] = []

        for step in archived:
            summary = step.outcome_summary or self._generate_summary(step)
            result.append({
                "role": step.role,
                "summary": summary,
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "step_index": step.step_index,
                "archived": True,
            })

        for step in recent:
            result.append({
                "role": step.role,
                "content": step.content,
                "tool_name": step.tool_name,
                "step_id": step.step_id,
                "step_index": step.step_index,
                "archived": False,
            })

        return result

    def _generate_summary(self, step: AgentStep) -> str:
        """Generate a single-sentence summary for an archived step."""
        role = step.role
        tool = step.tool_name or ""

        if role == "tool":
            content = step.content
            lines = content.split("\n")
            if len(lines) > 3:
                return f"Tool '{tool}': returned {len(lines)} lines of output (truncated)"
            return f"Tool '{tool}': {content[:200]}"

        elif role == "assistant":
            if tool:
                return f"Assistant: called tool '{tool}'"
            return f"Assistant: {step.content[:200]}"

        elif role == "thinking":
            return f"Thinking: {step.content[:150]}..."

        elif role == "user":
            return f"User: {step.content[:150]}"

        return f"{role}: {step.content[:150]}"

    def _infer_subtask(self, step: AgentStep) -> str | None:
        """Infer subtask from step content when not explicitly provided."""
        match = re.search(r"# Subtask: (.+)", step.content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        if step.tool_name:
            return step.tool_name

        return None

    def serialize(self) -> str:
        """Serialize state store to JSON for persistence."""
        return json.dumps({
            "steps": [s.to_dict() for s in self.steps],
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "subtask_index": dict(self.subtask_index),
            "tool_index": dict(self.tool_index),
        })

    @classmethod
    def deserialize(cls, data: str) -> AgentStateStore:
        """Deserialize state store from JSON."""
        store = cls()
        parsed = json.loads(data)
        for s in parsed.get("steps", []):
            step = AgentStep.from_dict(s)
            idx = len(store.steps)
            step.step_index = idx
            store.steps.append(step)
            store._step_index[step.step_id] = idx
            fingerprint = cls.step_fingerprint(step)
            store._step_hashes[idx] = fingerprint
            store._step_hash_set.add(fingerprint)
            if step.tool_name:
                store.tool_index[step.tool_name].append(step.step_id)
            subtask = step.metadata.get("subtask") or store._infer_subtask(step)
            if subtask:
                store.subtask_index[subtask].append(step.step_id)
        for k, v in parsed.get("goals", {}).items():
            store.goals[k] = GoalNode(**v)
            store._goal_id = k  # Restore goal reference
        for k, v in parsed.get("subtask_index", {}).items():
            store.subtask_index[k] = v
        for k, v in parsed.get("tool_index", {}).items():
            store.tool_index[k] = v
        return store
