"""StateBasedRAG — Graph-indexed retrieval (not flat embeddings).

Uses the AgentStateStore graph to retrieve context by:
  1. Goal proximity — steps related to the current goal
  2. Subtask affinity — steps from the same subtask
  3. Tool lineage — steps that used the same tools
  4. Temporal decay — older related steps get lower priority

MOE context integrity:
  - RAG context is injected as a SEPARATE user message (never into assistant)
  - Format uses model-friendly structure: "step N: {role} - {summary}"
  - Avoids arbitrary markers like "role#index:content" that the model
    was not trained to recognize
"""

from __future__ import annotations

import re

from moeptimizer.models import AgentStep
from moeptimizer.state_store import AgentStateStore


class StateBasedRAG:
    """
    State-Based RAG for agentic workflows.

    Instead of semantic similarity (which fails across structurally different
    steps), this uses the AgentStateStore graph to retrieve context by
    structural relationships.

    Context is injected as a separate user message to preserve the model's
    expected chat template (ăssistant\n reasoning\n response).
    """

    def __init__(self, store: AgentStateStore) -> None:
        self.store = store

    def get_context_for_step(self, current_step: AgentStep) -> str:
        """
        Build a context injection string from structurally related steps.

        Format is model-friendly: "step N: {role} - {summary}"
        This matches how the model was trained to see conversation history.

        CRITICAL: This context is injected as a SEPARATE user message
        (never into assistant content) to preserve the model's expected
        chat template pattern and avoid KV-cache refills.
        """
        related = self.store.get_related_context(current_step)
        if not related:
            return ""

        def relevance_score(step: AgentStep) -> float:
            score = 0.0
            if step.metadata.get("subtask") == current_step.metadata.get("subtask"):
                score += 10.0
            if step.tool_name == current_step.tool_name:
                score += 5.0
            if step.role == current_step.role:
                score += 1.0
            score += 1.0 / (1.0 + abs(step.step_index - current_step.step_index))
            return score

        related.sort(key=relevance_score, reverse=True)

        # Model-friendly format: "step N: role - summary"
        lines: list[str] = []
        for step in related[:6]:
            if step.role == "tool":
                content = step.outcome_summary or step.content[:100]
                lines.append(f"step {step.step_index}: tool - {content}")
            elif step.role == "assistant":
                # Skip reasoning content, focus on action
                content = step.content
                # Strip reasoning tags for cleaner summary
                content = _strip_reasoning(content)
                lines.append(f"step {step.step_index}: assistant - {content[:100]}")
            elif step.role == "thinking":
                lines.append(f"step {step.step_index}: thinking - {_strip_reasoning(step.content)[:80]}...")
            else:
                lines.append(f"step {step.step_index}: {step.role} - {step.content[:100]}")

        return "\n".join(lines)


def _strip_reasoning(text: str) -> str:
    """Strip reasoning tags from text to get the action/result."""
    # Qwen-native tags
    text = _strip_qwen_reasoning(text)
    # XML-style tags
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_qwen_reasoning(text: str) -> str:
    """Strip Qwen-native <think>/</think> tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
