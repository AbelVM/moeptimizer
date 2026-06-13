"""StateBasedRAG — Graph-indexed retrieval (not flat embeddings).

Uses the AgentStateStore graph to retrieve context by:
  1. Goal proximity — steps related to the current goal
  2. Subtask affinity — steps from the same subtask
  3. Tool lineage — steps that used the same tools
  4. Temporal decay — older related steps get lower priority
  5. Dependency graph — related files and symbols

MOE context integrity:
  - RAG context is injected as a SEPARATE user message (never into assistant)
  - Format uses model-friendly structure: "step N: {role} - {summary}"
  - Avoids arbitrary markers like "role#index:content" that the model
    was not trained to recognize
"""

from __future__ import annotations

import re
from typing import Any

from moeptimizer.models import AgentStep
from moeptimizer.state_store import AgentStateStore
from moeptimizer.symbol_index import SymbolIndex


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
        self._symbol_index = SymbolIndex()
        self._dependency_graph: dict[str, set[str]] = {}

    def build_dependency_graph(
        self,
        file_contents: dict[str, str],
    ) -> None:
        """Build dependency graph from file contents.

        Maps file → imported/required files for context prefetching.
        """
        for file_path, content in file_contents.items():
            imports = self._extract_imports(content, file_path)
            self._dependency_graph[file_path] = imports
            # Also index symbols in this file
            self._symbol_index.add_file(file_path, content)

    def _extract_imports(
        self,
        content: str,
        file_path: str,
    ) -> set[str]:
        """Extract import statements to build dependency graph."""
        imports: set[str] = set()

        # Python imports
        for match in re.finditer(r"^from\s+(\S+)\s+import", content, re.MULTILINE):
            module = match.group(1)
            # Convert module to file path
            imports.add(module.replace(".", "/") + ".py")

        for match in re.finditer(r"^import\s+(\S+)", content, re.MULTILINE):
            module = match.group(1)
            imports.add(module.replace(".", "/") + ".py")

        # JavaScript/TypeScript imports
        for match in re.finditer(r"^import\s+.*from\s+['\"]([^'\"]+)['\"]", content, re.MULTILINE):
            module = match.group(1)
            if not module.startswith("."):
                continue
            # Relative import
            base = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
            imports.add(base + "/" + module.lstrip("./") + ".ts")

        return imports

    def get_context_for_step(
        self,
        current_step: AgentStep,
    ) -> str:
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

    def get_dependency_context(
        self,
        file_path: str,
        max_files: int = 3,
    ) -> str:
        """Get context for files that the given file depends on.

        Uses dependency graph to prefetch related files.
        """
        if file_path not in self._dependency_graph:
            return ""

        deps = list(self._dependency_graph[file_path])[:max_files]
        if not deps:
            return ""

        lines = [f"# Dependencies of {file_path}:"]
        for dep in deps:
            symbols = self._symbol_index.get_symbols_in_file(dep)
            if symbols:
                lines.append(f"  {dep}: {', '.join(s['name'] for s in symbols[:5])}")

        return "\n".join(lines)

    def find_related_symbols(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find symbols related to query using fuzzy matching."""
        return self._symbol_index.find_symbol(query, fuzzy=True, max_results=max_results)


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
