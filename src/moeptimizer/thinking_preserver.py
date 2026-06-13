"""ThinkingPreserver — Pass-through for front-loading eviction.

Since the compactor now uses pure eviction (dropping entire turns), there is
no need to compress reasoning content. The ThinkingPreserver preserves all
messages as-is, ensuring the MTP heads see exactly the token sequences they
were trained on.

The `protect_recent` parameter is retained for API compatibility but has no
functional effect — all messages are preserved regardless of recency.

Enhanced with:
- Syntax-stable MTP prompt engineering
- Code structure markers
- Indentation pattern stabilization
"""

from __future__ import annotations

import re
from typing import Any


class ThinkingPreserver:
    """
    Pass-through reasoning preserver for front-loading eviction.

    Since eviction drops entire turns (no summarization), reasoning tags
    are never modified. This preserves the exact token sequences the model
    was trained on, preventing MTP head disruption and Vulkan prefills.

    The `protect_recent` parameter is retained for API compatibility.
    """

    def __init__(self, protect_recent: int | None = None) -> None:
        # Parameter retained for API compatibility; no functional effect
        # since eviction drops whole turns rather than compressing content.
        self._protect_recent = protect_recent

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Return messages unchanged.

        With front-loading eviction, reasoning content is never compressed.
        Old turns are dropped entirely (not summarized), so all preserved
        messages retain their original token sequences.
        """
        return [dict(msg) for msg in messages]

    def preseed_reasoning_prefix(
        self,
        prompt: str,
        task_type: str = "default",
    ) -> str:
        """
        Pre-seed reasoning prefix for MTP head optimization.

        Adds task-specific reasoning scaffolding to improve MTP convergence.
        Only adds content for larger contexts where benefit outweighs cost.
        """
        # For short prompts, don't add preseeding overhead
        # The preseeding adds ~100 tokens, so only use for substantial prompts
        if len(prompt) < 200:
            return prompt

        # Add task-specific reasoning hint
        task_hints = {
            "debug": "Analyze the error, identify root cause, propose fix.",
            "refactor": "Review structure, identify improvements, implement changes.",
            "feature": "Plan implementation, write code, verify correctness.",
            "test": "Identify test cases, write tests, check coverage.",
            "default": "Analyze context, plan approach, implement solution.",
        }

        hint = task_hints.get(task_type, task_hints["default"])

        # Pre-seed the reasoning block
        return f"{prompt}\n\n<thought>\n{hint}\n\n"

    def inject_syntax_markers(
        self,
        code: str,
        language: str = "python",
    ) -> str:
        """
        Inject syntax markers for MTP stability.

        Adds predictable patterns that help MTP heads converge faster.
        """
        lines = code.split("\n")
        result = []
        in_code_block = False

        for i, line in enumerate(lines):
            if line.strip().startswith("```") and not in_code_block:
                in_code_block = True
                result.append(line)
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                result.append(line)
            elif in_code_block:
                # Add section markers for common patterns
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("function "):
                    # Extract function name
                    match = re.match(r"(def|function)\s+(\w+)", stripped)
                    if match:
                        name = match.group(2)
                        result.append(f"# SECTION: function {name}")
                elif stripped.startswith("class "):
                    match = re.match(r"class\s+(\w+)", stripped)
                    if match:
                        name = match.group(2)
                        result.append(f"# SECTION: class {name}")
                result.append(line)
            else:
                result.append(line)

        return "\n".join(result)

    def stabilize_indentation(
        self,
        code: str,
    ) -> str:
        """
        Stabilize indentation patterns for MTP prediction.

        Ensures consistent indentation that MTP heads can predict.
        """
        lines = code.split("\n")
        result = []

        for line in lines:
            # Normalize tabs to spaces
            normalized = line.replace("\t", "    ")
            result.append(normalized)

        return "\n".join(result)
