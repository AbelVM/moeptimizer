"""OutputShaper — response-length shaping (Headroom pattern).

The proxy compacts ONLY the input context (correct per the architectural
constraint), but the benchmark shows proxy responses are **3.6x longer** than
direct (`length_ratio.mean = 3.6244`). This destroys TPS and inflates cost.

OutputShaper does not touch the input path. It only mutates the request
parameters sent to the backend:

  1. Appends a concise "be terse" instruction to the system prompt. Because
     the instruction is appended *after* the frozen prefix, it does not shift
     the token boundaries the backend hashes — the prefix cache is preserved.
  2. Maps turn-class (new question vs. tool result vs. error) to
     ``reasoning_effort`` / ``max_tokens`` clamping via ``extra_body``.

The shaping is applied in ``app.py`` just before the backend request is sent,
so the optimizer's input-compaction work is unaffected.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TurnClass(Enum):
    """Classification of the current turn for output shaping."""

    NEW_QUESTION = "new_question"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CONTINUATION = "continuation"


# Default max-tokens clamp per turn-class. Values are conservative: they cap
# the *maximum* the model can generate, not the target. The model can always
# stop earlier.
_TURN_CLASS_MAX_TOKENS: dict[TurnClass, int] = {
    TurnClass.NEW_QUESTION: 4096,
    TurnClass.TOOL_RESULT: 2048,
    TurnClass.ERROR: 1024,
    TurnClass.CONTINUATION: 2048,
}

# Default reasoning_effort per turn-class (OpenAI-compatible field).
_TURN_CLASS_REASONING_EFFORT: dict[TurnClass, str] = {
    TurnClass.NEW_QUESTION: "medium",
    TurnClass.TOOL_RESULT: "low",
    TurnClass.ERROR: "low",
    TurnClass.CONTINUATION: "low",
}

# Concise system-prompt tail instruction. Appended after the frozen prefix so
# the backend's prefix cache is preserved (the leading bytes are unchanged).
_TERSE_INSTRUCTION = (
    "Be concise. Answer directly without restating context. "
    "Use minimal words; omit filler."
)


class OutputShaper:
    """Shape backend response length without touching the input path.

    Applies two levers:
      - System-prompt tail instruction (cache-safe append).
      - Per-turn-class max_tokens / reasoning_effort clamping.
    """

    def __init__(
        self,
        terse_instruction: str = _TERSE_INSTRUCTION,
        turn_class_max_tokens: dict[TurnClass, int] | None = None,
        turn_class_reasoning_effort: dict[TurnClass, str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._terse_instruction = terse_instruction
        self._max_tokens = dict(_TURN_CLASS_MAX_TOKENS)
        if turn_class_max_tokens:
            self._max_tokens.update(turn_class_max_tokens)
        self._reasoning_effort = dict(_TURN_CLASS_REASONING_EFFORT)
        if turn_class_reasoning_effort:
            self._reasoning_effort.update(turn_class_reasoning_effort)
        self._enabled = enabled

    def shape_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Mutate ``body`` in place with output-shaping parameters.

        Returns the mutated body for chaining.
        """
        if not self._enabled:
            return body

        turn_class = self._classify_turn(body.get("messages", []))

        # 1) Append terse instruction to the system prompt (cache-safe: appended
        #    after the frozen prefix, so the leading bytes are unchanged).
        body = self._inject_terse_instruction(body)

        # 2) Clamp max_tokens per turn-class.
        max_tokens = self._max_tokens.get(turn_class)
        if max_tokens is not None:
            existing = body.get("max_tokens")
            if existing is None or (isinstance(existing, int) and existing > max_tokens):
                body["max_tokens"] = max_tokens

        # 3) Clamp reasoning_effort per turn-class (OpenAI-compatible).
        reasoning = self._reasoning_effort.get(turn_class)
        if reasoning is not None:
            extra_body = dict(body.get("extra_body") or {})
            extra_body.setdefault("reasoning_effort", reasoning)
            body["extra_body"] = extra_body

        return body

    def _classify_turn(self, messages: list[dict[str, Any]]) -> TurnClass:
        """Classify the current turn for output shaping."""
        if not messages:
            return TurnClass.NEW_QUESTION

        last = messages[-1]
        role = last.get("role", "")

        if role == "user":
            # Check if this is a tool-result user turn (contains tool_call_id
            # or looks like a tool output summary).
            content = last.get("content") or ""
            if isinstance(content, str) and (
                "tool_result" in content.lower()
                or "tool_call_id" in str(last)
                or content.strip().startswith(("Output:", "Result:", "stdout:", "stderr:"))
            ):
                return TurnClass.TOOL_RESULT
            # Check if this is an error turn.
            if isinstance(content, str) and content.strip().lower().startswith("error:"):
                return TurnClass.ERROR
            # If the previous turn was assistant/tool, this is a continuation.
            if len(messages) >= 2 and messages[-2].get("role") in ("assistant", "tool"):
                return TurnClass.CONTINUATION
            return TurnClass.NEW_QUESTION

        if role == "assistant":
            # Check if the previous turn was a tool result (continuation).
            if len(messages) >= 2 and messages[-2].get("role") == "tool":
                return TurnClass.CONTINUATION
            return TurnClass.CONTINUATION

        if role == "tool":
            return TurnClass.TOOL_RESULT

        return TurnClass.NEW_QUESTION

    def _inject_terse_instruction(self, body: dict[str, Any]) -> dict[str, Any]:
        """Append the terse instruction to the system prompt.

        Cache-safe: the instruction is appended after the existing system
        content, so the frozen prefix bytes are unchanged.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        # Find the last system message and append the instruction.
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "system":
                content = msg.get("content") or ""
                if isinstance(content, str) and _TERSE_INSTRUCTION not in content:
                    new_msg = dict(msg)
                    new_msg["content"] = f"{content}\n\n{_TERSE_INSTRUCTION}"
                    messages = list(messages)
                    messages[i] = new_msg
                    body["messages"] = messages
                break

        return body
