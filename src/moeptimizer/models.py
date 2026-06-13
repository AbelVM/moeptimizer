"""Data models for the MoE optimizer."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StepRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    THINKING = "thinking"


@dataclass
class AgentStep:
    """Represents a single step in the agent loop."""

    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str = "assistant"
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    outcome_summary: str | None = None
    thinking_archived: bool = False
    timestamp: float = field(default_factory=time.time)
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AgentStep:
        return cls(**{k: v for k, v in d.items() if k != "timestamp" or isinstance(v, (int, float))})


@dataclass
class GoalNode:
    """Root goal that anchors the entire agent session."""

    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    original_prompt: str = ""
    subtasks: list[str] = field(default_factory=list)
    completed: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoopWarning:
    """Indicates the agent may be stuck in a loop."""

    loop_type: str
    tool_name: str | None = None
    repeat_count: int = 0
    message: str = ""


@dataclass
class ProgressSnapshot:
    """Current progress toward the agent's goal."""

    total_steps: int = 0
    completed_subtasks: list[str] = field(default_factory=list)
    active_subtasks: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    estimated_completion: float = 0.0
    is_complete: bool = False
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tools_used"] = sorted(self.tools_used)
        return d
