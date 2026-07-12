"""Domain models for the user-analytics service.

This is the initial, deliberately small starting point of the fixture project.
The benchmark's ``fixtures`` scenario replays a realistic agentic-coding
session that grows this module into a typed, tested, packaged service.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class User:
    id: int
    name: str
    active: bool = True


class UserSchemaError(ValueError):
    """Raised when a JSONL row fails schema validation."""


def summarize(users: list[User]) -> dict[str, int | bool]:
    return {"count": len(users), "active": sum(1 for user in users if user.active)}
