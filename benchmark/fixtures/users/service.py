"""Application service with dependency injection.

Wraps the repository so callers (CLI, API) depend on an interface rather than
concrete IO, which is what makes the package unit-testable without touching disk.
"""

from __future__ import annotations

from .models import User
from .repository import UserRepository


class SummarizerService:
    def __init__(self, repository: UserRepository) -> None:
        self.repository = repository

    def summarize(self) -> dict[str, int | bool]:
        return _summarize(self.repository.load())

    def active_count(self, users: list[User]) -> int:
        return sum(1 for u in users if u.active)


def _summarize(users: list[User]) -> dict[str, int | bool]:
    from collections import Counter

    counts = Counter(u.active for u in users)
    return {"count": len(users), "active": counts.get(True, 0)}
