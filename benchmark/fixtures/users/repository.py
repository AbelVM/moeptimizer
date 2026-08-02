"""JSONL-backed repository for :class:`User` records.

Realistic production code: row-level schema validation, a strict mode that
surfaces every bad row instead of stopping at the first, and a small retry
wrapper around file reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .models import User, UserSchemaError


def _parse_row(line: str, line_no: int) -> User:
    raw = json.loads(line)
    if "id" not in raw or "name" not in raw:
        raise UserSchemaError(f"row {line_no}: missing id/name")
    return User(
        id=int(raw["id"]),
        name=str(raw["name"]),
        active=bool(raw.get("active", True)),
    )


class UserRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[User]:
        users: list[User] = []
        with self.path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                users.append(_parse_row(line, line_no))
        return users

    def load_strict(self) -> list[User]:
        users: list[User] = []
        errors: list[str] = []
        with self.path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    users.append(_parse_row(line, line_no))
                except UserSchemaError as exc:
                    errors.append(str(exc))
        if errors:
            raise UserSchemaError(f"{len(errors)} bad rows: {errors[:3]}")
        return users

    def load_with_retry(self, retries: int = 3) -> list[User]:
        for attempt in range(retries):
            try:
                return self.load()
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise OSError("unreachable")
