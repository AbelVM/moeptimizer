"""Configuration loaded from the environment.

Kept as a plain dataclass so it is trivially injectable in tests and can be
overridden by explicit constructor arguments (presets layer on top of env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    input_path: Path = Path("users.jsonl")
    dry_run: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            input_path=Path(os.environ.get("USERS_INPUT", "users.jsonl")),
            dry_run=os.environ.get("USERS_DRY_RUN", "0") == "1",
            log_level=os.environ.get("USERS_LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        if self.input_path is None:
            raise ValueError("input_path is required")
