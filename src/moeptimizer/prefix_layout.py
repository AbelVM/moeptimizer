"""PrefixLayout dataclass extracted from optimizer.py (E1 god-object decomposition)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixLayout:
    """The immutable, evictable, and protected prompt regions."""

    system_anchor: list[dict[str, Any]]
    evictable_body: list[dict[str, Any]]
    protected_tail: list[dict[str, Any]]
