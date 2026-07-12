"""In-memory metrics and structured logging for the service.

Lightweight on purpose: no external dependency, just enough signal to watch
load counts, parse errors, and summary latency in production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("users")


@dataclass
class Metrics:
    loads: int = 0
    parse_errors: int = 0
    summary_ms: float = 0.0

    def record_load(self, n: int) -> None:
        self.loads += n

    def record_parse_error(self) -> None:
        self.parse_errors += 1

    def record_summary(self, ms: float) -> None:
        self.summary_ms += ms


def log_event(step: str, **fields: object) -> None:
    logger.info(json.dumps({"step": step, **fields}))
