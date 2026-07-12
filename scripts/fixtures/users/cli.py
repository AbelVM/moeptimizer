"""Command-line entry point.

Thin wrapper over :class:`SummarizerService`; all real logic lives behind the
service/repository so the CLI stays easy to test and the proxy's frozen prefix
(the system prompt) stays stable across turns.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import Config
from .metrics import Metrics, log_event
from .repository import UserRepository
from .service import SummarizerService


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize users from a JSONL file")
    parser.add_argument("--input", default="users.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)
    cfg = Config(input_path=Path(args.input), dry_run=args.dry_run)
    cfg.validate()
    metrics = Metrics()
    repo = UserRepository(cfg.input_path)
    service = SummarizerService(repo)

    start = time.perf_counter()
    summary = service.summarize()
    metrics.record_summary((time.perf_counter() - start) * 1000)
    metrics.record_load(summary["count"])
    log_event("summarize", **summary)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(json.dumps(summary))
    else:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
