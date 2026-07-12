"""Entry point for `python -m moeptimizer`."""

from __future__ import annotations

import sys

import uvicorn

from moeptimizer.app import create_app
from moeptimizer.config import get_config


def main() -> None:
    if "--check-config" in sys.argv[1:]:
        from moeptimizer.config_check import main as check_main

        raise SystemExit(check_main())

    config = get_config()
    app = create_app(config)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
