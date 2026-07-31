#!/usr/bin/env bash
# Install project dependencies and run tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Installing moeptimizer..."
pip install -e ".[dev]" --break-system-packages

echo "Running tests..."
pytest tests/ -v --tb=short

echo "Running linter..."
ruff check src/ tests/

echo "Running type checker..."
mypy src/moeptimizer/

# Optional cache-stability gate (review §4.12.2): replay the opencode scenario
# through the proxy's dry-run endpoint and fail if prefix-cache breaks exceed the
# expected rolling-summary fold budget. The fold breaks the cache by design on fold
# turns, so this gates on the break COUNT (a regression like the every-turn eviction
# cliff produces ~18 breaks; a healthy run ~7). Requires a live backend; skipped
# when none is reachable so CI without a backend still passes. Tune the budget via
# CACHE_GATE_MAX_BREAKS.
CACHE_GATE_URL="${MOEPT_SERVER__URL:-http://localhost:13305/api/v1}"
CACHE_GATE_MAX_BREAKS="${CACHE_GATE_MAX_BREAKS:-12}"
if curl -sf -m 3 "${CACHE_GATE_URL%/}/health" >/dev/null 2>&1; then
    echo "Running cache-stability gate (dry-run, max-breaks=${CACHE_GATE_MAX_BREAKS})..."
    python scripts/diag_dryrun_opencode.py --persistent-session --turns 30 \
        --max-breaks "${CACHE_GATE_MAX_BREAKS}"
else
    echo "Skipping cache-stability gate (no backend at ${CACHE_GATE_URL})."
fi

echo "All checks passed."
