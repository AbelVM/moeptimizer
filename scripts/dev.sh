#!/usr/bin/env bash
# Install project dependencies and run tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Installing moeptimizer..."
pip install -e ".[dev]"

echo "Running tests..."
pytest tests/ -v --tb=short

echo "Running linter..."
ruff check src/ tests/

echo "Running type checker..."
mypy src/moeptimizer/

echo "All checks passed."
