#!/usr/bin/env bash
# Start the MoE Optimizer middleware
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Starting MoE Optimizer on http://127.0.0.1:8080"
echo "Backend: ${MOEPT_SERVER__URL:-http://localhost:13305/api/v1}"
echo "Model: ${MOEPT_SERVER__LLM_MODEL:-Qwen3.6-35B-A3B-MTP-GGUF}"

exec python -m moeptimizer
