#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${1:-3.12}"

uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}"
uv sync --dev

echo "Done. Activate with: source .venv/bin/activate"
