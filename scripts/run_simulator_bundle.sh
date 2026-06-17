#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "$ROOT_DIR"

echo "=============================================="
echo "  Simulator Quick Check"
echo "=============================================="
echo ""

"$PYTHON_BIN" -m unittest discover -s tests -v

echo ""
echo "All checks passed."
