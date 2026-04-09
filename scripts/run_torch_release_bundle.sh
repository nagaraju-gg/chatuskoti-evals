#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

BUNDLE_ROOT="artifacts/strong_v1_2_torch"

"$PYTHON_BIN" -m unittest discover -s tests -v

"$PYTHON_BIN" -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output "$BUNDLE_ROOT/canonical_failure"

"$PYTHON_BIN" scripts/generate_failure_figure.py \
  "$BUNDLE_ROOT/canonical_failure/failure_injection/failure_results.json"

"$PYTHON_BIN" -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --iterations 4 \
  --seeds 3 \
  --output "$BUNDLE_ROOT/challenge_compare"

"$PYTHON_BIN" -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output "$BUNDLE_ROOT/ablations"

"$PYTHON_BIN" -m chatuskoti_evals.cli run-calibration \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output "$BUNDLE_ROOT/calibration"

"$PYTHON_BIN" scripts/verify_release_bundle.py "$BUNDLE_ROOT"
