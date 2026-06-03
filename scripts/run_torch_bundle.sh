#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

VERSION=$(grep -m1 '^version = ' "$ROOT_DIR/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
DEFAULT_BUNDLE="artifacts/strong_v${VERSION}_torch"
BUNDLE_ROOT="${1:-$DEFAULT_BUNDLE}"
ITERATIONS="${2:-15}"
SEEDS="${3:-3}"
COOLDOWN="${4:-300}"

cd "$ROOT_DIR"
mkdir -p "$BUNDLE_ROOT"

# ──────────────────────────────────────────────────
echo "=============================================="
echo "  Torch Evidence Bundle (v$VERSION)"
echo "=============================================="
echo "Bundle root: $BUNDLE_ROOT"
TRAIN_MIN_PER_SEED=5
ITER_TRAIN_MIN=$((TRAIN_MIN_PER_SEED * SEEDS))
ITER_COOLDOWN_MIN=$((COOLDOWN / 60))
ITER_TOTAL_MIN=$((ITER_TRAIN_MIN + ITER_COOLDOWN_MIN))
TOTAL_MIN=$((ITER_TOTAL_MIN * ITERATIONS))
echo "  $ITERATIONS iters × $SEEDS seeds × ${TRAIN_MIN_PER_SEED}min + ${COOLDOWN}s cooldown = ~${TOTAL_MIN}min (~$((TOTAL_MIN / 60))h)"
echo "  Yields $ITERATIONS trajectory steps = $ITERATIONS annotation cases for the ground-truth study"
echo ""

# ──────────────────────────────────────────────────
# 1. Lead-time analysis
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 1/4: Lead-time analysis"
echo "  Iterations: $ITERATIONS | Seeds: $SEEDS | Cooldown: ${COOLDOWN}s"
echo "  Est: ~${TOTAL_MIN}min (~$((TOTAL_MIN / 60))h)"
echo "=============================================="
echo ""

"$PYTHON_BIN" -m unittest discover -s tests -v

"$PYTHON_BIN" -m chatuskoti_evals.cli lead-time \
  --backend torch \
  --data-dir "$ROOT_DIR/data" \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds "$SEEDS" \
  --iterations "$ITERATIONS" \
  --action stochastic_depth_high \
  --window 5 \
  --tau 0.4 \
  --cooldown "$COOLDOWN" \
  --output "$BUNDLE_ROOT/lead_time"

echo ""
echo "Lead-time complete."
echo ""

# ──────────────────────────────────────────────────
# 2. Extract annotation cases
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 2/4: Extract annotation cases"
echo "  Cooldown ${COOLDOWN}s before next step"
echo "=============================================="
echo ""

echo "Waiting ${COOLDOWN}s before extract-cases..."
sleep "$COOLDOWN"

"$PYTHON_BIN" -m chatuskoti_evals.cli extract-cases \
  --bundle "$BUNDLE_ROOT" \
  --output "$BUNDLE_ROOT/annotation_cases.csv"

echo ""
echo "Extract complete: $BUNDLE_ROOT/annotation_cases.csv"
echo ""

# ──────────────────────────────────────────────────
# 3. Ablation bundle (all axis variants)
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 3/4: Ablation bundle (axis variants)"
echo "  Cooldown ${COOLDOWN}s before next step"
echo "=============================================="
echo ""

echo "Waiting ${COOLDOWN}s before ablation bundle..."
sleep "$COOLDOWN"

"$PYTHON_BIN" -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --data-dir "$ROOT_DIR/data" \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds "$SEEDS" \
  --output "$BUNDLE_ROOT/ablations"

echo ""
echo "Ablation bundle complete."
echo ""

# ──────────────────────────────────────────────────
# 4. Stage artifacts in git (replaces old tracked bundle)
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 4/4: Stage artifacts in git"
echo "=============================================="
echo ""

git rm -r --cached artifacts/strong_v* 2>/dev/null || true
git add "$BUNDLE_ROOT"
echo "Staged: $BUNDLE_ROOT (replaced any previously tracked strong_v bundle)"

echo ""
echo "=============================================="
echo "  All 4 steps complete."
echo "  Artifacts under: $BUNDLE_ROOT/"
echo "    lead-time:   $BUNDLE_ROOT/lead_time/lead_time_analysis/"
echo "    cases:       $BUNDLE_ROOT/annotation_cases.csv"
echo "    ablations:   $BUNDLE_ROOT/ablations/summary.md"
echo "=============================================="
echo "  Next: git commit to save the updated artifacts."
