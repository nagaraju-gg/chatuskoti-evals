#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

VERSION=$(grep -m1 '^version = ' "$ROOT_DIR/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
DEFAULT_BUNDLE="artifacts/strong_v${VERSION}_simulator"
BUNDLE_ROOT="${1:-$DEFAULT_BUNDLE}"
LEAD_ITERATIONS="${2:-10}"
TRAJ_TRAJECTORIES="${3:-500}"
TRAJ_ITERATIONS="${4:-6}"
TRAJ_SEEDS="${5:-2}"

cd "$ROOT_DIR"
mkdir -p "$BUNDLE_ROOT"

# ──────────────────────────────────────────────────
echo "=============================================="
echo "  Simulator Evidence Bundle (v$VERSION)"
echo "=============================================="
echo "Bundle root: $BUNDLE_ROOT"
echo "  lead-time:           $LEAD_ITERATIONS iterations"
echo "  trajectory-pred:     $TRAJ_TRAJECTORIES trajectories × $TRAJ_ITERATIONS iters × $TRAJ_SEEDS seeds"
echo "  (simulator, no cooldown needed)"
echo ""

# ──────────────────────────────────────────────────
# 1. Lead-time analysis (simulator)
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 1/3: Lead-time analysis (simulator)"
echo "  Iterations: $LEAD_ITERATIONS"
echo "=============================================="
echo ""

"$PYTHON_BIN" -m chatuskoti_evals.cli lead-time \
  --backend simulator \
  --data-dir "$ROOT_DIR/data" \
  --seeds 1 \
  --iterations "$LEAD_ITERATIONS" \
  --action stochastic_depth_high \
  --window auto \
  --tau auto \
  --output "$BUNDLE_ROOT/lead_time"

echo ""
echo "Lead-time complete."
echo ""

# ──────────────────────────────────────────────────
# 2. Trajectory prediction (simulator)
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 2/3: Trajectory prediction (simulator)"
echo "  Trajectories: $TRAJ_TRAJECTORIES | Iters: $TRAJ_ITERATIONS | Seeds: $TRAJ_SEEDS"
echo "=============================================="
echo ""

"$PYTHON_BIN" -m chatuskoti_evals.cli trajectory-prediction \
  --backend simulator \
  --data-dir "$ROOT_DIR/data" \
  --trajectories "$TRAJ_TRAJECTORIES" \
  --iterations "$TRAJ_ITERATIONS" \
  --window 3 \
  --ridge-alpha 1.0 \
  --seeds "$TRAJ_SEEDS" \
  --output "$BUNDLE_ROOT/trajectory_prediction"

echo ""
echo "Trajectory prediction complete."
echo ""

# ──────────────────────────────────────────────────
# 3. Stage artifacts in git
# ──────────────────────────────────────────────────
echo "=============================================="
echo "  STEP 3/3: Stage artifacts in git"
echo "=============================================="
echo ""

git rm -r --cached artifacts/strong_v* 2>/dev/null || true
git add "$BUNDLE_ROOT"
echo "Staged: $BUNDLE_ROOT (replaced any previously tracked strong_v bundle)"

echo ""
echo "=============================================="
echo "  All 3 steps complete."
echo "  Artifacts under: $BUNDLE_ROOT/"
echo "    lead-time:           $BUNDLE_ROOT/lead_time/lead_time_analysis/"
echo "    trajectory-pred:     $BUNDLE_ROOT/trajectory_prediction/"
echo "=============================================="
echo "  Next: git commit to save the updated artifacts."
