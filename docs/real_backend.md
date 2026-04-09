# Real PyTorch Backend

The simulator remains the default backend so the repo stays runnable without third-party packages. For a real evidence pass on a GPU-capable machine, use the torch backend.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-torch.txt
```

## Run

Compare both controllers on the real backend:

```bash
python3 -m chatuskoti_evals.cli compare \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --output artifacts/strong_v1_2_torch/challenge_compare
```

Run the canonical failure benchmark:

```bash
python3 -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output artifacts/strong_v1_2_torch/canonical_failure
```

Run the nearby-threshold calibration sweep:

```bash
python3 -m chatuskoti_evals.cli run-calibration \
  --backend torch \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output artifacts/strong_v1_2_torch/calibration
```

## Notes

- The backend trains `torchvision.models.resnet18(weights=None, num_classes=100)`.
- `stochastic_depth_*` now applies residual drop-path across the ResNet blocks on the torch path.
- `dropout_high` remains an explicit classifier-dropout stress action.
- `eval_tta` changes the evaluation hash and should drive validity below the merge threshold as intended.
- The adapter logs real metrics for validation accuracy, train/val loss gap, gradient statistics, weight distance, and proxy metrics.
