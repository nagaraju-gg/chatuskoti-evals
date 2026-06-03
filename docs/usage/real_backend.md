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
python3 -m chatuskoti_evals.cli lead-time \
  --backend torch \
  --data-dir data \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --iterations 15 \
  --action stochastic_depth_high \
  --window 5 \
  --tau 0.4 \
  --output artifacts/strong_v1_3_torch/lead_time
```

Extract annotation cases:

```bash
python3 -m chatuskoti_evals.cli extract-cases \
  --bundle artifacts/strong_v1_3_torch \
  --output artifacts/strong_v1_3_torch/annotation_cases.csv
```

Run the ablation bundle:

```bash
python3 -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --device auto \
  --epochs 10 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --seeds 3 \
  --output artifacts/strong_v1_3_torch/ablations
```

## Notes

- The backend trains `torchvision.models.resnet18(weights=None, num_classes=100)`.
- `stochastic_depth_*` now applies residual drop-path across the ResNet blocks on the torch path.
- `dropout_high` remains an explicit classifier-dropout stress action.
- `eval_tta` changes the evaluation hash and should drive validity below the merge threshold as intended.
- The adapter logs real metrics for validation accuracy, train/val loss gap, gradient statistics, weight distance, and proxy metrics.
