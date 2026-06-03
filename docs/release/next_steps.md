# Next Steps

This repo is now organized around a bundle release workflow that generates `artifacts/strong_v{version}_torch` on a machine with `torch` and `torchvision` installed.

## 1. Prepare the machine

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-torch.txt
.venv/bin/python scripts/check_torch_env.py
```

Success looks like:

- `torch` imports cleanly
- `torchvision` imports cleanly
- at least one GPU backend is available, ideally CUDA

## 2. Run the smoke test first

Before the real comparison, validate the backend with the smallest useful run:

```bash
bash scripts/run_torch_smoke.sh
```

This uses:

- `1` epoch
- `1` iteration
- `1` seed
- `num_workers=0`

If that succeeds, move to the real comparison.

## 3. Run the bundle

Use the prepared wrapper:

```bash
bash scripts/run_torch_bundle.sh
```

That script:

- runs the unit test suite first
- generates lead-time trajectory analysis
- extracts annotation cases
- generates ablation reports
- stages the artifacts in git (replaces any previously tracked `strong_v*` bundle)

## 4. Or run the main pieces directly

If you want to run pieces one at a time:

```bash
.venv/bin/python -m chatuskoti_evals.cli lead-time \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --iterations 15 \
  --action stochastic_depth_high \
  --window 5 \
  --tau 0.4 \
  --output artifacts/strong_v1_3_torch/lead_time
```

```bash
.venv/bin/python -m chatuskoti_evals.cli extract-cases \
  --bundle artifacts/strong_v1_3_torch \
  --output artifacts/strong_v1_3_torch/annotation_cases.csv
```

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_3_torch/ablations
```

Expected outputs:

- `artifacts/strong_v1_3_torch/lead_time/lead_time_analysis/summary.md`
- `artifacts/strong_v1_3_torch/annotation_cases.csv`
- `artifacts/strong_v1_3_torch/ablations/summary.md`
- JSON manifests, per-seed metrics, and SVG charts

If the smoke test succeeds and the machine is CUDA-capable, then scale up to `--epochs 30`.

## 5. Check the evidence gate

Do not call it publish-ready yet unless all are true:

- the canonical failure benchmark matches all intended branches
- binary still adopts pyrrhic, Goodhart-style, and incomparable benchmark cases that `Vec3` blocks
- the challenge comparison still shows structural invalid merges being accepted by binary
- the ablation summary weakens in the expected directions
- the lead-time trajectory shows the evaluator providing meaningful lead steps before bad merges would occur
- outputs are reproducible across reruns within a reasonable tolerance band

## 6. Tighten before publishing

After the first run, inspect and likely refine:

- lead-time window size and tau threshold
- annotation case quality and coverage
- paper wording so claims match real evidence rather than simulator behavior
- package/release metadata so the package version, docs, manifests, and artifact bundle name stay aligned

## 7. Minimum publish package

For a credible first public drop, include:

- the repo
- exact run command
- one saved `strong_v1_3_torch` bundle
- the lead-time trajectory summary
- the annotation cases
- the ablation summary
- one short limitations section that says this is a benchmark-specific calibrated evaluation framework
