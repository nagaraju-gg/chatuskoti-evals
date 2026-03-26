# Catuskoti AutoResearcher V1

This repository implements a deterministic V1 prototype for the `Vec3` research loop described in the draft: `truthness`, `coherence`, and `comparability`, plus a separate `goodhart_score`.

## Current status

The strongest current result in this repo is the torch-backed adversarial calibration benchmark, not the open-loop controller runs.

- canonical benchmark summary: [summary.md](artifacts/torch_failure_set_e10_v5/failure_injection/summary.md)
- benchmark figure: [benchmark_figure.svg](artifacts/torch_failure_set_e10_v5/failure_injection/benchmark_figure.svg)
- paper-ready section: [paper_figure_1.md](docs/paper_figure_1.md)

Headline result:

- `4/4` benchmark expectations matched
- `binary` would adopt `3/4` adversarial cases
- `Vec3` would adopt `0/4`

The prototype is deliberately narrow:

- benchmark target: `CIFAR-100 + ResNet-18`
- implementation mode: deterministic simulation with `PyTorch`-shaped metrics
- proposal space: typed interventions only
- controllers: `binary` and `vec3`
- outputs: JSONL history, per-seed metrics, SVG plots, Markdown summaries

The reports distinguish between:

- controller eval metric: what the controller would have accepted in its own loop
- canonical benchmark metric: the anchored comparison metric used to compare controllers fairly

## Failure injection set

The repo now treats detector stress tests as a first-class artifact.

- We include deliberately adversarial interventions to stress-test detectors.
- The named failure injection set lives in [catuskoti_ar/scenarios.py](catuskoti_ar/scenarios.py).
- The goal is to prove that `Vec3` distinguishes clean wins from pyrrhic, Goodhart, broken, and incomparable outcomes under a controlled benchmark-specific setup.

## Canonical demo cases

These are the three paper/demo anchor cases hard-coded into the repo metadata:

- pyrrhic win example
- Goodhart example
- recovery via Vec3 history

They are defined in [catuskoti_ar/scenarios.py](catuskoti_ar/scenarios.py) and are intended to become the core figures for the paper and demo.

## Interpretability outputs

For every decision the history now logs:

- which detector signals fired
- the resolver action
- the explicit resolver reason string that explains why that action was chosen

This makes the generated `history.jsonl` and `summary.md` reviewer-friendly instead of leaving decisions implicit.

## Scope and claims

This should be presented as a benchmark-specific calibrated prototype.

- Do claim deterministic evaluation, richer history, and reproducible benchmark evidence.
- Do not claim general validity across all automated research systems yet.

## Why a simulator?

This workspace does not have `torch`, `torchvision`, or plotting packages installed, so the first implementation is a self-contained stdlib simulation that preserves the control-flow and detector logic. The adapter boundary is explicit, so a real training adapter can replace the simulator later without changing the loop, resolver, reporting, or tests.

## Quickstart

Fastest way to see the strongest result:

```bash
.venv/bin/python -m catuskoti_ar.cli run-failure-set --backend torch --epochs 10 --seeds 1 --num-workers 0 --output artifacts/torch_failure_set_e10_v5
python3 scripts/generate_failure_figure.py artifacts/torch_failure_set_e10_v5/failure_injection/failure_results.json
```

Run the side-by-side demo:

```bash
python3 -m catuskoti_ar.cli compare --output artifacts/demo_run
```

Run the stronger benchmark-aware open-loop comparison:

```bash
python3 -m catuskoti_ar.cli compare --mode challenge --output artifacts/challenge_compare
```

Run against the real PyTorch backend on a machine with `torch` installed:

```bash
python3 -m catuskoti_ar.cli compare --backend torch --epochs 30 --output artifacts/torch_compare
```

Check the machine first:

```bash
.venv/bin/python scripts/check_torch_env.py
```

Or use the prepared wrapper:

```bash
bash scripts/run_torch_smoke.sh
```

Then move to:

```bash
bash scripts/run_torch_compare.sh
```

`run_torch_smoke.sh` is the true first-run check. `run_torch_compare.sh` is much heavier because it runs both controllers across multiple iterations and seeds.

Run one controller only:

```bash
python3 -m catuskoti_ar.cli run-loop --controller vec3 --output artifacts/vec3_only
```

Run the named failure-injection set:

```bash
python3 -m catuskoti_ar.cli run-failure-set --output artifacts/failure_set
```

On the torch backend, the failure-injection set now includes explicit adversarial probes for:

- pyrrhic behavior
- metric-gaming / Goodhart-like behavior
- broken failure
- incomparable evaluation

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

## What gets generated

Each controller writes:

- `history.jsonl`
- `seed_metrics.json`
- `wisdom.json`
- `summary.md`
- `metric_trajectory.svg`
- `action_counts.svg`
- `outcome_regions.svg`

The comparison report uses the canonical benchmark metric so that a controller cannot "win" just by adopting an incomparable evaluation regime.

You can also run a calibration loop that deliberately tries adversarial actions first:

```bash
python3 -m catuskoti_ar.cli run-loop --controller vec3 --mode calibration --output artifacts/vec3_calibration
```

For open-loop comparison, `challenge` mode is the better current narrative than the plain default loop because it runs both controllers through the same benchmark-aware action schedule and makes the divergence easier to interpret.

## Canonical benchmark artifact

The strongest current project artifact is the torch-backed adversarial calibration benchmark:

- canonical run summary: [summary.md](artifacts/torch_failure_set_e10_v5/failure_injection/summary.md)
- benchmark figure: [benchmark_figure.svg](artifacts/torch_failure_set_e10_v5/failure_injection/benchmark_figure.svg)
- benchmark caption: [benchmark_figure_caption.md](artifacts/torch_failure_set_e10_v5/failure_injection/benchmark_figure_caption.md)
- paper-ready condensation: [paper_ready_summary.md](artifacts/torch_failure_set_e10_v5/failure_injection/paper_ready_summary.md)
- paper/demo figure section: [paper_figure_1.md](docs/paper_figure_1.md)
- claims and limitations note: [canonical_failure_benchmark.md](docs/canonical_failure_benchmark.md)
- release checklist: [release_checklist.md](docs/release_checklist.md)
- license: [LICENSE](LICENSE)
- citation metadata: [CITATION.cff](CITATION.cff)

## Real backend

The repo now includes a lazy-loaded real backend in [catuskoti_ar/torch_backend.py](catuskoti_ar/torch_backend.py).

- Default backend: `simulator`
- Real backend: `torch`
- Install file: [requirements-torch.txt](requirements-torch.txt)
- Setup notes: [docs/real_backend.md](docs/real_backend.md)

The torch backend is intentionally lazy-loaded so the simulator path and test suite still work in environments that do not have `torch` or `torchvision` installed.

On the torch path:

- `stochastic_depth_*` uses residual drop-path on the ResNet blocks
- `dropout_high` is a separate classifier-dropout stress action

The operational handoff is documented in [docs/next_steps.md](docs/next_steps.md).

## Expected runtime and cost

Current stdlib simulator in this repo:

- runtime: under 1 minute on a laptop for `compare`
- GPU cost: none

Planned real `CIFAR-100 + ResNet-18` backend estimate for the same narrative artifact:

- detector calibration pass: roughly `3-5 GPU hours`
- one `binary` vs `vec3` comparison run at 4 iterations and 3 seeds: roughly `6-10 GPU hours`
- total first evidence pass: roughly `10-16 GPU hours` on a single mid-range GPU class machine

The comparison run also writes:

- `comparison.md`
- `controller_comparison.svg`

## Repo structure

- [catuskoti_ar/benchmark.py](catuskoti_ar/benchmark.py)
- [catuskoti_ar/scoring.py](catuskoti_ar/scoring.py)
- [catuskoti_ar/resolver.py](catuskoti_ar/resolver.py)
- [catuskoti_ar/proposals.py](catuskoti_ar/proposals.py)
- [catuskoti_ar/reporting.py](catuskoti_ar/reporting.py)
