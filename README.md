# Chatuskoti Evals

`Chatuskoti Evals` is a benchmark-specific evaluation framework for research loops on `CIFAR-100 + ResNet-18`.

In plain language: an eval is a repeatable way to decide whether a change is actually good enough to trust. This repo focuses on a failure mode that shows up everywhere in AI work: a run, model tweak, or tool policy can make a number go up while still being the wrong thing to merge, ship, or learn from.

That is the core problem here. Instead of asking only "did the metric improve?", this repo asks a more useful control question:

- is the result actually better
- is it stable enough to trust
- is it valid enough to compare to the baseline at all

Those distinctions matter for technical readers because many expensive mistakes in research loops look like progress at first:

- a training change improves top-line accuracy but makes the run unstable
- a search policy finds a better score by exploiting a proxy
- an evaluation setting changes and makes before/after numbers incomparable
- a controller keeps merging superficially positive results that later turn into wasted iteration

The repo name comes from [Chatuskoti](https://en.wikipedia.org/wiki/Catu%E1%B9%A3ko%E1%B9%ADi), a four-way logic lens. In this codebase that idea is used in a practical way, not a philosophical one: it is a way to avoid collapsing several importantly different outcomes into the same "good" bucket.

## What This Repo Is

This is not a general autonomous researcher. It is a calibrated evaluator plus controller logic for one concrete benchmark setting.

Today the repo provides:

- a benchmark adapter for a simulated backend and a real torch backend
- a three-axis evaluator over `truthness`, `reliability`, and `validity`
- resolver actions such as `adopt`, `hold`, `reframe`, `reject`, `rollback`, and `keep_going`
- canonical failure cases designed to separate clean gains from pyrrhic, gamed, broken, and incomparable outcomes
- comparison reports, ablations, calibration sweeps, plots, manifests, and release scripts

The narrow claim is the important one: if your controller only sees the headline metric, it will often make the wrong research-loop decision.

## Why AI Researchers And Tool Builders Should Care

Many evals answer "which system scored higher?" This repo is about a different question: "what should the loop do next?"

That matters if you care about:

- automated model search, where unstable wins can poison the next baseline
- agentic tooling, where a score bump may come from exploiting the harness rather than improving the behavior you wanted
- benchmark maintenance, where silent eval changes create fake progress
- reproducible research, where a result is only useful if someone can explain and regenerate it

Put differently: this repo treats evaluation as decision support, not only as measurement.

## The Core Idea

`Chatuskoti Evals` decomposes a candidate run into three axes:

| Axis | Question | Why it matters |
| --- | --- | --- |
| `truthness` (`T`) | Did the anchored benchmark metric actually improve? | Keeps the ordinary "better or worse" question explicit. |
| `reliability` (`R`) | Did the run stay stable and reproducible enough to trust? | Separates clean gains from pyrrhic ones. |
| `validity` (`V`) | Is the gain meaningful, rather than gamed or comparison-invalid? | Separates real progress from Goodhart-style or eval-regime failures. |

Those axes map to controller actions that a binary metric gate would collapse together:

| Outcome type | What a metric-only gate sees | `Vec3` action | Why |
| --- | --- | --- | --- |
| Clean gain | metric up | `adopt` | Better, stable, and comparable |
| Pyrrhic gain | metric up | `hold` | Improvement is not yet trustworthy |
| Metric-gamed gain | metric up | `reframe` | The loop learned the wrong lesson |
| Broken result | metric down | `rollback` | This is damage, not just failure to improve |
| Incomparable gain | metric up | `reframe` | The comparison itself is invalid |

If you want the shortest framing of the project, it is this:

> not every positive metric delta is the same kind of outcome, so not every positive metric delta deserves the same control action

## What The Repo Demonstrates Today

The main evidence in the repo is a controlled benchmark story, not a broad leaderboard claim.

The strongest checked-in torch-backed evidence is now the V1.2 bundle under [artifacts/strong_v1_2_torch](artifacts/strong_v1_2_torch). In that bundle:

- the canonical failure benchmark matches `4/4` intended cases
- binary evaluation would adopt `3/4` benchmark-aware bad cases
- `Vec3` routes those cases to `hold`, `reframe`, and `rollback`
- the challenge comparison shows binary reaching the higher metric by taking merges the benchmark says should not be merged

That is the current headline claim this repo can support well:

- structured evaluation preserves benchmark validity better than a metric-only gate on this benchmark

## Why Version 1.2 Matters

Version `1.2.0` is important, but not because it changes the public claim into something broader. It matters because it makes the repo easier to trust, inspect, and regenerate as a technical artifact.

`1.2.0` adds:

- a threshold calibration sweep via `run-calibration`
- richer manifests with package version, git commit, backend config, detector config, and benchmark spec id
- a release bundle script that runs tests, generates artifacts, and verifies the output structure
- a release demo page that gives a clearer reading order for the evidence bundle

For a technical reader, the practical meaning is:

- you can inspect not just the result, but the conditions that produced it
- you can test whether nearby threshold choices preserve the intended benchmark behavior
- you can regenerate a release bundle with stronger provenance checks

The important nuance is that V1.2 does not broaden the scientific claim. It strengthens the same benchmark story with calibration and provenance, and it checks in the corresponding `artifacts/strong_v1_2_torch` bundle.

## How To Read The Repo

If you are new to the project, do not start with the CLI.

Start in this order:

1. Read the canonical failure benchmark: [artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md](artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
2. Look at the companion loop comparison: [artifacts/strong_v1_2_torch/challenge_compare/comparison.md](artifacts/strong_v1_2_torch/challenge_compare/comparison.md)
3. Check whether the extra axes matter: [artifacts/strong_v1_2_torch/ablations/summary.md](artifacts/strong_v1_2_torch/ablations/summary.md)
4. Then read the release/demo framing: [docs/release_demo.md](docs/release_demo.md)

Supporting docs:

- plain-language overview: [docs/why_four_states.md](docs/why_four_states.md)
- guided walkthrough: [docs/demo.md](docs/demo.md)
- release runbook: [docs/next_steps.md](docs/next_steps.md)
- torch backend notes: [docs/real_backend.md](docs/real_backend.md)

## Quickstart

### Check the environment

```bash
.venv/bin/python scripts/check_torch_env.py
```

### Run the canonical failure benchmark

```bash
.venv/bin/python -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_2_torch/canonical_failure
```

### Run the challenge comparison

```bash
.venv/bin/python -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --epochs 10 \
  --iterations 4 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_2_torch/challenge_compare
```

### Run the ablation bundle

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_2_torch/ablations
```

### Run the new calibration sweep

```bash
.venv/bin/python -m chatuskoti_evals.cli run-calibration \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1_2_torch/calibration
```

### Or generate the whole V1.2 release bundle

```bash
bash scripts/run_torch_release_bundle.sh
```

## Repo Tour

The repo is easier to navigate if you think of it in layers.

### 1. Benchmark and backends

- [chatuskoti_evals/benchmark.py](chatuskoti_evals/benchmark.py): benchmark adapter interface plus the simulator-backed benchmark logic
- [chatuskoti_evals/torch_backend.py](chatuskoti_evals/torch_backend.py): real PyTorch execution path for `CIFAR-100 + ResNet-18`
- [chatuskoti_evals/scenarios.py](chatuskoti_evals/scenarios.py): canonical failure cases and benchmark scenarios

### 2. Scoring and decision logic

- [chatuskoti_evals/scoring.py](chatuskoti_evals/scoring.py): computes `T/R/V`, detector outputs, and fired signals
- [chatuskoti_evals/resolver.py](chatuskoti_evals/resolver.py): turns scores into actions such as `adopt`, `hold`, or `reframe`
- [chatuskoti_evals/models.py](chatuskoti_evals/models.py): typed result objects, manifests, and report payloads

### 3. Loop orchestration

- [chatuskoti_evals/proposals.py](chatuskoti_evals/proposals.py): proposes the next action in the loop
- [chatuskoti_evals/runner.py](chatuskoti_evals/runner.py): runs comparisons, failure sets, ablations, and calibration bundles
- [chatuskoti_evals/wisdom.py](chatuskoti_evals/wisdom.py): simple offline memory for action-family outcomes

### 4. Reporting and release workflow

- [chatuskoti_evals/reporting.py](chatuskoti_evals/reporting.py): markdown reports, JSON outputs, and SVG plots
- [chatuskoti_evals/cli.py](chatuskoti_evals/cli.py): package entrypoint
- [scripts/run_torch_release_bundle.sh](scripts/run_torch_release_bundle.sh): end-to-end V1.2 release bundle generation
- [scripts/verify_release_bundle.py](scripts/verify_release_bundle.py): verifies the generated release artifacts

### 5. Evidence and docs

- [artifacts/index.md](artifacts/index.md): entrypoint for checked-in evidence
- [docs/why_four_states.md](docs/why_four_states.md): conceptual explanation
- [docs/release_demo.md](docs/release_demo.md): intended public reading order

## How This Compares With Other Evals

This repo is not trying to replace general benchmark suites. It is doing a narrower job.

Compared with common eval styles:

| Eval style | Typical question | What this repo adds |
| --- | --- | --- |
| Pass/fail or leaderboard evals | "Which system scored higher?" | Distinguishes several kinds of "higher score" before turning that score into a control action. |
| Regression tests | "Did we break something obvious?" | Adds structured handling for unstable, gamed, and incomparable apparent wins. |
| Safety or adversarial evals | "Can the system fail in dangerous ways?" | Focuses on benchmark-aware decision errors inside the research loop itself. |
| Process or trajectory evals | "What happened during the run?" | Compresses process evidence into explicit `T/R/V` axes and resolver actions. |

The practical difference is that `Chatuskoti Evals` sits between a benchmark and a controller. It is not only measuring outcomes; it is deciding how those outcomes should change the next state of the loop.

## Scope And Limits

The current project should be understood as:

- benchmark-specific
- calibrated to `CIFAR-100 + ResNet-18`
- evaluation-first
- interpretable and reproducible

It should not yet be presented as:

- a universal eval layer for all AI agents
- proof that `Vec3` beats binary gates on every benchmark
- a learned, domain-general research controller
- a replacement for broader task suites or end-to-end agent benchmarks

That narrowness is a feature. The repo is trying to make one hard decision problem legible before generalizing it.

## Future Scope

Natural next steps for the repo include:

- expanding beyond one benchmark family while keeping strict benchmark definitions
- growing the canonical failure set, especially around harder `both` and `neither`-style cases
- comparing against a wider range of evaluator baselines, not only a binary gate
- learning or calibrating thresholds from accumulated offline evidence more systematically
- extending the checked-in V1.2 torch bundle with broader reruns or stronger hardware-backed replications
- testing whether the same decision framing transfers to agentic tool-use or code-generation benchmarks

That future would make the project more like a broader Chatuskoti-style eval benchmark. The current repo is the earlier and more concrete step: a benchmark-specific framework for making research-loop decisions less naive.
