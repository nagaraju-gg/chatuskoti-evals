# Chatuskoti Eval Framework

Four-state AI eval framework for research outcomes and benchmark decisions. It catches pyrrhic gains, gamed metrics, broken runs, and invalid comparisons that binary pass/fail misses, calibrated on `CIFAR-100 + ResNet-18`.

Modern AI evals usually assume every question has one correct answer and every output can be scored on a single pass/fail axis. That works for straightforward facts, but it breaks down in the real world.

Some questions are:

- true
- false
- both, because evidence or interpretation conflicts
- neither, because the question is ill-posed or not meaningfully evaluable

That is the intuition behind Gautama's [Chatuskoti](https://en.wikipedia.org/wiki/Catu%E1%B9%A3ko%E1%B9%ADi), an ancient Indian logical system that extends simple true/false reasoning into a four-way lens.

Here is the basic motivation:

| Question | Correct mode | Why |
| --- | --- | --- |
| "Paris is the capital of France" | True | Plain factual statement |
| "Paris is the capital of Germany" | False | Plain factual error |
| "Light is a wave" | Both | Useful in one frame, incomplete in another |
| "What is the color of number 7?" | Neither | Ill-posed category error |

Most benchmarks are good at the first two and weak on the latter two. The same failure shows up in research loops: not every "gain" is the same kind of gain.

In this repo, the Chatuskoti idea is not implemented as four hard-coded labels. Instead, it is decomposed into three evaluation axes:

| Axis | What it asks | Why this decomposition matters for four-state logic |
| --- | --- | --- |
| `truthness` | Did the benchmark really improve? | Preserves the ordinary true/false question instead of throwing it away. |
| `coherence` | Is the result internally stable and consistent enough to trust? | Separates ordinary improvement from contradiction-like cases where evidence or behavior pulls in opposing directions. |
| `comparability` | Is the before/after comparison even valid? | Separates meaningful evaluation from cases that should be treated as neither cleanly true nor false because the comparison itself broke down. |

Those axes generate the practically important outcome types that binary evaluation collapses together:

- clean gain
- pyrrhic gain
- gamed gain
- broken result
- incomparable result

`goodhart_score` is kept separate to catch suspicious metric gains that are still technically comparable but likely won through proxy gaming or decoupling.

That is why this project uses a Chatuskoti-inspired evaluation logic instead of a single binary accept/reject rule.

## What this repo proves today

This repo is a benchmark-specific calibrated evaluation framework over `CIFAR-100 + ResNet-18`, with:

- typed intervention proposals
- deterministic `Vec3` scoring
- explicit resolver actions: `adopt`, `reject`, `hold`, `rollback`, `reframe`, `keep_going`
- machine-readable histories, plots, manifests, and markdown reports

The current public evidence package shows that a metric-only controller can merge benchmark-aware bad cases that a structured evaluator should treat differently.

## Quickstart

Environment check:

```bash
.venv/bin/python scripts/check_torch_env.py
```

Regenerate the strongest real benchmark:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-failure-set \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/canonical_failure
```

Run the benchmark-aware open-loop companion:

```bash
.venv/bin/python -m chatuskoti_evals.cli compare \
  --mode challenge \
  --backend torch \
  --epochs 10 \
  --iterations 4 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/challenge_compare
```

Run the failure-benchmark ablation sweep:

```bash
.venv/bin/python -m chatuskoti_evals.cli run-ablation \
  --backend torch \
  --epochs 10 \
  --seeds 3 \
  --num-workers 0 \
  --output artifacts/strong_v1/ablations
```

Or regenerate the whole release bundle:

```bash
bash scripts/run_torch_release_bundle.sh
```

## Why four states beat binary eval

| Case | What happened | Binary eval | Chatuskoti Eval Framework | Why it matters |
| --- | --- | --- | --- | --- |
| Clean gain | Metric improves and internals stay sane | `adopt` | `adopt` | Good changes should still flow through quickly |
| Pyrrhic gain | Metric rises while instability widens | `adopt` | `hold` | Prevents shipping seductive but unhealthy updates |
| Gamed gain | Metric rises while proxy alignment collapses | `adopt` | `reject` | Catches Goodhart-style false positives |
| Broken result | Metric drops and internals are damaged | `reject` | `rollback` | Separates passive rejection from active damage control |
| Incomparable gain | Eval protocol changed | `adopt` | `reframe` | Stops invalid leaderboard wins from being treated as real progress |

The point is not “more philosophy.” The point is that a single number often throws away the structure that tells you what to do next.

## Current evidence bundle

The public-facing evidence bundle is organized under [artifacts/strong_v1](artifacts/strong_v1).

The checked-in bundle is a curated torch-backed benchmark package built from the strongest current saved runs in the repo. The release script regenerates the same bundle in a stronger `3`-seed form on a faster machine.

- canonical failure benchmark: [summary.md](artifacts/strong_v1/canonical_failure/failure_injection/summary.md)
  This is the main result. It shows how the framework separates pyrrhic, gamed, broken, and incomparable cases that binary evaluation collapses.
- canonical figure: [benchmark_figure.svg](artifacts/strong_v1/canonical_failure/failure_injection/benchmark_figure.svg)
  This is the fastest visual summary of the failure benchmark.
- challenge comparison: [comparison.md](artifacts/strong_v1/challenge_compare/comparison.md)
  This is the companion run that shows binary can post a higher metric by merging benchmark-aware invalid cases.
- challenge case table: [challenge_cases.md](artifacts/strong_v1/challenge_compare/challenge_cases.md)
  This breaks down the cases that `binary` adopted and `Vec3` blocked.
- ablation summary: [summary.md](artifacts/strong_v1/ablations/summary.md)
  This shows which axes matter by removing `coherence`, `comparability`, `goodhart_score`, `wisdom`, and spread gating one at a time.
- artifact landing page: [index.md](artifacts/index.md)
- plain-language explanation: [why_four_states.md](docs/why_four_states.md)
- guided walkthrough: [demo.md](docs/demo.md)

## Headline result

The canonical failure benchmark is the main result to lead with.

- binary evaluation would adopt the pyrrhic, gamed, and incomparable benchmark cases
- `Chatuskoti Eval Framework` routes those same cases to `hold`, `reject`, and `reframe`
- the damaged failure case is escalated to `rollback`, not just passively rejected

The benchmark-aware `challenge` comparison is companion evidence:

- binary can post the higher metric by accepting benchmark-aware invalid merges
- `Vec3` preserves structural validity even when that means refusing superficially better numbers

## What gets written

Each controller or benchmark bundle emits:

- `history.jsonl`
- `seed_metrics.json`
- `summary.md`
- SVG plots
- aggregate summaries
- `manifest.json`

The stronger public bundles also emit:

- challenge case tables
- benchmark figures
- ablation summaries
- top-level artifact index pages

## Package and CLI

The Python package is now `chatuskoti_evals`.

- module entrypoint: `python -m chatuskoti_evals.cli`
- project name: `chatuskoti-evals`

The most useful commands are:

- `compare`
- `run-loop`
- `run-failure-set`
- `run-ablation`

## Scope and claims

This repo should be presented as:

- benchmark-specific
- calibrated to `CIFAR-100 + ResNet-18`
- evaluation-first
- reproducible and interpretable

This repo should not yet be presented as:

- a general automated researcher
- universally calibrated across domains
- proof that `Vec3` always beats binary controllers on any benchmark

## Future: full Chatuskoti eval benchmark

This repo is not yet a full general-purpose Chatuskoti eval benchmark in the strict sense.

It would cross that line when it has:

1. locked task definitions with strict output schemas
2. a larger high-quality dataset, including adversarial `both` and `neither` cases
3. deterministic scoring and formalized rubrics
4. published baseline model comparisons
5. clone-and-run reproducibility with stable headline numbers

That is a natural future for this repo. The current version is a benchmark-specific calibrated evaluation framework that moves in that direction.

## Repo structure

- [chatuskoti_evals/benchmark.py](chatuskoti_evals/benchmark.py)
- [chatuskoti_evals/scoring.py](chatuskoti_evals/scoring.py)
- [chatuskoti_evals/resolver.py](chatuskoti_evals/resolver.py)
- [chatuskoti_evals/proposals.py](chatuskoti_evals/proposals.py)
- [chatuskoti_evals/reporting.py](chatuskoti_evals/reporting.py)
- [docs/why_four_states.md](docs/why_four_states.md)
- [docs/demo.md](docs/demo.md)
- [docs/release_checklist.md](docs/release_checklist.md)

## Test suite

```bash
python3 -m unittest discover -s tests -v
```
