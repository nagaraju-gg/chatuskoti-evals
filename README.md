# Chatuskoti Evals

**A three-axis evaluation framework for research-loop decisions.**

> Not every positive metric delta deserves the same control action.

Chatuskoti Evals explores a simple representational question:

> If two outcomes require different decisions, should they be represented as the same evaluation state?

Most evaluation systems reduce outcomes to a single number. For many research-loop decisions, that is enough. But some outcomes share the same metric result while requiring different actions.

A gain may be:

* genuinely useful,
* unstable and not yet trustworthy,
* produced by exploiting a proxy,
* or incomparable to the baseline altogether.

Scalar evaluation collapses these into the same outcome. Chatuskoti Evals represents them separately using three axes:

* **Truthness (T):** Did the metric improve?
* **Reliability (R):** Is the result stable and reproducible?
* **Validity (V):** Is the comparison meaningful?

The framework is inspired by *Chatuskoti*, a classical four-way logical framework. Here the inspiration is practical rather than philosophical: outcomes that look identical under a scalar evaluation may require different decisions.

The central claim of this repository is straightforward:

> An evaluation representation should preserve distinctions that matter for action.

---

## The Vec3 Representation

Chatuskoti Evals represents outcomes using three non-redundant axes:

| Axis        | Symbol | Question                                |
| ----------- | ------ | --------------------------------------- |
| Truthness   | `T`    | Did the anchored metric improve?        |
| Reliability | `R`    | Was the result stable and reproducible? |
| Validity    | `V`    | Is the gain meaningful and comparable?  |

These axes define eight distinguishable outcome regions.

A scalar representation distinguishes only two.

### Expanded Axis State

The implementation now treats a Vec3 result as more than three bare floats. Each axis carries a status:

* `measured`: computed from detector evidence
* `imputed`: intentionally filled by policy, such as the historical `0.75` value for a disabled axis
* `undefined`: absent from the state, not equivalent to zero or any other coordinate

This makes the earlier implementation a subset of the expanded evaluator. The historical behavior is recovered by using
absolute-delta truthness, weighted R/V aggregation, and imputed disabled axes. Paper-aligned behavior can be expressed by choosing
relative-delta truthness, worst-case R/V aggregation, paper thresholds, and undefined disabled axes.

The Vec3 prototype configuration is pinned in `configs/vec3_prototype.json` and exposed in Python as `vec3_prototype_detector_config()`.

### Resolver Actions

Vec3 maps outcome states to controller actions:

| Outcome Type      | Action       |
| ----------------- | ------------ |
| Clean gain        | `adopt`      |
| Pyrrhic gain      | `hold`       |
| Metric-gamed gain | `reframe`    |
| Incomparable gain | `reframe`    |
| Broken result     | `rollback`   |
| No clear signal   | `keep_going` |

The central claim is representational:

> Evaluation representations should be judged by the decisions they support, not only by the metrics they summarize.

---

## Evidence

The repository contains canonical failure cases demonstrating that scalar evaluation merges outcomes requiring different actions.

| Case              | Binary Action | Vec3 Action |
| ----------------- | ------------- | ----------- |
| Pyrrhic Pass      | Adopt         | Hold        |
| Orthogonal Pass   | Adopt         | Reframe     |
| Incomparable Pass | Adopt         | Reframe     |
| Broken Failure    | Reject        | Rollback    |

**Result:** 4/4 canonical failure cases classified correctly. Binary evaluation would adopt 3/4 of these failure cases; Vec3 rejects or defers all 4.

### Ablation Results

Removing axes causes previously distinct states to collapse.

| Configuration | Matched Cases |
| ------------- | ------------- |
| T + R + V     | 4/4           |
| T + R         | 2/4           |
| T + V         | 2/4           |
| T only        | 0/4           |

This is a representation argument, not a performance argument. Removing axes reduces distinguishability capacity.

---

## What This Repository Contains

* Vec3 (Truthness, Reliability, Validity) evaluation
* Resolver actions (`adopt`, `hold`, `reframe`, `rollback`, `keep_going`)
* Canonical failure benchmarks
* Representational ablations
* CIFAR-100 / ResNet-18 demonstration environment
* Reporting and artifact generation
* Reproducible research bundles

---

## Quick Start

```bash
git clone https://github.com/your-org/chatuskoti-evals.git
cd chatuskoti-evals

python -m venv .venv
source .venv/bin/activate

pip install -e .

python -m chatuskoti_evals
```

Artifacts are written to `artifacts/`.

---

## Reading Guide

1. `docs/design/why_four_states.md`
2. Run `python -m chatuskoti_evals`
3. `docs/design/distinguishability_matrix.md`
4. `docs/design/vec3_prototype_notes.md`

---

## Scope

### This repository is

* A demonstration that scalar evaluation is representationally insufficient for some research-loop decisions
* A concrete implementation of a three-axis evaluation representation
* A benchmark-specific research artifact

### This repository is not

* A proof that Vec3 is optimal for all domains
* A universal evaluation framework
* A large-scale benchmark comparison
* A general autonomous researcher

---

## Citation

```bibtex
@software{chatuskoti_evals2026,
  title = {Chatuskoti Evals: A Three-Axis Representation for Research-Loop Decisions},
  author = {G, Nagaraju},
  year = {2026},
  version = {1.3.0}
}
```

## License

MIT License.
