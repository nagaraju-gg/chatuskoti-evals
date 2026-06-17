<div align="center">

# Chatuskoti Evals

**A three-axis evaluation representation (Truthness, Reliability, Validity) that preserves distinctions scalar evaluation necessarily destroys.**
</div>

> **One-liner:** Not every positive metric delta deserves the same control action.

Chatuskoti Evals is a benchmark-specific evaluation framework for research loops on CIFAR-100 + ResNet-18.

The repository explores a simple question:

> If two outcomes require different decisions, should they be represented as the same evaluation state?

Most evaluation systems answer a single question:

> Did the metric improve?

This repository argues that this is often insufficient for research-loop control because multiple action-distinct outcomes can share the same positive metric result.

The name *Chatuskoti* is inspired by the classical four-way logical framework. Here the idea is used operationally: outcomes that appear identical under scalar evaluation may require fundamentally different actions.

---

## The Problem: Representational Collapse

Most evaluation systems map outcomes to a single scalar.

| State           | T (Truthness) | Correct action |
| --------------- | ------------- | -------------- |
| Clean Pass      | +             | Accept         |
| Pyrrhic Pass    | +             | Hold           |
| Orthogonal Pass | +             | Reframe        |
| Paradox         | +             | Audit          |

All four states share the same positive truth signal but imply different decisions.

**T alone is not action-sufficient.**

A sufficient evaluation representation must distinguish states that require different actions. Scalar evaluation cannot—it collapses four distinct states into one bit.

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
4. `docs/design/paper_notes.md`

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
