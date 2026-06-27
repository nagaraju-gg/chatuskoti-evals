# Documentation

> 📖 Everything you need to understand, use, and extend Chatuskoti Evals.

---

## Getting Started

| Guide | Description |
|---|---|
| [Quickstart](usage/demo.md) | Walk through a full run |

## Design & Concepts

| Document | Description |
|---|---|
| [Why four states?](design/why_four_states.md) | Practical motivation with concrete failure cases |
| [Canonical failure benchmark](design/canonical_failure_benchmark.md) | 8 canonical cases, metrics, and Vec3 actions |
| [Distinguishability matrix](design/distinguishability_matrix.md) | Formal sufficiency criterion with state-pair analysis |
| [Scalar projection diagram](figures/scalar_projection.svg) | Visual: Vec3 cube collapsed onto T axis |
| [Vec3 prototype notes](design/vec3_prototype_notes.md) | Framing guide, figures to create, language to avoid |

## Releases & Roadmap

| Document | Description |
|---|---|
| [Failure injection set](release/failure_injection_set.md) | The 4 canonical cases and scope/claim boundaries |
| [Paper figure 1](release/paper_figure_1.md) | Proposed layout for the action-collapse figure |
| [Next steps](release/next_steps.md) | Paper prep tasks and technical todos |
| [Release checklist](release/release_checklist.md) | Pre-release and paper readiness checklist |
| [CHANGELOG](../CHANGELOG.md) | Version history |

## Reference

| File | Description |
|---|---|
| [README](../README.md) | Project overview, install, architecture |
| [CONTRIBUTING](../CONTRIBUTING.md) | How to contribute |
| [SECURITY](../SECURITY.md) | Security policy |
| [CITATION.cff](../CITATION.cff) | Citation metadata |

## Architecture

```
chatuskoti_evals/
├── core/           # Vec3, configs, coupling, wisdom
├── evaluation/     # Scoring, resolver, actions, scenarios, runner, reporting
└── __main__.py     # Entry point
```

See [README → architecture section](../README.md#7-repo-architecture) for details.
