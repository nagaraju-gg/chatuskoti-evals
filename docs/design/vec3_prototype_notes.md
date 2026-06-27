# Paper revision notes for V1

## Core framing

The paper should be framed as a **representation** argument, not a benchmark/performance comparison.

### Argument structure

1. **Representation Sufficiency Criterion**: A representation is sufficient for decision set D if states requiring different D-actions remain distinguishable.
2. **Observation**: The four canonical states (Clean Pass, Pyrrhic Pass, Orthogonal Pass, Paradox) all share T > 0 but require different actions.
3. **Conclusion**: T (scalar evaluation) is not sufficient for research-loop decisions.
4. **Proposal**: Vec3 (T/R/V) restores the lost distinctions.
5. **Evidence**: Ablation proves that removing axes causes the predicted representational collapse.

### Paper structure (recommended flow)

1. Research-loop decision problem
2. Sufficiency criterion for evaluation representations
3. Scalar collapse observation
4. Vec3 representation
5. Projection and action-collapse figure
6. Resolver
7. Implementation (Chatuskoti)
8. Canonical cases (as consequences, not primary evidence)
9. Ablations (as representation analysis)

### Key figures

1. **Scalar projection diagram**: Vec3 cube projected onto T line — multiple states collapse to one bit
2. **Action-collapse table**: 4 states with identical T requiring 4 different actions
3. **Distinguishability matrix**: formalizes which state pairs are merged by scalar but separated by Vec3
4. **Ablation cascade**: removing R, V, or both causes progressive state collapse

### What to avoid

- Avoid "Vec3 beats binary" or "Vec3 improves benchmarks" language
- Avoid lead time, trajectory prediction, or predictive performance claims
- Avoid calibration sweeps as primary evidence
- Avoid any claim about general superiority without explicit scope

## Implementation compatibility

The implementation should be described as an expanded Vec3 evaluator rather than as two separate modes. The older Chatuskoti behavior
is a subset obtained by the following policies:

- Truthness: absolute metric delta scaled by `truth_delta_scale`
- R/V aggregation: weighted detector blend
- Disabled axes: explicit imputation, historically `0.75`
- Resolver: staged instrument classification with the historical action priority

The paper-aligned behavior is obtained by different policies inside the same evaluator:

- Truthness: relative metric delta scaled by `relative_truth_scale = 4`
- R/V aggregation: worst-case detector aggregation
- Disabled axes: `undefined`, preserving partial states instead of projecting them to a point
- Thresholds: set to the paper priors where the paper uses concrete values

Reports and run logs should expose axis status so imputed and undefined coordinates are never silently confused with measured coordinates.

`configs/vec3_prototype.json` is the pinned Vec3 prototype configuration. It should be cited for reproducibility rather than asking readers to infer
the prototype settings from the working default.

The Instrument Tradeoff overlay is implemented as the paper's trajectory check: mean sign of `Delta R * Delta V` over a rolling window
of `n = 5`, firing when the coupling is `<= -0.4`. This is distinct from the Goodhart structural pre-check, which remains only partially
represented by detector signals and should still be caveated unless implemented as its own named structural-distance check.
