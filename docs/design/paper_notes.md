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
