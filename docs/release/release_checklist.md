# Release Checklist

## Pre-release verification

1. [ ] Run `python -m pytest tests/ -v` — all pass
2. [ ] Run `python -m chatuskoti_evals`
3. [ ] Verify 4/4 failure cases matched
4. [ ] Verify ablation shows representational collapse

## Paper readiness

- [ ] Scalar projection diagram created
- [ ] Action-collapse table created
- [ ] Distinguishability matrix created
- [ ] Ablation-as-representation analysis written
- [ ] Canonical cases documented (4 cases)
- [ ] Claims reviewed: no performance-comparison language
- [ ] README updated for representation framing
