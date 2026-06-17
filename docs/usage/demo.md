# Demo Walkthrough

This demo walks through the representation argument using the simulator backend.

```bash
# 1. Run everything
.venv/bin/python -m chatuskoti_evals

# 2. Check the results
cat artifacts/failure_set/failure_injection/summary.md
cat artifacts/ablation_bundle/summary.md
```

## What to look for

- The failure case report shows 4 cases where T > 0 but actions differ
- The ablation report shows that removing axes causes representational collapse (fewer cases matched)
