# Paper Figure 1: Action-Collapse Under Scalar Evaluation

## Concept

Show four states with identical T values that require different resolver actions.

## Suggested layout

Left side: Vec3 cube with four points highlighted (Clean, Pyrrhic, Orthogonal, Paradox).
Right side: Projection onto T axis, showing all four merge to one point.

Below: Table mapping T value to required action.

```
Vec3 space:                    Scalar projection:

  (+,+,+) Clean               PASS: ● (merged)
     ●
       (-,+,+) Pyrrhic
         ●
           (+,-,+) Orthogonal
             ●
               (-,-,+) Paradox
                 ●
```

All four states have T > 0, yet:

| State | T | Required Action |
|---|---|---|
| Clean Pass | + | Accept |
| Pyrrhic Pass | + | Hold |
| Orthogonal Pass | + | Reframe |
| Paradox | + | Rollback |

**Caption:** Scalar evaluation collapses four decision-relevant states into one "pass" category. Vec3 preserves the distinctions required for correct research-loop decisions.
