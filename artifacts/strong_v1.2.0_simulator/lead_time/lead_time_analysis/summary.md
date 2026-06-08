# Lead-Time Analysis

- Total steps observed: `10`
- Coupling window: `4`, coupling threshold τ: `{'tv': -1.0, 'tr': -1.0}`
- Goodhart pre-check fired at step: `0`
- First coupling warning at window end: `4`
- Lead time (steps): `-4`

## Per-step Deltas

| Step | ΔT | ΔR | ΔV |
| --- | --- | --- | --- |
| 1 | `-0.24337` | `0.41409` | `0.51326` |
| 2 | `0.00000` | `0.02765` | `0.18674` |
| 3 | `0.00000` | `0.08892` | `0.10869` |
| 4 | `0.00000` | `0.02634` | `0.05143` |
| 5 | `0.00000` | `0.00000` | `0.00000` |
| 6 | `0.00000` | `0.00000` | `0.00000` |
| 7 | `0.00000` | `0.00000` | `0.00000` |
| 8 | `0.00000` | `0.00000` | `0.00000` |
| 9 | `0.00000` | `0.00000` | `0.00000` |

## Sliding-Window Coupling

| Window End | T-V Coupling | T-R Coupling | Goodhart Warning | Pyrrhic Warning |
| --- | --- | --- | --- | --- |
| 4 | `-1.00000` | `-1.00000` | 🚨 | 🚨 |
| 5 | `0.00000` | `0.00000` | — | — |
| 6 | `0.00000` | `0.00000` | — | — |
| 7 | `0.00000` | `0.00000` | — | — |
| 8 | `0.00000` | `0.00000` | — | — |
| 9 | `0.00000` | `0.00000` | — | — |

**Result**: No measurable lead time — coupling warning did not precede Goodhart pre-check.
