# Distinguishability Matrix

The distinguishability matrix formalizes the claim that scalar evaluation (T-only) collapses states that require different actions.

## Criterion

A representation R is **sufficient** for decision set D if states requiring different D-actions remain distinguishable under R.

## Matrix

| State A | State B | Same T? | Same Binary Outcome? | Same Action Required? | Vec3 Distinguishes? |
|---|---|---|---|---|---|
| Clean Pass | Pyrrhic Pass | Yes | Yes | No | Yes |
| Clean Pass | Orthogonal Pass | Yes | Yes | No | Yes |
| Clean Pass | Paradox | Yes | Yes | No | Yes |
| Pyrrhic Pass | Orthogonal Pass | Yes | Yes | No | Yes |
| Pyrrhic Pass | Paradox | Yes | Yes | No | Yes |
| Orthogonal Pass | Paradox | Yes | Yes | No | Yes |
| Clean Failure | Broken Failure | Yes | Yes | No | Yes |

## Key observations

Every row has:
- `Same T?` = Yes (scalar evaluation cannot separate them)
- `Same Binary Outcome?` = Yes (both would be treated as "pass" or "fail")
- `Same Action Required?` = No (different resolver actions needed)
- `Vec3 Distinguishes?` = Yes (T/R/V profile differs)

## Formal statement

Let f(s) be the scalar evaluation of state s, and g(s) be the Vec3 evaluation.

For any pair of states (A, B) in the canonical set that require different actions:

f(A) = f(B)   (scalar collapses them)
g(A) ≠ g(B)   (Vec3 preserves the distinction)

Therefore scalar evaluation is not action-sufficient for the research-loop decision set D.
