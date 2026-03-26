# Why Four States

Most evaluation systems make one coarse distinction: improvement or no improvement.

That is too lossy for research loops. A model update can improve the reported metric while still being the wrong thing to merge. `Chatuskoti Eval Framework` separates those cases so the control decision matches the kind of result we actually observed.

## The five practical cases

| Case | Short version | Resolver action |
| --- | --- | --- |
| Clean gain | Better metric, healthy internals, valid comparison | `adopt` |
| Pyrrhic gain | Better metric, worse training dynamics | `hold` |
| Gamed gain | Better metric, suspicious proxy behavior | `reject` |
| Broken result | Worse metric, obvious damage | `rollback` |
| Incomparable gain | Better metric under a changed eval regime | `reframe` |

The important shift is this:

- binary evaluation asks “did the number go up?”
- four-state evaluation asks “what kind of outcome is this, and what action fits it?”

## How the scoring works

`Chatuskoti Eval Framework` scores each outcome along:

| Axis | What it asks | Why it helps resolve the four-state lens |
| --- | --- | --- |
| `truthness` | Did the anchored benchmark metric really improve? | Keeps the basic true/false direction of the result visible. |
| `coherence` | Did internals remain healthy? | Distinguishes stable results from contradiction-like cases where the metric and the internal story diverge. |
| `comparability` | Is this result even comparable to the baseline? | Separates normal evaluation from cases that are effectively neither cleanly true nor false because the comparison itself is invalid. |
| `goodhart_score` | Did the metric move for suspicious reasons? | Catches metric-gaming behavior that can look superficially true on the top-line number alone. |

That extra structure turns evaluation into control logic. A pyrrhic gain should not be treated like a clean gain, and an incomparable gain should not even be treated as a valid before/after result.

## Why this matters for research loops

In real loops, many bad decisions do not look bad at first:

- the metric gets a little better
- the training process gets less stable
- the proxy metrics decouple
- the evaluation quietly changes

If the controller only sees the top-line number, it will merge exactly the kinds of results that later become regressions, misleading wins, or wasted search effort.

That is what this repo is trying to fix.
