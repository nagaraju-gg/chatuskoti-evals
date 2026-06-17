# Why Four States

Most evaluation systems make one coarse distinction: improvement or no improvement.

That is too lossy for research loops. A model update can improve the reported metric while still being the wrong thing to merge. Vec3 separates those cases so the control decision matches the kind of result actually observed.

## The five practical cases

| Case | Short version | Resolver action |
| --- | --- | --- |
| Clean gain | Better metric, healthy internals, valid comparison | `adopt` |
| Pyrrhic gain | Better metric, worse training dynamics | `hold` |
| Orthogonal gain | Better metric, suspicious proxy behavior | `reframe` |
| Broken result | Worse metric, obvious damage | `rollback` |
| Incomparable gain | Better metric under a changed eval regime | `reframe` |

The important shift is this:

- binary evaluation asks "did the number go up?"
- Vec3 evaluation asks "what kind of outcome is this, and what action fits it?"

## How the scoring works

Vec3 scores each outcome along three non-redundant axes:

| Axis | What it asks | Why it helps |
| --- | --- | --- |
| `truthness` (`T`) | Did the anchored metric really improve? | Keeps the basic direction visible |
| `reliability` (`R`) | Did internals remain healthy? | Distinguishes stable from pyrrhic |
| `validity` (`V`) | Is this result meaningful? | Separates real progress from gamed/invalid wins |

That extra structure turns evaluation into action mapping. A pyrrhic gain should not be treated like a clean gain, and an incomparable gain should not be treated as a decision-ready result.

## Why this matters

In real loops, many bad decisions do not look bad at first:

- the metric gets a little better
- the training process gets less stable
- the proxy metrics decouple
- the evaluation quietly changes

If the controller only sees the top-line number, it will merge exactly the kinds of results that later become regressions or misleading wins.

That is what Vec3 aims to fix — not by improving the metric, but by preserving the distinctions scalar evaluation loses.
