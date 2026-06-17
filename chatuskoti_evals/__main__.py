from __future__ import annotations

import time
from pathlib import Path

from chatuskoti_evals.core.config import ExperimentConfig
from chatuskoti_evals.evaluation.runner import run_ablation_bundle, run_failure_injection_set

OUTPUT_ROOT = Path("artifacts")


def _run() -> None:
    print()
    print("  Chatuskoti Evals  T/R/V evaluation framework")
    print("  " + "\u2500" * 50)

    cfg = ExperimentConfig()
    total_start = time.monotonic()

    # --- Failure injection set ---
    print()
    print("  [1/2] Failure Injection Set")
    print("  " + "\u2500" * 40)

    start = time.monotonic()
    results = run_failure_injection_set(OUTPUT_ROOT / "failure_set", cfg, seeds=3)
    elapsed = time.monotonic() - start

    matched = 0
    for r in results:
        ok = r.matched_expectation
        if ok:
            matched += 1
        icon = "\u2713" if ok else "\u2717"
        status = "PASS" if ok else "FAIL"
        print(f"  {icon}  {r.scenario_name:<30s}  expected={r.expected_resolution:<10s}  got={r.resolution.action:<10s}  {status}")
    print(f"  {matched}/{len(results)} cases matched  \u2022  {elapsed:.1f}s")

    # --- Ablation bundle ---
    print()
    print("  [2/2] Ablation Bundle")
    print("  " + "\u2500" * 40)

    start = time.monotonic()
    summaries = run_ablation_bundle(OUTPUT_ROOT / "ablation_bundle", cfg, seeds=3)
    elapsed = time.monotonic() - start

    print(f"  {'Variant':<20s}  {'Matched':<10s}  {'Mean T':<8s}  {'Mean R':<8s}  {'Mean V':<8s}")
    print("  " + "\u2500" * 60)
    for s in summaries:
        print(f"  {s.label:<20s}  {s.matched_expectations}/{s.total_cases:<5}  {s.mean_truthness:+.3f}  {s.mean_reliability:+.3f}  {s.mean_validity:+.3f}")
    print(f"  {len(summaries)} variants  \u2022  {elapsed:.1f}s")

    # --- Summary ---
    total_elapsed = time.monotonic() - total_start
    print()
    print("  " + "\u2500" * 50)
    print(f"  Done  \u2022  {total_elapsed:.1f}s total")
    print(f"  Output: {OUTPUT_ROOT.resolve()}")
    print()


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
