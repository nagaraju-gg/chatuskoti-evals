"""Generate a distinguishability matrix from the canonical failure case set.

Usage:
    python3 scripts/generate_distinguishability_matrix.py

Output:
    Prints a markdown table showing which state pairs are collapsed by scalar
    evaluation but distinguished by Vec3.
"""

from chatuskoti_evals.core.config import DetectorConfig, ExperimentConfig
from chatuskoti_evals.evaluation.actions import ACTION_INDEX
from chatuskoti_evals.evaluation.benchmark import SimulatedCIFAR100ResNet18Adapter
from chatuskoti_evals.evaluation.resolver import resolve_vec3
from chatuskoti_evals.evaluation.scenarios import FAILURE_INJECTION_SET
from chatuskoti_evals.evaluation.scoring import score_run_metrics


def collect_states():
    experiment_cfg = ExperimentConfig()
    adapter = SimulatedCIFAR100ResNet18Adapter(experiment_cfg.simulation)
    baseline = adapter.record_baseline([0, 1, 2])
    detector_cfg = DetectorConfig()
    states = []
    for scenario in FAILURE_INJECTION_SET:
        action = ACTION_INDEX[scenario.action_name]
        candidate_metrics, _ = adapter.execute(action, [0, 1, 2])
        run_score, _ = score_run_metrics(candidate_metrics, baseline.metrics, detector_cfg)
        resolution = resolve_vec3(run_score, detector_cfg)
        t = run_score.mean.truthness
        t_bin = "+" if t >= 0 else "-"
        is_pass = resolution.action == "adopt"
        states.append({
            "name": scenario.name,
            "scenario": scenario,
            "run_score": run_score,
            "resolution": resolution,
            "t": t,
            "t_bin": t_bin,
            "is_pass": is_pass,
            "t_r_v": (t, run_score.mean.reliability, run_score.mean.validity),
        })
    return states


def main() -> int:
    states = collect_states()
    n = len(states)

    print("# Distinguishability Matrix\n")
    print("Generated from canonical failure case set.\n")
    print("| State A | State B | Same T? | Same Binary? | Same Action? | Vec3 Distinguishes? |")
    print("| --- | --- | --- | --- | --- | --- |")

    for i in range(n):
        for j in range(i + 1, n):
            a = states[i]
            b = states[j]
            same_t = a["t_bin"] == b["t_bin"]
            same_binary = a["is_pass"] == b["is_pass"]
            same_action = a["resolution"].action == b["resolution"].action
            vec3_diff = a["t_r_v"] != b["t_r_v"]

            a_name = a["name"].replace("_injection", "").replace("_", " ").title()
            b_name = b["name"].replace("_injection", "").replace("_", " ").title()

            print(
                f"| {a_name} | {b_name} | {'Yes' if same_t else 'No'}"
                f" | {'Yes' if same_binary else 'No'}"
                f" | {'No' if same_action else 'Yes'}"
                f" | {'Yes' if vec3_diff else 'No'} |"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
