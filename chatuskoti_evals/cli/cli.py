from __future__ import annotations

import json
import argparse
from dataclasses import replace
from pathlib import Path

from chatuskoti_evals.actions import ACTION_INDEX
from chatuskoti_evals.config import AblationConfig, ExperimentConfig, LoopConfig
from chatuskoti_evals.runner import (
    run_ablation_bundle,
    run_calibration_bundle,
    run_comparison,
    run_failure_injection_set,
    run_lead_time_analysis,
    run_single_loop,
)


def _auto_or_float(val: str) -> float | str:
    if val.lower() == "auto":
        return "auto"
    return float(val)


def _auto_or_int(val: str) -> int | str:
    if val.lower() == "auto":
        return "auto"
    return int(val)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark-specific evaluator for research-loop outcomes on CIFAR-100 + ResNet-18"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Run both controllers and generate a comparison report")
    compare.add_argument("--output", type=Path, default=Path("artifacts/demo_run"))
    add_backend_args(compare)

    run_loop = subparsers.add_parser("run-loop", help="Run a single controller")
    run_loop.add_argument("--controller", choices=["vec3", "binary"], required=True)
    run_loop.add_argument("--output", type=Path, default=Path("artifacts/single_run"))
    add_backend_args(run_loop)

    failure_set = subparsers.add_parser("run-failure-set", help="Run the named failure injection set and generate a report")
    failure_set.add_argument("--output", type=Path, default=Path("artifacts/failure_set"))
    add_backend_args(failure_set)

    run_ablation = subparsers.add_parser("run-ablation", help="Run the canonical failure benchmark ablation bundle")
    run_ablation.add_argument("--output", type=Path, default=Path("artifacts/ablation_bundle"))
    add_backend_args(run_ablation)

    run_calibration = subparsers.add_parser(
        "run-calibration",
        help="Run a nearby-threshold calibration sweep over the canonical failure benchmark",
    )
    run_calibration.add_argument("--output", type=Path, default=Path("artifacts/calibration_bundle"))
    add_backend_args(run_calibration)

    lead_time = subparsers.add_parser(
        "lead-time",
        help="Run lead-time analysis: measure coupling warning lead over Goodhart pre-check",
        description="Runs a multi-step progressive trajectory, computes sliding-window coupling, "
                    "and measures lead time between the coupling warning and the snapshot Goodhart pre-check. "
                    "Use --cooldown 300 between iterations when using the torch backend (real training). "
                    "Progress is printed live with per-step TRV, coupling values, elapsed time, and ETA.",
    )
    lead_time.add_argument("--output", type=Path, default=Path("artifacts/lead_time"))
    lead_time.add_argument("--iterations", type=int, default=10)
    lead_time.add_argument("--action", default="stochastic_depth_high", choices=[a.name for a in ACTION_INDEX.values()])
    lead_time.add_argument("--window", type=_auto_or_int, default=5, help="Sliding window size, or 'auto' to select from data")
    lead_time.add_argument("--tau", type=_auto_or_float, default=0.4, help="Coupling warning threshold, or 'auto' for 15th-percentile")
    lead_time.add_argument("--seeds", type=int, default=1)
    lead_time.add_argument("--cooldown", type=float, default=0.0, help="Seconds to wait between iterations (default 0, suggest 300 for torch backend)")
    lead_time.add_argument("--backend", choices=["simulator", "torch"], default="simulator")
    lead_time.add_argument("--data-dir", type=Path, default=Path("data"))
    lead_time.add_argument("--device", default="auto")
    lead_time.add_argument("--epochs", type=int, default=3)
    lead_time.add_argument("--batch-size", type=int, default=128)
    lead_time.add_argument("--eval-batch-size", type=int, default=256)
    lead_time.add_argument("--num-workers", type=int, default=2)
    lead_time.add_argument("--mode", choices=["default", "calibration", "challenge"], default="default")
    lead_time.add_argument(
        "--ablation",
        choices=[
            "full", "no_reliability", "no_validity", "no_wisdom", "no_spread_gate",
            "no_coherence", "no_comparability", "no_goodhart",
            "t_only", "t_r", "t_v", "t_r_v",
        ],
        default="full",
    )

    traj_pred = subparsers.add_parser(
        "trajectory-prediction",
        help="Generate trajectory dataset, train predictors, and evaluate trajectory vs endpoint forecasting",
        description="Generates a dataset of (trajectory, action, next-delta) by running many simulator "
                    "loops, then compares two linear predictors (endpoint-only vs trajectory-aware) on "
                    "a held-out set split by trajectory. Reports MSE and runs an endpoint-matched pair test.",
    )
    traj_pred.add_argument("--output", type=Path, default=Path("artifacts/trajectory_prediction"))
    traj_pred.add_argument("--trajectories", type=int, default=50, help="Number of simulator trajectories to generate")
    traj_pred.add_argument("--iterations", type=int, default=6, help="Steps per trajectory")
    traj_pred.add_argument("--window", type=int, default=3, help="Trajectory window k (number of past Vec3 states)")
    traj_pred.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength")
    traj_pred.add_argument("--seeds", type=int, default=3, help="Seeds per trajectory step")
    traj_pred.add_argument("--backend", choices=["simulator", "torch"], default="simulator")
    traj_pred.add_argument("--data-dir", type=Path, default=Path("data"))
    traj_pred.add_argument("--device", default="auto")
    traj_pred.add_argument("--epochs", type=int, default=3)
    traj_pred.add_argument("--batch-size", type=int, default=128)
    traj_pred.add_argument("--eval-batch-size", type=int, default=256)
    traj_pred.add_argument("--num-workers", type=int, default=2)
    traj_pred.add_argument("--cooldown", type=float, default=0.0, help="Seconds to sleep every N trajectories (suggest 300 for torch backend)")
    traj_pred.add_argument("--cooldown-interval", type=int, default=2, help="Apply cooldown every N trajectories")
    traj_pred.add_argument("--mode", choices=["default", "calibration", "challenge"], default="default")
    traj_pred.add_argument(
        "--ablation",
        choices=[
            "full", "no_reliability", "no_validity", "no_wisdom", "no_spread_gate",
            "no_coherence", "no_comparability", "no_goodhart",
            "t_only", "t_r", "t_v", "t_r_v",
        ],
        default="full",
    )

    extract = subparsers.add_parser("extract-cases", help="Extract annotation cases from a torch artifact bundle for human rater labeling")
    extract.add_argument("--bundle", type=Path, required=True)
    extract.add_argument("--output", type=Path, default=Path("artifacts/annotation_cases.csv"))

    return parser


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["simulator", "torch"], default="simulator")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mode", choices=["default", "calibration", "challenge"], default="default")
    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "no_reliability",
            "no_validity",
            "no_wisdom",
            "no_spread_gate",
            "no_coherence",
            "no_comparability",
            "no_goodhart",
            "t_only",
            "t_r",
            "t_v",
            "t_r_v",
        ],
        default="full",
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    torch_cfg = replace(
        cfg.torch,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    ablation_cfg = AblationConfig(name=args.ablation)
    detector_cfg = ablation_cfg.apply(cfg.detector)
    return replace(
        cfg,
        backend=args.backend,
        torch=torch_cfg,
        detector=detector_cfg,
        ablation=AblationConfig(name=ablation_cfg.normalized_name),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "extract-cases":
            from chatuskoti_evals.annotation import extract_cases_to_csv
            path = extract_cases_to_csv(args.bundle, args.output)
            print(f"Wrote annotation cases to {path}")
            return

        cfg = build_config(args)

        if args.command == "compare":
            run_comparison(args.output, cfg, iterations=args.iterations, seeds=args.seeds, mode=args.mode)
            print(f"Wrote comparison artifacts to {args.output}")
            return
        if args.command == "run-loop":
            result = run_single_loop(
                cfg,
                LoopConfig(controller=args.controller, max_iterations=args.iterations, n_seeds=args.seeds, mode=args.mode),
                args.output,
            )
            print(f"Wrote {args.controller} artifacts to {result.output_dir}")
            return
        if args.command == "run-failure-set":
            results = run_failure_injection_set(args.output, cfg, seeds=args.seeds)
            print(f"Wrote failure injection artifacts to {args.output} ({len(results)} cases)")
            return
        if args.command == "run-ablation":
            summaries = run_ablation_bundle(args.output, cfg, seeds=args.seeds)
            print(f"Wrote ablation bundle to {args.output} ({len(summaries)} variants)")
            return
        if args.command == "run-calibration":
            summaries = run_calibration_bundle(args.output, cfg, seeds=args.seeds)
            print(f"Wrote calibration bundle to {args.output} ({len(summaries)} threshold profiles)")
            return
        if args.command == "lead-time":
            result = run_lead_time_analysis(
                args.output, cfg, seeds=args.seeds, iterations=args.iterations,
                goodhart_action=args.action, window=args.window, tau=args.tau,
                cooldown=args.cooldown,
            )
            lt = result.get("lead_time_steps")
            pre = result.get("precheck_step")
            warn = result.get("first_coupling_warning_step")
            print(f"Lead time: coupling warning at step {warn}, pre-check at step {pre}, lead = {lt} steps")
            print(f"Wrote lead-time analysis to {args.output}")
            return
        if args.command == "trajectory-prediction":
            from chatuskoti_evals.evaluation.trajectory_prediction import (
                evaluate_predictors,
                generate_trajectory_dataset,
                save_dataset,
                write_prediction_charts,
            )
            dataset = generate_trajectory_dataset(
                cfg,
                n_trajectories=args.trajectories,
                iterations=args.iterations,
                seeds_per_run=args.seeds,
                cooldown=args.cooldown,
                cooldown_interval=args.cooldown_interval,
            )
            result = evaluate_predictors(
                dataset,
                window=args.window,
                ridge_alpha=args.ridge_alpha,
            )
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "prediction_result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            save_dataset(dataset, args.output / "trajectory_dataset.json")
            chart_path = write_prediction_charts(result, args.output)
            _imp = result.get("improvement_pct", {})
            pair = result.get("pair_test", {})
            print(f"Trajectory prediction: endpoint MSE={result['endpoint_mse']['total']}, "
                  f"trajectory MSE={result['trajectory_mse']['total']}, "
                  f"improvement={_imp.get('total', 0)}%")
            if pair.get("n_pairs_found", 0) > 0:
                print(f"  Endpoint-matched pairs found: {pair['n_pairs_found']}")
            print(f"Wrote trajectory prediction artifacts to {args.output}")
            print(f"  Chart: {chart_path}")
            return
    except ImportError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
