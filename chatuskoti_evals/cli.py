from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chatuskoti_evals.config import AblationConfig, ExperimentConfig, LoopConfig
from chatuskoti_evals.runner import (
    run_ablation_bundle,
    run_calibration_bundle,
    run_comparison,
    run_failure_injection_set,
    run_single_loop,
)


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
    cfg = build_config(args)

    try:
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
    except ImportError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
