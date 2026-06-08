from __future__ import annotations

import unittest

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import RunMetrics, RunScore, Vec3
from chatuskoti_evals.resolver import adaptive_detector_config, resolve_vec3
from chatuskoti_evals.scoring import score_run_metrics


def make_metrics(
    *,
    run_id: str,
    primary_metric: float,
    train_loss: float,
    val_loss: float,
    train_val_gap: float,
    grad_norm_mean: float,
    grad_norm_std: float,
    eval_hash: str = "baseline",
    objective_family: str = "cross_entropy",
    proxy_corr: float = 0.86,
) -> RunMetrics:
    return RunMetrics(
        run_id=run_id,
        seed=0,
        primary_metric=primary_metric,
        train_loss=train_loss,
        val_loss=val_loss,
        train_val_gap=train_val_gap,
        grad_norm_mean=grad_norm_mean,
        grad_norm_std=grad_norm_std,
        weight_distance=0.2,
        param_count=11_200_000,
        eval_hash=eval_hash,
        model_family="resnet18",
        objective_family=objective_family,
        proxy_metrics={"proxy_metric_corr": proxy_corr, "calibration": max(0.0, proxy_corr - 0.05)},
        detector_inputs={},
    )


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DetectorConfig()
        self.baseline = make_metrics(
            run_id="baseline",
            primary_metric=0.636,
            train_loss=1.45,
            val_loss=1.63,
            train_val_gap=0.18,
            grad_norm_mean=2.1,
            grad_norm_std=0.11,
        )

    def score(self, metric: RunMetrics):
        run_score, _ = score_run_metrics([metric, metric, metric], self.baseline, self.cfg)
        return run_score

    def test_clean_win_routes_to_adopt(self) -> None:
        metric = make_metrics(
            run_id="clean-win",
            primary_metric=0.671,
            train_loss=1.36,
            val_loss=1.49,
            train_val_gap=0.13,
            grad_norm_mean=1.9,
            grad_norm_std=0.10,
        )
        run_score = self.score(metric)
        self.assertGreater(run_score.mean.truthness, 0.25)
        self.assertGreater(run_score.mean.reliability, 0.0)
        self.assertGreater(run_score.mean.validity, 0.0)
        resolution = resolve_vec3(run_score, self.cfg)
        self.assertEqual(resolution.action, "adopt")
        self.assertIn("exceeds adopt threshold", resolution.reason)

    def test_pyrrhic_win_routes_to_hold(self) -> None:
        metric = make_metrics(
            run_id="pyrrhic",
            primary_metric=0.668,
            train_loss=1.30,
            val_loss=1.72,
            train_val_gap=0.42,
            grad_norm_mean=2.0,
            grad_norm_std=0.10,
        )
        run_score = self.score(metric)
        self.assertIn("instability_gap", run_score.fired_signals)
        self.assertLess(run_score.mean.reliability, self.cfg.reliability_threshold)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "hold")

    def test_metric_gaming_win_routes_to_reframe(self) -> None:
        metric = make_metrics(
            run_id="goodhart",
            primary_metric=0.669,
            train_loss=1.32,
            val_loss=1.54,
            train_val_gap=0.22,
            grad_norm_mean=1.7,
            grad_norm_std=0.02,
            proxy_corr=0.42,
        )
        run_score = self.score(metric)
        self.assertIn("proxy_decoupling", run_score.fired_signals)
        self.assertLess(run_score.mean.validity, self.cfg.validity_threshold)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "reframe")

    def test_incomparable_routes_to_reframe(self) -> None:
        metric = make_metrics(
            run_id="incomparable",
            primary_metric=0.649,
            train_loss=1.39,
            val_loss=1.57,
            train_val_gap=0.18,
            grad_norm_mean=1.95,
            grad_norm_std=0.10,
            eval_hash="changed",
            objective_family="focal_loss",
        )
        run_score = self.score(metric)
        self.assertLess(run_score.mean.validity, self.cfg.validity_threshold)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "reframe")

    def test_broken_failure_routes_to_rollback(self) -> None:
        metric = make_metrics(
            run_id="broken",
            primary_metric=0.580,
            train_loss=1.68,
            val_loss=1.96,
            train_val_gap=0.28,
            grad_norm_mean=5.9,
            grad_norm_std=0.32,
        )
        run_score = self.score(metric)
        self.assertLess(run_score.mean.truthness, -self.cfg.adopt_truth_threshold)
        self.assertLess(run_score.mean.reliability, self.cfg.reliability_threshold)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "rollback")

    def test_noisy_lucky_seed_routes_to_keep_going(self) -> None:
        metrics = [
            make_metrics(
                run_id=f"noisy-{index}",
                primary_metric=value,
                train_loss=1.38,
                val_loss=1.55,
                train_val_gap=0.17,
                grad_norm_mean=1.95,
                grad_norm_std=0.09,
            )
            for index, value in enumerate([0.70, 0.61, 0.67])
        ]
        run_score, _ = score_run_metrics(metrics, self.baseline, self.cfg)
        self.assertGreater(run_score.spread, self.cfg.max_spread)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "keep_going")

    def test_positive_metric_with_isolated_grad_mean_spike_can_still_adopt(self) -> None:
        metric = make_metrics(
            run_id="adamw-like",
            primary_metric=0.670,
            train_loss=1.31,
            val_loss=1.45,
            train_val_gap=0.14,
            grad_norm_mean=11.8,
            grad_norm_std=0.14,
            proxy_corr=0.88,
        )
        run_score = self.score(metric)
        self.assertNotIn("exploding_gradients", run_score.fired_signals)
        self.assertGreater(run_score.mean.reliability, 0.0)
        self.assertEqual(resolve_vec3(run_score, self.cfg).action, "adopt")

    def test_logs_axis_components_for_reliability_and_validity(self) -> None:
        metric = make_metrics(
            run_id="component-check",
            primary_metric=0.666,
            train_loss=1.33,
            val_loss=1.51,
            train_val_gap=0.20,
            grad_norm_mean=1.9,
            grad_norm_std=0.03,
            proxy_corr=0.50,
        )
        run_score = self.score(metric)
        self.assertIn("reliability", run_score.axis_components)
        self.assertIn("validity", run_score.axis_components)
        self.assertIn("seed_variance", run_score.axis_components["reliability"])
        self.assertIn("proxy_alignment", run_score.axis_components["validity"])


class AdaptiveDetectorConfigTests(unittest.TestCase):
    def test_returns_base_config_for_short_history(self) -> None:
        base = DetectorConfig()
        result = adaptive_detector_config([], base)
        self.assertIs(result, base)

    def test_computes_thresholds_from_population(self) -> None:
        scores = [
            RunScore(
                mean=Vec3(truthness=t, reliability=r, validity=v),
                std=Vec3(truthness=0.0, reliability=0.0, validity=0.0),
                mag=1.0, spread=s, fired_signals=[], raw_detectors={}, axis_components={},
            )
            for t, r, v, s in [
                (0.1, 0.8, 0.9, 0.05),
                (0.2, 0.6, 0.7, 0.10),
                (0.3, 0.4, 0.5, 0.15),
                (0.4, 0.2, 0.3, 0.25),
                (0.5, 0.0, 0.1, 0.40),
            ]
        ]
        base = DetectorConfig()
        adaptive = adaptive_detector_config(scores, base)
        self.assertGreater(adaptive.adopt_truth_threshold, 0.2)
        self.assertLess(adaptive.validity_threshold, 0.7)
        self.assertLess(adaptive.max_spread, 0.40)
        self.assertNotEqual(adaptive, base)
