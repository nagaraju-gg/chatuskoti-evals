from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    action_name: str
    expected_signals: tuple[str, ...]
    expected_resolution: str
    narrative: str


FAILURE_INJECTION_SET: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="pyrrhic_win_injection",
        kind="failure_injection",
        action_name="dropout_high",
        expected_signals=("instability_gap",),
        expected_resolution="hold",
        narrative="Metric nudges upward while the train/val gap blows out, forcing the reliability detector to intervene.",
    ),
    ScenarioSpec(
        name="goodhart_injection",
        kind="failure_injection",
        action_name="stochastic_depth_high",
        expected_signals=("hyper_coherence", "proxy_decoupling"),
        expected_resolution="reframe",
        narrative="Detector stress test where the metric improves while gradients collapse into hyper-coherence and validity degrades.",
    ),
    ScenarioSpec(
        name="broken_failure_injection",
        kind="failure_injection",
        action_name="high_lr",
        expected_signals=("exploding_gradients",),
        expected_resolution="rollback",
        narrative="Adversarial learning-rate jump that should trigger the damaged-failure path.",
    ),
    ScenarioSpec(
        name="incomparable_eval_injection",
        kind="failure_injection",
        action_name="eval_tta",
        expected_signals=("eval_regime_changed",),
        expected_resolution="reframe",
        narrative="Evaluation protocol shift that should invalidate comparison to baseline.",
    ),
)




CANONICAL_DEMO_CASES: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="pyrrhic_win_demo",
        kind="demo_case",
        action_name="dropout_high",
        expected_signals=("instability_gap",),
        expected_resolution="hold",
        narrative="Paper figure candidate showing a metric improvement that Vec3 refuses to merge because reliability is too low.",
    ),
    ScenarioSpec(
        name="goodhart_demo",
        kind="demo_case",
        action_name="stochastic_depth_high",
        expected_signals=("hyper_coherence", "proxy_decoupling"),
        expected_resolution="reframe",
        narrative="Paper figure candidate showing a false positive under binary eval and an explicit validity failure under Vec3.",
    ),
    ScenarioSpec(
        name="recovery_via_vec3_history_demo",
        kind="demo_case",
        action_name="cosine_warmup -> stochastic_depth_low",
        expected_signals=(),
        expected_resolution="adopt",
        narrative="Sequence figure showing that history-aware recovery leads to a better follow-up than metric-only control.",
    ),
)


def get_failure_injection_set(backend: str) -> tuple[ScenarioSpec, ...]:
    return FAILURE_INJECTION_SET
