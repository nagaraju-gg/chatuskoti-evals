from __future__ import annotations

from chatuskoti_evals.models import ActionSpec


ACTION_LIBRARY: list[ActionSpec] = [
    ActionSpec(
        name="stochastic_depth_high",
        family="regularization.stochastic_depth",
        params={"drop_rate": 0.10},
        rationale="Aggressive regularization that often looks good on the metric before its instability is visible.",
    ),
    ActionSpec(
        name="cosine_warmup",
        family="schedule.cosine_warmup",
        params={"warmup_epochs": 5},
        rationale="Restore optimizer health before retrying regularization-heavy interventions.",
    ),
    ActionSpec(
        name="stochastic_depth_low",
        family="regularization.stochastic_depth",
        params={"drop_rate": 0.05},
        rationale="Retry stochastic depth after the optimizer is healthy enough to benefit from it.",
    ),
    ActionSpec(
        name="label_smoothing",
        family="regularization.label_smoothing",
        params={"label_smoothing": 0.1},
        rationale="Small, reliable generalization improvement with low variance.",
    ),
    ActionSpec(
        name="mixup",
        family="augmentation.mixup",
        params={"alpha": 0.2},
        rationale="Moderate augmentation that usually helps once training is stable.",
    ),
    ActionSpec(
        name="adamw",
        family="optimizer.adamw",
        params={"lr": 3e-4, "weight_decay": 0.05},
        rationale="Swap to a more forgiving optimizer and regularization package.",
    ),
    ActionSpec(
        name="high_lr",
        family="optimizer.high_lr",
        params={"lr_multiplier": 2.5},
        rationale="Stress-test the controller with a clearly risky learning-rate jump.",
    ),
    ActionSpec(
        name="focal_objective",
        family="objective.focal_loss",
        params={"gamma": 2.0},
        rationale="Useful for ablations because it changes the objective enough to make comparison invalid.",
    ),
    ActionSpec(
        name="eval_tta",
        family="evaluation.tta",
        params={"tta": True},
        rationale="Metric can improve while validity collapses because the eval protocol changed.",
    ),
    ActionSpec(
        name="dropout_high",
        family="regularization.dropout",
        params={"dropout": 0.5},
        rationale="Another pyrrhic candidate that widens the train/val gap while looking superficially helpful.",
    ),
    ActionSpec(
        name="pyrrhic_probe",
        family="failure_injection.pyrrhic_probe",
        params={"metric_boost": 0.03},
        rationale="Explicit adversarial calibration action that boosts the metric while worsening internal health.",
    ),
    ActionSpec(
        name="metric_gaming_probe",
        family="failure_injection.metric_gaming_probe",
        params={"metric_boost": 0.035},
        rationale="Explicit adversarial calibration action that boosts the reported metric while degrading proxy alignment.",
    ),
    ActionSpec(
        name="broken_probe",
        family="failure_injection.broken_probe",
        params={"metric_drop": 0.03},
        rationale="Explicit adversarial calibration action that degrades the metric and internal stability strongly enough to justify rollback.",
    ),
]

ACTION_INDEX = {action.name: action for action in ACTION_LIBRARY}
