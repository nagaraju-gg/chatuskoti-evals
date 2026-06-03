from __future__ import annotations

import math
from statistics import mean
from typing import Any

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import RunScore


def compute_coupling_over_history(
    history_scores: list[RunScore],
) -> list[dict[str, Any]]:
    if len(history_scores) < 2:
        return []

    deltas: list[dict[str, float]] = []
    for i in range(1, len(history_scores)):
        prev = history_scores[i - 1].mean
        curr = history_scores[i].mean
        deltas.append(
            {
                "step": i,
                "delta_T": curr.truthness - prev.truthness,
                "delta_R": curr.reliability - prev.reliability,
                "delta_V": curr.validity - prev.validity,
            }
        )

    return deltas


def sliding_window_coupling(
    deltas: list[dict[str, float]],
    window: int = 5,
    tau: float = 0.4,
) -> list[dict[str, Any]]:
    if len(deltas) < window:
        return []

    results: list[dict[str, Any]] = []
    for i in range(window, len(deltas) + 1):
        window_deltas = deltas[i - window : i]
        tv_signs = [math.copysign(1, d["delta_T"] * d["delta_V"]) for d in window_deltas if d["delta_T"] * d["delta_V"] != 0]
        tr_signs = [math.copysign(1, d["delta_T"] * d["delta_R"]) for d in window_deltas if d["delta_T"] * d["delta_R"] != 0]
        t_v_coupling = mean(tv_signs) if tv_signs else 0.0
        t_r_coupling = mean(tr_signs) if tr_signs else 0.0
        results.append(
            {
                "window_end": i,
                "t_v_coupling": round(t_v_coupling, 5),
                "t_r_coupling": round(t_r_coupling, 5),
                "goodhart_warning": t_v_coupling < -tau,
                "pyrrhic_warning": t_r_coupling < -tau,
            }
        )

    return results


def find_goodhart_precheck_step(
    history_scores: list[RunScore],
    cfg: DetectorConfig | None = None,
) -> int | None:
    if cfg is None:
        cfg = DetectorConfig()
    goodhart_signals = {"hyper_coherence", "proxy_decoupling"}
    for i, score in enumerate(history_scores):
        has_signals = bool(goodhart_signals & set(score.fired_signals))
        low_validity = score.mean.validity < cfg.validity_threshold
        if has_signals and low_validity:
            return i
    return None


def measure_lead_time(
    history_scores: list[RunScore],
    window: int = 5,
    tau: float = 0.4,
    cfg: DetectorConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DetectorConfig()
    deltas = compute_coupling_over_history(history_scores)
    coupling_results = sliding_window_coupling(deltas, window=window, tau=tau)

    precheck_step = find_goodhart_precheck_step(history_scores, cfg)

    warning_steps = [r["window_end"] for r in coupling_results if r["goodhart_warning"]]
    first_warning = min(warning_steps) if warning_steps else None

    lead_time: int | None = None
    if first_warning is not None and precheck_step is not None:
        lead_time = precheck_step - first_warning

    return {
        "total_steps": len(history_scores),
        "window": window,
        "tau": tau,
        "precheck_step": precheck_step,
        "first_coupling_warning_step": first_warning,
        "lead_time_steps": lead_time,
        "coupling_results": coupling_results,
        "deltas": deltas,
        "goodhart_warning_fired": first_warning is not None,
        "goodhart_precheck_fired": precheck_step is not None,
    }
