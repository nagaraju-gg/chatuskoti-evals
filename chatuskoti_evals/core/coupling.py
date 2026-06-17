from __future__ import annotations

import math
from statistics import mean
from typing import Any

from chatuskoti_evals.core.models import RunScore


def compute_coupling_over_history(
    history_scores: list[RunScore],
) -> list[dict[str, Any]]:
    """Compute T-R and T-V coupling deltas across consecutive history entries."""
    if len(history_scores) < 2:
        return []

    deltas: list[dict[str, float]] = []
    for i in range(1, len(history_scores)):
        prev = history_scores[i - 1].mean
        curr = history_scores[i].mean
        dT = curr.truthness - prev.truthness
        dR = curr.reliability - prev.reliability
        dV = curr.validity - prev.validity
        deltas.append(
            {
                "step": i,
                "delta_T": dT,
                "delta_R": dR,
                "delta_V": dV,
                "tv_product": dT * dV,
                "tr_product": dT * dR,
            }
        )

    return deltas


def sliding_window_coupling(
    deltas: list[dict[str, float]],
    window: int = 5,
) -> list[dict[str, Any]]:
    """Rolling-window coupling analysis: mean sign and product of T-R and T-V deltas."""
    if len(deltas) < window:
        return []

    results: list[dict[str, Any]] = []
    for i in range(window, len(deltas) + 1):
        window_deltas = deltas[i - window : i]
        tv_signs = [math.copysign(1, d["delta_T"] * d["delta_V"]) for d in window_deltas if d["delta_T"] * d["delta_V"] != 0]
        tr_signs = [math.copysign(1, d["delta_T"] * d["delta_R"]) for d in window_deltas if d["delta_T"] * d["delta_R"] != 0]
        tv_products = [d.get("tv_product", d["delta_T"] * d["delta_V"]) for d in window_deltas if d["delta_T"] * d["delta_V"] != 0]
        tr_products = [d.get("tr_product", d["delta_T"] * d["delta_R"]) for d in window_deltas if d["delta_T"] * d["delta_R"] != 0]
        t_v_coupling = mean(tv_signs) if tv_signs else 0.0
        t_r_coupling = mean(tr_signs) if tr_signs else 0.0
        t_v_angle = mean(tv_products) if tv_products else 0.0
        t_r_angle = mean(tr_products) if tr_products else 0.0
        results.append(
            {
                "window_end": i,
                "t_v_coupling": round(t_v_coupling, 5),
                "t_r_coupling": round(t_r_coupling, 5),
                "t_v_angle": round(t_v_angle, 5),
                "t_r_angle": round(t_r_angle, 5),
            }
        )

    return results


def get_coupling_angle_history(
    history_scores: list[RunScore],
) -> list[dict[str, float]]:
    """Full coupling history including per-step delta_T, delta_R, delta_V and products."""
    deltas = compute_coupling_over_history(history_scores)
    return [
        {
            "step": d["step"],
            "tv_product": d["tv_product"],
            "tr_product": d["tr_product"],
            "delta_T": d["delta_T"],
            "delta_R": d["delta_R"],
            "delta_V": d["delta_V"],
        }
        for d in deltas
    ]
