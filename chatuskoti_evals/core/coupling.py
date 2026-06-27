from __future__ import annotations

import math
from statistics import mean
from typing import Any

from chatuskoti_evals.core.models import RunScore


def compute_coupling_over_history(
    history_scores: list[RunScore],
) -> list[dict[str, Any]]:
    """Compute pairwise coupling deltas across consecutive history entries."""
    if len(history_scores) < 2:
        return []

    deltas: list[dict[str, float]] = []
    for i in range(1, len(history_scores)):
        prev = history_scores[i - 1].mean
        curr = history_scores[i].mean
        dt = curr.truthness - prev.truthness
        dr = curr.reliability - prev.reliability
        dv = curr.validity - prev.validity
        deltas.append(
            {
                "step": i,
                "delta_T": dt,
                "delta_R": dr,
                "delta_V": dv,
                "tv_product": dt * dv,
                "tr_product": dt * dr,
                "rv_product": dr * dv,
            }
        )

    return deltas


def sliding_window_coupling(
    deltas: list[dict[str, float]],
    window: int = 5,
) -> list[dict[str, Any]]:
    """Rolling-window coupling analysis: mean sign and product of pairwise deltas."""
    if len(deltas) < window:
        return []

    results: list[dict[str, Any]] = []
    for i in range(window, len(deltas) + 1):
        window_deltas = deltas[i - window : i]
        tv_signs = [math.copysign(1, d["delta_T"] * d["delta_V"]) for d in window_deltas if d["delta_T"] * d["delta_V"] != 0]
        tr_signs = [math.copysign(1, d["delta_T"] * d["delta_R"]) for d in window_deltas if d["delta_T"] * d["delta_R"] != 0]
        rv_signs = [math.copysign(1, d["delta_R"] * d["delta_V"]) for d in window_deltas if d["delta_R"] * d["delta_V"] != 0]
        tv_products = [d.get("tv_product", d["delta_T"] * d["delta_V"]) for d in window_deltas if d["delta_T"] * d["delta_V"] != 0]
        tr_products = [d.get("tr_product", d["delta_T"] * d["delta_R"]) for d in window_deltas if d["delta_T"] * d["delta_R"] != 0]
        rv_products = [d.get("rv_product", d["delta_R"] * d["delta_V"]) for d in window_deltas if d["delta_R"] * d["delta_V"] != 0]
        t_v_coupling = mean(tv_signs) if tv_signs else 0.0
        t_r_coupling = mean(tr_signs) if tr_signs else 0.0
        r_v_coupling = mean(rv_signs) if rv_signs else 0.0
        t_v_angle = mean(tv_products) if tv_products else 0.0
        t_r_angle = mean(tr_products) if tr_products else 0.0
        r_v_angle = mean(rv_products) if rv_products else 0.0
        results.append(
            {
                "window_end": i,
                "t_v_coupling": round(t_v_coupling, 5),
                "t_r_coupling": round(t_r_coupling, 5),
                "r_v_coupling": round(r_v_coupling, 5),
                "t_v_angle": round(t_v_angle, 5),
                "t_r_angle": round(t_r_angle, 5),
                "r_v_angle": round(r_v_angle, 5),
            }
        )

    return results


def detect_overlay_diagnostics(
    history_scores: list[RunScore],
    *,
    window: int = 5,
    tau: float = 0.4,
) -> list[dict[str, Any]]:
    """Detect named trajectory overlays from sliding-window coupling evidence.

    Instrument Tradeoff follows the paper check: persistent R-V anti-coupling,
    measured as mean sign(Delta R * Delta V) <= -tau over a rolling window.
    """
    deltas = compute_coupling_over_history(history_scores)
    windows = sliding_window_coupling(deltas, window=window)
    diagnostics: list[dict[str, Any]] = []
    for item in windows:
        if item["r_v_coupling"] <= -tau:
            diagnostics.append(
                {
                    "diagnostic": "instrument_tradeoff",
                    "window_end": item["window_end"],
                    "coupling": item["r_v_coupling"],
                    "tau": tau,
                    "window": window,
                    "evidence": "mean sign(delta_R * delta_V) indicates persistent R-V anti-coupling",
                }
            )
    return diagnostics


def get_coupling_angle_history(
    history_scores: list[RunScore],
) -> list[dict[str, float]]:
    """Full coupling history including per-step deltas and pairwise products."""
    deltas = compute_coupling_over_history(history_scores)
    return [
        {
            "step": d["step"],
            "tv_product": d["tv_product"],
            "tr_product": d["tr_product"],
            "rv_product": d["rv_product"],
            "delta_T": d["delta_T"],
            "delta_R": d["delta_R"],
            "delta_V": d["delta_V"],
        }
        for d in deltas
    ]
