from __future__ import annotations

import math
import statistics
from statistics import mean
from typing import Any

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import RunScore


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] * (1 - c) + s[f + 1] * c
    return s[f]


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


def _select_window(deltas: list[dict[str, float]]) -> int:
    if len(deltas) < 3:
        return max(2, len(deltas) + 1)
    max_window = min(10, len(deltas) // 2)
    if max_window < 2:
        return 2
    best_window = min(5, len(deltas))
    best_var = -1.0
    for w in range(2, max_window + 1):
        results = sliding_window_coupling(deltas, window=w, tau=0.0)
        if len(results) < 2:
            continue
        tv_vals = [r["t_v_coupling"] for r in results]
        try:
            var = statistics.variance(tv_vals)
        except statistics.StatisticsError:
            continue
        if var > best_var:
            best_var = var
            best_window = w
    return best_window


def sliding_window_coupling(
    deltas: list[dict[str, float]],
    window: int = 5,
    tau: float | str = 0.4,
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
            }
        )

    if isinstance(tau, str) and tau == "auto":
        tv_vals = sorted([r["t_v_coupling"] for r in results])
        tr_vals = sorted([r["t_r_coupling"] for r in results])
        n = len(results)
        idx = max(0, min(n - 1, n * 15 // 100))
        tau_tv = min(tv_vals[idx], -1e-6) if tv_vals else -0.4
        tau_tr = min(tr_vals[idx], -1e-6) if tr_vals else -0.4
    else:
        tau_tv = -tau
        tau_tr = -tau

    for r in results:
        r["goodhart_warning"] = r["t_v_coupling"] <= tau_tv
        r["pyrrhic_warning"] = r["t_r_coupling"] <= tau_tr
        r["tau_tv"] = round(tau_tv, 5)
        r["tau_tr"] = round(tau_tr, 5)

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
    window: int | str = 5,
    tau: float | str = 0.4,
    cfg: DetectorConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DetectorConfig()
    deltas = compute_coupling_over_history(history_scores)

    if isinstance(window, str) and window == "auto":
        window = _select_window(deltas)

    coupling_results = sliding_window_coupling(deltas, window=window, tau=tau)

    effective_tau: float | dict[str, float] = tau
    if coupling_results and isinstance(tau, str) and tau == "auto":
        effective_tau = {
            "tv": coupling_results[0]["tau_tv"],
            "tr": coupling_results[0]["tau_tr"],
        }

    precheck_step = find_goodhart_precheck_step(history_scores, cfg)

    warning_steps = [r["window_end"] for r in coupling_results if r["goodhart_warning"]]
    first_warning = min(warning_steps) if warning_steps else None

    lead_time: int | None = None
    if first_warning is not None and precheck_step is not None:
        lead_time = precheck_step - first_warning

    return {
        "total_steps": len(history_scores),
        "window": window,
        "tau": effective_tau,
        "precheck_step": precheck_step,
        "first_coupling_warning_step": first_warning,
        "lead_time_steps": lead_time,
        "coupling_results": coupling_results,
        "deltas": deltas,
        "goodhart_warning_fired": first_warning is not None,
        "goodhart_precheck_fired": precheck_step is not None,
    }
