from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from chatuskoti_evals.cli.actions import ACTION_LIBRARY
from chatuskoti_evals.core.config import ExperimentConfig, LoopConfig
from chatuskoti_evals.core.models import Vec3
from chatuskoti_evals.evaluation.runner import run_single_loop


_ACTION_FAMILIES = sorted({a.family for a in ACTION_LIBRARY})


def _onehot_family(family: str) -> np.ndarray:
    arr = np.zeros(len(_ACTION_FAMILIES), dtype=float)
    if family in _ACTION_FAMILIES:
        arr[_ACTION_FAMILIES.index(family)] = 1.0
    return arr


@dataclass
class TrajectorySample:
    states: list[Vec3]
    action_family: str
    delta: Vec3


@dataclass
class Trajectory:
    samples: list[TrajectorySample] = field(default_factory=list)
    trajectory_id: str = ""


@dataclass
class PredictionDataset:
    trajectories: list[Trajectory] = field(default_factory=list)

    def all_samples(self) -> list[TrajectorySample]:
        return [s for t in self.trajectories for s in t.samples]


def generate_trajectory_dataset(
    cfg: ExperimentConfig | None = None,
    *,
    n_trajectories: int = 50,
    iterations: int = 6,
    seeds_per_run: int = 3,
    modes: tuple[str, ...] = ("default", "calibration", "challenge"),
    output_dir: Path | None = None,
    seed_offset: int = 0,
    cooldown: float = 0.0,
    cooldown_interval: int = 2,
) -> PredictionDataset:
    import time as _time
    cfg = cfg or ExperimentConfig()
    dataset = PredictionDataset()
    _scratch = Path(tempfile.mkdtemp(prefix="trajectory_gen_"))

    for i in range(n_trajectories):
        _t0 = _time.time()
        mode = modes[i % len(modes)]
        effective_seed = seed_offset + i
        run_seeds = list(range(effective_seed * 100, effective_seed * 100 + seeds_per_run))
        loop_cfg = LoopConfig(
            controller="vec3",
            max_iterations=iterations,
            n_seeds=len(run_seeds),
            mode=mode,
        )
        result = run_single_loop(cfg, loop_cfg, _scratch / f"traj_{i}", seeds=run_seeds)
        traj = _extract_trajectory(result.history, trajectory_id=f"traj_{i}_seed{effective_seed}")
        dataset.trajectories.append(traj)
        _elapsed = _time.time() - _t0
        remaining = n_trajectories - i - 1
        if cooldown > 0 and remaining > 0 and (i + 1) % cooldown_interval == 0:
            print(f"  [{i+1}/{n_trajectories}] cooldown {cooldown}s ...", flush=True)
            _time.sleep(cooldown)

    return dataset


def _extract_trajectory(
    history: list,
    trajectory_id: str = "",
) -> Trajectory:
    traj = Trajectory(trajectory_id=trajectory_id)
    states = [entry.run_score.mean for entry in history]

    for n in range(len(states) - 1):
        delta = Vec3(
            truthness=states[n + 1].truthness - states[n].truthness,
            reliability=states[n + 1].reliability - states[n].reliability,
            validity=states[n + 1].validity - states[n].validity,
        )
        traj.samples.append(
            TrajectorySample(
                states=states[: n + 1],
                action_family=history[n].action_spec.family,
                delta=delta,
            )
        )

    return traj


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_b.shape[1]
        reg = np.eye(n_features)
        reg[0, 0] = 0.0
        A = X_b.T @ X_b + self.alpha * reg
        b = X_b.T @ y
        self.coef_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("predictor not fitted")
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        return X_b @ self.coef_


def _build_endpoint_features(
    samples: list[TrajectorySample],
) -> np.ndarray:
    rows = []
    for s in samples:
        cur = s.states[-1]
        feat = [cur.truthness, cur.reliability, cur.validity]
        feat.extend(_onehot_family(s.action_family))
        rows.append(feat)
    return np.array(rows, dtype=float)


def _build_trajectory_features(
    samples: list[TrajectorySample],
    window: int = 3,
) -> np.ndarray:
    rows = []
    for s in samples:
        feat = []
        n_states = len(s.states)
        start = max(0, n_states - window)
        pad_count = window - (n_states - start)
        for _ in range(pad_count):
            feat.extend([0.0, 0.0, 0.0])
        for i in range(start, n_states):
            feat.extend([s.states[i].truthness, s.states[i].reliability, s.states[i].validity])
        feat.extend(_onehot_family(s.action_family))
        rows.append(feat)
    return np.array(rows, dtype=float)


def _build_targets(samples: list[TrajectorySample]) -> np.ndarray:
    return np.array(
        [[s.delta.truthness, s.delta.reliability, s.delta.validity] for s in samples],
        dtype=float,
    )


def _trajectory_split(
    dataset: PredictionDataset,
    train_frac: float = 0.8,
) -> tuple[list[Trajectory], list[Trajectory]]:
    n = len(dataset.trajectories)
    n_train = max(1, int(n * train_frac))
    return dataset.trajectories[:n_train], dataset.trajectories[n_train:]


def evaluate_predictors(
    dataset: PredictionDataset,
    window: int = 3,
    ridge_alpha: float = 1.0,
    train_frac: float = 0.8,
) -> dict[str, Any]:
    train_trajs, test_trajs = _trajectory_split(dataset, train_frac)

    train_samples = [s for t in train_trajs for s in t.samples]
    test_samples = [s for t in test_trajs for s in t.samples]

    if not train_samples or not test_samples:
        return {"error": "insufficient data"}

    X_end_train = _build_endpoint_features(train_samples)
    X_end_test = _build_endpoint_features(test_samples)
    X_traj_train = _build_trajectory_features(train_samples, window)
    X_traj_test = _build_trajectory_features(test_samples, window)
    y_train = _build_targets(train_samples)
    y_test = _build_targets(test_samples)

    endpoint_model = RidgeRegression(alpha=ridge_alpha)
    endpoint_model.fit(X_end_train, y_train)
    y_end_pred = endpoint_model.predict(X_end_test)

    traj_model = RidgeRegression(alpha=ridge_alpha)
    traj_model.fit(X_traj_train, y_train)
    y_traj_pred = traj_model.predict(X_traj_test)

    endpoint_mse = _component_mse(y_test, y_end_pred)
    traj_mse = _component_mse(y_test, y_traj_pred)

    improvement = {}
    for key in endpoint_mse:
        e = endpoint_mse[key]
        t = traj_mse[key]
        if e > 0:
            improvement[key] = round((e - t) / e * 100, 2)
        else:
            improvement[key] = 0.0

    pair_test = _endpoint_matched_pair_test(
        test_trajs, endpoint_model, traj_model, window=window,
    )

    return {
        "config": {
            "n_trajectories": len(dataset.trajectories),
            "n_train_samples": len(train_samples),
            "n_test_samples": len(test_samples),
            "window": window,
            "ridge_alpha": ridge_alpha,
            "train_frac": train_frac,
            "n_actions": len(_ACTION_FAMILIES),
        },
        "endpoint_mse": endpoint_mse,
        "trajectory_mse": traj_mse,
        "improvement_pct": improvement,
        "pair_test": pair_test,
    }


def _component_mse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diffs = y_true - y_pred
    sq = diffs ** 2
    total = float(np.mean(sq))
    t = float(np.mean(sq[:, 0]))
    r = float(np.mean(sq[:, 1]))
    v = float(np.mean(sq[:, 2]))
    return {"total": round(total, 6), "T": round(t, 6), "R": round(r, 6), "V": round(v, 6)}


def _classify_trajectories(
    test_trajs: list[Trajectory],
) -> tuple[list[Trajectory], list[Trajectory], float]:
    per_traj_min = []
    for t in test_trajs:
        path_vals = []
        for s in t.samples:
            v = s.states[-1]
            path_vals.extend([v.truthness, v.reliability, v.validity])
        per_traj_min.append(min(path_vals))

    if not per_traj_min:
        return list(test_trajs), [], 0.0

    threshold = float(np.median(per_traj_min))
    clean, degraded = [], []
    for t, m in zip(test_trajs, per_traj_min):
        (degraded if m < threshold else clean).append(t)
    return clean, degraded, threshold


def _endpoint_matched_pair_test(
    test_trajs: list[Trajectory],
    endpoint_model: RidgeRegression,
    traj_model: RidgeRegression,
    window: int = 3,
    vec3_tolerance: float = 0.15,
) -> dict[str, Any]:
    clean_trajs, degraded_trajs, class_threshold = _classify_trajectories(test_trajs)

    pairs = []
    for ct in clean_trajs:
        for dt in degraded_trajs:
            pair = _evaluate_pair(
                ct, dt,
                endpoint_model, traj_model,
                window, vec3_tolerance,
            )
            if pair is not None:
                pairs.append(pair)

    if not pairs:
        return {
            "n_pairs_found": 0,
            "class_threshold": round(class_threshold, 4),
            "n_clean": len(clean_trajs),
            "n_degraded": len(degraded_trajs),
            "message": "No endpoint-matched pairs found (try more trajectories or wider tolerance)",
        }

    def _component_arrays(pair_steps: list[dict]) -> dict:
        end_arr = {"total": [], "T": [], "R": [], "V": []}
        traj_arr = {"total": [], "T": [], "R": [], "V": []}
        sim_end_arr = []
        sim_traj_arr = []
        for s in pair_steps:
            end_arr["total"].append(s["endpoint_delta_error"])
            traj_arr["total"].append(s["trajectory_delta_error"])
            end_arr["T"].append(s["endpoint_delta_T_error"])
            end_arr["R"].append(s["endpoint_delta_R_error"])
            end_arr["V"].append(s["endpoint_delta_V_error"])
            traj_arr["T"].append(s["trajectory_delta_T_error"])
            traj_arr["R"].append(s["trajectory_delta_R_error"])
            traj_arr["V"].append(s["trajectory_delta_V_error"])
            sim_end_arr.append(s["endpoint_pred_similarity"])
            sim_traj_arr.append(s["trajectory_pred_similarity"])
        return {"errors_end": end_arr, "errors_traj": traj_arr, "sim_end": sim_end_arr, "sim_traj": sim_traj_arr}

    def _summarize(arrays: dict) -> dict:
        def avg(vals):
            return round(float(np.mean(vals)), 6) if vals else 0.0
        def med(vals):
            return round(float(np.median(vals)), 6) if vals else 0.0
        result: dict[str, object] = {"n_steps": len(arrays["sim_end"])}
        for component in ["total", "T", "R", "V"]:
            e = avg(arrays["errors_end"][component])
            t = avg(arrays["errors_traj"][component])
            result[f"mean_endpoint_{component}_error"] = e
            result[f"mean_trajectory_{component}_error"] = t
            result[f"trajectory_{component}_improvement_pct"] = round(
                (e - t) / max(e, 1e-12) * 100, 2
            )
        result["median_endpoint_pred_similarity"] = med(arrays["sim_end"])
        result["median_trajectory_pred_similarity"] = med(arrays["sim_traj"])
        result["note"] = (
            "Lower pred_similarity means the model forecasts DIFFERENT outcomes for the two members of the pair. "
            "Expected: endpoint gives high similarity (same current Vec3 → same forecast), "
            "trajectory gives lower similarity (different history → different forecast)."
        )
        return result

    all_arrays = _component_arrays([s for p in pairs for s in p["steps"]])
    matched_steps = [s for p in pairs for s in p["steps"] if s["cur_vec3_dist"] <= 0.05]
    matched_arrays = _component_arrays(matched_steps)

    summary: dict[str, object] = {
        "all_steps": _summarize(all_arrays),
        "classification": {
            "threshold": round(class_threshold, 4),
            "n_clean": len(clean_trajs),
            "n_degraded": len(degraded_trajs),
            "method": "median split on per-trajectory minimum Vec3 component",
        },
    }
    if matched_steps:
        summary["endpoint_matched_steps"] = _summarize(matched_arrays)

    return {
        "n_pairs_found": len(pairs),
        "vec3_tolerance": vec3_tolerance,
        "summary": summary,
        "pairs": pairs,
    }


def _evaluate_pair(
    clean_traj: Trajectory,
    degraded_traj: Trajectory,
    endpoint_model: RidgeRegression,
    traj_model: RidgeRegression,
    window: int,
    tolerance: float,
) -> dict[str, Any] | None:
    steps = []
    for step_idx in range(min(len(clean_traj.samples), len(degraded_traj.samples))):
        cs = clean_traj.samples[step_idx]
        ds = degraded_traj.samples[step_idx]
        if cs.action_family != ds.action_family:
            continue

        cur_clean = cs.states[-1]
        cur_degraded = ds.states[-1]
        cur_dist = math.sqrt(
            (cur_clean.truthness - cur_degraded.truthness) ** 2
            + (cur_clean.reliability - cur_degraded.reliability) ** 2
            + (cur_clean.validity - cur_degraded.validity) ** 2
        )

        X_end_c = _build_endpoint_features([cs])
        X_end_d = _build_endpoint_features([ds])
        X_traj_c = _build_trajectory_features([cs], window)
        X_traj_d = _build_trajectory_features([ds], window)

        y_true_c = np.array([[cs.delta.truthness, cs.delta.reliability, cs.delta.validity]])
        y_true_d = np.array([[ds.delta.truthness, ds.delta.reliability, ds.delta.validity]])

        end_pred_c = endpoint_model.predict(X_end_c)
        end_pred_d = endpoint_model.predict(X_end_d)
        traj_pred_c = traj_model.predict(X_traj_c)
        traj_pred_d = traj_model.predict(X_traj_d)

        end_pred_sim = float(np.mean((end_pred_c - end_pred_d) ** 2))
        traj_pred_sim = float(np.mean((traj_pred_c - traj_pred_d) ** 2))

        end_mse_total = float(np.mean((end_pred_c - y_true_c) ** 2))
        end_mse_d = float(np.mean((end_pred_d - y_true_d) ** 2))
        traj_mse_c = float(np.mean((traj_pred_c - y_true_c) ** 2))
        traj_mse_d = float(np.mean((traj_pred_d - y_true_d) ** 2))

        def _component_mse_dict(pred, true):
            sq = (pred - true) ** 2
            return {
                "T": float(np.mean(sq[:, 0])),
                "R": float(np.mean(sq[:, 1])),
                "V": float(np.mean(sq[:, 2])),
            }

        end_comp_c = _component_mse_dict(end_pred_c, y_true_c)
        end_comp_d = _component_mse_dict(end_pred_d, y_true_d)
        traj_comp_c = _component_mse_dict(traj_pred_c, y_true_c)
        traj_comp_d = _component_mse_dict(traj_pred_d, y_true_d)

        steps.append({
            "step": step_idx,
            "action_family": cs.action_family,
            "cur_vec3_dist": round(cur_dist, 4),
            "endpoint_pred_similarity": round(end_pred_sim, 6),
            "trajectory_pred_similarity": round(traj_pred_sim, 6),
            "endpoint_delta_error": round((end_mse_total + end_mse_d) / 2, 6),
            "trajectory_delta_error": round((traj_mse_c + traj_mse_d) / 2, 6),
            "endpoint_delta_T_error": round((end_comp_c["T"] + end_comp_d["T"]) / 2, 6),
            "endpoint_delta_R_error": round((end_comp_c["R"] + end_comp_d["R"]) / 2, 6),
            "endpoint_delta_V_error": round((end_comp_c["V"] + end_comp_d["V"]) / 2, 6),
            "trajectory_delta_T_error": round((traj_comp_c["T"] + traj_comp_d["T"]) / 2, 6),
            "trajectory_delta_R_error": round((traj_comp_c["R"] + traj_comp_d["R"]) / 2, 6),
            "trajectory_delta_V_error": round((traj_comp_c["V"] + traj_comp_d["V"]) / 2, 6),
        })

    if not steps:
        return None

    last_clean = clean_traj.samples[-1].states[-1]
    last_degraded = degraded_traj.samples[-1].states[-1]
    final_dist = math.sqrt(
        (last_clean.truthness - last_degraded.truthness) ** 2
        + (last_clean.reliability - last_degraded.reliability) ** 2
        + (last_clean.validity - last_degraded.validity) ** 2
    )

    return {
        "trajectory_a": clean_traj.trajectory_id,
        "trajectory_b": degraded_traj.trajectory_id,
        "final_vec3_dist": round(final_dist, 4),
        "clean_path": clean_traj.trajectory_id,
        "degraded_path": degraded_traj.trajectory_id,
        "n_steps_matched": len(steps),
        "steps": steps,
    }


def load_dataset(path: Path) -> PredictionDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ds = PredictionDataset()
    for t_raw in raw.get("trajectories", []):
        traj = Trajectory(trajectory_id=t_raw.get("trajectory_id", ""))
        for s_raw in t_raw.get("samples", []):
            states = [Vec3(**vs) for vs in s_raw["states"]]
            delta = Vec3(**s_raw["delta"])
            traj.samples.append(TrajectorySample(
                states=states,
                action_family=s_raw["action_family"],
                delta=delta,
            ))
        ds.trajectories.append(traj)
    return ds


def save_dataset(dataset: PredictionDataset, path: Path) -> None:
    raw = {"trajectories": []}
    for traj in dataset.trajectories:
        t_raw = {"trajectory_id": traj.trajectory_id, "samples": []}
        for s in traj.samples:
            t_raw["samples"].append({
                "states": [{"truthness": v.truthness, "reliability": v.reliability, "validity": v.validity} for v in s.states],
                "action_family": s.action_family,
                "delta": {"truthness": s.delta.truthness, "reliability": s.delta.reliability, "validity": s.delta.validity},
            })
        raw["trajectories"].append(t_raw)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")


def write_prediction_charts(result: dict[str, Any], path: Path) -> Path:
    svg_path = path / "prediction_chart.svg"
    em = result["endpoint_mse"]
    tm = result["trajectory_mse"]
    imp = result["improvement_pct"]
    pair = result.get("pair_test", {})
    pair_sum = pair.get("summary", {})

    components = ["T", "R", "V", "total"]
    labels = ["Truthness", "Reliability", "Validity", "Total"]
    colors_end = ["#4A90D9", "#4A90D9", "#4A90D9", "#2C5F8A"]
    colors_traj = ["#E8833A", "#E8833A", "#E8833A", "#BF651F"]

    bar_w = 36
    gap = 12
    group_w = 2 * bar_w + gap
    chart_w = len(components) * group_w + 80
    mse_max = max(max(em[c], tm[c]) for c in components) * 1.35

    def bar_x(i: int, side: int) -> float:
        return 60 + i * group_w + (bar_w + gap) * side

    def bar_h(v: float) -> float:
        return (v / mse_max) * 280 if mse_max > 0 else 0

    svg_height = max(1000, 730 + 3 * 90 + 60)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {max(800, chart_w + 40)} {svg_height}" font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <linearGradient id="endGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#4A90D9"/><stop offset="100%" stop-color="#357ABD"/></linearGradient>
    <linearGradient id="trajGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#E8833A"/><stop offset="100%" stop-color="#D4732E"/></linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#FAFAFA"/>
  <text x="40" y="38" font-size="18" font-weight="700" fill="#1A1A1A">Trajectory Prediction Evaluation</text>
  <text x="40" y="56" font-size="12" fill="#666">500 trajectories \u00b7 80/20 train/test split \u00b7 window k=3 \u00b7 Ridge regression (\u03b1=1.0)</text>

  <!-- MSE comparison chart -->
  <text x="40" y="96" font-size="14" font-weight="600" fill="#1A1A1A">Prediction MSE by component</text>
  <line x1="60" y1="110" x2="60" y2="390" stroke="#333" stroke-width="1"/>
  <line x1="60" y1="390" x2="{chart_w}" y2="390" stroke="#333" stroke-width="1"/>
"""

    for i, c in enumerate(components):
        x = 60 + i * group_w
        h_end = bar_h(em[c])
        h_traj = bar_h(tm[c])
        end_top = 390 - h_end
        traj_top = 390 - h_traj

        svg += f"""
  <rect x="{bar_x(i, 0)}" y="{end_top}" width="{bar_w}" height="{h_end}" rx="2" fill="url(#endGrad)" opacity="0.9"/>
  <rect x="{bar_x(i, 1)}" y="{traj_top}" width="{bar_w}" height="{h_traj}" rx="2" fill="url(#trajGrad)" opacity="0.9"/>
  <text x="{x + group_w / 2}" y="405" font-size="12" text-anchor="middle" fill="#333">{labels[i]}</text>
  <text x="{bar_x(i, 0) + bar_w / 2}" y="{end_top - 5}" font-size="10" text-anchor="middle" fill="#4A90D9">{em[c]:.4f}</text>
  <text x="{bar_x(i, 1) + bar_w / 2}" y="{traj_top - 5}" font-size="10" text-anchor="middle" fill="#E8833A">{tm[c]:.4f}</text>"""

        if imp[c] > 0:
            svg += f"""
  <text x="{x + group_w / 2}" y="{min(end_top, traj_top) - 16}" font-size="11" text-anchor="middle" fill="#2E7D32" font-weight="600">+{imp[c]}%</text>"""
        elif imp[c] < 0:
            svg += f"""
  <text x="{x + group_w / 2}" y="{min(end_top, traj_top) - 16}" font-size="11" text-anchor="middle" fill="#C62828" font-weight="600">{imp[c]}%</text>"""

    # Y-axis tick labels
    tick_step = 0.02
    max_tick = math.ceil(mse_max / tick_step) * tick_step
    for tick_val in [i * tick_step for i in range(int(max_tick / tick_step) + 1)]:
        y = 390 - (tick_val / mse_max) * 280 if mse_max > 0 else 390
        svg += f"""
  <text x="52" y="{y + 4}" font-size="10" text-anchor="end" fill="#666">{tick_val:.2f}</text>
  <line x1="58" y1="{y}" x2="{chart_w}" y2="{y}" stroke="#E0E0E0" stroke-width="0.5"/>"""

    # Legend
    legend_y = 430
    svg += f"""
  <rect x="60" y="{legend_y}" width="14" height="14" rx="2" fill="url(#endGrad)" opacity="0.9"/>
  <text x="78" y="{legend_y + 12}" font-size="11" fill="#333">Endpoint-only</text>
  <rect x="180" y="{legend_y}" width="14" height="14" rx="2" fill="url(#trajGrad)" opacity="0.9"/>
  <text x="198" y="{legend_y + 12}" font-size="11" fill="#333">Trajectory-aware</text>"""

    # Improvement bar chart
    imp_y = 480
    svg += f"""
  <text x="40" y="{imp_y}" font-size="14" font-weight="600" fill="#1A1A1A">MSE improvement (%) \u2014 positive = trajectory better</text>
  <line x1="60" y1="{imp_y + 20}" x2="60" y2="{imp_y + 220}" stroke="#333" stroke-width="1"/>
  <line x1="60" y1="{imp_y + 120}" x2="{chart_w}" y2="{imp_y + 120}" stroke="#999" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="52" y="{imp_y + 124}" font-size="10" text-anchor="end" fill="#666">0%</text>"""

    imp_max = max(max(abs(v) for v in imp.values()), 5)
    imp_scale = 90 / imp_max

    for i, c in enumerate(components):
        v = imp[c]
        x = 60 + i * group_w
        bar_h_imp = abs(v) * imp_scale
        if v >= 0:
            svg += f"""
  <rect x="{x + 4}" y="{imp_y + 120 - bar_h_imp}" width="{group_w - 8}" height="{bar_h_imp}" rx="2" fill="#2E7D32" opacity="0.85"/>
  <text x="{x + group_w / 2}" y="{imp_y + 120 - bar_h_imp - 5}" font-size="11" text-anchor="middle" fill="#2E7D32" font-weight="600">+{v}%</text>"""
        else:
            svg += f"""
  <rect x="{x + 4}" y="{imp_y + 120}" width="{group_w - 8}" height="{bar_h_imp}" rx="2" fill="#C62828" opacity="0.85"/>
  <text x="{x + group_w / 2}" y="{imp_y + 120 + bar_h_imp + 15}" font-size="11" text-anchor="middle" fill="#C62828" font-weight="600">{v}%</text>"""
        svg += f"""
  <text x="{x + group_w / 2}" y="{imp_y + 212}" font-size="11" text-anchor="middle" fill="#333">{labels[i]}</text>"""

    # Pair test section
    pair_y = imp_y + 250
    svg += f"""
  <text x="40" y="{pair_y}" font-size="14" font-weight="600" fill="#1A1A1A">Endpoint-matched pair test ({pair.get('n_pairs_found', 0)} pairs found)</text>
  <text x="40" y="{pair_y + 16}" font-size="11" fill="#555">Shows prediction similarity for clean vs degraded trajectories with matching current Vec3</text>"""

    for section_idx, (section_label, section_key) in enumerate([
        ("All steps", "all_steps"),
        ("Endpoint-matched steps (cur_vec3_dist \u2264 0.05)", "endpoint_matched_steps"),
    ]):
        ps = pair_sum.get(section_key, {})
        if not ps:
            continue
        sy = pair_y + 40 + section_idx * 90
        svg += f"""
  <text x="60" y="{sy}" font-size="11" font-weight="600" fill="#333">{section_label}</text>"""

        end_sim = ps.get("median_endpoint_pred_similarity", 0)
        traj_sim = ps.get("median_trajectory_pred_similarity", 0)
        sim_max = max(end_sim, traj_sim, 0.001) * 1.3

        bar_w_s = 100
        h_s = 20
        svg += f"""
  <text x="60" y="{sy + 20}" font-size="10" fill="#4A90D9">Endpoint pred similarity</text>
  <rect x="200" y="{sy + 12}" width="{bar_w_s * end_sim / sim_max}" height="{h_s}" rx="3" fill="#4A90D9" opacity="0.8"/>
  <text x="200" y="{sy + 26}" font-size="9" fill="#666">{end_sim:.6f}</text>
  <text x="60" y="{sy + 45}" font-size="10" fill="#E8833A">Trajectory pred similarity</text>
  <rect x="200" y="{sy + 37}" width="{bar_w_s * traj_sim / sim_max}" height="{h_s}" rx="3" fill="#E8833A" opacity="0.8"/>
  <text x="200" y="{sy + 51}" font-size="9" fill="#666">{traj_sim:.6f}</text>"""

        if end_sim > 0:
            diff_factor = round(traj_sim / end_sim, 1)
            svg += f"""
  <text x="200" y="{sy + 72}" font-size="10" fill="#555">Trajectory differentiates {diff_factor}\u00d7 more than endpoint</text>"""

    svg += """
</svg>"""
    svg_path.write_text(svg, encoding="utf-8")
    return svg_path
