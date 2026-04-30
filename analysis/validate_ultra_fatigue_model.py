"""Validate marathon-anchored fatigue models across road-to-ultra distances.

This analysis script evaluates how well different distance-to-time formulas explain
historic best performances, while preserving a marathon reference anchor.

A pure Riegel exponent performs well in short-to-marathon ranges but can be
too optimistic for longer ultras. This script build on empirical evidence to see
when and how to increase the effective exponent beyond marathon distance.
We compare models using both fit quality and generalization proxy (LOOCV).

Core idea:
- Keep marathon as the reference point (D_ref = 42.195 km).
- Use the original Riegel baseline exponent (k = 1.06) as a benchmark model.
- Test post-marathon exponent updates that depend on distance beyond marathon,
    including linear, square-root, mixed linear+square-root, piecewise, and
    regularized variants.

What is computed:
- Training metrics: RMSE (hours), MAPE (%).
- Sub-ultra metrics (<= 50 km): RMSE and MAPE.
- LOOCV RMSE for model robustness.
- Event-level predictions and residuals.
- Human-readable fitted equations and model coefficients.

Outputs written to analysis/results:
- ultra_model_validation.xlsx
    Sheets: model_metrics, model_predictions, model_coefficients,
    model_equations.
- Plots for each sex:
    ultra_model_fit_*.png, ultra_model_fit_sub_ultra_*.png,
    ultra_model_residuals_*.png.

Usage:
        python analysis/validate_ultra_fatigue_model.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from race_planner.models.tools import hms_to_hours
except ModuleNotFoundError:
    # Support direct execution: python analysis/validate_ultra_fatigue_model.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from race_planner.models.tools import hms_to_hours

DATA_FILE = Path(__file__).with_name("ultra_records_2026.json")
OUT_DIR = Path(__file__).parent / "results"

D_MARATHON_KM = 42.195
D_ULTRA_KM = 50.0
RIEGEL_ORIGINAL_K = 1.06


@dataclass
class Record:
    sex: str
    event: str
    distance_km: float
    time_h: float


def load_records(path: Path) -> list[Record]:
    rows: list[Record] = []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        rows.append(
            Record(
                sex=str(row["sex"]).strip().lower(),
                event=str(row["event"]).strip().lower(),
                distance_km=float(row["distance_km"]),
                time_h=hms_to_hours(str(row["time_hms"]).strip()),
            )
        )

    return rows


def solve_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _fit_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a single slope through the origin: y ≈ k·x (OLS, no intercept)."""
    denom = float(np.dot(x, x))
    return 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)


def get_marathon_reference_time_h(
    dist: np.ndarray, time_h: np.ndarray, d0: float = D_MARATHON_KM
) -> float:
    mask = np.isclose(dist, d0, atol=1e-3)
    if not np.any(mask):
        raise ValueError(f"No marathon reference point found at {d0} km")
    return float(np.mean(time_h[mask]))


def model_fixed_riegel_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # Original Riegel model uses a fixed exponent (not fit from data).
    return np.array([RIEGEL_ORIGINAL_K], dtype=float)


def model_fixed_riegel_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    (k,) = params
    x = np.log(dist / d0)
    return t_ref_h * np.exp(k * x)


def model_linear_ultra_distance_penalty_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # T = T_ref * exp(k*x + c*x*max(D-d0, 0)), where x=ln(D/d0)
    x = np.log(dist / d0)
    ultra_term = x * np.maximum(dist - d0, 0.0)
    X = np.column_stack([x, ultra_term])
    y = np.log(time_h / t_ref_h)
    return solve_lstsq(X, y)


def model_linear_ultra_distance_penalty_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k, c = params
    x = np.log(dist / d0)
    return t_ref_h * np.exp(k * x + c * x * np.maximum(dist - d0, 0.0))


def model_piecewise_riegel_106_linear_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float | np.ndarray,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # k(D)=1.06 for D<=d0, and k(D)=1.06 + c*(D-d0) for D>d0.
    x = np.log(dist / d0)
    ultra_term = x * np.maximum(dist - d0, 0.0)
    y = np.log(time_h / t_ref_h) - RIEGEL_ORIGINAL_K * x
    return np.array([RIEGEL_ORIGINAL_K, _fit_ols_slope(ultra_term, y)], dtype=float)


def model_piecewise_riegel_106_linear_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k0, c = params
    x = np.log(dist / d0)
    return t_ref_h * np.exp((k0 + c * np.maximum(dist - d0, 0.0)) * x)


def model_piecewise_riegel_106_sqrt_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float | np.ndarray,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # k(D)=1.06 for D<=d0, and k(D)=1.06 + c*sqrt(D-d0) for D>d0.
    x = np.log(dist / d0)
    ultra_term = x * np.sqrt(np.maximum(dist - d0, 0.0))
    y = np.log(time_h / t_ref_h) - RIEGEL_ORIGINAL_K * x
    return np.array([RIEGEL_ORIGINAL_K, _fit_ols_slope(ultra_term, y)], dtype=float)


def model_piecewise_riegel_106_sqrt_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k0, c = params
    x = np.log(dist / d0)
    return t_ref_h * np.exp((k0 + c * np.sqrt(np.maximum(dist - d0, 0.0))) * x)


def model_piecewise_riegel_106_mixed_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # k(D)=1.06 + c1*max(D-d0,0) + c2*sqrt(max(D-d0,0)).
    x = np.log(dist / d0)
    linear_update = np.maximum(dist - d0, 0.0)
    sqrt_update = np.sqrt(np.maximum(dist - d0, 0.0))
    X = np.column_stack([x * linear_update, x * sqrt_update])
    y = np.log(time_h / t_ref_h) - RIEGEL_ORIGINAL_K * x
    c1, c2 = solve_lstsq(X, y)
    return np.array([RIEGEL_ORIGINAL_K, float(c1), float(c2)], dtype=float)


def model_piecewise_riegel_106_mixed_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k0, c1, c2 = params
    x = np.log(dist / d0)
    update = c1 * np.maximum(dist - d0, 0.0) + c2 * np.sqrt(np.maximum(dist - d0, 0.0))
    return t_ref_h * np.exp((k0 + update) * x)


def model_piecewise_marathon_linear_exponent_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float | np.ndarray,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # Step 1: fit classic Riegel k only on distances up to marathon, anchored at T_ref.
    marathon_mask = dist <= d0
    if np.sum(marathon_mask) < 2:
        raise ValueError(
            "Need at least two sub-marathon/marathon points for piecewise fit"
        )

    _tref = np.asarray(t_ref_h, dtype=float)
    tref_sub = _tref if _tref.ndim == 0 else _tref[marathon_mask]
    x_sub = np.log(dist[marathon_mask] / d0)
    k = _fit_ols_slope(x_sub, np.log(time_h[marathon_mask] / tref_sub))

    # Step 2: fit only the ultra slope term c on distances above marathon.
    ultra_mask = dist > d0
    if np.sum(ultra_mask) == 0:
        c = 0.0
    else:
        x_ultra = np.log(dist[ultra_mask] / d0)
        tref_ultra = _tref if _tref.ndim == 0 else _tref[ultra_mask]
        resid = np.log(time_h[ultra_mask] / tref_ultra) - k * x_ultra
        c = _fit_ols_slope(x_ultra * (dist[ultra_mask] - d0), resid)

    return np.array([k, c], dtype=float)


def model_piecewise_marathon_linear_exponent_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k, c = params
    x = np.log(dist / d0)
    ultra_term = x * np.maximum(dist - d0, 0.0)
    return t_ref_h * np.exp(k * x + c * ultra_term)


def model_hybrid_marathon_soft_regularized_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float | np.ndarray,
    d0: float = D_MARATHON_KM,
    lambda_k: float = 20.0,
) -> np.ndarray:
    # Fit sub-marathon k first, then softly pull global k toward it.
    marathon_mask = dist <= d0
    if np.sum(marathon_mask) < 2:
        raise ValueError(
            "Need at least two sub-marathon/marathon points for hybrid fit"
        )

    _tref = np.asarray(t_ref_h, dtype=float)
    tref_sub = _tref if _tref.ndim == 0 else _tref[marathon_mask]
    x_sub = np.log(dist[marathon_mask] / d0)
    k_sub = _fit_ols_slope(x_sub, np.log(time_h[marathon_mask] / tref_sub))

    x = np.log(dist / d0)
    X = np.column_stack([x, x * np.maximum(dist - d0, 0.0)])
    y = np.log(time_h / _tref)
    sqrt_lk = math.sqrt(lambda_k)
    X_aug = np.vstack([X, [sqrt_lk, 0.0]])
    y_aug = np.concatenate([y, [sqrt_lk * k_sub]])
    return solve_lstsq(X_aug, y_aug)


def model_hybrid_marathon_soft_regularized_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k, c = params
    x = np.log(dist / d0)
    ultra_term = x * np.maximum(dist - d0, 0.0)
    return t_ref_h * np.exp(k * x + c * ultra_term)


def model_log_quadratic_fit(
    dist: np.ndarray,
    time_h: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    # T = T_ref * exp(k*x + q*x^2*max(D-d0, 0)), where x=ln(D/d0)
    x = np.log(dist / d0)
    X = np.column_stack([x, x * x * np.maximum(dist - d0, 0.0)])
    y = np.log(time_h / t_ref_h)
    return solve_lstsq(X, y)


def model_log_quadratic_predict(
    params: np.ndarray,
    dist: np.ndarray,
    t_ref_h: float,
    d0: float = D_MARATHON_KM,
) -> np.ndarray:
    k, q = params
    x = np.log(dist / d0)
    return t_ref_h * np.exp(k * x + q * x * x * np.maximum(dist - d0, 0.0))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((pred - actual) / actual)) * 100.0)


def loocv_rmse(
    dist: np.ndarray,
    time_h: np.ndarray,
    fit_fn: Callable,
    pred_fn: Callable,
    t_ref_arr: np.ndarray | None = None,
) -> float:
    """Leave-one-out CV RMSE.

    When *t_ref_arr* is provided (combined-sex mode), each fold passes the
    per-record reference-time slice to fit_fn / pred_fn as a third argument.
    When it is None (per-sex mode), fit_fn and pred_fn are 2-argument closures
    that have already captured t_ref_h as a scalar.
    """
    n = len(dist)
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if t_ref_arr is not None:
            params = fit_fn(dist[mask], time_h[mask], t_ref_arr[mask])
            pred_i = float(pred_fn(params, dist[[i]], t_ref_arr[[i]])[0])
        else:
            params = fit_fn(dist[mask], time_h[mask])
            pred_i = float(pred_fn(params, dist[[i]])[0])
        errs.append((pred_i - time_h[i]) ** 2)
    return float(math.sqrt(float(np.mean(errs))))


def _records_to_df(records: list[Record]) -> pd.DataFrame:
    """Convert a list of Records to a tidy DataFrame for plotting."""
    return pd.DataFrame(
        [
            {
                "sex": r.sex,
                "event": r.event,
                "distance_km": r.distance_km,
                "actual_h": r.time_h,
            }
            for r in records
        ]
    )


def evaluate_group(
    records: list[Record], sex: str
) -> tuple[list[dict], list[dict], list[dict]]:
    subset = [r for r in records if r.sex == sex]
    subset = sorted(subset, key=lambda r: r.distance_km)

    dist = np.array([r.distance_km for r in subset], dtype=float)
    time_h = np.array([r.time_h for r in subset], dtype=float)
    t_ref_h = get_marathon_reference_time_h(dist, time_h)

    models = [
        (
            "fixed_riegel",
            lambda d, t: model_fixed_riegel_fit(d, t, t_ref_h),
            lambda p, d: model_fixed_riegel_predict(p, d, t_ref_h),
        ),
        (
            "piecewise_riegel_106_linear",
            lambda d, t: model_piecewise_riegel_106_linear_fit(d, t, t_ref_h),
            lambda p, d: model_piecewise_riegel_106_linear_predict(p, d, t_ref_h),
        ),
        (
            "piecewise_riegel_106_sqrt",
            lambda d, t: model_piecewise_riegel_106_sqrt_fit(d, t, t_ref_h),
            lambda p, d: model_piecewise_riegel_106_sqrt_predict(p, d, t_ref_h),
        ),
        (
            "piecewise_riegel_106_mixed",
            lambda d, t: model_piecewise_riegel_106_mixed_fit(d, t, t_ref_h),
            lambda p, d: model_piecewise_riegel_106_mixed_predict(p, d, t_ref_h),
        ),
        (
            "linear_ultra_distance_penalty",
            lambda d, t: model_linear_ultra_distance_penalty_fit(d, t, t_ref_h),
            lambda p, d: model_linear_ultra_distance_penalty_predict(p, d, t_ref_h),
        ),
        (
            "hybrid_marathon_soft_regularized",
            lambda d, t: model_hybrid_marathon_soft_regularized_fit(d, t, t_ref_h),
            lambda p, d: model_hybrid_marathon_soft_regularized_predict(p, d, t_ref_h),
        ),
        (
            "piecewise_marathon_linear_exponent",
            lambda d, t: model_piecewise_marathon_linear_exponent_fit(d, t, t_ref_h),
            lambda p, d: model_piecewise_marathon_linear_exponent_predict(
                p, d, t_ref_h
            ),
        ),
        (
            "log_quadratic_exponent",
            lambda d, t: model_log_quadratic_fit(d, t, t_ref_h),
            lambda p, d: model_log_quadratic_predict(p, d, t_ref_h),
        ),
    ]

    metrics: list[dict] = []
    all_preds: list[dict] = []
    coeff_rows: list[dict] = []
    sub_ultra_mask = dist <= D_ULTRA_KM

    for name, fit_fn, pred_fn in models:
        params = fit_fn(dist, time_h)
        pred = pred_fn(params, dist)

        coeff_rows.append(
            {
                "sex": sex,
                "model": name,
                "t_ref_h": t_ref_h,
                "param_k": float(params[0]),
                "param_c": float(params[1]) if len(params) > 1 else np.nan,
                "param_d": float(params[2]) if len(params) > 2 else np.nan,
            }
        )

        metrics.append(
            {
                "sex": sex,
                "model": name,
                "train_rmse_h": rmse(time_h, pred),
                "train_mape_pct": mape(time_h, pred),
                "sub_ultra_rmse_h": rmse(time_h[sub_ultra_mask], pred[sub_ultra_mask]),
                "sub_ultra_mape_pct": mape(
                    time_h[sub_ultra_mask], pred[sub_ultra_mask]
                ),
                "loocv_rmse_h": loocv_rmse(dist, time_h, fit_fn, pred_fn),
            }
        )

        for r, p in zip(subset, pred):
            all_preds.append(
                {
                    "sex": sex,
                    "model": name,
                    "event": r.event,
                    "distance_km": r.distance_km,
                    "actual_h": r.time_h,
                    "pred_h": float(p),
                    "error_pct": float((p - r.time_h) / r.time_h * 100.0),
                }
            )

    return metrics, all_preds, coeff_rows


def evaluate_combined(
    records: list[Record],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Fit each model with shared coefficients across both sexes; only T_ref differs by sex.

    The combined fit pools all records after normalising each one by its sex-specific marathon
    reference time.  Free parameters (k, c, …) are thus identical for men and women; only the
    anchor T_ref distinguishes predictions between sexes.

    Returns:
        metrics   – one row per (sex, model); sex ∈ {"combined", "men", "women"}.
                    LOOCV is computed on the full combined pool; per-sex LOOCV is left as NaN
                    because the shared-coefficient model is trained on both sexes jointly.
        preds     – per-record predictions using shared coefficients + sex-specific T_ref.
        coeff_rows – one row per model with the single set of shared coefficients.
    """
    # Sex-specific T_ref (fixed from per-sex marathon records)
    t_ref_by_sex: dict[str, float] = {}
    for sex in ("men", "women"):
        subset = [r for r in records if r.sex == sex]
        d_s = np.array([r.distance_km for r in subset], dtype=float)
        t_s = np.array([r.time_h for r in subset], dtype=float)
        t_ref_by_sex[sex] = get_marathon_reference_time_h(d_s, t_s)

    all_sorted = sorted(records, key=lambda r: (r.distance_km, r.sex))
    dist = np.array([r.distance_km for r in all_sorted], dtype=float)
    time_h = np.array([r.time_h for r in all_sorted], dtype=float)
    t_ref_arr = np.array([t_ref_by_sex[r.sex] for r in all_sorted], dtype=float)

    # The module-level fit/predict functions now accept t_ref_h as either a scalar float
    # or a per-record ndarray, so they can be called directly here with t_ref_arr.
    models = [
        ("fixed_riegel", model_fixed_riegel_fit, model_fixed_riegel_predict),
        (
            "piecewise_riegel_106_linear",
            model_piecewise_riegel_106_linear_fit,
            model_piecewise_riegel_106_linear_predict,
        ),
        (
            "piecewise_riegel_106_sqrt",
            model_piecewise_riegel_106_sqrt_fit,
            model_piecewise_riegel_106_sqrt_predict,
        ),
        (
            "piecewise_riegel_106_mixed",
            model_piecewise_riegel_106_mixed_fit,
            model_piecewise_riegel_106_mixed_predict,
        ),
        (
            "linear_ultra_distance_penalty",
            model_linear_ultra_distance_penalty_fit,
            model_linear_ultra_distance_penalty_predict,
        ),
        (
            "hybrid_marathon_soft_regularized",
            model_hybrid_marathon_soft_regularized_fit,
            model_hybrid_marathon_soft_regularized_predict,
        ),
        (
            "piecewise_marathon_linear_exponent",
            model_piecewise_marathon_linear_exponent_fit,
            model_piecewise_marathon_linear_exponent_predict,
        ),
        (
            "log_quadratic_exponent",
            model_log_quadratic_fit,
            model_log_quadratic_predict,
        ),
    ]

    sub_ultra_mask = dist <= D_ULTRA_KM
    sex_masks = {
        s: np.array([r.sex == s for r in all_sorted]) for s in ("men", "women")
    }

    metrics: list[dict] = []
    all_preds: list[dict] = []
    coeff_rows: list[dict] = []

    for name, fit_fn, pred_fn in models:
        params = fit_fn(dist, time_h, t_ref_arr)
        pred = pred_fn(params, dist, t_ref_arr)

        coeff_rows.append(
            {
                "model": name,
                "t_ref_men_h": t_ref_by_sex["men"],
                "t_ref_women_h": t_ref_by_sex["women"],
                "param_k": float(params[0]),
                "param_c": float(params[1]) if len(params) > 1 else np.nan,
                "param_d": float(params[2]) if len(params) > 2 else np.nan,
            }
        )

        # Combined-pool metrics (including LOOCV)
        loocv_val = loocv_rmse(dist, time_h, fit_fn, pred_fn, t_ref_arr=t_ref_arr)
        metrics.append(
            {
                "sex": "combined",
                "model": name,
                "train_rmse_h": rmse(time_h, pred),
                "train_mape_pct": mape(time_h, pred),
                "sub_ultra_rmse_h": rmse(time_h[sub_ultra_mask], pred[sub_ultra_mask]),
                "sub_ultra_mape_pct": mape(
                    time_h[sub_ultra_mask], pred[sub_ultra_mask]
                ),
                "loocv_rmse_h": loocv_val,
            }
        )

        # Per-sex metrics using the shared coefficients
        for sex in ("men", "women"):
            sm = sex_masks[sex]
            ssm = sm & sub_ultra_mask
            metrics.append(
                {
                    "sex": sex,
                    "model": name,
                    "train_rmse_h": rmse(time_h[sm], pred[sm]),
                    "train_mape_pct": mape(time_h[sm], pred[sm]),
                    "sub_ultra_rmse_h": rmse(time_h[ssm], pred[ssm]),
                    "sub_ultra_mape_pct": mape(time_h[ssm], pred[ssm]),
                    "loocv_rmse_h": np.nan,
                }
            )

        for r, p in zip(all_sorted, pred):
            all_preds.append(
                {
                    "sex": r.sex,
                    "model": name,
                    "event": r.event,
                    "distance_km": r.distance_km,
                    "actual_h": r.time_h,
                    "pred_h": float(p),
                    "error_pct": float((p - r.time_h) / r.time_h * 100.0),
                }
            )

    return metrics, all_preds, coeff_rows


def write_excel(
    path: Path,
    metrics_rows: list[dict],
    pred_rows: list[dict],
    coeff_rows: list[dict],
    equation_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(pred_rows)
    coeff_df = pd.DataFrame(coeff_rows)
    equation_df = pd.DataFrame(equation_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        preds_df.to_excel(writer, sheet_name="model_predictions", index=False)
        coeff_df.to_excel(writer, sheet_name="model_coefficients", index=False)
        equation_df.to_excel(writer, sheet_name="model_equations", index=False)


def build_equation_rows(
    coeff_rows: list[dict], d0: float = D_MARATHON_KM
) -> list[dict]:
    out: list[dict] = []
    for row in coeff_rows:
        sex = str(row["sex"])
        model = str(row["model"])
        t_ref = float(row["t_ref_h"])
        k = float(row["param_k"])
        c_val = row["param_c"]
        has_c = not pd.isna(c_val)
        c = float(c_val) if has_c else None
        d_val = row.get("param_d", np.nan)
        has_d = not pd.isna(d_val)
        d = float(d_val) if has_d else None

        if model == "fixed_riegel":
            symbolic = f"T(D)=T_ref*(D/{d0:.3f})^k"
            fitted = f"T(D)={t_ref:.6f}*(D/{d0:.3f})^{k:.6f}"
            notes = "Classic reference-distance Riegel anchored at marathon (k fixed at original value 1.06)"
        elif model == "piecewise_riegel_106_linear":
            symbolic = (
                f"k(D)=1.06 + c*max(D-{d0:.3f},0); " f"T(D)=T_ref*(D/{d0:.3f})^k(D)"
            )
            fitted = (
                f"k(D)=1.06 + {c:.6f}*max(D-{d0:.3f},0); "
                f"T(D)={t_ref:.6f}*(D/{d0:.3f})^k(D)"
            )
            notes = "Requested piecewise coefficient update: fixed 1.06 up to marathon, linear increase after marathon"
        elif model == "piecewise_riegel_106_sqrt":
            symbolic = (
                f"k(D)=1.06 + c*sqrt(max(D-{d0:.3f},0)); "
                f"T(D)=T_ref*(D/{d0:.3f})^k(D)"
            )
            fitted = (
                f"k(D)=1.06 + {c:.6f}*sqrt(max(D-{d0:.3f},0)); "
                f"T(D)={t_ref:.6f}*(D/{d0:.3f})^k(D)"
            )
            notes = "Requested piecewise coefficient update: fixed 1.06 up to marathon, square-root increase after marathon"
        elif model == "piecewise_riegel_106_mixed":
            symbolic = (
                f"k(D)=1.06 + c1*max(D-{d0:.3f},0) + c2*sqrt(max(D-{d0:.3f},0)); "
                f"T(D)=T_ref*(D/{d0:.3f})^k(D)"
            )
            fitted = (
                f"k(D)=1.06 + {c:.6f}*max(D-{d0:.3f},0) + {d:.6f}*sqrt(max(D-{d0:.3f},0)); "
                f"T(D)={t_ref:.6f}*(D/{d0:.3f})^k(D)"
            )
            notes = "Mixed post-marathon coefficient update with linear and square-root terms"
        elif model == "log_quadratic_exponent":
            symbolic = (
                f"T(D)=T_ref*exp(k*log(D/{d0:.3f})"
                f"+q*(log(D/{d0:.3f}))^2*max(D-{d0:.3f},0))"
            )
            fitted = (
                f"T(D)={t_ref:.6f}*exp({k:.6f}*log(D/{d0:.3f})"
                f"+{c:.6f}*(log(D/{d0:.3f}))^2*max(D-{d0:.3f},0))"
            )
            notes = "Piecewise log-quadratic: quadratic term active only above marathon"
        elif model == "hybrid_marathon_soft_regularized":
            symbolic = (
                f"T(D)=T_ref*exp(k*log(D/{d0:.3f})"
                f"+c*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            fitted = (
                f"T(D)={t_ref:.6f}*exp({k:.6f}*log(D/{d0:.3f})"
                f"+{c:.6f}*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            notes = "Same functional form as linear-ultra; k softly regularized to sub-marathon fit"
        elif model == "piecewise_marathon_linear_exponent":
            symbolic = (
                f"T(D)=T_ref*exp(k*log(D/{d0:.3f})"
                f"+c*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            fitted = (
                f"T(D)={t_ref:.6f}*exp({k:.6f}*log(D/{d0:.3f})"
                f"+{c:.6f}*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            notes = "k fit on up-to-marathon data; c fit on ultra data"
        else:
            symbolic = (
                f"T(D)=T_ref*exp(k*log(D/{d0:.3f})"
                f"+c*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            fitted = (
                f"T(D)={t_ref:.6f}*exp({k:.6f}*log(D/{d0:.3f})"
                f"+{c:.6f}*log(D/{d0:.3f})*max(D-{d0:.3f},0))"
            )
            notes = "Global linear ultra-distance penalty"

        out.append(
            {
                "sex": sex,
                "model": model,
                "fit_scope": row.get("fit_scope", "sex_specific"),
                "equation_symbolic": symbolic,
                "equation_fitted": fitted,
                "notes": notes,
            }
        )

    return out


def write_plots(out_dir: Path, records: list[Record], pred_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df = _records_to_df(records)
    preds_df = pd.DataFrame(pred_rows)

    for sex in ("men", "women"):
        sex_actual = records_df[records_df["sex"] == sex].sort_values("distance_km")
        sex_pred = preds_df[preds_df["sex"] == sex].sort_values("distance_km")

        # Plot 1: top panel shows times, bottom panel shows percent errors by distance.
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )

        ax_top.scatter(
            sex_actual["distance_km"],
            sex_actual["actual_h"],
            color="black",
            s=40,
            label="actual records",
            zorder=3,
        )

        for model_name, group in sex_pred.groupby("model"):
            group = group.sort_values("distance_km")
            ax_top.plot(
                group["distance_km"],
                group["pred_h"],
                marker="o",
                linewidth=2,
                label=model_name,
            )

            err_by_dist = (
                group.groupby("distance_km", as_index=False)["error_pct"]
                .mean()
                .sort_values("distance_km")
            )
            ax_bottom.plot(
                err_by_dist["distance_km"],
                err_by_dist["error_pct"],
                marker="o",
                linewidth=1.8,
                label=model_name,
            )

        ax_top.set_title(f"Ultra model fits ({sex})")
        ax_top.set_ylabel("Time (hours)")
        ax_top.grid(alpha=0.3)
        ax_top.legend(loc="best")

        ax_bottom.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_bottom.set_xlabel("Distance (km)")
        ax_bottom.set_ylabel("Error (%)")
        ax_bottom.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"ultra_model_fit_{sex}.png", dpi=160)
        plt.close(fig)

        # Plot 1b: sub-ultra zoom for distances where classic Riegel is intended to work.
        sub_actual = sex_actual[sex_actual["distance_km"] <= D_ULTRA_KM]
        sub_pred = sex_pred[sex_pred["distance_km"] <= D_ULTRA_KM]
        if not sub_actual.empty:
            fig, (ax_top, ax_bottom) = plt.subplots(
                2,
                1,
                figsize=(10, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [2.2, 1.2]},
            )

            ax_top.scatter(
                sub_actual["distance_km"],
                sub_actual["actual_h"],
                color="black",
                s=40,
                label="actual records",
                zorder=3,
            )

            for model_name, group in sub_pred.groupby("model"):
                group = group.sort_values("distance_km")
                ax_top.plot(
                    group["distance_km"],
                    group["pred_h"],
                    marker="o",
                    linewidth=2,
                    label=model_name,
                )

                err_by_dist = (
                    group.groupby("distance_km", as_index=False)["error_pct"]
                    .mean()
                    .sort_values("distance_km")
                )
                ax_bottom.plot(
                    err_by_dist["distance_km"],
                    err_by_dist["error_pct"],
                    marker="o",
                    linewidth=1.8,
                    label=model_name,
                )

            ax_top.set_title(f"Sub-ultra model fits ({sex}, <= {D_ULTRA_KM:.0f} km)")
            ax_top.set_ylabel("Time (hours)")
            ax_top.grid(alpha=0.3)
            ax_top.legend(loc="best")

            ax_bottom.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            ax_bottom.set_xlabel("Distance (km)")
            ax_bottom.set_ylabel("Error (%)")
            ax_bottom.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(out_dir / f"ultra_model_fit_sub_ultra_{sex}.png", dpi=160)
            plt.close(fig)

        # Plot 2: residual error (%) by event and model
        pivot = sex_pred.pivot_table(
            index="event",
            columns="model",
            values="error_pct",
            aggfunc="mean",
        ).sort_index()
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_title(f"Model residuals by event ({sex})")
        ax.set_xlabel("Event")
        ax.set_ylabel("Error (%)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"ultra_model_residuals_{sex}.png", dpi=160)
        plt.close()


def write_plots_combined(
    out_dir: Path, records: list[Record], pred_rows: list[dict]
) -> None:
    """Produce an overlay plot for the combined-sex shared-coefficient analysis.

    Both men and women actual records are shown on the same axes together with model
    predictions.  Because only T_ref differs, model curves for men and women differ only
    by a vertical scaling factor.  Men's curves are drawn solid and women's dashed so
    both families are visually distinguishable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df = _records_to_df(records)
    preds_df = pd.DataFrame(pred_rows)

    men_actual = records_df[records_df["sex"] == "men"].sort_values("distance_km")
    women_actual = records_df[records_df["sex"] == "women"].sort_values("distance_km")
    men_pred = preds_df[preds_df["sex"] == "men"].sort_values("distance_km")
    women_pred = preds_df[preds_df["sex"] == "women"].sort_values("distance_km")

    # --- full-distance overlay plot -------------------------------------------------
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax_top.scatter(
        men_actual["distance_km"],
        men_actual["actual_h"],
        color="steelblue",
        marker="o",
        s=50,
        label="men actual",
        zorder=5,
    )
    ax_top.scatter(
        women_actual["distance_km"],
        women_actual["actual_h"],
        color="tomato",
        marker="^",
        s=50,
        label="women actual",
        zorder=5,
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_names = preds_df["model"].unique()
    for idx, model_name in enumerate(model_names):
        color = prop_cycle[idx % len(prop_cycle)]

        m_grp = men_pred[men_pred["model"] == model_name].sort_values("distance_km")
        w_grp = women_pred[women_pred["model"] == model_name].sort_values("distance_km")

        ax_top.plot(
            m_grp["distance_km"],
            m_grp["pred_h"],
            linestyle="-",
            linewidth=1.8,
            color=color,
            label=f"{model_name} (men)",
        )
        ax_top.plot(
            w_grp["distance_km"],
            w_grp["pred_h"],
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=f"{model_name} (women)",
        )

        for grp, ls, sex_label in [(m_grp, "-", "men"), (w_grp, "--", "women")]:
            err = (
                grp.groupby("distance_km", as_index=False)["error_pct"]
                .mean()
                .sort_values("distance_km")
            )
            ax_bottom.plot(
                err["distance_km"],
                err["error_pct"],
                linestyle=ls,
                linewidth=1.5,
                color=color,
            )

    ax_top.set_title("Ultra model fits – shared coefficients, sex-specific T_ref")
    ax_top.set_ylabel("Time (hours)")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=7, ncol=2)

    ax_bottom.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_bottom.set_xlabel("Distance (km)")
    ax_bottom.set_ylabel("Error (%)")
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ultra_model_fit_combined.png", dpi=160)
    plt.close(fig)

    # --- sub-ultra zoom (<=50 km) ---------------------------------------------------
    men_sub_actual = men_actual[men_actual["distance_km"] <= D_ULTRA_KM]
    women_sub_actual = women_actual[women_actual["distance_km"] <= D_ULTRA_KM]
    men_sub_pred = men_pred[men_pred["distance_km"] <= D_ULTRA_KM]
    women_sub_pred = women_pred[women_pred["distance_km"] <= D_ULTRA_KM]

    if not men_sub_actual.empty:
        fig2, (ax2_top, ax2_bottom) = plt.subplots(
            2,
            1,
            figsize=(11, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )

        ax2_top.scatter(
            men_sub_actual["distance_km"],
            men_sub_actual["actual_h"],
            color="steelblue",
            marker="o",
            s=50,
            label="men actual",
            zorder=5,
        )
        ax2_top.scatter(
            women_sub_actual["distance_km"],
            women_sub_actual["actual_h"],
            color="tomato",
            marker="^",
            s=50,
            label="women actual",
            zorder=5,
        )

        for idx, model_name in enumerate(model_names):
            color = prop_cycle[idx % len(prop_cycle)]
            m_grp = men_sub_pred[men_sub_pred["model"] == model_name].sort_values(
                "distance_km"
            )
            w_grp = women_sub_pred[women_sub_pred["model"] == model_name].sort_values(
                "distance_km"
            )

            ax2_top.plot(
                m_grp["distance_km"],
                m_grp["pred_h"],
                linestyle="-",
                linewidth=1.8,
                color=color,
                label=f"{model_name} (men)",
            )
            ax2_top.plot(
                w_grp["distance_km"],
                w_grp["pred_h"],
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=f"{model_name} (women)",
            )

            for grp, ls in [(m_grp, "-"), (w_grp, "--")]:
                err = (
                    grp.groupby("distance_km", as_index=False)["error_pct"]
                    .mean()
                    .sort_values("distance_km")
                )
                ax2_bottom.plot(
                    err["distance_km"],
                    err["error_pct"],
                    linestyle=ls,
                    linewidth=1.5,
                    color=color,
                )

        ax2_top.set_title(
            f"Sub-ultra model fits – shared coefficients (<= {D_ULTRA_KM:.0f} km)"
        )
        ax2_top.set_ylabel("Time (hours)")
        ax2_top.grid(alpha=0.3)
        ax2_top.legend(loc="upper left", fontsize=7, ncol=2)

        ax2_bottom.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax2_bottom.set_xlabel("Distance (km)")
        ax2_bottom.set_ylabel("Error (%)")
        ax2_bottom.grid(alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(out_dir / "ultra_model_fit_combined_sub_ultra.png", dpi=160)
        plt.close(fig2)


def main() -> None:
    records = load_records(DATA_FILE)

    metrics_all: list[dict] = []
    preds_all: list[dict] = []
    coeff_all: list[dict] = []

    for sex in ("men", "women"):
        metrics, preds, coeff_rows = evaluate_group(records, sex)
        metrics_all.extend(metrics)
        preds_all.extend(preds)
        coeff_all.extend(coeff_rows)

    # Rank models per sex by LOOCV RMSE (generalization)
    print("Model validation summary (lower is better):")
    for sex in ("men", "women"):
        group = [m for m in metrics_all if m["sex"] == sex]
        group = sorted(group, key=lambda m: m["loocv_rmse_h"])
        print(f"\n[{sex.upper()}]")
        for m in group:
            print(
                f"  {m['model']:<24} "
                f"train_rmse={m['train_rmse_h']:.3f}h  "
                f"train_mape={m['train_mape_pct']:.2f}%  "
                f"sub_ultra_rmse={m['sub_ultra_rmse_h']:.3f}h  "
                f"sub_ultra_mape={m['sub_ultra_mape_pct']:.2f}%  "
                f"loocv_rmse={m['loocv_rmse_h']:.3f}h"
            )

    # --- combined-sex shared-coefficient analysis -----------------------------------
    combined_metrics, combined_preds, combined_coeff = evaluate_combined(records)

    # Merge combined rows into the existing sheet outputs and tag their origin.
    metrics_sheet_rows = [
        {**row, "fit_scope": "sex_specific"} for row in metrics_all
    ] + [{**row, "fit_scope": "combined_shared"} for row in combined_metrics]

    preds_sheet_rows = [{**row, "fit_scope": "sex_specific"} for row in preds_all] + [
        {**row, "fit_scope": "combined_shared"} for row in combined_preds
    ]

    coeff_sheet_rows = [{**row, "fit_scope": "sex_specific"} for row in coeff_all]
    for row in combined_coeff:
        for sex in ("men", "women"):
            coeff_sheet_rows.append(
                {
                    "sex": sex,
                    "model": row["model"],
                    "t_ref_h": row[f"t_ref_{sex}_h"],
                    "param_k": row["param_k"],
                    "param_c": row["param_c"],
                    "param_d": row["param_d"],
                    "fit_scope": "combined_shared",
                }
            )

    equation_rows = build_equation_rows(coeff_sheet_rows)

    print("\nCombined-sex model summary (shared coefficients, only T_ref differs):")
    combined_pool = [m for m in combined_metrics if m["sex"] == "combined"]
    combined_pool = sorted(combined_pool, key=lambda m: m["loocv_rmse_h"])
    for m in combined_pool:
        print(
            f"  {m['model']:<24} "
            f"train_rmse={m['train_rmse_h']:.3f}h  "
            f"train_mape={m['train_mape_pct']:.2f}%  "
            f"sub_ultra_rmse={m['sub_ultra_rmse_h']:.3f}h  "
            f"sub_ultra_mape={m['sub_ultra_mape_pct']:.2f}%  "
            f"loocv_rmse={m['loocv_rmse_h']:.3f}h"
        )

    out_file = OUT_DIR / "ultra_model_validation.xlsx"
    write_excel(
        out_file,
        metrics_sheet_rows,
        preds_sheet_rows,
        coeff_sheet_rows,
        equation_rows,
    )
    write_plots(OUT_DIR, records, preds_all)
    write_plots_combined(OUT_DIR, records, combined_preds)

    print("\nFitted coefficients by model:")
    coeff_df = pd.DataFrame(coeff_all).sort_values(["sex", "model"])
    for sex in ("men", "women"):
        print(f"\n[{sex.upper()}]")
        group = coeff_df[coeff_df["sex"] == sex]
        for _, row in group.iterrows():
            c_text = "NA" if pd.isna(row["param_c"]) else f"{row['param_c']:.6f}"
            d_text = (
                "NA" if pd.isna(row.get("param_d", np.nan)) else f"{row['param_d']:.6f}"
            )
            print(
                f"  {row['model']:<28} "
                f"T_ref={row['t_ref_h']:.6f}h  k={row['param_k']:.6f}  c={c_text}  d={d_text}"
            )

    print(f"\nWrote: {out_file}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_men.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_women.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_sub_ultra_men.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_sub_ultra_women.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_residuals_men.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_residuals_women.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_combined.png'}")
    print(f"Wrote: {OUT_DIR / 'ultra_model_fit_combined_sub_ultra.png'}")


if __name__ == "__main__":
    main()
