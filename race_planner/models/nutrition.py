"""Nutrition models for race fueling planning."""

from math import log

import numpy as np


def caffeine_concentration_mg_per_kg(
    time_h: np.ndarray,
    ingestion_plan: list[tuple[float, float]],
    weight_kg: float,
    absorption_lag_h: float,
    half_life_h: float,
) -> np.ndarray:
    """Return caffeine concentration curve in mg/kg over race time.

    Models first-order absorption (step at absorption_lag_h after each dose)
    and first-order elimination (exponential decay with the given half-life).

    Args:
        time_h: Array of race-time points in hours.
        ingestion_plan: List of (dose_time_h, dose_mg) tuples.
        weight_kg: Athlete body weight in kg.
        absorption_lag_h: Time after ingestion at which full dose enters circulation.
        half_life_h: Caffeine plasma half-life in hours.

    Returns:
        Array of caffeine concentration values in mg/kg, same shape as time_h.
    """
    elimination_k = log(2.0) / half_life_h
    concentration = np.zeros_like(time_h, dtype=float)

    for dose_time_h, dose_mg in ingestion_plan:
        absorbed_time_h = dose_time_h + absorption_lag_h
        dt = time_h - absorbed_time_h
        active = dt >= 0.0
        concentration[active] += (dose_mg / weight_kg) * np.exp(-elimination_k * dt[active])

    return concentration
