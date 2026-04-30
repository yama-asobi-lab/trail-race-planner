"""Tests for race_planner.models.nutrition"""

import numpy as np
import pytest

from race_planner.models.nutrition import caffeine_concentration_mg_per_kg

WEIGHT_KG = 70.0
HALF_LIFE_H = 5.5
ABSORPTION_LAG_H = 0.5
SINGLE_DOSE_MG = 200.0


def test_concentration_is_zero_before_absorption():
    """No caffeine in blood before the absorption lag has elapsed."""
    time_h = np.array([0.0, 0.25, 0.49])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert np.all(conc == 0.0)


def test_concentration_peaks_at_absorption_lag():
    """Concentration is highest exactly at the absorption lag (no decay yet)."""
    # Sample time points: just before, at, and after absorption
    lag = ABSORPTION_LAG_H
    time_h = np.array([lag - 1e-9, lag, lag + 1.0])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    peak = SINGLE_DOSE_MG / WEIGHT_KG
    assert conc[1] == pytest.approx(peak)
    assert conc[2] < conc[1]


def test_concentration_halves_after_one_half_life():
    """Concentration at absorption_lag + half_life should be half the peak."""
    peak_time = ABSORPTION_LAG_H
    half_life_time = peak_time + HALF_LIFE_H
    time_h = np.array([peak_time, half_life_time])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert conc[1] == pytest.approx(conc[0] / 2.0, rel=1e-6)


def test_concentration_superposition_of_two_doses():
    """Two simultaneous doses should produce exactly double the concentration."""
    time_h = np.linspace(0.0, 10.0, 50)
    single = caffeine_concentration_mg_per_kg(
        time_h, [(0.0, SINGLE_DOSE_MG)], WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    double = caffeine_concentration_mg_per_kg(
        time_h,
        [(0.0, SINGLE_DOSE_MG), (0.0, SINGLE_DOSE_MG)],
        WEIGHT_KG,
        ABSORPTION_LAG_H,
        HALF_LIFE_H,
    )
    np.testing.assert_allclose(double, 2.0 * single)


def test_concentration_is_non_negative_for_any_plan():
    """Concentration must never go negative regardless of dosing plan."""
    time_h = np.linspace(0.0, 30.0, 300)
    ingestion_plan = [(0.0, 100.0), (5.0, 200.0), (12.0, 150.0)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert np.all(conc >= 0.0)


def test_empty_ingestion_plan_yields_zero_concentration():
    time_h = np.linspace(0.0, 10.0, 100)
    conc = caffeine_concentration_mg_per_kg(time_h, [], WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H)
    assert np.all(conc == 0.0)
