"""Tests for pure PacingModel logic."""

import numpy as np
import pytest

from race_planner.models import PacingModel


@pytest.fixture
def model():
    return PacingModel(ref_dist_km=42.195, ref_time_s=2 * 3600 + 50 * 60)


def test_from_athlete_config(carlos_config):
    m = PacingModel.from_athlete_config(carlos_config)
    assert m.ref_dist_km == pytest.approx(42.195)
    assert m.ref_time_s == pytest.approx(2 * 3600 + 50 * 60)


def test_piecewise_riegel_uses_sqrt_term(model):
    t_pred = model.predict_riegel_flat_race_time_sec(100.0)
    ultra_excess = 100.0 - model.ref_dist_km
    exponent = model.RIEGEL_BASE_EXPONENT + model.PIECEWISE_RIEGEL_106_SQRT_C * np.sqrt(
        ultra_excess
    )
    t_expected = model.ref_time_s * (100.0 / model.ref_dist_km) ** exponent
    assert t_pred == pytest.approx(t_expected)


def test_grade_correction_constant_vertical_speed_tail(model):
    up_cutoff = model.GAP_UPHILL_CUTOFF_GRADE
    c_up_cutoff = model.grade_correction(np.array([up_cutoff]))[0]
    c_up_steep = model.grade_correction(np.array([0.30]))[0]
    assert c_up_steep == pytest.approx(c_up_cutoff * 0.30 / up_cutoff)

    down_cutoff = model.GAP_DOWNHILL_CUTOFF_GRADE
    c_down_cutoff = model.grade_correction(np.array([down_cutoff]))[0]
    c_down_steep = model.grade_correction(np.array([-0.30]))[0]
    assert c_down_steep == pytest.approx(c_down_cutoff * -0.30 / down_cutoff)
