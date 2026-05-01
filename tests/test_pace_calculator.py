"""Tests for PaceCalculator."""

import numpy as np
import pytest

from race_planner.planner import PaceCalculator
from race_planner.course.course import Course


@pytest.fixture
def carlos_calc():
    """PaceCalculator with Carlos marathon reference: 42.195 km / 2:50:00."""
    return PaceCalculator(ref_dist_km=42.195, ref_time_s=2 * 3600 + 50 * 60)


@pytest.fixture
def tgt_course(sample_gpx_path):
    return Course(sample_gpx_path, resample_m=5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_athlete_config_carlos(carlos_config):
    calc = PaceCalculator.from_athlete_config(carlos_config)
    assert calc.ref_dist_km == pytest.approx(42.195)
    assert calc.ref_time_s == pytest.approx(2 * 3600 + 50 * 60)


def test_from_athlete_config_yas(yas_config):
    calc = PaceCalculator.from_athlete_config(yas_config)
    assert calc.ref_dist_km == pytest.approx(42.195)
    assert calc.ref_time_s == pytest.approx(4 * 3600)


def test_invalid_ref_dist():
    with pytest.raises(ValueError, match="ref_dist_km"):
        PaceCalculator(ref_dist_km=0, ref_time_s=12600)


def test_invalid_ref_time():
    with pytest.raises(ValueError, match="ref_time_s"):
        PaceCalculator(ref_dist_km=42.195, ref_time_s=0)


def test_custom_gap_curve_is_sorted():
    """Custom curve provided in arbitrary order should be sorted ascending."""
    curve = [[-0.01, 0.97], [0.01, 1.08], [0.00, 1.00]]
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600, gap_curve=curve)
    assert list(calc.gap_curve[:, 0]) == sorted(calc.gap_curve[:, 0])


# ---------------------------------------------------------------------------
# Riegel formula
# ---------------------------------------------------------------------------


def test_riegel_at_reference_distance(carlos_calc):
    """Riegel at the reference distance must return the reference time."""
    result = carlos_calc.predict_riegel_flat_race_time_sec(42.195)
    assert result == pytest.approx(2 * 3600 + 50 * 60, rel=1e-6)


def test_riegel_longer_distance_takes_more_time(carlos_calc):
    """A longer flat race should take proportionally more time."""
    t_half = carlos_calc.predict_riegel_flat_race_time_sec(21.0975)
    t_full = carlos_calc.predict_riegel_flat_race_time_sec(42.195)
    assert t_full > t_half
    # Below the reference distance, the piecewise model reduces to the base 1.06 exponent.
    assert t_full / t_half == pytest.approx(
        2**carlos_calc.RIEGEL_BASE_EXPONENT,
        rel=0.01,
    )


def test_riegel_uses_piecewise_sqrt_exponent(carlos_calc):
    """Above the reference distance, the piecewise sqrt exponent should be used."""
    t_pred = carlos_calc.predict_riegel_flat_race_time_sec(100.0)
    ultra_excess = 100.0 - carlos_calc.ref_dist_km
    exponent = carlos_calc.RIEGEL_BASE_EXPONENT + carlos_calc.PIECEWISE_RIEGEL_106_SQRT_C * np.sqrt(
        ultra_excess
    )
    t_expected = carlos_calc.ref_time_s * (100.0 / carlos_calc.ref_dist_km) ** exponent
    assert t_pred == pytest.approx(t_expected)


# ---------------------------------------------------------------------------
# Flat equivalent distance
# ---------------------------------------------------------------------------


def test_fed_flat_course(carlos_calc):
    """FED of a flat course equals the course distance."""
    assert carlos_calc.flat_equivalent_distance_km(100.0, gain_m=0) == pytest.approx(100.0)


def test_fed_adds_gain(carlos_calc):
    """1000 m of gain adds 10 km FED (100 m/km factor)."""
    fed = carlos_calc.flat_equivalent_distance_km(100.0, gain_m=1000.0)
    assert fed == pytest.approx(110.0)


def test_riegel_fed_greater_than_riegel(carlos_calc):
    """FED-based Riegel always predicts more time than raw-distance Riegel for mountain courses."""
    t_flat = carlos_calc.predict_riegel_flat_race_time_sec(160.0)
    t_fed = carlos_calc.predict_riegel_fed_race_time_sec(160.0, elevation_gain_m=11000.0)
    assert t_fed > t_flat


# ---------------------------------------------------------------------------
# Grade correction
# ---------------------------------------------------------------------------


def test_grade_correction_at_knots():
    """Correction at exact knot points must match the table values exactly."""
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    table = calc.DEFAULT_GAP_CURVE

    grades = table[:, 0]
    expected = table[:, 1]
    result = calc.grade_correction(grades)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_grade_correction_flat_is_one():
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    result = calc.grade_correction(np.array([0.0]))
    assert result[0] == pytest.approx(1.0)


def test_grade_correction_uphill_gt_one():
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    result = calc.grade_correction(np.array([0.05]))  # 5 % uphill
    assert result[0] > 1.0


def test_grade_correction_gentle_downhill_lt_one():
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    result = calc.grade_correction(np.array([-0.03]))  # 3 % downhill
    assert result[0] < 1.0


def test_grade_correction_steep_downhill_gt_one():
    """Very steep descent (beyond -7 %) should slow down (correction > 1)."""
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    result = calc.grade_correction(np.array([-0.20]))  # 20 % downhill
    assert result[0] > 1.0


def test_grade_correction_interpolation():
    """Interpolated value between -1 % and -5 % (e.g. -3.5 %) should lie between their corrections."""
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    c_m1 = calc.grade_correction(np.array([-0.01]))[0]  # 0.97
    c_m5 = calc.grade_correction(np.array([-0.05]))[0]  # 0.85
    c_m35 = calc.grade_correction(np.array([-0.035]))[0]  # should be between
    assert c_m5 < c_m35 < c_m1


def test_grade_correction_uphill_constant_vertical_speed_tail():
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    cutoff = calc.GAP_UPHILL_CUTOFF_GRADE
    c_cutoff = calc.grade_correction(np.array([cutoff]))[0]

    steep_uphill = 0.30
    c_steep = calc.grade_correction(np.array([steep_uphill]))[0]
    expected = c_cutoff * steep_uphill / cutoff

    assert c_steep == pytest.approx(expected, rel=1e-9)

    # Vertical speed is proportional to g / c(g) for a fixed flat pace.
    vspeed_ratio_cutoff = cutoff / c_cutoff
    vspeed_ratio_steep = steep_uphill / c_steep
    assert vspeed_ratio_steep == pytest.approx(vspeed_ratio_cutoff, rel=1e-9)


def test_grade_correction_downhill_constant_vertical_speed_tail():
    calc = PaceCalculator(ref_dist_km=42.195, ref_time_s=12600)
    cutoff = calc.GAP_DOWNHILL_CUTOFF_GRADE
    c_cutoff = calc.grade_correction(np.array([cutoff]))[0]

    steep_downhill = -0.30
    c_steep = calc.grade_correction(np.array([steep_downhill]))[0]
    expected = c_cutoff * steep_downhill / cutoff

    assert c_steep == pytest.approx(expected, rel=1e-9)

    # Vertical speed is proportional to g / c(g) for a fixed flat pace.
    vspeed_ratio_cutoff = cutoff / c_cutoff
    vspeed_ratio_steep = steep_downhill / c_steep
    assert vspeed_ratio_steep == pytest.approx(vspeed_ratio_cutoff, rel=1e-9)


# ---------------------------------------------------------------------------
# calculate_pacing
# ---------------------------------------------------------------------------


def test_calculate_pacing_returns_dataframe(carlos_calc, tgt_course, race_config):
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations)
    assert len(df) == len(aid_stations)


def test_calculate_pacing_columns(carlos_calc, tgt_course, race_config):
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations)
    expected_cols = [
        "Point Name",
        "Total Distance (km)",
        "Elevation (m)",
        "Accum. Elevation Gain (m)",
        "Segment Distance (km)",
        "Segment Elevation Gain (m)",
        "Segment Elevation Loss (m)",
        "Segment Running Time",
        "Stop Time",
        "Elapsed Time",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_calculate_pacing_start_row(carlos_calc, tgt_course, race_config):
    """Start row has zero segment distance and zero running time."""
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations)
    assert df.iloc[0]["Segment Distance (km)"] == 0.0
    assert df.iloc[0]["Segment Running Time"] == "0:00:00"


def test_calculate_pacing_elapsed_time_monotonic(carlos_calc, tgt_course, race_config):
    """Elapsed time must be strictly increasing across the course."""
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations)

    def hms_to_s(t):
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s

    times = [hms_to_s(t) for t in df["Elapsed Time"]]
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


def test_calculate_pacing_total_time_attrs(carlos_calc, tgt_course, race_config):
    """DataFrame attrs should carry total_time_s and riegel_method."""
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations, use_fed=True)
    assert "total_time_s" in df.attrs
    assert df.attrs["riegel_method"] == "FED"
    assert df.attrs["total_time_s"] > 0


def test_calculate_pacing_flat_distance_mode(carlos_calc, tgt_course, race_config):
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations, use_fed=False)
    assert df.attrs["riegel_method"] == "flat-distance"


def test_calculate_pacing_fed_matches_fed_riegel_target(carlos_calc, tgt_course, race_config):
    """FED mode should expose both approximation and integrated running totals."""
    aid_stations = race_config["aid_stations"]
    df_fed = carlos_calc.calculate_pacing(tgt_course, aid_stations, use_fed=True)
    expected_running_s = carlos_calc.predict_riegel_fed_race_time_sec(
        tgt_course.total_distance_km,
        tgt_course.total_elevation_gain_m,
    )
    assert df_fed.attrs["riegel_running_time_approx_s"] == pytest.approx(
        expected_running_s, rel=0.01
    )
    assert df_fed.attrs["grade_adjusted_running_time_s"] == pytest.approx(
        df_fed.attrs["total_running_time_s"], rel=1e-9
    )
    assert df_fed.attrs["total_running_time_s"] > 0


def test_calculate_pacing_reasonable_total_time(carlos_calc, tgt_course, race_config):
    """Carlos (3:30 marathon) on TGT 160km/11000m should finish in 20–40 h."""
    aid_stations = race_config["aid_stations"]
    df = carlos_calc.calculate_pacing(tgt_course, aid_stations, use_fed=True)
    total_hours = df.attrs["total_time_s"] / 3600
    assert 20 <= total_hours <= 40, f"Unexpected total time: {total_hours:.1f} h"


def test_calculate_pacing_from_athlete_config(carlos_config, tgt_course, race_config):
    """End-to-end: build calculator from athlete YAML and run pacing plan."""
    calc = PaceCalculator.from_athlete_config(carlos_config)
    aid_stations = race_config["aid_stations"]
    df = calc.calculate_pacing(tgt_course, aid_stations)
    assert len(df) == len(aid_stations)
    assert df.attrs["total_time_s"] > 0
