"""Tests for shared utility helpers."""

import numpy as np
import pytest

from race_planner.models.tools import (
    hhmm_to_hours,
    pace_from_constant_vertical_speed,
    pace_to_seconds_per_km,
    pace_to_speed_kmh,
    race_offset_to_clock_hhmm,
    seconds_per_km_to_pace,
    vertical_speed_m_per_h,
)


def test_pace_to_seconds_per_km_parses_mm_ss():
    assert pace_to_seconds_per_km("3:50/km") == pytest.approx(230.0)


def test_pace_to_seconds_per_km_parses_hh_mm_ss():
    assert pace_to_seconds_per_km("0:03:50/km") == pytest.approx(230.0)


def test_seconds_per_km_to_pace_formats_expected_string():
    assert seconds_per_km_to_pace(230.4) == "3:50/km"


def test_pace_to_seconds_per_km_rejects_invalid_format():
    with pytest.raises(ValueError, match="Unsupported pace format"):
        pace_to_seconds_per_km("350/km")


def test_speed_and_vertical_speed_helpers_are_consistent():
    pace_sec_per_km = np.array([300.0])
    grade = np.array([0.10])

    speed_kmh = pace_to_speed_kmh(pace_sec_per_km)
    vertical_speed = vertical_speed_m_per_h(grade, pace_sec_per_km)
    recovered_pace = pace_from_constant_vertical_speed(grade, float(vertical_speed[0]))

    assert speed_kmh[0] == pytest.approx(12.0)
    assert vertical_speed[0] == pytest.approx(1200.0)
    assert recovered_pace[0] == pytest.approx(pace_sec_per_km[0])


# ---------------------------------------------------------------------------
# hhmm_to_hours
# ---------------------------------------------------------------------------


def test_hhmm_to_hours_typical_cases():
    assert hhmm_to_hours("12:00") == pytest.approx(12.0)
    assert hhmm_to_hours("00:00") == pytest.approx(0.0)
    assert hhmm_to_hours("16:30") == pytest.approx(16.5)


def test_hhmm_to_hours_rejects_hms_format():
    with pytest.raises(ValueError, match="Expected HH:MM"):
        hhmm_to_hours("16:30:00")


def test_hhmm_to_hours_rejects_out_of_range_hour():
    with pytest.raises(ValueError, match="Invalid time value"):
        hhmm_to_hours("25:00")


def test_hhmm_to_hours_rejects_out_of_range_minute():
    with pytest.raises(ValueError, match="Invalid time value"):
        hhmm_to_hours("10:60")


# ---------------------------------------------------------------------------
# race_offset_to_clock_hhmm
# ---------------------------------------------------------------------------


def test_race_offset_to_clock_hhmm_same_day():
    # Start 16:00, offset 2.5 h → 18:30
    assert race_offset_to_clock_hhmm(16.0, 2.5) == "18:30"


def test_race_offset_to_clock_hhmm_midnight_rollover():
    # Start 22:00, offset 3 h → 01:00 next day
    assert race_offset_to_clock_hhmm(22.0, 3.0) == "01:00"
    # Start 22:00, offset 25 h → 23:00 next day
    assert race_offset_to_clock_hhmm(22.0, 25.0) == "23:00"


def test_race_offset_to_clock_hhmm_zero_offset():
    assert race_offset_to_clock_hhmm(16.0, 0.0) == "16:00"
