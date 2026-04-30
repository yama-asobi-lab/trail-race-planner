"""Tests for shared utility helpers."""

import numpy as np
import pytest

from race_planner.models.tools import (
    pace_from_constant_vertical_speed,
    pace_to_seconds_per_km,
    seconds_per_km_to_pace,
    speed_kmh_from_pace,
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

    speed_kmh = speed_kmh_from_pace(pace_sec_per_km)
    vertical_speed = vertical_speed_m_per_h(grade, pace_sec_per_km)
    recovered_pace = pace_from_constant_vertical_speed(grade, float(vertical_speed[0]))

    assert speed_kmh[0] == pytest.approx(12.0)
    assert vertical_speed[0] == pytest.approx(1200.0)
    assert recovered_pace[0] == pytest.approx(pace_sec_per_km[0])
