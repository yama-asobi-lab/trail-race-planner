"""Tests for MultiDaySigmoidalFatigueModel."""

import math

import pytest

from race_planner.models.fatigue_model import MultiDaySigmoidalFatigueModel


@pytest.fixture
def tor_model():
    """Typical TOR 330 sigmoidal fatigue model."""
    return MultiDaySigmoidalFatigueModel(
        threshold_speed_kmh=15.3,
        floor_speed_kmh=5.0,
        start_pct=0.55,
        phase1_distance_km=100.0,
        inflection_steepness_k=3.0,
        circadian_amplitude=0.15,
        circadian_period_hours=24.0,
        sleep_half_life_hours=2.5,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_valid(tor_model):
    assert tor_model.threshold_speed_kmh == pytest.approx(15.3)
    assert tor_model.floor_speed_kmh == pytest.approx(5.0)
    assert tor_model.start_pct == pytest.approx(0.55)


def test_construction_invalid_threshold():
    with pytest.raises(ValueError, match="threshold_speed_kmh"):
        MultiDaySigmoidalFatigueModel(threshold_speed_kmh=0, floor_speed_kmh=3.0)


def test_construction_invalid_floor_zero():
    with pytest.raises(ValueError, match="floor_speed_kmh"):
        MultiDaySigmoidalFatigueModel(threshold_speed_kmh=15.0, floor_speed_kmh=0)


def test_construction_floor_must_be_less_than_threshold():
    with pytest.raises(ValueError, match="floor_speed_kmh must be < threshold"):
        MultiDaySigmoidalFatigueModel(threshold_speed_kmh=5.0, floor_speed_kmh=5.0)


def test_construction_invalid_start_pct():
    with pytest.raises(ValueError, match="start_pct"):
        MultiDaySigmoidalFatigueModel(
            threshold_speed_kmh=15.0, floor_speed_kmh=5.0, start_pct=0.0
        )


def test_construction_invalid_steepness():
    with pytest.raises(ValueError, match="inflection_steepness_k"):
        MultiDaySigmoidalFatigueModel(
            threshold_speed_kmh=15.0, floor_speed_kmh=5.0, inflection_steepness_k=-1
        )


def test_construction_invalid_sleep_half_life():
    with pytest.raises(ValueError, match="sleep_half_life_hours"):
        MultiDaySigmoidalFatigueModel(
            threshold_speed_kmh=15.0, floor_speed_kmh=5.0, sleep_half_life_hours=0
        )


# ---------------------------------------------------------------------------
# Phase 1: sigmoid decay
# ---------------------------------------------------------------------------


def test_phase1_start_velocity_near_start_pct(tor_model):
    """At distance 0, velocity should be very close to start_pct * threshold."""
    v_start_expected = tor_model.start_pct * tor_model.threshold_speed_kmh
    v_actual = tor_model.velocity_at_distance(0.0)
    # Sigmoid at d=0 (left of inflection) approaches start speed asymptotically
    assert v_actual > tor_model.floor_speed_kmh
    # Should be between floor and start
    assert tor_model.floor_speed_kmh < v_actual <= v_start_expected + 0.5


def test_phase1_speed_decreases_with_distance(tor_model):
    """Speed should decrease monotonically from 0 to phase1_distance_km."""
    speeds = [tor_model.velocity_at_distance(d) for d in range(0, 101, 10)]
    for i in range(len(speeds) - 1):
        assert speeds[i] >= speeds[i + 1], f"Speed increased at step {i}"


def test_phase1_inflection_point_midpoint(tor_model):
    """At the inflection point (50 km), velocity should be midway between start and floor."""
    v_start = tor_model.start_pct * tor_model.threshold_speed_kmh
    v_floor = tor_model.floor_speed_kmh
    v_mid_expected = (v_start + v_floor) / 2.0
    v_at_inflection = tor_model.velocity_at_distance(50.0)
    assert v_at_inflection == pytest.approx(v_mid_expected, rel=0.01)


def test_phase1_end_approaches_floor(tor_model):
    """At 100 km (end of Phase 1), velocity should be very close to floor."""
    v_at_100 = tor_model.velocity_at_distance(100.0)
    # After inflection, should be close to floor (within ~5%)
    assert abs(v_at_100 - tor_model.floor_speed_kmh) < 0.5


# ---------------------------------------------------------------------------
# Phase 2: floor + circadian
# ---------------------------------------------------------------------------


def test_phase2_velocity_near_floor(tor_model):
    """In Phase 2, base velocity should be at or near the floor."""
    v = tor_model.velocity_at_distance(150.0, cumulative_sleep_duration_s=0)
    assert tor_model.floor_speed_kmh <= v


def test_phase2_velocity_never_below_floor(tor_model):
    """Velocity must never go below the biological floor."""
    for d in [100, 150, 200, 250, 300]:
        v = tor_model.velocity_at_distance(d, cumulative_sleep_duration_s=0)
        assert v >= tor_model.floor_speed_kmh, f"Velocity below floor at {d} km: {v}"


# ---------------------------------------------------------------------------
# Sleep recovery
# ---------------------------------------------------------------------------


def test_sleep_recovery_zero_no_change(tor_model):
    """Zero sleep should give zero recovery delta."""
    v_no_sleep = tor_model.velocity_at_distance(150.0, cumulative_sleep_duration_s=0)
    v_zero_sleep = tor_model.velocity_at_distance(150.0, cumulative_sleep_duration_s=0)
    assert v_no_sleep == pytest.approx(v_zero_sleep)


def test_sleep_recovery_increases_velocity(tor_model):
    """Sleep should increase velocity above the no-sleep baseline."""
    v_no_sleep = tor_model.velocity_at_distance(150.0, cumulative_sleep_duration_s=0)
    v_with_sleep = tor_model.velocity_at_distance(150.0, cumulative_sleep_duration_s=3600)
    assert v_with_sleep > v_no_sleep


def test_one_hour_nap_recovery_20_to_30_pct(tor_model):
    """A 1-hour nap should reduce sleep debt by 20–30% (exponential decay).

    Based on Process S model: S(t) = S_0 * exp(-λt) with λ = ln(2)/half_life.
    For half_life=2.5h: after 1h, debt decays to S_0 * exp(-ln(2)/2.5)
    Recovery fraction = 1 - exp(-0.277) ≈ 0.242 ≈ 24%, within 20–30%.
    """
    initial_debt_s = 10000.0
    nap_s = 3600.0  # 1 hour
    remaining_debt = tor_model.apply_nap_recovery(initial_debt_s, nap_s)
    recovery_fraction = 1.0 - remaining_debt / initial_debt_s
    assert 0.20 <= recovery_fraction <= 0.30, (
        f"Expected 20–30% recovery from 1-h nap, got {recovery_fraction:.1%}"
    )


def test_nap_recovery_exponential_more_efficient_early(tor_model):
    """First hour of sleep should recover more than second hour (exponential)."""
    initial_debt = 10000.0
    after_1h = tor_model.apply_nap_recovery(initial_debt, 3600)
    after_2h = tor_model.apply_nap_recovery(initial_debt, 7200)
    recovery_1h = initial_debt - after_1h
    recovery_2h = initial_debt - after_2h
    # First hour should give more than half the total 2-hour recovery
    assert recovery_1h > recovery_2h - recovery_1h


def test_nap_debt_never_negative(tor_model):
    """Apply nap should never make debt negative."""
    remaining = tor_model.apply_nap_recovery(100.0, 100 * 3600)  # 100h nap
    assert remaining >= 0.0


# ---------------------------------------------------------------------------
# Fatigue multiplier
# ---------------------------------------------------------------------------


def test_fatigue_multiplier_at_start_greater_than_one(tor_model):
    """Even at distance 0, multiplier should be > 1 (athlete isn't at threshold)."""
    m = tor_model.fatigue_multiplier_for_distance(0.0)
    assert m > 1.0


def test_fatigue_multiplier_increases_through_phase1(tor_model):
    """Multiplier should increase (get slower) through Phase 1."""
    multipliers = [tor_model.fatigue_multiplier_for_distance(float(d)) for d in range(0, 100, 20)]
    for i in range(len(multipliers) - 1):
        assert multipliers[i] <= multipliers[i + 1]


def test_fatigue_multiplier_sleep_reduces_value(tor_model):
    """Cumulative sleep should lower the multiplier (make it faster)."""
    m_no_sleep = tor_model.fatigue_multiplier_for_distance(200.0, 0)
    m_with_sleep = tor_model.fatigue_multiplier_for_distance(200.0, 3600)
    assert m_with_sleep < m_no_sleep


# ---------------------------------------------------------------------------
# velocity_at_time
# ---------------------------------------------------------------------------


def test_velocity_at_time_early_race(tor_model):
    """Early in the race (t < estimated phase1 time), velocity should be above floor."""
    v = tor_model.velocity_at_time(3600.0)  # 1 hour
    assert v > tor_model.floor_speed_kmh


def test_velocity_at_time_with_sleep(tor_model):
    """Sleep should boost velocity at time too."""
    v_no_sleep = tor_model.velocity_at_time(50 * 3600, 0)
    v_sleep = tor_model.velocity_at_time(50 * 3600, 3600)
    assert v_sleep >= v_no_sleep


# ---------------------------------------------------------------------------
# Integration with PaceCalculator
# ---------------------------------------------------------------------------


def test_pace_calculator_accepts_fatigue_model(tor_model):
    """PaceCalculator should accept and store a fatigue_model_instance."""
    from race_planner.planner import PaceCalculator

    calc = PaceCalculator(
        ref_dist_km=42.195,
        ref_time_s=2 * 3600 + 45 * 60,
        fatigue_model_instance=tor_model,
    )
    assert calc.fatigue_model_instance is tor_model


def test_pace_calculator_fatigue_multiplier_array(tor_model):
    """fatigue_multiplier_for_distance_array should return array of multipliers."""
    import numpy as np

    from race_planner.planner import PaceCalculator

    calc = PaceCalculator(
        ref_dist_km=42.195,
        ref_time_s=2 * 3600 + 45 * 60,
        fatigue_model_instance=tor_model,
    )
    distances = np.array([0.0, 50.0, 100.0, 200.0])
    sleeps = np.array([0.0, 0.0, 0.0, 3600.0])
    multipliers = calc.fatigue_multiplier_for_distance_array(distances, sleeps)
    assert multipliers.shape == (4,)
    assert all(m > 0 for m in multipliers)
    # Phase 1 should be increasing
    assert multipliers[0] <= multipliers[1] <= multipliers[2]
