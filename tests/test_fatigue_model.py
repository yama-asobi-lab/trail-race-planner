"""Tests for fatigue models (LinearFatigueModel and MultiDaySigmoidalFatigueModel)."""

import pytest

from race_planner.models.fatigue_model import LinearFatigueModel, MultiDaySigmoidalFatigueModel


@pytest.fixture
def tor_model():
    """Sigmoidal fatigue model representative of a TOR-class race."""
    return MultiDaySigmoidalFatigueModel(
        threshold_speed_kmh=15.3,
        floor_speed_kmh=5.0,
        start_pct=0.55,
        s_0=0.65,
        t_0_hours=18.0,
        k_0=0.45,
        circadian_amplitude=0.15,
        circadian_period_hours=24.0,
        sleep_half_life_hours=2.5,
    )


# ---------------------------------------------------------------------------
# LinearFatigueModel
# ---------------------------------------------------------------------------


def test_linear_no_decay():
    m = LinearFatigueModel(total_decay_pct=0)
    assert m.fatigue_multiplier(0.0) == pytest.approx(1.0)
    assert m.fatigue_multiplier(1.0) == pytest.approx(1.0)


def test_linear_multiplier_increases_with_progress():
    m = LinearFatigueModel(total_decay_pct=14)
    assert m.fatigue_multiplier(0.0) == pytest.approx(1.0)
    assert m.fatigue_multiplier(1.0) > m.fatigue_multiplier(0.5)


def test_linear_invalid_decay():
    with pytest.raises(ValueError):
        LinearFatigueModel(total_decay_pct=101)


# ---------------------------------------------------------------------------
# MultiDaySigmoidalFatigueModel — construction
# ---------------------------------------------------------------------------


def test_sigmoid_construction_valid(tor_model):
    assert tor_model.threshold_speed_kmh == pytest.approx(15.3)
    assert tor_model.floor_speed_kmh == pytest.approx(5.0)
    assert tor_model.start_pct == pytest.approx(0.55)


def test_sigmoid_construction_invalid():
    with pytest.raises(ValueError, match="floor_speed_kmh must be < threshold"):
        MultiDaySigmoidalFatigueModel(threshold_speed_kmh=5.0, floor_speed_kmh=5.0)


# ---------------------------------------------------------------------------
# Velocity behaviour
# ---------------------------------------------------------------------------


def test_velocity_decreases_over_time(tor_model):
    """Velocity should decline from the early race to many hours in."""
    v_early = tor_model.velocity_at_time(1 * 3600)
    v_late = tor_model.velocity_at_time(50 * 3600)
    assert v_early > v_late


def test_velocity_never_below_floor(tor_model):
    for t_hours in [1, 10, 30, 80, 150]:
        v = tor_model.velocity_at_time(t_hours * 3600)
        assert v >= tor_model.floor_speed_kmh, f"velocity below floor at {t_hours}h: {v}"


def test_distance_time_inversion_matches_integrated_velocity(tor_model):
    """Distance-time conversion must be consistent with the actual velocity curve."""
    target_distance_km = 100.0
    elapsed_hours = tor_model.elapsed_hours_for_distance(target_distance_km)
    distance_from_time = tor_model.distance_travelled_in_time(elapsed_hours)
    assert distance_from_time == pytest.approx(target_distance_km, rel=1e-3)
    assert tor_model.velocity_at_distance(target_distance_km) == pytest.approx(
        tor_model.velocity_at_time(elapsed_hours * 3600.0), rel=1e-6
    )


def test_elapsed_time_increases_monotonically_with_distance(tor_model):
    distances = [0.0, 10.0, 50.0, 100.0, 200.0]
    times = [tor_model.elapsed_hours_for_distance(d) for d in distances]
    assert times == pytest.approx(sorted(times), rel=1e-12)


def test_inflection_at_t0_for_reference_intensity():
    """At S == S_0, inflection should occur at t_0_hours."""
    model = MultiDaySigmoidalFatigueModel(
        threshold_speed_kmh=15.3,
        floor_speed_kmh=5.0,
        start_pct=0.65,  # equals s_0
        s_0=0.65,
        t_0_hours=18.0,
        k_0=0.45,
        circadian_amplitude=0.0,  # disable circadian for clarity
    )
    v_start = model._v_start
    v_mid_expected = (v_start + model.floor_speed_kmh) / 2.0
    v_at_inflection = model.velocity_at_time(18.0 * 3600)
    assert v_at_inflection == pytest.approx(v_mid_expected, rel=0.01)


def test_faster_start_inflects_earlier():
    """A higher starting intensity should produce an earlier inflection."""
    common = dict(
        threshold_speed_kmh=15.3,
        floor_speed_kmh=5.0,
        s_0=0.65,
        t_0_hours=18.0,
        k_0=0.45,
        circadian_amplitude=0.0,
    )
    m_conservative = MultiDaySigmoidalFatigueModel(start_pct=0.55, **common)
    m_aggressive = MultiDaySigmoidalFatigueModel(start_pct=0.75, **common)
    assert m_aggressive._t_inflection < m_conservative._t_inflection


# ---------------------------------------------------------------------------
# Sleep recovery
# ---------------------------------------------------------------------------


def test_sleep_increases_velocity(tor_model):
    v_no_sleep = tor_model.velocity_at_time(50 * 3600, cumulative_sleep_duration_s=0)
    v_with_sleep = tor_model.velocity_at_time(50 * 3600, cumulative_sleep_duration_s=3600)
    assert v_with_sleep > v_no_sleep


def test_one_hour_nap_recovery_20_to_30_pct(tor_model):
    """A 1-hour nap should clear 20–30% of accumulated sleep debt."""
    initial_debt_s = 10000.0
    remaining = tor_model.apply_nap_recovery(initial_debt_s, 3600.0)
    recovery_fraction = 1.0 - remaining / initial_debt_s
    assert 0.20 <= recovery_fraction <= 0.30, f"got {recovery_fraction:.1%}"


# ---------------------------------------------------------------------------
# Fatigue multiplier
# ---------------------------------------------------------------------------


def test_fatigue_multiplier_increases_with_distance():
    """Multiplier should increase (get slower) through early race without circadian noise."""
    model = MultiDaySigmoidalFatigueModel(
        threshold_speed_kmh=15.3,
        floor_speed_kmh=5.0,
        start_pct=0.55,
        s_0=0.65,
        t_0_hours=18.0,
        k_0=0.45,
        circadian_amplitude=0.0,  # isolate sigmoid decay
    )
    distances = [float(d) for d in range(0, 81, 10)]
    multipliers = [model.pace_multiplier_for_distance(d) for d in distances]
    for i in range(len(multipliers) - 1):
        assert multipliers[i] <= multipliers[i + 1], (
            f"multiplier decreased from {multipliers[i]:.4f} to {multipliers[i+1]:.4f} "
            f"at d={distances[i+1]}"
        )


def test_fatigue_multiplier_sleep_reduces_value(tor_model):
    m_no_sleep = tor_model.pace_multiplier_for_distance(200.0, 0)
    m_with_sleep = tor_model.pace_multiplier_for_distance(200.0, 3600)
    assert m_with_sleep < m_no_sleep


# ---------------------------------------------------------------------------
# Integration with PaceCalculator
# ---------------------------------------------------------------------------


def test_pace_calculator_accepts_fatigue_model(tor_model):
    from race_planner.planner import PaceCalculator

    calc = PaceCalculator(
        ref_dist_km=42.195,
        ref_time_s=2 * 3600 + 45 * 60,
        fatigue_model_instance=tor_model,
    )
    assert calc.fatigue_model_instance is tor_model


def test_pace_calculator_fatigue_multiplier_array(tor_model):
    import numpy as np

    from race_planner.planner import PaceCalculator

    calc = PaceCalculator(
        ref_dist_km=42.195,
        ref_time_s=2 * 3600 + 45 * 60,
        fatigue_model_instance=tor_model,
    )
    distances = np.array([0.0, 50.0, 100.0, 200.0])
    sleeps = np.array([0.0, 0.0, 0.0, 3600.0])
    multipliers = calc.fatigue_multiplier(
        np.zeros(4),  # progress_fraction_values (unused for sigmoid)
        distances,
        sleeps,
    )
    assert multipliers.shape == (4,)
    assert all(m > 0 for m in multipliers)
