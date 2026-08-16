"""
Bi-phasic sigmoidal fatigue model for expedition-length mountain races.

This module implements ``MultiDaySigmoidalFatigueModel``, a physiologically-grounded
fatigue model designed for ultra-endurance events such as Tor des Géants (TOR 330).

Unlike single-sigmoid or linear decay models, this model accounts for:

1. **Bi-phasic decay**: The body undergoes a different degradation profile during
   the first ~100 km (Phase 1) than during the remainder of the race (Phase 2).

2. **Low starting intensity**: TOR-class athletes begin at 50–60 % of Lactate
   Threshold—already close to the biological fatigue floor—meaning the velocity
   delta between start and floor is narrow.

3. **Sleep-recovery modelling**: Once the fatigue floor is reached, circadian
   rhythm oscillation and exponential recovery from nap blocks govern pace.

Mathematical foundation
-----------------------
Phase 1 (0 → ``phase1_distance_km``):
    Logistic (sigmoid) decay from starting speed to floor speed::

        V(d) = V_floor + (V_start - V_floor) / (1 + exp(k * (d - d_inflection)))

    where ``d_inflection`` = ``phase1_distance_km / 2`` and ``k`` =
    ``inflection_steepness_k / phase1_distance_km``.

Phase 2 (``phase1_distance_km`` → finish):
    Oscillation around floor with exponential sleep recovery::

        V(t) = V_floor · (1 + A·sin(2π(t - φ) / T)) + f(sleep_recovery)

    Sleep recovery (Process S exponential decay)::

        ΔV_sleep = (V_start - V_floor) · (1 - exp(-λ·nap_hours))

    where ``λ = ln(2) / sleep_half_life_hours``.

Tuning guidelines
-----------------
* ``start_pct``            – 0.55–0.60 for TOR, 0.75–0.80 for 100-mile ultras.
* ``inflection_steepness_k`` – higher values produce a sharper Phase 1 drop.
  Values of 3–5 are typical.
* ``circadian_amplitude``  – ±10–20 % around the floor; 0.15 is a good default.
* ``sleep_half_life_hours`` – 2.0–3.0 h for the Process S model; 2.5 h default.
"""

from __future__ import annotations

import math
from typing import Optional


class MultiDaySigmoidalFatigueModel:
    """
    Bi-phasic sigmoidal fatigue model for expedition-length mountain races.

    Instantiate from a race/athlete configuration and inject into
    :class:`~race_planner.planner.pace_calculator.PaceCalculator` via the
    ``fatigue_model_instance`` parameter.

    Args:
        threshold_speed_kmh:
            Lactate Threshold speed in km/h. Used as the upper reference for
            the velocity range.
        floor_speed_kmh:
            Minimum sustainable speed (biological floor) in km/h. The athlete
            will not drop below this under normal circumstances.
        start_pct:
            Starting speed as a fraction of ``threshold_speed_kmh``.
            Typical for TOR: 0.55 (55 %).
        phase1_distance_km:
            Distance (km) over which Phase 1 degradation occurs.  Degradation
            is modelled as a sigmoid centred at half this distance.
            Default: 100 km.
        inflection_steepness_k:
            Sharpness of the Phase 1 sigmoid.  Higher values produce a sharper
            speed drop.  Default: 3.0.
        circadian_amplitude:
            Amplitude of circadian rhythm oscillation as a fraction of
            ``floor_speed_kmh``.  Default: 0.15 (±15 %).
        circadian_period_hours:
            Period of the circadian cycle in hours.  Default: 24.0.
        circadian_phase_offset_hours:
            Phase shift of the circadian sine wave in hours.  A value of 6
            means peak performance occurs at hour 6 (i.e., mid-morning for a
            10:00 start).  Default: 6.0.
        sleep_half_life_hours:
            Half-life for exponential sleep-pressure recovery (Process S
            model).  Default: 2.5 hours.
        floor_vertical_ascent_speed_kmh:
            Optional: vertical ascent speed at the fatigue floor (m/h) for
            GAP calculations.  Currently informational; may be used in future
            gradient-adjusted pace extensions.
    """

    def __init__(
        self,
        threshold_speed_kmh: float,
        floor_speed_kmh: float,
        start_pct: float = 0.55,
        phase1_distance_km: float = 100.0,
        inflection_steepness_k: float = 3.0,
        circadian_amplitude: float = 0.15,
        circadian_period_hours: float = 24.0,
        circadian_phase_offset_hours: float = 6.0,
        sleep_half_life_hours: float = 2.5,
        floor_vertical_ascent_speed_kmh: Optional[float] = None,
    ) -> None:
        if threshold_speed_kmh <= 0:
            raise ValueError("threshold_speed_kmh must be > 0")
        if floor_speed_kmh <= 0:
            raise ValueError("floor_speed_kmh must be > 0")
        if floor_speed_kmh >= threshold_speed_kmh:
            raise ValueError("floor_speed_kmh must be < threshold_speed_kmh")
        if not 0.0 < start_pct <= 1.0:
            raise ValueError("start_pct must be in (0, 1]")
        if phase1_distance_km <= 0:
            raise ValueError("phase1_distance_km must be > 0")
        if inflection_steepness_k <= 0:
            raise ValueError("inflection_steepness_k must be > 0")
        if circadian_amplitude < 0:
            raise ValueError("circadian_amplitude must be >= 0")
        if circadian_period_hours <= 0:
            raise ValueError("circadian_period_hours must be > 0")
        if sleep_half_life_hours <= 0:
            raise ValueError("sleep_half_life_hours must be > 0")

        self.threshold_speed_kmh = float(threshold_speed_kmh)
        self.floor_speed_kmh = float(floor_speed_kmh)
        self.start_pct = float(start_pct)
        self.phase1_distance_km = float(phase1_distance_km)
        self.inflection_steepness_k = float(inflection_steepness_k)
        self.circadian_amplitude = float(circadian_amplitude)
        self.circadian_period_hours = float(circadian_period_hours)
        self.circadian_phase_offset_hours = float(circadian_phase_offset_hours)
        self.sleep_half_life_hours = float(sleep_half_life_hours)
        self.floor_vertical_ascent_speed_kmh = (
            float(floor_vertical_ascent_speed_kmh)
            if floor_vertical_ascent_speed_kmh is not None
            else None
        )

        self._v_start = self.threshold_speed_kmh * self.start_pct
        self._v_delta = self._v_start - self.floor_speed_kmh
        # Pre-compute sigmoid parameters for Phase 1
        self._d_inflection = self.phase1_distance_km / 2.0
        self._k_scaled = self.inflection_steepness_k / self.phase1_distance_km
        # Sleep recovery rate constant (ln(2) / half-life)
        self._sleep_lambda = math.log(2.0) / self.sleep_half_life_hours

    # ------------------------------------------------------------------
    # Core velocity methods
    # ------------------------------------------------------------------

    def velocity_at_distance(
        self,
        distance_km: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Return estimated velocity (km/h) at a given race distance.

        Combines Phase 1 sigmoid decay, Phase 2 circadian oscillation, and
        exponential sleep-pressure recovery.

        Args:
            distance_km: Cumulative race distance in km.
            cumulative_sleep_duration_s: Total sleep accumulated up to this
                point in the race, in seconds.

        Returns:
            Estimated velocity in km/h.
        """
        phase1_velocity = self._phase1_velocity(distance_km)

        if distance_km < self.phase1_distance_km:
            base_velocity = phase1_velocity
        else:
            base_velocity = self.floor_speed_kmh

        sleep_recovery = self._sleep_recovery_delta(cumulative_sleep_duration_s)

        return max(self.floor_speed_kmh, base_velocity + sleep_recovery)

    def velocity_at_time(
        self,
        elapsed_time_s: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Return estimated velocity (km/h) at a given elapsed race time.

        Uses average speed to approximate distance from time, then applies
        the distance-based model.  For Phase 2, also adds circadian oscillation.

        Args:
            elapsed_time_s: Elapsed race time in seconds.
            cumulative_sleep_duration_s: Total sleep accumulated, in seconds.

        Returns:
            Estimated velocity in km/h.
        """
        elapsed_hours = elapsed_time_s / 3600.0
        sleep_recovery = self._sleep_recovery_delta(cumulative_sleep_duration_s)

        # Use average of start and floor to estimate distance covered
        avg_speed = (self._v_start + self.floor_speed_kmh) / 2.0
        approx_distance_km = avg_speed * elapsed_hours

        if approx_distance_km < self.phase1_distance_km:
            base_velocity = self._phase1_velocity(approx_distance_km)
        else:
            circadian_delta = self._circadian_delta(elapsed_hours)
            base_velocity = self.floor_speed_kmh + circadian_delta

        return max(self.floor_speed_kmh, base_velocity + sleep_recovery)

    def fatigue_multiplier_for_distance(
        self,
        distance_km: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Return pace multiplier (>1 means slower) relative to flat threshold.

        A multiplier > 1 indicates the athlete is moving slower than threshold.
        This is the main interface for ``PaceCalculator``.

        Args:
            distance_km: Cumulative race distance in km.
            cumulative_sleep_duration_s: Total sleep accumulated, in seconds.

        Returns:
            Pace multiplier ≥ 1.0.  Values approaching 1 mean near-threshold
            pace; higher values mean slower pace.
        """
        v = self.velocity_at_distance(distance_km, cumulative_sleep_duration_s)
        if v <= 0:
            return float("inf")
        return self.threshold_speed_kmh / v

    def apply_nap_recovery(
        self,
        current_sleep_debt_s: float,
        nap_duration_s: float,
    ) -> float:
        """Apply exponential recovery to accumulated sleep debt.

        The sleep-debt model is based on Process S from the two-process model
        (Borbély, 1982). Recovery follows an exponential decay::

            S_after = S_before · exp(-λ · nap_hours)

        Args:
            current_sleep_debt_s: Current accumulated sleep debt, in seconds.
            nap_duration_s: Duration of the nap block, in seconds.

        Returns:
            Reduced sleep debt in seconds after the nap.
        """
        nap_hours = nap_duration_s / 3600.0
        return current_sleep_debt_s * math.exp(-self._sleep_lambda * nap_hours)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _phase1_velocity(self, distance_km: float) -> float:
        """Phase 1 sigmoid velocity decay from start to floor speed.

        Uses the standard logistic function::

            V(d) = V_floor + (V_start - V_floor) / (1 + exp(k · (d - d₀)))

        where ``d₀ = phase1_distance_km / 2`` and
        ``k = inflection_steepness_k / phase1_distance_km``.
        """
        exponent = self._k_scaled * (distance_km - self._d_inflection)
        # Clamp exponent to avoid overflow in exp for very large values
        exponent = max(-500.0, min(500.0, exponent))
        return self.floor_speed_kmh + self._v_delta / (1.0 + math.exp(exponent))

    def _circadian_delta(self, elapsed_hours: float) -> float:
        """Circadian oscillation around the floor speed.

        Returns the additive velocity delta due to circadian rhythm::

            Δ = A · V_floor · sin(2π(t - φ) / T)

        where ``φ`` is the phase offset (peak performance time), ``T`` is the
        period (24 h), and ``A`` is the amplitude fraction.
        """
        phase = 2.0 * math.pi * (elapsed_hours - self.circadian_phase_offset_hours)
        phase /= self.circadian_period_hours
        return self.circadian_amplitude * self.floor_speed_kmh * math.sin(phase)

    def _sleep_recovery_delta(self, cumulative_sleep_duration_s: float) -> float:
        """Velocity boost from accumulated sleep (exponential Process S recovery).

        Uses the exponential recovery model::

            ΔV = (V_start - V_floor) · (1 - exp(-λ · sleep_hours))

        This ensures the first hour of sleep gives the largest recovery gain,
        consistent with the steep initial drop of Process S.

        Args:
            cumulative_sleep_duration_s: Total sleep in seconds.

        Returns:
            Additive velocity boost in km/h.
        """
        if cumulative_sleep_duration_s <= 0.0:
            return 0.0
        sleep_hours = cumulative_sleep_duration_s / 3600.0
        return self._v_delta * (1.0 - math.exp(-self._sleep_lambda * sleep_hours))
