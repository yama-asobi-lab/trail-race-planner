"""
Fatigue models for long-distance running events.

This module provides two fatigue models:

- ``LinearFatigueModel``: a simple percentage-based pace decay applied linearly
  over a race. Useful as a lightweight default for any distance.

- ``MultiDaySigmoidalFatigueModel``: a physiologically-grounded time-based sigmoid
  model applicable to any long-distance event, from 100-mile ultras to multi-day
  stage races. The model accounts for:

  1. **Sigmoidal velocity decay** governed by the athlete's relative starting
     intensity and athlete-specific calibration parameters.
  2. **Circadian rhythm oscillation** in Phase 2 (after the fatigue floor is
     reached).
  3. **Sleep recovery** modelled via Borbély's Process S exponential decay;
     this term is simply zero for single-day events where no sleep occurs.

Mathematical foundation — sigmoid model
----------------------------------------
Velocity as a function of elapsed race time *t* (hours)::

    V(t) = max(V_floor,
               V_floor + (V_start − V_floor) / (1 + exp(k · (t − t₀)))
               + ΔV_circadian(t)
               + ΔV_sleep(t))

Inflection time and steepness are derived from the athlete's relative starting
intensity *S = V_start / V_threshold* via power laws calibrated to a reference
intensity *S₀*:

    t₀ = t_0_hours · (S₀ / S) ** 3.1
    k  = k_0       · (S  / S₀) ** 2.5   (h⁻¹)

A conservative start (small *S*) shifts the inflection later and flattens the
decay curve; an aggressive start (large *S*) brings forward the inflection and
steepens the drop — consistent with physiological field observations.

Circadian oscillation::

    ΔV_circadian(t) = A · V_floor · sin(2π(t − φ) / T)

Sleep recovery (Process S)::

    ΔV_sleep = (V_start − V_floor) · (1 − exp(−λ · sleep_hours))
    λ = ln(2) / sleep_half_life_hours
"""

from __future__ import annotations

import math

# Power-law exponent for inflection-time scaling: t₀ = t_0_hours · (S₀/S)^_T0_EXPONENT
_T0_EXPONENT: float = 3.1
# Power-law exponent for sigmoid-steepness scaling: k = k_0 · (S/S₀)^_K_EXPONENT
_K_EXPONENT: float = 2.5


class LinearFatigueModel:
    """Simple linear percentage-based pace decay over a race.

    The pace multiplier increases linearly from 1.0 at the start to
    ``1 / (1 - total_decay_pct/100)`` at the finish.

    Args:
        total_decay_pct: Total speed reduction from start to finish, expressed
            as a percentage (0–100).  A value of 14 means the athlete arrives at
            the finish moving 14 % slower than at the start.
    """

    def __init__(self, total_decay_pct: float) -> None:
        if not 0.0 <= total_decay_pct <= 100.0:
            raise ValueError("total_decay_pct must be in [0, 100]")
        self.total_decay_pct = float(total_decay_pct)

    def fatigue_multiplier(self, progress_fraction: float) -> float:
        """Return pace multiplier (≥ 1) at the given race progress fraction.

        Args:
            progress_fraction: Race completion fraction in [0, 1].

        Returns:
            Pace multiplier ≥ 1.  Values > 1 indicate a pace slower than the
            starting pace.
        """
        progress_fraction = max(0.0, min(1.0, progress_fraction))
        decay_at_point = (self.total_decay_pct / 100.0) * progress_fraction
        denominator = 1.0 - decay_at_point
        if denominator <= 0.0:
            return float("inf")
        return 1.0 / denominator


class MultiDaySigmoidalFatigueModel:
    """Sigmoidal fatigue model for long-distance running events.

    The velocity decay is governed by a logistic sigmoid in the time domain,
    with inflection time and steepness derived from the athlete's relative
    starting intensity via power laws.

    Instantiate from a race/athlete configuration and inject into
    :class:`~race_planner.planner.pace_calculator.PaceCalculator` via the
    ``fatigue_model_instance`` parameter.

    Args:
        threshold_speed_kmh:
            Lactate Threshold speed in km/h.
        floor_speed_kmh:
            Minimum sustainable speed (biological floor) in km/h.
        start_pct:
            Starting speed as a fraction of ``threshold_speed_kmh``
            (*S = V_start / V_threshold*).
        s_0:
            Reference relative intensity used for calibration.  The power-law
            parameters ``t_0_hours`` and ``k_0`` are defined at this intensity.
        t_0_hours:
            Inflection time (hours) at the reference intensity ``s_0``.  At a
            different starting intensity *S*, the inflection time scales as
            ``t_0_hours · (s_0 / S) ** 3.1``.
        k_0:
            Sigmoid steepness (h⁻¹) at the reference intensity ``s_0``.  At a
            different *S* it scales as ``k_0 · (S / s_0) ** 2.5``.
        circadian_amplitude:
            Amplitude of circadian rhythm oscillation as a fraction of
            ``floor_speed_kmh``.  Default: 0.15 (±15 %).
        circadian_period_hours:
            Period of the circadian cycle in hours.  Default: 24.0.
        circadian_phase_offset_hours:
            Phase shift of the circadian sine wave in hours.  Default: 6.0.
        sleep_half_life_hours:
            Half-life for exponential sleep-pressure recovery (Process S).
            Set to a large value (or omit sleep inputs) for single-day events.
            Default: 2.5 hours.
    """

    def __init__(
        self,
        threshold_speed_kmh: float,
        floor_speed_kmh: float,
        start_pct: float = 0.55,
        s_0: float = 0.65,
        t_0_hours: float = 18.0,
        k_0: float = 0.45,
        circadian_amplitude: float = 0.15,
        circadian_period_hours: float = 24.0,
        circadian_phase_offset_hours: float = 6.0,
        sleep_half_life_hours: float = 2.5,
    ) -> None:
        if threshold_speed_kmh <= 0:
            raise ValueError("threshold_speed_kmh must be > 0")
        if floor_speed_kmh <= 0:
            raise ValueError("floor_speed_kmh must be > 0")
        if floor_speed_kmh >= threshold_speed_kmh:
            raise ValueError("floor_speed_kmh must be < threshold_speed_kmh")
        if not 0.0 < start_pct <= 1.0:
            raise ValueError("start_pct must be in (0, 1]")
        if s_0 <= 0.0:
            raise ValueError("s_0 must be > 0")
        if t_0_hours <= 0:
            raise ValueError("t_0_hours must be > 0")
        if k_0 <= 0:
            raise ValueError("k_0 must be > 0")
        if circadian_amplitude < 0:
            raise ValueError("circadian_amplitude must be >= 0")
        if circadian_period_hours <= 0:
            raise ValueError("circadian_period_hours must be > 0")
        if sleep_half_life_hours <= 0:
            raise ValueError("sleep_half_life_hours must be > 0")

        self.threshold_speed_kmh = float(threshold_speed_kmh)
        self.floor_speed_kmh = float(floor_speed_kmh)
        self.start_pct = float(start_pct)
        self.s_0 = float(s_0)
        self.t_0_hours = float(t_0_hours)
        self.k_0 = float(k_0)
        self.circadian_amplitude = float(circadian_amplitude)
        self.circadian_period_hours = float(circadian_period_hours)
        self.circadian_phase_offset_hours = float(circadian_phase_offset_hours)
        self.sleep_half_life_hours = float(sleep_half_life_hours)

        self._v_start = self.threshold_speed_kmh * self.start_pct
        self._v_delta = self._v_start - self.floor_speed_kmh
        # Derive time-domain sigmoid parameters from power laws
        S = self.start_pct
        self._t_inflection = self.t_0_hours * (self.s_0 / S) ** _T0_EXPONENT
        self._k_h = self.k_0 * (S / self.s_0) ** _K_EXPONENT  # h⁻¹
        # Approximate average speed for distance ↔ time conversion
        self._avg_speed = (self._v_start + self.floor_speed_kmh) / 2.0
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

        Combines sigmoid velocity decay, circadian oscillation, and
        exponential sleep-pressure recovery.  Distance is converted to
        elapsed time using the average of start and floor speeds.

        Args:
            distance_km: Cumulative race distance in km.
            cumulative_sleep_duration_s: Total sleep accumulated up to this
                point in the race, in seconds.

        Returns:
            Estimated velocity in km/h.
        """
        elapsed_hours = distance_km / self._avg_speed if self._avg_speed > 0 else 0.0
        return self.velocity_at_time(elapsed_hours * 3600.0, cumulative_sleep_duration_s)

    def velocity_at_time(
        self,
        elapsed_time_s: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Return estimated velocity (km/h) at a given elapsed race time.

        The sigmoid decay, circadian oscillation, and sleep recovery are all
        evaluated in the time domain.

        Args:
            elapsed_time_s: Elapsed race time in seconds.
            cumulative_sleep_duration_s: Total sleep accumulated, in seconds.

        Returns:
            Estimated velocity in km/h.
        """
        elapsed_hours = elapsed_time_s / 3600.0
        sigmoid_velocity = self._sigmoid_velocity(elapsed_hours)
        circadian_delta = self._circadian_delta(elapsed_hours)
        sleep_recovery = self._sleep_recovery_delta(cumulative_sleep_duration_s)

        return max(self.floor_speed_kmh, sigmoid_velocity + circadian_delta + sleep_recovery)

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

    def _sigmoid_velocity(self, elapsed_hours: float) -> float:
        """Sigmoid velocity as a function of elapsed time.

        Uses the time-domain logistic function::

            V(t) = V_floor + (V_start − V_floor) / (1 + exp(k · (t − t₀)))

        where ``t₀ = _t_inflection`` and ``k = _k_h`` (h⁻¹).
        """
        exponent = self._k_h * (elapsed_hours - self._t_inflection)
        # Clamp exponent to avoid overflow in exp for very large values
        exponent = max(-500.0, min(500.0, exponent))
        return self.floor_speed_kmh + self._v_delta / (1.0 + math.exp(exponent))

    def _circadian_delta(self, elapsed_hours: float) -> float:
        """Circadian oscillation around the floor speed.

        Returns the additive velocity delta due to circadian rhythm::

            Δ = A · V_floor · sin(2π(t − φ) / T)

        where ``φ`` is the phase offset (peak performance time), ``T`` is the
        period (24 h), and ``A`` is the amplitude fraction.
        """
        phase = 2.0 * math.pi * (elapsed_hours - self.circadian_phase_offset_hours)
        phase /= self.circadian_period_hours
        return self.circadian_amplitude * self.floor_speed_kmh * math.sin(phase)

    def _sleep_recovery_delta(self, cumulative_sleep_duration_s: float) -> float:
        """Velocity boost from accumulated sleep (exponential Process S recovery).

        Uses the exponential recovery model::

            ΔV = (V_start − V_floor) · (1 − exp(−λ · sleep_hours))

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
