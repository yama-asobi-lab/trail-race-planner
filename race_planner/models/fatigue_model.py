"""Fatigue models for long-distance running events.

This module provides two fatigue models:

- ``LinearFatigueModel``: a simple percentage-based pace decay applied linearly
  over a race.
- ``MultiDaySigmoidalFatigueModel``: a physiologically-grounded time-based
  sigmoid model applicable to long-distance events. The model accounts for:

  1. sigmoidal velocity decay governed by the athlete's relative starting
     intensity and athlete-specific calibration parameters,
  2. circadian rhythm oscillation after the fatigue floor is reached, and
  3. sleep recovery modelled via Borbély's Process S exponential decay.

The sigmoid is defined in elapsed-time space::

    V(t) = max(V_floor,
               V_floor + (V_start - V_floor) / (1 + exp(k * (t - t0)))
               + ΔV_circadian(t)
               + ΔV_sleep(t))

For planner-facing pacing, the raw circadian rhythm is intentionally suppressed.
The positive phase of the oscillation is not used in a single-event pacing
model because it reverses the general fatigue trend and makes the athlete run
faster again later in the race, which is not the intended race-planning
behavior.

The inflection time and steepness are derived from the athlete's relative
starting intensity S = V_start / V_threshold via power laws::

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

The planner converts the time-domain model into pace multipliers at the
integration boundary.
"""

from __future__ import annotations

import math
from functools import lru_cache

# Power-law exponent for inflection-time scaling: t₀ = t_0_hours · (S₀/S)^_T0_EXPONENT
_T0_EXPONENT: float = 3.1
# Power-law exponent for sigmoid-steepness scaling: k = k_0 · (S/S₀)^_K_EXPONENT
_K_EXPONENT: float = 2.5


class LinearFatigueModel:
    """Simple linear percentage-based pace decay over a race.

    The pace multiplier increases linearly from 1.0 at the start to
    ``1 / (1 - total_decay_pct / 100)`` at the finish, with the entire race
    treated as a single percentage-based decay in pace.
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
    starting intensity via power laws. A circadian oscillation and Process S
    sleep recovery are added in the same time domain before the planner converts
    the result into a pace multiplier.

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
            ``floor_speed_kmh``.  Default: 0.04 (±4 %).
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
        start_pct: float,
        s_0: float = 0.65,
        t_0_hours: float = 18.0,
        k_0: float = 0.45,
        circadian_amplitude: float = 0.04,
        circadian_period_hours: float = 24.0,
        circadian_phase_offset_hours: float = 6.0,
        sleep_half_life_hours: float = 2.5,
        fatigue_reaccumulation_half_life_hours: float = 8.0,
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
        if fatigue_reaccumulation_half_life_hours <= 0:
            raise ValueError("fatigue_reaccumulation_half_life_hours must be > 0")

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
        self.fatigue_reaccumulation_half_life_hours = float(fatigue_reaccumulation_half_life_hours)

        self._v_start = self.threshold_speed_kmh * self.start_pct
        self._v_delta = self._v_start - self.floor_speed_kmh
        # Derive time-domain sigmoid parameters from power laws
        S = self.start_pct
        self._t_inflection = self.t_0_hours * (self.s_0 / S) ** _T0_EXPONENT
        self._k_h = self.k_0 * (S / self.s_0) ** _K_EXPONENT  # h⁻¹
        # Distance-time conversion is solved numerically below.
        self._sleep_lambda = math.log(2.0) / self.sleep_half_life_hours
        self._reaccum_lambda = math.log(2.0) / self.fatigue_reaccumulation_half_life_hours
        self._pace_multiplier_cache: dict[tuple[float, float], float] = {}

    # ------------------------------------------------------------------
    # Core velocity methods
    # ------------------------------------------------------------------

    def distance_travelled_in_time(
        self,
        elapsed_hours: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Integrate modeled speed over elapsed time.

        This is the physically consistent distance-time inverse of the fatigue
        model and honors fatigue, circadian modulation, and sleep recovery.
        """
        if elapsed_hours < 0.0:
            raise ValueError("elapsed_hours must be non-negative")
        if elapsed_hours == 0.0:
            return 0.0

        rounded_elapsed_hours = round(elapsed_hours, 6)
        rounded_sleep_s = round(cumulative_sleep_duration_s, 3)
        return self._distance_travelled_in_time_cached(
            rounded_elapsed_hours,
            rounded_sleep_s,
        )

    def base_sigmoidal_velocity(self, elapsed_hours: float) -> float:
        """Pure sigmoidal decay without circadian or sleep modifications."""
        sigmoid_factor = 1.0 / (1.0 + math.exp(self._k_h * (elapsed_hours - self._t_inflection)))
        return self.floor_speed_kmh + self._v_delta * sigmoid_factor

    @lru_cache(maxsize=2048)
    def _distance_travelled_in_time_cached(
        self,
        elapsed_hours: float,
        cumulative_sleep_duration_s: float,
    ) -> float:
        n_steps = max(60, min(4000, int(math.ceil(elapsed_hours * 60.0))))
        dt = elapsed_hours / float(n_steps)
        distance_km = 0.0

        for i in range(n_steps):
            t0 = i * dt
            t1 = (i + 1) * dt
            v0 = self.velocity_at_time(t0 * 3600.0, cumulative_sleep_duration_s)
            v1 = self.velocity_at_time(t1 * 3600.0, cumulative_sleep_duration_s)
            distance_km += 0.5 * (v0 + v1) * dt

        return distance_km

    def elapsed_hours_for_distance(
        self,
        distance_km: float,
        cumulative_sleep_duration_s: float = 0.0,
        tolerance_hours: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        """Solve for elapsed time at a given cumulative distance.

        The inversion is done by integrating the modeled speed curve and
        bisectioning on the cumulative distance, rather than using a fixed
        average-speed shortcut.
        """
        if distance_km < 0.0:
            raise ValueError("distance_km must be non-negative")
        if distance_km == 0.0:
            return 0.0

        # A lower bound of zero is valid, and the upper bound is chosen using the
        # floor speed as the worst-case sustainable speed to ensure a bracket.
        lower_hours = 0.0
        upper_hours = max(1.0, distance_km / max(self.floor_speed_kmh, 1e-9))
        while (
            self.distance_travelled_in_time(upper_hours, cumulative_sleep_duration_s) < distance_km
        ):
            upper_hours *= 2.0
            if upper_hours > 1e6:
                raise RuntimeError("distance-to-time solver failed to bracket the target distance")

        for _ in range(max_iter):
            mid_hours = 0.5 * (lower_hours + upper_hours)
            if (
                self.distance_travelled_in_time(mid_hours, cumulative_sleep_duration_s)
                < distance_km
            ):
                lower_hours = mid_hours
            else:
                upper_hours = mid_hours
            if upper_hours - lower_hours <= tolerance_hours:
                break

        return 0.5 * (lower_hours + upper_hours)

    def velocity_at_distance(
        self,
        distance_km: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Estimated speed at a given cumulative distance.

        The elapsed time is recovered from the distance integral of the
        time-domain fatigue model, so the mapping remains consistent with the
        underlying speed curve.
        """
        if distance_km <= 0.0:
            return self.velocity_at_time(0.0, cumulative_sleep_duration_s)
        elapsed_hours = self.elapsed_hours_for_distance(distance_km, cumulative_sleep_duration_s)
        return self.velocity_at_time(elapsed_hours * 3600.0, cumulative_sleep_duration_s)

    def velocity_at_time(
        self,
        elapsed_time_s: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Calculate physiological running speed in km/h at elapsed_seconds.

        Args:
            elapsed_seconds: Total elapsed race time in seconds.
            sleep_events: List of (nap_start_time_hours, nap_duration_hours) tuples.
        """
        elapsed_hours = elapsed_time_s / 3600.0
        sigmoid_velocity = self.base_sigmoidal_velocity(elapsed_hours)

        # 1. Circadian oscillation
        circadian_phase = (
            2.0
            * math.pi
            * (elapsed_hours - self.circadian_phase_offset_hours)
            / self.circadian_period_hours
        )
        v_circ = self.circadian_amplitude * self.floor_speed_kmh * math.sin(circadian_phase)

        # 2. Stateful decaying sleep recovery
        v_sleep_boost = 0.0
        if sleep_events:
            for nap_start_h, nap_duration_h in sleep_events:
                nap_end_h = nap_start_h + nap_duration_h
                if elapsed_hours >= nap_end_h:
                    # Velocity lost at time of nap
                    v_lost = self._v_start - self.base_sigmoidal_velocity(nap_start_h)
                    # Recovery fraction from nap
                    eta_sleep = 1.0 - math.exp(-self._sleep_lambda * nap_duration_h)
                    initial_boost = eta_sleep * max(0.0, v_lost)
                    # Exponential decay of boost as continuous running resumes
                    time_since_nap = elapsed_hours - nap_end_h
                    v_sleep_boost += initial_boost * math.exp(
                        -self._reaccum_lambda * time_since_nap
                    )

        # 3. Combine and clamp strictly to [V_floor, V_start]
        v_total = sigmoid_velocity + v_circ + v_sleep_boost
        return max(self.floor_speed_kmh, min(self._v_start, v_total))

    def pace_multiplier_at_time(
        self,
        elapsed_time_s: float,
        cumulative_sleep_duration_s: float = 0.0,
    ) -> float:
        """Planner-facing pace penalty at a given elapsed time.

        Values larger than 1.0 indicate slower-than-threshold running pace;
        values below 1.0 would indicate faster-than-threshold pace.
        """
        speed = self.velocity_at_time(elapsed_time_s, cumulative_sleep_duration_s)
        if speed <= 0:
            return float("inf")
        return self.threshold_speed_kmh / speed

    def pace_multiplier_for_distance(
        self,
        distance_km: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Return pace multiplier (>= 1.0) relative to starting speed."""
        t_h = self.elapsed_hours_for_distance(distance_km, sleep_events)
        v_t = self.velocity_at_time(t_h * 3600.0, sleep_events)
        return self._v_start / max(v_t, 1e-6)

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

    def _circadian_delta(self, elapsed_hours: float) -> float:
        """Raw circadian oscillation around the floor speed.

        The term is ``A * V_floor * sin(2π(t - φ) / T)`` with phase offset
        ``φ`` and period ``T``. It captures the late-day performance rhythm that
        emerges after the athlete has reached the fatigue floor. The planner
        uses only the negative half-cycle, because the positive phase can
        reverse the overall fatigue progression within a single race plan.
        """
        phase = 2.0 * math.pi * (elapsed_hours - self.circadian_phase_offset_hours)
        phase /= self.circadian_period_hours
        return self.circadian_amplitude * self.floor_speed_kmh * math.sin(phase)

    def _sleep_recovery_delta(self, cumulative_sleep_duration_s: float) -> float:
        """Velocity boost from cumulative sleep.

        This is the Process S recovery term: sleep reduces the accumulated fatigue
        debt and increases speed by a fraction of the gap between the starting
        speed and the floor speed.

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
