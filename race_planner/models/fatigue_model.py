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

DEFAULT_S0_START_THRESHOLD_FRACTION: float = 0.85
DEFAULT_T0_SIGMOID_INFLECTION_HOURS: float = 10.0  # hours
DEFAULT_K0_SIGMOID_STEEPNESS: float = 0.45  # h⁻¹

DEFAULT_CIRCADIAN_AMPLITUDE: float = 0.02  # ±2% of floor speed
DEFAULT_CIRCADIAN_PERIOD_HOURS: float = 24.0
DEFAULT_CIRCADIAN_PHASE_OFFSET_HOURS: float = 6.0

DEFAULT_SLEEP_HALF_LIFE_HOURS: float = 4  # hours
DEFAULT_FATIGUE_REACCUMULATION_HALF_LIFE_HOURS: float = 6.0  # hours

EPS = 1e-4


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
            The transition width (going from 10% to 90%) is approximately ``4.4 / k`` hours.
        circadian_amplitude:
            Amplitude of circadian rhythm oscillation as a fraction of
            ``floor_speed_kmh``.
        circadian_period_hours:
            Period of the circadian cycle in hours.
        circadian_phase_offset_hours:
            Phase shift of the circadian sine wave in hours.
        sleep_half_life_hours:
            Half-life for exponential sleep-pressure recovery (Process S).
            Set to a large value (or omit sleep inputs) for single-day events.
    """

    def __init__(
        self,
        threshold_speed_kmh: float,
        floor_speed_kmh: float,
        start_pct: float,
        s_0: float = DEFAULT_S0_START_THRESHOLD_FRACTION,
        t_0_hours: float = DEFAULT_T0_SIGMOID_INFLECTION_HOURS,
        k_0: float = DEFAULT_K0_SIGMOID_STEEPNESS,
        circadian_amplitude: float = DEFAULT_CIRCADIAN_AMPLITUDE,
        circadian_period_hours: float = DEFAULT_CIRCADIAN_PERIOD_HOURS,
        circadian_phase_offset_hours: float = DEFAULT_CIRCADIAN_PHASE_OFFSET_HOURS,
        sleep_half_life_hours: float = DEFAULT_SLEEP_HALF_LIFE_HOURS,
        fatigue_reaccumulation_half_life_hours: float = DEFAULT_FATIGUE_REACCUMULATION_HALF_LIFE_HOURS,
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
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Integrate modeled speed over elapsed time using sleep events.

        This is the physically consistent distance-time inverse of the fatigue
        model and honors fatigue, circadian modulation, and sleep recovery.
        Args:
            elapsed_hours: Total elapsed race time in hours.
            sleep_events: List of (nap_start_time_hours, nap_duration_hours) tuples.
        """
        if elapsed_hours < 0.0:
            raise ValueError("elapsed_hours must be non-negative")
        if elapsed_hours == 0.0:
            return 0.0

        rounded_elapsed_hours = round(
            elapsed_hours, 3
        )  # Round to avoid cache misses due to floating-point noise
        # lru_cache requires hashable args. Passing lists/dicts will raise TypeError, pass to tuple.
        events_tuple = tuple(sleep_events) if sleep_events else ()
        return self._distance_travelled_in_time_cached(
            rounded_elapsed_hours,
            events_tuple,
        )

    @lru_cache(maxsize=2048)
    def _distance_travelled_in_time_cached(
        self,
        elapsed_hours: float,
        sleep_events_tuple: tuple[tuple[float, float], ...],
    ) -> float:
        # Adaptive integration: fewer steps for long durations, reuse velocity evaluations.
        # Keep a reasonable minimum and cap to avoid excessive work.
        n_steps = max(80, min(2000, int(math.ceil(elapsed_hours * 60.0))))
        dt = elapsed_hours / float(n_steps)
        distance_km = 0.0
        events_list = list(sleep_events_tuple) if sleep_events_tuple else None

        # Precompute velocities, returning 0.0 during sleep to prevent distance accumulation
        times_s = [i * dt * 3600.0 for i in range(n_steps + 1)]
        v_vals = []
        for t in times_s:
            t_h = t / 3600.0
            is_sleeping = False
            if events_list:
                for nap_start, nap_dur in events_list:
                    if nap_start <= t_h < nap_start + nap_dur:
                        is_sleeping = True
                        break
            if is_sleeping:
                v_vals.append(0.0)
            else:
                v_vals.append(self.velocity_at_time(t, events_list))

        # Trapezoidal integration using precomputed velocities
        for i in range(n_steps):
            distance_km += 0.5 * (v_vals[i] + v_vals[i + 1]) * dt

        return distance_km

    def base_sigmoidal_velocity(self, elapsed_hours: float) -> float:
        """Pure sigmoidal decay without circadian or sleep modifications."""
        arg = self._k_h * (elapsed_hours - self._t_inflection)

        # Clamp the argument to prevent OverflowError in math.exp
        if arg > 700.0:
            sigmoid_factor = 0.0
        elif arg < -700.0:
            sigmoid_factor = 1.0
        else:
            sigmoid_factor = 1.0 / (1.0 + math.exp(arg))
        return self.floor_speed_kmh + self._v_delta * sigmoid_factor

    def elapsed_hours_for_distance(
        self,
        target_distance_km: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Solve for elapsed time at a given cumulative distance."""
        from scipy.optimize import root_scalar

        def distance_error(t_hours: float) -> float:
            # We use the cached integration function here
            return self.distance_travelled_in_time(t_hours, sleep_events) - target_distance_km

        # Bracket up to 200 hours to safely cover the TOR330 time limit
        res = root_scalar(
            distance_error,
            bracket=[0.0, 200.0],
            method='brentq',
            xtol=1e-4,  # Ensure high precision on the time boundary
        )

        if not res.converged:
            raise RuntimeError(f"Solver failed to find arrival time for {target_distance_km}km")

        return res.root

    def velocity_at_distance(
        self,
        distance_km: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Estimated speed at a given cumulative distance.

        The elapsed time is recovered from the distance integral of the
        time-domain fatigue model, so the mapping remains consistent with the
        underlying speed curve.
        """
        if distance_km <= 0.0:
            return self.velocity_at_time(0.0, sleep_events)
        elapsed_hours = self.elapsed_hours_for_distance(distance_km, sleep_events)
        return self.velocity_at_time(elapsed_hours * 3600.0, sleep_events)

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
        # Base fatigue is driven strictly by time spent moving
        moving_time = elapsed_hours - self._cumulative_sleep_at_time(elapsed_hours, sleep_events)
        sigmoid_velocity = self.base_sigmoidal_velocity(moving_time)

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
        sorted_naps = sorted(sleep_events or [], key=lambda x: x[0])

        if sorted_naps:
            # Precompute each nap's initial_boost in chronological order, accounting for prior nap contributions.
            nap_infos: list[dict] = []
            for nap_start_h, nap_duration_h in sorted_naps:
                nap_end_h = nap_start_h + nap_duration_h
                # velocity just before nap: base sigmoid + circadian + decayed contributions from earlier naps
                # Convert absolute nap_start_h to moving time so the base sigmoid doesn't decay during prior sleep
                moving_time_at_nap = nap_start_h - self._cumulative_sleep_at_time(
                    nap_start_h, sleep_events
                )
                v_before = self.base_sigmoidal_velocity(moving_time_at_nap) + self._circadian_delta(
                    nap_start_h
                )
                # add decayed contributions from earlier naps that ended before this nap_start
                for prev in nap_infos:
                    prev_end = prev["nap_end_h"]
                    if prev_end + EPS < nap_start_h:
                        moving_time_at_prev_end = prev_end - self._cumulative_sleep_at_time(
                            prev_end, sleep_events
                        )
                        time_since_prev_moving = max(
                            0.0, moving_time_at_nap - moving_time_at_prev_end
                        )
                        v_before += prev["initial_boost"] * math.exp(
                            -self._reaccum_lambda * time_since_prev_moving
                        )

                v_lost = max(0.0, self._v_start - v_before)
                eta_sleep = 1.0 - math.exp(-self._sleep_lambda * nap_duration_h)
                initial_boost = eta_sleep * v_lost
                nap_infos.append(
                    {
                        "nap_start_h": nap_start_h,
                        "nap_end_h": nap_end_h,
                        "initial_boost": initial_boost,
                    }
                )

            # Now sum contributions of naps that have finished before elapsed_hours
            for info in nap_infos:
                if elapsed_hours > info["nap_end_h"] - EPS:
                    moving_time_at_end = info["nap_end_h"] - self._cumulative_sleep_at_time(
                        info["nap_end_h"], sleep_events
                    )
                    time_since_nap_moving = max(0.0, moving_time - moving_time_at_end)
                    v_sleep_boost += info["initial_boost"] * math.exp(
                        -self._reaccum_lambda * time_since_nap_moving
                    )

        # 3. Combine and clamp strictly to [V_floor, V_start]
        v_total = sigmoid_velocity + v_circ + v_sleep_boost
        return max(self.floor_speed_kmh, min(self._v_start, v_total))

    def pace_multiplier_at_time(
        self,
        elapsed_time_s: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Planner-facing pace penalty at a given elapsed time.

        Values larger than 1.0 indicate slower-than-baseline running pace;
        values below 1.0 would indicate faster-than-baseline pace.
        """
        speed = self.velocity_at_time(elapsed_time_s, sleep_events)
        if speed <= 0:
            return float("inf")
        return self._v_start / speed

    def pace_multiplier_for_distance(
        self,
        distance_km: float,
        sleep_events: list[tuple[float, float]] = None,
    ) -> float:
        """Return pace multiplier (>= 1.0) relative to starting speed."""
        t_h = self.elapsed_hours_for_distance(distance_km, sleep_events)
        v_t = self.velocity_at_time(t_h * 3600.0, sleep_events)
        return self._v_start / max(v_t, EPS)

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

    def _cumulative_sleep_at_time(
        self, elapsed_hours, sleep_events: list[tuple[float, float]] = None
    ):
        sleep_hours = 0.0
        if sleep_events is None:
            sleep_events = []
        sorted_naps = sorted(sleep_events)
        for nap_start_h, nap_duration_h in sorted_naps:
            if elapsed_hours > nap_start_h:
                sleep_hours += min(elapsed_hours - nap_start_h, nap_duration_h)
        return sleep_hours
