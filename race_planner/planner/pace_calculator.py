"""
Pace calculator for trail race planning.

Implements Peter Riegel's endurance formula combined with per-point grade-adjusted
pace (GAP) correction to produce a segment-by-segment pacing plan.

-----------------------------------------------------------------------
Piecewise Riegel:
Use 1.06 exponent from Riegel's formula for sub-marathon distance, 
and update the exponent for longer distance based on sqrt of the distance beyond marathon.
-----------------------------------------------------------------------
    k(D) = 1.06 + c * sqrt(max(D - D_ref, 0))
    T    = T_ref * (D / D_ref) ^ k(D)

where c = 0.013422 in this implementation.

-----------------------------------------------------------------------
Flat Equivalent Distance (FED)
-----------------------------------------------------------------------
Mountain races last much longer than a flat race of the same distance.
A 160 km / 11 000 m course might take 28 h, while 160 km flat would take
only ~15 h.  If we apply Riegel with the raw 160 km, the formula anchors
the predicted effort at a "15 h race" when the actual effort is "28 h" —
underestimating fatigue-induced pace degradation.

The fix: replace the raw distance with a *flat equivalent distance* (FED)
that converts elevation gain into additional horizontal effort:

    FED = D_km + gain_m / FED_VERT_FACTOR

where FED_VERT_FACTOR = 100 m/km, i.e. 100 m of gain ≡ 1 km of flat running
(standard trail-running planning rule; Naismith / ITRA convention).

For TGT 2026 (160 km / 11 000 m):  FED ≈ 270 km → Riegel ≈ 25 h  ✓

When use_fed=True (recommended for mountain races) the FED replaces the raw
distance in Riegel.

As a "duration-based re-expression" of Riegel: the FED approach is
equivalent to asking "what flat race of duration T_ref×(FED/D_ref)^e takes
the same effort as this mountain course?", anchoring the fatigue model to
the actual physical workload rather than the horizontal distance alone.

-----------------------------------------------------------------------
Grade Adjusted Pace (GAP) correction
-----------------------------------------------------------------------
Per-point pace is multiplied by a correction factor interpolated from a
calibration table:

    correction > 1  → slower than flat (uphills)
    correction < 1  → faster than flat (gentle downhills)
    correction > 1  → slower again    (very steep downhills, braking)

Default table (adapted from an empirical trail-running tool):

    grade (decimal) | correction
    ----------------+----------
      +0.01  (+1%)  |  1.08     ← 8% slower; linear extrapolation for steeper
       0.00  ( 0%)  |  1.00
      -0.01  (-1%)  |  0.965    ← 3.5% faster
      -0.06  (-6%)  |  0.79     ← 21% faster  (empirical optimum)
      -0.07  (-7%)  |  0.86     ← 14% faster  (braking begins)

Endpoints are extrapolated linearly.
Note: the uphill branch has only one reference point (+1%), so steep uphills
(> ~15%) are extrapolated fairly aggressively.  Custom points per-athlete
can be supplied to override the default table.

Grade model assessment
-----------------------------
The model is conceptually sound:
  • Uphills: pace slows roughly proportionally to grade — consistent with
    Minetti et al. metabolic cost data and empirical race records.  The
    linear slope of 8 per decimal unit is somewhat steeper than the
    pure metabolic model (~5.4) to account for technical terrain.
  • Downhills: the non-monotonic shape (fastest at -6%, then slowdown) is
    well supported in race records research (Minetti et al. 2002;
    "Pace and critical gradient for hill runners", Loughborough 2019).
  • Weakness: with only one uphill knot (+ one shared flat), the table
    extrapolates linearly for grades > 1%.  This works acceptably up to
    ~20-25% but becomes imprecise on very steep technical terrain.
    Adding more knot points (e.g. 5%, 10%, 20%) would improve accuracy.

References:
  - Riegel, P.S. (1981) Athletic records and human endurance.
  - Minetti, A.E. et al. (2002) J. Exp. Biol. 205:3041-3047.
  - https://repository.lboro.ac.uk/articles/journal_contribution/
    Pace_and_critical_gradient_for_hill_runners_an_analysis_of_race_records/9387719
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from race_planner.course.course import Course
from race_planner.models.tools import seconds_to_hms, time_to_seconds


class PaceCalculator:
    """
    Calculates a per-segment pacing plan for a trail race.

    Two calculation modes are supported (``use_fed`` parameter):

    **use_fed=False**:
        Riegel is applied to the raw course distance.  The gradient
        correction per GPS point determines both the total time and its
        distribution across terrain.  Conceptually simple; may
        under-predict total time for courses dominated by elevation.

    **use_fed=True** (recommended for mountain ultras):
        Riegel is applied to the flat equivalent distance (FED).  This
        anchors the fatigue model to the actual effort duration.  The
        gradient corrections are *normalised* to preserve the FED-based
        total, so they control only the *distribution* of time across
        segments, not the overall scale.

    Args:
        ref_dist_km: Reference flat race distance in km (e.g. 42.195).
        ref_time_s:  Reference flat race time in seconds.
        gap_curve:   Optional custom GAP curve, shape (N, 2), columns
                     [grade_decimal, correction_factor].  Rows need not be
                     sorted.  Defaults to the built-in table described in
                     the module docstring.
    """

    # Built-in default GAP correction table (grade in decimal, correction factor).
    DEFAULT_GAP_CURVE = np.array(
        [
            [0.01, 1.08],  # +1 %: 8 % slower (linear extrapolation above this)
            [0.00, 1.00],  # flat
            [-0.01, 0.97],  # −1 %: 3 % faster
            [-0.05, 0.85],  # −5 %: 15 % faster (optimal descent)
            [-0.06, 0.9],  # −6 %: braking effect starts
        ],
        dtype=float,
    )

    # Piecewise Riegel parameters (combined-sex robust fit from analysis):
    #   k(D) = 1.06 + c*sqrt(max(D-D_ref, 0))
    RIEGEL_BASE_EXPONENT: float = 1.06
    PIECEWISE_RIEGEL_106_SQRT_C: float = 0.013422

    # Flat Equivalent Distance factor: metres of gain per 1 km FED.
    # 100 m gain ↔ 1 km flat (standard trail-running convention).
    FED_VERT_FACTOR_M_PER_KM: float = 100.0

    def __init__(
        self,
        ref_dist_km: float,
        ref_time_s: float,
        gap_curve: Optional[np.ndarray] = None,
    ) -> None:
        if ref_dist_km <= 0:
            raise ValueError("ref_dist_km must be positive")
        if ref_time_s <= 0:
            raise ValueError("ref_time_s must be positive")

        self.ref_dist_km = float(ref_dist_km)
        self.ref_time_s = float(ref_time_s)

        raw = (
            np.array(gap_curve, dtype=float)
            if gap_curve is not None
            else self.DEFAULT_GAP_CURVE.copy()
        )
        # Sort ascending by grade for interpolation
        self.gap_curve = raw[raw[:, 0].argsort()]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_athlete_config(cls, athlete_config: Dict) -> 'PaceCalculator':
        """
        Build a PaceCalculator from an athlete YAML config dict.

        The config must contain::

            athlete:
              reference_performance:
                distance_km: <float>
                time: "HH:MM:SS"
              gap_curve:
                points: []   # empty = use default, or list of [grade, factor] pairs

        Args:
            athlete_config: Loaded athlete YAML as a Python dict.

        Returns:
            PaceCalculator instance.
        """
        athlete = athlete_config.get('athlete', {})
        ref = athlete.get('reference_performance', {})

        ref_dist_km = ref.get('distance_km')
        ref_time_str = ref.get('time')

        if ref_dist_km is None or ref_time_str is None:
            raise ValueError(
                "Athlete config must contain 'athlete.reference_performance.distance_km' "
                "and 'athlete.reference_performance.time'"
            )

        ref_time_s = float(time_to_seconds(str(ref_time_str)))

        # Optional custom GAP curve
        custom_points = athlete.get('gap_curve', {}).get('points') or []
        gap_curve = np.array(custom_points, dtype=float) if custom_points else None

        return cls(
            ref_dist_km=float(ref_dist_km),
            ref_time_s=ref_time_s,
            gap_curve=gap_curve,
        )

    # ------------------------------------------------------------------
    # Riegel's formula
    # ------------------------------------------------------------------

    def flat_equivalent_distance_km(self, dist_km: float, gain_m: float) -> float:
        """
        Convert a mountain course to its flat equivalent distance (FED).

        Only elevation *gain* contributes to FED; descents are less taxing
        and are not included in the standard trail-running FED formula.

        Args:
            dist_km: Horizontal course distance in km.
            gain_m:  Total elevation gain in metres.

        Returns:
            Flat equivalent distance in km.
        """
        return dist_km + gain_m / self.FED_VERT_FACTOR_M_PER_KM

    def predict_riegel_race_time_sec(
        self,
        target_distance_km: float,
        elevation_gain_m: float = 0.0,
        use_flat_equivalent_distance: bool = False,
    ) -> float:
        """
        Predict race time (seconds) using piecewise Riegel.

        The exponent is distance-dependent and follows the validated
        piecewise model:

            k(D) = 1.06 + c*sqrt(max(D-D_ref, 0))

        where ``c = PIECEWISE_RIEGEL_106_SQRT_C`` and ``D_ref`` is
        ``self.ref_dist_km``. If ``use_flat_equivalent_distance`` is true,
        FED replaces horizontal distance before applying the formula.

        Args:
            target_distance_km: Horizontal course distance in km.
            elevation_gain_m:   Total elevation gain in metres.
            use_flat_equivalent_distance: If true, apply FED before Riegel.

        Returns:
            Predicted total race time in seconds (running only, no stops).
        """
        effective_distance_km = target_distance_km
        if use_flat_equivalent_distance:
            effective_distance_km = self.flat_equivalent_distance_km(
                target_distance_km,
                elevation_gain_m,
            )

        if effective_distance_km <= 0:
            raise ValueError("target_distance_km must be positive")

        distance_ratio = effective_distance_km / self.ref_dist_km
        ultra_excess_km = max(effective_distance_km - self.ref_dist_km, 0.0)
        exponent = (
            self.RIEGEL_BASE_EXPONENT
            + self.PIECEWISE_RIEGEL_106_SQRT_C * np.sqrt(ultra_excess_km)
        )

        return self.ref_time_s * (distance_ratio**exponent)

    def predict_riegel_flat_race_time_sec(self, target_distance_km: float) -> float:
        """Predict flat race time (seconds) for a horizontal target distance."""
        return self.predict_riegel_race_time_sec(target_distance_km)

    def predict_riegel_fed_race_time_sec(
        self, target_distance_km: float, elevation_gain_m: float
    ) -> float:
        """Predict race time (seconds) using FED-adjusted Riegel distance."""
        return self.predict_riegel_race_time_sec(
            target_distance_km,
            elevation_gain_m,
            use_flat_equivalent_distance=True,
        )

    # Backward-compatible aliases
    def riegel(self, dist_km: float) -> float:
        """Backward-compatible alias for ``predict_riegel_flat_race_time_sec``."""
        return self.predict_riegel_flat_race_time_sec(dist_km)

    def flat_equivalent_dist_km(self, dist_km: float, gain_m: float) -> float:
        """Backward-compatible alias for ``flat_equivalent_distance_km``."""
        return self.flat_equivalent_distance_km(dist_km, gain_m)

    def riegel_fed(self, dist_km: float, gain_m: float, loss_m: float = 0.0) -> float:
        """Backward-compatible alias for ``predict_riegel_fed_race_time_sec``."""
        return self.predict_riegel_fed_race_time_sec(dist_km, gain_m)

    # ------------------------------------------------------------------
    # Grade correction
    # ------------------------------------------------------------------

    def grade_correction(self, grades: np.ndarray) -> np.ndarray:
        """
        Compute GAP correction factors for an array of grade values.

        Uses piecewise linear interpolation with linear extrapolation at both
        ends of the table.

        Args:
            grades: 1-D array of grade values as decimal rise/run
                    (e.g. 0.10 for a 10 % uphill, -0.06 for a 6 % downhill).

        Returns:
            Array of correction factors, same length as *grades*.
        """
        g = self.gap_curve[:, 0]  # sorted ascending
        c = self.gap_curve[:, 1]

        grades = np.asarray(grades, dtype=float)
        result = np.empty_like(grades)

        for i, gr in enumerate(grades.flat):
            if gr <= g[0]:
                slope = (c[1] - c[0]) / (g[1] - g[0])
                result.flat[i] = c[0] + slope * (gr - g[0])
            elif gr >= g[-1]:
                slope = (c[-1] - c[-2]) / (g[-1] - g[-2])
                result.flat[i] = c[-1] + slope * (gr - g[-1])
            else:
                result.flat[i] = float(np.interp(gr, g, c))

        return result

    # ------------------------------------------------------------------
    # Pacing plan
    # ------------------------------------------------------------------

    def calculate_pacing(
        self,
        course: Course,
        aid_stations: List[Dict],
        use_fed: bool = True,
        override_total_running_time_s: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Calculate a per-segment pacing plan.

        For each segment between consecutive aid stations the method:

        1. Extracts the GPS points from the course DataFrame.
        2. Computes per-point grade corrections (grade column is in %, converted
           to decimal).
        3. Computes per-point running times from a baseline pace and inverse-GAP
            adjustments.
        4. Adds the aid station stop time (``stop_time_s`` from race YAML,
           defaults to 0 if absent).
        5. Accumulates elapsed time throughout.

        When ``use_fed=True`` the method first computes an adjusted-Riegel total
        on FED distance, converts it to a flat-equivalent average pace
        (seconds per FED-km), then applies inverse-GAP factors point-by-point
        along the full course. The resulting integrated running time is used as
        the final total.

        When ``use_fed=False`` the raw-distance Riegel result is used as the
        baseline flat pace and the same per-point GAP process is applied.

        Args:
            course:       ``Course`` object with loaded GPX data.
            aid_stations: List of aid-station dicts from the race YAML.
            use_fed:      Use FED-adjusted Riegel (default ``True``).
                          Ignored when *override_total_running_time_s* is set.
            override_total_running_time_s:
                          If given, skip Riegel entirely and use this value as
                          the total running time (seconds).  Grade corrections
                          still distribute the time across segments.  Use this
                          for ``target_time`` and ``target_itra`` planning modes.

        Returns:
            DataFrame with one row per aid station and columns:

            - Point Name
            - Total Distance (km)
            - Elevation (m)
            - Accum. Elevation Gain (m)
            - Segment Distance (km)
            - Segment Elevation Gain (m)
            - Segment Elevation Loss (m)
            - Segment Running Time  (H:MM:SS)
            - Stop Time             (H:MM:SS)
            - Elapsed Time          (H:MM:SS)

            The DataFrame's ``.attrs`` dict carries summary metadata:

            - ``total_running_time_s``
            - ``total_stop_time_s``
            - ``total_time_s``
            - ``riegel_method``  (``'FED'`` or ``'flat-distance'``)
            - ``riegel_running_time_approx_s`` (FED adjusted-Riegel approximation)
            - ``grade_adjusted_running_time_s`` (integrated course running time)
        """
        full_df = course.df

        # Convert grade (%) → decimal for the GAP curve
        grades_decimal = full_df['grade'].values / 100.0
        corrections = self.grade_correction(grades_decimal)
        dist_km_per_point = full_df['dist_m'].values / 1000.0

        riegel_running_time_approx_s: float | None = None

        if override_total_running_time_s is not None:
            total_running_time_s = float(override_total_running_time_s)
            riegel_method = 'target-override'
            weights = dist_km_per_point * corrections
            total_weight = weights.sum()
            time_per_weight = (
                total_running_time_s / total_weight if total_weight > 0 else 0.0
            )
            point_times_s = weights * time_per_weight
        elif use_fed:
            # 1) Adjusted-Riegel total approximation on FED distance.
            riegel_running_time_approx_s = self.predict_riegel_race_time_sec(
                target_distance_km=course.total_distance_km,
                elevation_gain_m=course.total_elevation_gain_m,
                use_flat_equivalent_distance=True,
            )

            # 2) Convert approximation into flat-equivalent average pace.
            fed_distance_km = self.flat_equivalent_distance_km(
                course.total_distance_km,
                course.total_elevation_gain_m,
            )
            if fed_distance_km <= 0:
                raise ValueError('Computed FED distance must be positive')
            flat_equiv_avg_pace_s_per_km = (
                riegel_running_time_approx_s / fed_distance_km
            )

            # 3) Apply inverse-GAP pace adjustments point-by-point and integrate.
            point_times_s = (
                dist_km_per_point * flat_equiv_avg_pace_s_per_km * corrections
            )
            riegel_method = 'FED'
        else:
            # Flat pace from raw-distance Riegel.
            flat_time_s = self.predict_riegel_race_time_sec(
                target_distance_km=course.total_distance_km,
                use_flat_equivalent_distance=False,
            )
            flat_pace_s_per_km = flat_time_s / course.total_distance_km
            point_times_s = dist_km_per_point * flat_pace_s_per_km * corrections
            riegel_method = 'flat-distance'

        cum_dist = full_df['cum_dist_m'].values

        rows = []
        cumulative_time_s = 0.0
        total_stop_s = 0.0

        for i, aid in enumerate(aid_stations):
            name = aid.get('name', 'Unknown')
            jap_name = aid.get('jap_name', '')
            full_name = f"{name} ({jap_name})" if jap_name else name

            dist_km = float(aid.get('distance_km', 0.0))
            dist_m = dist_km * 1000.0
            stop_time_s = float(aid.get('stop_time_s', 0))

            point = course.get_point_at_distance(dist_m)
            elevation_m = float(point['ele_m'])
            cum_ele_gain_m = float(point['cum_ele_gain_m'])

            if i == 0:
                running_time_s = 0.0
                seg_dist_km = 0.0
                seg_gain_m = 0.0
                seg_loss_m = 0.0
            else:
                prev_dist_m = (
                    float(aid_stations[i - 1].get('distance_km', 0.0)) * 1000.0
                )
                mask = (cum_dist >= prev_dist_m) & (cum_dist <= dist_m)

                seg_dist_km = dist_km - float(
                    aid_stations[i - 1].get('distance_km', 0.0)
                )
                seg_gain_m = float(full_df['ele_gain_m'].values[mask].sum())
                seg_loss_m = float(full_df['ele_loss_m'].values[mask].sum())
                running_time_s = float(point_times_s[mask].sum())

            cumulative_time_s += running_time_s + stop_time_s
            total_stop_s += stop_time_s

            rows.append(
                {
                    'Point Name': full_name,
                    'Total Distance (km)': dist_km,
                    'Elevation (m)': round(elevation_m, 1),
                    'Accum. Elevation Gain (m)': round(cum_ele_gain_m, 0),
                    'Segment Distance (km)': round(seg_dist_km, 2),
                    'Segment Elevation Gain (m)': round(seg_gain_m, 0),
                    'Segment Elevation Loss (m)': round(seg_loss_m, 0),
                    'Segment Running Time': seconds_to_hms(running_time_s),
                    'Stop Time': seconds_to_hms(stop_time_s),
                    'Elapsed Time': seconds_to_hms(cumulative_time_s),
                }
            )

        df = pd.DataFrame(rows)

        total_running_s = cumulative_time_s - total_stop_s
        df.attrs['total_running_time_s'] = total_running_s
        df.attrs['total_stop_time_s'] = total_stop_s
        df.attrs['total_time_s'] = cumulative_time_s
        df.attrs['riegel_method'] = riegel_method
        df.attrs['riegel_running_time_approx_s'] = riegel_running_time_approx_s
        df.attrs['grade_adjusted_running_time_s'] = total_running_s

        return df
