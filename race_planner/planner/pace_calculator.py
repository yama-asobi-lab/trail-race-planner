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
    +0.20 (+20%)  |  2.60     ← uphill cutoff anchor
    +0.01  (+1%)  |  1.08     ← 8% slower
    +0.00  ( 0%)  |  1.00
    -0.01  (-1%)  |  0.965    ← 3.5% faster
    -0.06  (-6%)  |  0.79     ← 21% faster  (empirical optimum)
    -0.07  (-7%)  |  0.86     ← 14% faster  (braking begins)
    -0.20 (-20%)  |  1.60     ← downhill cutoff anchor

For grades beyond ±18%, correction follows a constant-vertical-speed rule:

    c(g) = c(g_cutoff) * g / g_cutoff

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
    • Tail handling: outside ±18% grade, corrections switch to a
        constant-vertical-speed rule anchored at the cutoff points. This avoids
        unrealistic tail growth from unconstrained linear extrapolation.

References:
  - Riegel, P.S. (1981) Athletic records and human endurance.
  - Minetti, A.E. et al. (2002) J. Exp. Biol. 205:3041-3047.
  - https://repository.lboro.ac.uk/articles/journal_contribution/
    Pace_and_critical_gradient_for_hill_runners_an_analysis_of_race_records/9387719
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from race_planner.course.course import Course
from race_planner.models.pacing_model import PacingModel
from race_planner.models.tools import seconds_per_km_to_mmss, seconds_to_hms


class PaceCalculator:
    """
    Calculates a per-segment pacing plan for a trail race.

    This planner class delegates pure formulas to ``PacingModel`` and keeps
    course traversal + segment aggregation responsibilities.

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

    # Expose model constants from the pure-model layer.
    DEFAULT_GAP_CURVE = PacingModel.DEFAULT_GAP_CURVE
    GAP_UPHILL_CUTOFF_GRADE: float = PacingModel.GAP_UPHILL_CUTOFF_GRADE
    GAP_DOWNHILL_CUTOFF_GRADE: float = PacingModel.GAP_DOWNHILL_CUTOFF_GRADE
    RIEGEL_BASE_EXPONENT: float = PacingModel.RIEGEL_BASE_EXPONENT
    PIECEWISE_RIEGEL_106_SQRT_C: float = PacingModel.PIECEWISE_RIEGEL_106_SQRT_C
    FED_VERT_FACTOR_M_PER_KM: float = PacingModel.FED_VERT_FACTOR_M_PER_KM

    def __init__(
        self,
        ref_dist_km: float,
        ref_time_s: float,
        gap_curve: Optional[np.ndarray] = None,
        fatigue_total_decay_pct: float = 0.0,
    ) -> None:
        self.model = PacingModel(
            ref_dist_km=ref_dist_km,
            ref_time_s=ref_time_s,
            gap_curve=gap_curve,
        )
        self.ref_dist_km = self.model.ref_dist_km
        self.ref_time_s = self.model.ref_time_s
        self.gap_curve = self.model.gap_curve
        self.fatigue_total_decay_pct = float(fatigue_total_decay_pct)
        if not 0.0 <= self.fatigue_total_decay_pct <= 100.0:
            raise ValueError("fatigue_total_decay_pct must be between 0 and 100")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_athlete_config(
        cls, athlete_config: Dict, fatigue_total_decay_pct: float = 0.0
    ) -> "PaceCalculator":
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
            fatigue_total_decay_pct: Linear fatigue decay (0-100); optional override.

        Returns:
            PaceCalculator instance.
        """
        model = PacingModel.from_athlete_config(athlete_config)
        return cls(
            ref_dist_km=model.ref_dist_km,
            ref_time_s=model.ref_time_s,
            gap_curve=model.gap_curve,
            fatigue_total_decay_pct=fatigue_total_decay_pct,
        )

    # ------------------------------------------------------------------
    # Compatibility wrappers around the pure-model layer
    # ------------------------------------------------------------------

    def flat_equivalent_distance_km(self, dist_km: float, gain_m: float) -> float:
        """Delegate FED calculation to the pure model layer."""
        return self.model.flat_equivalent_distance_km(dist_km, gain_m)

    def predict_riegel_race_time_sec(
        self,
        target_distance_km: float,
        elevation_gain_m: float = 0.0,
        use_flat_equivalent_distance: bool = False,
    ) -> float:
        """Delegate Riegel prediction to the pure model layer."""
        return self.model.predict_riegel_race_time_sec(
            target_distance_km,
            elevation_gain_m,
            use_flat_equivalent_distance,
        )

    def predict_riegel_flat_race_time_sec(self, target_distance_km: float) -> float:
        """Delegate flat-distance Riegel prediction to the pure model layer."""
        return self.model.predict_riegel_flat_race_time_sec(target_distance_km)

    def predict_riegel_fed_race_time_sec(
        self,
        target_distance_km: float,
        elevation_gain_m: float,
    ) -> float:
        """Delegate FED-adjusted Riegel prediction to the pure model layer."""
        return self.model.predict_riegel_fed_race_time_sec(
            target_distance_km,
            elevation_gain_m,
        )

    def grade_correction(self, grade_decimal_values: np.ndarray) -> np.ndarray:
        """Delegate GAP correction lookup to the pure model layer."""
        return self.model.grade_correction(grade_decimal_values)

    def grade_weighted_distance_km(
        self,
        course: Course,
        end_distance_km: float | None = None,
    ) -> float:
        """Return total GAP-weighted distance for the course in km.

        If ``end_distance_km`` is provided, only points up to that cumulative
        distance are included. This matches pacing plans that stop at the last
        configured aid-station distance instead of the GPX endpoint.
        """
        grade_decimal_values = course.df["grade"].values / 100.0
        grade_correction_factors = self.grade_correction(grade_decimal_values)
        point_distance_km_values = course.df["dist_m"].values / 1000.0
        weighted_distance_km_values = point_distance_km_values * grade_correction_factors

        if end_distance_km is None:
            return float(weighted_distance_km_values.sum())

        end_distance_m = float(end_distance_km) * 1000.0
        point_mask = course.df["cum_dist_m"].values <= end_distance_m
        return float(weighted_distance_km_values[point_mask].sum())

    def fatigue_multiplier(self, progress_fraction_values: np.ndarray) -> np.ndarray:
        """Converts “how far through the race am I?” into a pace slowdown multiplier.
        Return per-point pace multipliers for linear fatigue model.

        Pace multiplier rises linearly from 1.0 (start) to 1.0 + decay_fraction (finish).
        """
        if np.any(progress_fraction_values < 0.0) or np.any(progress_fraction_values > 1.0):
            raise ValueError("progress_fraction_values must be between 0 and 1")

        if self.fatigue_total_decay_pct == 0.0:
            return np.ones_like(progress_fraction_values)

        total_decay_fraction = self.fatigue_total_decay_pct / 100.0
        return 1.0 + total_decay_fraction * progress_fraction_values

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
        on FED distance, converts it to a FED-derived baseline pace
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
            - Avg Pace (mm:ss/km)
            - Avg Grade-Adjusted Pace (mm:ss/km)
            - Stop Time             (H:MM:SS)
            - Elapsed Time          (H:MM:SS)

            The DataFrame's ``.attrs`` dict carries summary metadata:

            - ``total_running_time_s``
            - ``total_stop_time_s``
            - ``total_time_s``
            - ``riegel_method``  (``'FED'`` or ``'flat-distance'``)
            - ``riegel_running_time_approx_s`` (FED adjusted-Riegel approximation)
            - ``grade_adjusted_running_time_s`` (integrated course running time)
            - ``overall_avg_pace_mmss``
            - ``overall_avg_grade_adjusted_pace_mmss``
            - ``total_grade_weighted_distance_km``
        """
        full_df = course.df

        # When no aid stations are configured treat the whole course as one segment.
        if not aid_stations:
            aid_stations = [
                {"name": "Start", "distance_km": 0.0},
                {"name": "Finish", "distance_km": course.total_distance_km},
            ]

        # Convert grade (%) → decimal for the GAP curve
        grade_decimal_values = full_df["grade"].values / 100.0
        grade_correction_factors = self.grade_correction(grade_decimal_values)
        point_distance_km_values = full_df["dist_m"].values / 1000.0
        point_grade_weighted_distance_km_values = (
            point_distance_km_values * grade_correction_factors
        )
        cumulative_distance_m_values = full_df["cum_dist_m"].values
        planned_finish_distance_km = float(
            aid_stations[-1].get("distance_km", course.total_distance_km)
        )
        planned_finish_distance_m = planned_finish_distance_km * 1000.0
        planned_point_mask = cumulative_distance_m_values <= planned_finish_distance_m
        total_grade_weighted_distance_km = float(
            point_grade_weighted_distance_km_values[planned_point_mask].sum()
        )

        # Compute fatigue multipliers based on progress through the course
        progress_distance_m_values = np.minimum(
            cumulative_distance_m_values,
            planned_finish_distance_m,
        )
        progress_fraction_values = (
            progress_distance_m_values / planned_finish_distance_m
            if planned_finish_distance_m > 0
            else np.zeros_like(cumulative_distance_m_values)
        )
        fatigue_multiplier_values = self.fatigue_multiplier(progress_fraction_values)

        # Effective distance = grade-weighted distance * fatigue multiplier
        point_effective_weighted_distance_km_values = (
            point_grade_weighted_distance_km_values * fatigue_multiplier_values
        )
        total_effective_weighted_distance_km = float(
            point_effective_weighted_distance_km_values[planned_point_mask].sum()
        )

        riegel_running_time_approx_s: float | None = None

        if override_total_running_time_s is not None:
            total_running_time_s = float(override_total_running_time_s)
            riegel_method = "target-override"
            seconds_per_weighted_km = (
                total_running_time_s / total_effective_weighted_distance_km
                if total_effective_weighted_distance_km > 0
                else 0.0
            )
            point_times_s = point_effective_weighted_distance_km_values * seconds_per_weighted_km
        elif use_fed:
            # 1) Adjusted-Riegel total approximation on FED distance.
            riegel_running_time_approx_s = self.predict_riegel_race_time_sec(
                target_distance_km=course.total_distance_km,
                elevation_gain_m=course.total_elevation_gain_m,
                use_flat_equivalent_distance=True,
            )

            # 2) Convert the FED-based total into a baseline pace per FED-km.
            fed_distance_km = self.flat_equivalent_distance_km(
                course.total_distance_km,
                course.total_elevation_gain_m,
            )
            if fed_distance_km <= 0:
                raise ValueError("Computed FED distance must be positive")
            fed_baseline_pace_s_per_km = riegel_running_time_approx_s / fed_distance_km

            # 3) Apply inverse-GAP pace adjustments point-by-point and integrate.
            point_times_s = (
                point_distance_km_values * fed_baseline_pace_s_per_km * grade_correction_factors
            )
            point_times_s = point_times_s * fatigue_multiplier_values
            riegel_method = "FED"
        else:
            # Flat pace from raw-distance Riegel.
            flat_time_s = self.predict_riegel_race_time_sec(
                target_distance_km=course.total_distance_km,
                use_flat_equivalent_distance=False,
            )
            flat_pace_s_per_km = flat_time_s / course.total_distance_km
            point_times_s = point_distance_km_values * flat_pace_s_per_km * grade_correction_factors
            point_times_s = point_times_s * fatigue_multiplier_values
            riegel_method = "flat-distance"

        rows = []
        cumulative_elapsed_time_s = 0.0
        total_stop_s = 0.0

        for i, aid in enumerate(aid_stations):
            name = aid.get("name", "Unknown")
            jap_name = aid.get("jap_name", "")
            full_name = f"{name} ({jap_name})" if jap_name else name

            aid_distance_km = float(aid.get("distance_km", 0.0))
            aid_distance_m = aid_distance_km * 1000.0
            stop_time_s = float(aid.get("stop_time_s", 0))

            aid_point = course.get_point_at_distance(aid_distance_m)
            elevation_m = float(aid_point["ele_m"])
            cum_ele_gain_m = float(aid_point["cum_ele_gain_m"])

            if i == 0:
                running_time_s = 0.0
                seg_dist_km = 0.0
                seg_gain_m = 0.0
                seg_loss_m = 0.0
                seg_avg_pace_mmss = "-"
                seg_avg_grade_adjusted_pace_mmss = "-"
            else:
                prev_aid_distance_m = float(aid_stations[i - 1].get("distance_km", 0.0)) * 1000.0
                segment_point_mask = (cumulative_distance_m_values >= prev_aid_distance_m) & (
                    cumulative_distance_m_values <= aid_distance_m
                )

                seg_dist_km = aid_distance_km - float(aid_stations[i - 1].get("distance_km", 0.0))
                seg_gain_m = float(full_df["ele_gain_m"].values[segment_point_mask].sum())
                seg_loss_m = float(full_df["ele_loss_m"].values[segment_point_mask].sum())
                running_time_s = float(point_times_s[segment_point_mask].sum())
                seg_grade_weighted_dist_km = float(
                    point_grade_weighted_distance_km_values[segment_point_mask].sum()
                )

                if seg_dist_km > 0:
                    seg_avg_pace_mmss = seconds_per_km_to_mmss(running_time_s / seg_dist_km)
                else:
                    seg_avg_pace_mmss = "-"

                if seg_grade_weighted_dist_km > 0:
                    seg_avg_grade_adjusted_pace_mmss = seconds_per_km_to_mmss(
                        running_time_s / seg_grade_weighted_dist_km
                    )
                else:
                    seg_avg_grade_adjusted_pace_mmss = "-"

            cumulative_elapsed_time_s += running_time_s + stop_time_s
            total_stop_s += stop_time_s

            rows.append(
                {
                    "Point Name": full_name,
                    "Total Distance (km)": aid_distance_km,
                    "Elevation (m)": round(elevation_m, 1),
                    "Accum. Elevation Gain (m)": round(cum_ele_gain_m, 0),
                    "Segment Distance (km)": round(seg_dist_km, 2),
                    "Segment Elevation Gain (m)": round(seg_gain_m, 0),
                    "Segment Elevation Loss (m)": round(seg_loss_m, 0),
                    "Segment Gain (%)": (
                        round(seg_gain_m / seg_dist_km / 10, 1) if seg_dist_km > 0 else "-"
                    ),
                    "Segment Running Time": seconds_to_hms(running_time_s),
                    "Avg Pace (mm:ss/km)": seg_avg_pace_mmss,
                    "Avg Grade-Adjusted Pace (mm:ss/km)": seg_avg_grade_adjusted_pace_mmss,
                    "Stop Time": seconds_to_hms(stop_time_s),
                    "Elapsed Time": seconds_to_hms(cumulative_elapsed_time_s),
                }
            )

        df = pd.DataFrame(rows)

        total_running_s = cumulative_elapsed_time_s - total_stop_s
        df.attrs["total_running_time_s"] = total_running_s
        df.attrs["total_stop_time_s"] = total_stop_s
        df.attrs["total_time_s"] = cumulative_elapsed_time_s
        df.attrs["riegel_method"] = riegel_method
        df.attrs["riegel_running_time_approx_s"] = riegel_running_time_approx_s
        df.attrs["grade_adjusted_running_time_s"] = total_running_s
        df.attrs["overall_avg_pace_mmss"] = (
            seconds_per_km_to_mmss(total_running_s / course.total_distance_km)
            if course.total_distance_km > 0
            else "-"
        )
        df.attrs["overall_avg_grade_adjusted_pace_mmss"] = (
            seconds_per_km_to_mmss(total_running_s / total_grade_weighted_distance_km)
            if total_grade_weighted_distance_km > 0
            else "-"
        )
        df.attrs["total_grade_weighted_distance_km"] = total_grade_weighted_distance_km
        df.attrs["fatigue_total_decay_pct"] = self.fatigue_total_decay_pct

        return df
