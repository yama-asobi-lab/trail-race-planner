"""Pure pacing model logic (Riegel, FED, and GAP correction).

This module intentionally contains only math/model behavior:

- piecewise Riegel race-time prediction,
- flat equivalent distance (FED) conversion,
- inverse-GAP correction interpolation with constant-vertical-speed tails.

Planner concerns (course traversal, segment accumulation, DataFrame assembly)
belong in the planner layer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from race_planner.models.tools import hms_to_seconds


class PacingModel:
    """Pure mathematical model used by the planner layer.

    This class contains no course traversal or tabular report assembly logic.
    """

    # Built-in default GAP correction table (grade in decimal, correction factor).
    #
    # correction > 1  -> slower than flat (uphills)
    # correction < 1  -> faster than flat (gentle downhills)
    # correction > 1  -> slower again (very steep downhills, braking)
    #
    # Notes:
    # - +20% and -20% are explicit tail anchors.
    # - The downhill branch is non-monotonic by design.
    # - Rows need not be sorted at input; they are sorted in __init__.
    DEFAULT_GAP_CURVE = np.array(
        [
            [0.20, 2.60],  # +20 % cutoff anchor
            [0.01, 1.08],  # +1 %
            [0.00, 1.00],  # flat
            [-0.01, 0.97],  # -1 %
            [-0.05, 0.85],  # -5 % (optimal descent zone)
            [-0.06, 0.9],  # -6 % (braking effect starts)
            [-0.20, 1.60],  # -20 % cutoff anchor
        ],
        dtype=float,
    )

    GAP_UPHILL_CUTOFF_GRADE: float = 0.20
    GAP_DOWNHILL_CUTOFF_GRADE: float = -0.20

    # Piecewise Riegel parameters (combined-sex robust fit from analysis):
    #   k(D) = 1.06 + c*sqrt(max(D-D_ref, 0))
    RIEGEL_BASE_EXPONENT: float = 1.06
    PIECEWISE_RIEGEL_106_SQRT_C: float = 0.013422

    # Flat Equivalent Distance factor: metres of gain per 1 km FED.
    # 100 m gain <-> 1 km flat (standard trail-running convention).
    FED_VERT_FACTOR_M_PER_KM: float = 100.0

    def __init__(
        self,
        ref_dist_km: float,
        ref_time_s: float,
        gap_curve: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a pure pacing model.

        Args:
            ref_dist_km: Reference flat race distance in km (e.g. 42.195).
            ref_time_s: Reference flat race time in seconds.
            gap_curve: Optional custom GAP curve with rows
                [grade_decimal, correction_factor].
        """
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
        self.gap_curve = raw[raw[:, 0].argsort()]

    @classmethod
    def from_athlete_config(cls, athlete_config: dict) -> "PacingModel":
        """Build model parameters from athlete YAML dict."""
        athlete = athlete_config.get("athlete", {})
        ref = athlete.get("reference_performance", {})

        ref_dist_km = ref.get("distance_km")
        ref_time_str = ref.get("time")

        if ref_dist_km is None or ref_time_str is None:
            raise ValueError(
                "Athlete config must contain "
                "'athlete.reference_performance.distance_km' and "
                "'athlete.reference_performance.time'"
            )

        ref_time_s = float(hms_to_seconds(str(ref_time_str)))
        custom_points = athlete.get("gap_curve", {}).get("points") or []
        gap_curve = np.array(custom_points, dtype=float) if custom_points else None

        return cls(
            ref_dist_km=float(ref_dist_km),
            ref_time_s=ref_time_s,
            gap_curve=gap_curve,
        )

    def flat_equivalent_distance_km(self, dist_km: float, gain_m: float) -> float:
        """Convert mountain course distance and gain to flat equivalent distance.

        Only elevation *gain* contributes to FED in this model; descents are less
        taxing and are not included in the standard trail-running FED formula.

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
        """Predict race time (seconds) using piecewise Riegel.

        Exponent model:
            k(D) = 1.06 + c*sqrt(max(D - D_ref, 0))

        where ``c = PIECEWISE_RIEGEL_106_SQRT_C`` and ``D_ref`` is
        ``self.ref_dist_km``. If ``use_flat_equivalent_distance`` is true, ``D``
        is flat equivalent distance (FED) instead of horizontal distance.

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
        exponent = self.RIEGEL_BASE_EXPONENT + self.PIECEWISE_RIEGEL_106_SQRT_C * np.sqrt(
            ultra_excess_km
        )
        return self.ref_time_s * (distance_ratio**exponent)

    def predict_riegel_flat_race_time_sec(self, target_distance_km: float) -> float:
        """Predict flat race time (seconds) for a horizontal target distance."""
        return self.predict_riegel_race_time_sec(target_distance_km)

    def predict_riegel_fed_race_time_sec(
        self, target_distance_km: float, elevation_gain_m: float
    ) -> float:
        """Predict race time (seconds) using FED-adjusted distance."""
        return self.predict_riegel_race_time_sec(
            target_distance_km,
            elevation_gain_m,
            use_flat_equivalent_distance=True,
        )

    def grade_correction(self, grade_decimal_values: np.ndarray) -> np.ndarray:
        """Compute GAP correction factors for an array of grade values, with
        constant-vspeed tails.

        Uses piecewise linear interpolation on the curve knots, and applies a
        constant-vertical-speed tail beyond the configured cutoff grades.

            c(g) = c(g_cutoff) * g / g_cutoff

        which preserves signed vertical speed for a fixed flat pace.

        Args:
            grade_decimal_values: 1-D array of grade values as decimal rise/run
                (e.g. 0.10 for a 10% uphill, -0.06 for a 6% downhill).

        Returns:
            Array of dimensionless correction factors, same length as *grade_decimal_values*.

        """

        curve_grade_decimal = self.gap_curve[:, 0]
        curve_correction_factor = self.gap_curve[:, 1]

        grade_decimal_values = np.asarray(grade_decimal_values, dtype=float)
        correction_factor_values = np.interp(
            grade_decimal_values,
            curve_grade_decimal,
            curve_correction_factor,
        )

        low_grade_mask = grade_decimal_values <= curve_grade_decimal[0]
        if np.any(low_grade_mask):
            low_edge_slope_factor_per_grade = (
                curve_correction_factor[1] - curve_correction_factor[0]
            ) / (curve_grade_decimal[1] - curve_grade_decimal[0])
            correction_factor_values[low_grade_mask] = curve_correction_factor[
                0
            ] + low_edge_slope_factor_per_grade * (
                grade_decimal_values[low_grade_mask] - curve_grade_decimal[0]
            )

        high_grade_mask = grade_decimal_values >= curve_grade_decimal[-1]
        if np.any(high_grade_mask):
            high_edge_slope_factor_per_grade = (
                curve_correction_factor[-1] - curve_correction_factor[-2]
            ) / (curve_grade_decimal[-1] - curve_grade_decimal[-2])
            correction_factor_values[high_grade_mask] = curve_correction_factor[
                -1
            ] + high_edge_slope_factor_per_grade * (
                grade_decimal_values[high_grade_mask] - curve_grade_decimal[-1]
            )

        uphill_cutoff_grade_decimal = self.GAP_UPHILL_CUTOFF_GRADE
        downhill_cutoff_grade_decimal = self.GAP_DOWNHILL_CUTOFF_GRADE

        uphill_cutoff_correction_factor = float(
            np.interp(
                uphill_cutoff_grade_decimal,
                curve_grade_decimal,
                curve_correction_factor,
            )
        )
        downhill_cutoff_correction_factor = float(
            np.interp(
                downhill_cutoff_grade_decimal,
                curve_grade_decimal,
                curve_correction_factor,
            )
        )

        uphill_tail_mask = grade_decimal_values > uphill_cutoff_grade_decimal
        if np.any(uphill_tail_mask):
            correction_factor_values[uphill_tail_mask] = (
                uphill_cutoff_correction_factor
                * grade_decimal_values[uphill_tail_mask]
                / uphill_cutoff_grade_decimal
            )

        downhill_tail_mask = grade_decimal_values < downhill_cutoff_grade_decimal
        if np.any(downhill_tail_mask):
            correction_factor_values[downhill_tail_mask] = (
                downhill_cutoff_correction_factor
                * grade_decimal_values[downhill_tail_mask]
                / downhill_cutoff_grade_decimal
            )

        return correction_factor_values
