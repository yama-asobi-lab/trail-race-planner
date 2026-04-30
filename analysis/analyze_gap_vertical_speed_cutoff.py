"""Inspect GAP extrapolation versus constant-vertical-speed cutoffs.

This script is a diagnostic tool for choosing where the inverse-GAP model
should stop extrapolating and instead preserve a constant signed vertical
speed for steeper slopes.

For a selected athlete, it:
- loads the athlete-specific GAP curve and threshold flat pace,
- evaluates the current extrapolated GAP correction from -50% to +50% grade,
- builds a modified curve that keeps signed vertical speed constant beyond
  configurable uphill/downhill cutoffs,
- plots correction factor, pace, horizontal speed, and vertical speed.

Outputs are written to analysis/results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pandas as pd
import yaml

try:
    from race_planner.models.tools import (
        pace_from_constant_vertical_speed,
        pace_to_seconds_per_km,
        seconds_per_km_to_pace,
        speed_kmh_from_pace,
        vertical_speed_m_per_h,
    )
    from race_planner.planner import PaceCalculator
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from race_planner.models.tools import (
        pace_from_constant_vertical_speed,
        pace_to_seconds_per_km,
        seconds_per_km_to_pace,
        speed_kmh_from_pace,
        vertical_speed_m_per_h,
    )
    from race_planner.planner import PaceCalculator


ATHLETE_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "athletes" / "carlos.yaml"
)
OUT_DIR = Path(__file__).parent / "results"

GRADE_MIN = -0.50
GRADE_MAX = 0.50
N_GRADE_POINTS = 1001

UPHILL_GRADE_CUTOFF = 0.18
DOWNHILL_GRADE_CUTOFF = -0.18


def build_export_grades() -> np.ndarray:
    outer_grades = np.concatenate(
        [
            np.arange(GRADE_MIN, -0.20, 0.05),
            np.arange(0.25, GRADE_MAX + 0.001, 0.05),
        ]
    )
    inner_grades = np.arange(-0.20, 0.20 + 0.001, 0.01)
    return np.unique(np.round(np.concatenate([outer_grades, inner_grades]), 6))


def build_cutoff_curve(
    grades: np.ndarray,
    threshold_flat_pace_sec_per_km: float,
    calc: PaceCalculator,
    uphill_cutoff: float,
    downhill_cutoff: float,
) -> dict[str, np.ndarray | float]:
    correction_extrap = calc.grade_correction(grades)
    pace_extrap = threshold_flat_pace_sec_per_km * correction_extrap

    pace_cutoff = pace_extrap.copy()

    uphill_cutoff_pace = float(
        threshold_flat_pace_sec_per_km
        * calc.grade_correction(np.array([uphill_cutoff]))[0]
    )
    downhill_cutoff_pace = float(
        threshold_flat_pace_sec_per_km
        * calc.grade_correction(np.array([downhill_cutoff]))[0]
    )

    uphill_cutoff_vspeed = float(
        vertical_speed_m_per_h(
            np.array([uphill_cutoff]), np.array([uphill_cutoff_pace])
        )[0]
    )
    downhill_cutoff_vspeed = float(
        vertical_speed_m_per_h(
            np.array([downhill_cutoff]), np.array([downhill_cutoff_pace])
        )[0]
    )

    uphill_mask = grades > uphill_cutoff
    downhill_mask = grades < downhill_cutoff

    if np.any(uphill_mask):
        pace_cutoff[uphill_mask] = pace_from_constant_vertical_speed(
            grades[uphill_mask], uphill_cutoff_vspeed
        )
    if np.any(downhill_mask):
        pace_cutoff[downhill_mask] = pace_from_constant_vertical_speed(
            grades[downhill_mask], downhill_cutoff_vspeed
        )

    correction_cutoff = pace_cutoff / threshold_flat_pace_sec_per_km

    return {
        "correction_extrap": correction_extrap,
        "pace_extrap": pace_extrap,
        "speed_extrap": speed_kmh_from_pace(pace_extrap),
        "vspeed_extrap": vertical_speed_m_per_h(grades, pace_extrap),
        "correction_cutoff": correction_cutoff,
        "pace_cutoff": pace_cutoff,
        "speed_cutoff": speed_kmh_from_pace(pace_cutoff),
        "vspeed_cutoff": vertical_speed_m_per_h(grades, pace_cutoff),
        "uphill_cutoff_pace": uphill_cutoff_pace,
        "downhill_cutoff_pace": downhill_cutoff_pace,
        "uphill_cutoff_vspeed": uphill_cutoff_vspeed,
        "downhill_cutoff_vspeed": downhill_cutoff_vspeed,
    }


def write_plot(
    out_path: Path,
    grades: np.ndarray,
    threshold_flat_pace_sec_per_km: float,
    curve: dict[str, np.ndarray | float],
    pace_label: str,
) -> None:
    grade_pct = grades * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax_corr, ax_pace, ax_speed, ax_vspeed = axes.ravel()

    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="major", axis="x", alpha=0.35)
        ax.grid(which="major", axis="y", alpha=0.3)
        ax.grid(which="minor", axis="y", alpha=0.18, linestyle=":")

    ax_corr.plot(
        grade_pct, curve["correction_extrap"], label="GAP extrapolation", linewidth=2
    )
    ax_corr.plot(
        grade_pct,
        curve["correction_cutoff"],
        label="constant-vspeed cutoff",
        linewidth=2,
    )
    ax_corr.axvline(
        UPHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_corr.axvline(
        DOWNHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_corr.set_title("Adjustment Factor vs Grade")
    ax_corr.set_ylabel("factor")
    ax_corr.legend(loc="best")

    ax_pace.plot(
        grade_pct, curve["pace_extrap"] / 60.0, label="GAP extrapolation", linewidth=2
    )
    ax_pace.plot(
        grade_pct,
        curve["pace_cutoff"] / 60.0,
        label="constant-vspeed cutoff",
        linewidth=2,
    )
    ax_pace.axhline(
        threshold_flat_pace_sec_per_km / 60.0,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label="threshold flat pace",
    )
    ax_pace.axvline(
        UPHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_pace.axvline(
        DOWNHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_pace.set_title("Pace vs Grade")
    ax_pace.set_ylabel("min/km")

    ax_speed.plot(
        grade_pct, curve["speed_extrap"], label="GAP extrapolation", linewidth=2
    )
    ax_speed.plot(
        grade_pct, curve["speed_cutoff"], label="constant-vspeed cutoff", linewidth=2
    )
    ax_speed.axvline(
        UPHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_speed.axvline(
        DOWNHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_speed.set_title("Horizontal Speed vs Grade")
    ax_speed.set_xlabel("grade (%)")
    ax_speed.set_ylabel("km/h")

    ax_vspeed.plot(
        grade_pct, curve["vspeed_extrap"], label="GAP extrapolation", linewidth=2
    )
    ax_vspeed.plot(
        grade_pct, curve["vspeed_cutoff"], label="constant-vspeed cutoff", linewidth=2
    )
    ax_vspeed.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    ax_vspeed.axvline(
        UPHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_vspeed.axvline(
        DOWNHILL_GRADE_CUTOFF * 100.0, color="black", linestyle="--", linewidth=1
    )
    ax_vspeed.set_title("Vertical Speed vs Grade")
    ax_vspeed.set_xlabel("grade (%)")
    ax_vspeed.set_ylabel("m/h")

    fig.suptitle(
        "Inverse-GAP extrapolation vs constant-vertical-speed cutoffs\n"
        f"Carlos {pace_label} flat pace = {seconds_per_km_to_pace(threshold_flat_pace_sec_per_km)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_outputs_for_pace(
    athlete: dict,
    calc: PaceCalculator,
    pace_sec_per_km: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    dict[str, np.ndarray | float],
    np.ndarray,
]:
    grades = np.linspace(GRADE_MIN, GRADE_MAX, N_GRADE_POINTS)
    export_grades = build_export_grades()
    curve = build_cutoff_curve(
        grades,
        pace_sec_per_km,
        calc,
        UPHILL_GRADE_CUTOFF,
        DOWNHILL_GRADE_CUTOFF,
    )
    export_curve = build_cutoff_curve(
        export_grades,
        pace_sec_per_km,
        calc,
        UPHILL_GRADE_CUTOFF,
        DOWNHILL_GRADE_CUTOFF,
    )

    out_df = pd.DataFrame(
        {
            "grade_decimal": export_grades,
            "grade_pct": export_grades * 100.0,
            "correction_extrap": export_curve["correction_extrap"],
            "correction_cutoff": export_curve["correction_cutoff"],
            "pace_extrap_sec_per_km": export_curve["pace_extrap"],
            "pace_cutoff_sec_per_km": export_curve["pace_cutoff"],
            "speed_extrap_kmh": export_curve["speed_extrap"],
            "speed_cutoff_kmh": export_curve["speed_cutoff"],
            "vertical_speed_extrap_m_per_h": export_curve["vspeed_extrap"],
            "vertical_speed_cutoff_m_per_h": export_curve["vspeed_cutoff"],
        }
    )

    summary_grades = np.array(
        [
            GRADE_MIN,
            DOWNHILL_GRADE_CUTOFF,
            -0.15,
            -0.10,
            -0.05,
            0.00,
            0.05,
            0.10,
            0.15,
            UPHILL_GRADE_CUTOFF,
            GRADE_MAX,
        ]
    )
    summary_curve = build_cutoff_curve(
        summary_grades,
        pace_sec_per_km,
        calc,
        UPHILL_GRADE_CUTOFF,
        DOWNHILL_GRADE_CUTOFF,
    )
    summary_df = pd.DataFrame(
        {
            "grade_pct": summary_grades * 100.0,
            "pace_extrap": [
                seconds_per_km_to_pace(value) for value in summary_curve["pace_extrap"]
            ],
            "pace_cutoff": [
                seconds_per_km_to_pace(value) for value in summary_curve["pace_cutoff"]
            ],
            "speed_extrap_kmh": summary_curve["speed_extrap"],
            "speed_cutoff_kmh": summary_curve["speed_cutoff"],
            "vertical_speed_extrap_m_per_h": summary_curve["vspeed_extrap"],
            "vertical_speed_cutoff_m_per_h": summary_curve["vspeed_cutoff"],
        }
    )

    metadata_df = pd.DataFrame(
        [
            {
                "athlete": athlete.get("name", "unknown"),
                "flat_pace_per_km": seconds_per_km_to_pace(pace_sec_per_km),
                "uphill_cutoff_pct": UPHILL_GRADE_CUTOFF * 100.0,
                "downhill_cutoff_pct": DOWNHILL_GRADE_CUTOFF * 100.0,
                "uphill_cutoff_pace": seconds_per_km_to_pace(
                    float(curve["uphill_cutoff_pace"])
                ),
                "downhill_cutoff_pace": seconds_per_km_to_pace(
                    float(curve["downhill_cutoff_pace"])
                ),
                "uphill_cutoff_vertical_speed_m_per_h": float(
                    curve["uphill_cutoff_vspeed"]
                ),
                "downhill_cutoff_vertical_speed_m_per_h": float(
                    curve["downhill_cutoff_vspeed"]
                ),
            }
        ]
    )
    return out_df, summary_df, metadata_df, grades, curve, summary_grades


def main() -> None:
    with ATHLETE_CONFIG.open("r", encoding="utf-8") as f:
        athlete_config = yaml.safe_load(f)

    athlete = athlete_config.get("athlete", {})
    preferences = athlete.get("preferences", {})
    threshold_pace_str = preferences.get("threshold_flat_pace_per_km")
    aerobic_threshold_pace_str = preferences.get("aerobic_threshold_flat_pace_per_km")
    if threshold_pace_str is None:
        raise ValueError(
            f"Missing athlete.preferences.threshold_flat_pace_per_km in {ATHLETE_CONFIG}"
        )
    if aerobic_threshold_pace_str is None:
        raise ValueError(
            f"Missing athlete.preferences.aerobic_threshold_flat_pace_per_km in {ATHLETE_CONFIG}"
        )

    threshold_flat_pace_sec_per_km = pace_to_seconds_per_km(str(threshold_pace_str))
    aerobic_threshold_flat_pace_sec_per_km = pace_to_seconds_per_km(
        str(aerobic_threshold_pace_str)
    )
    calc = PaceCalculator.from_athlete_config(athlete_config)

    (
        out_df,
        summary_df,
        metadata_df,
        grades,
        curve,
        _,
    ) = build_outputs_for_pace(
        athlete,
        calc,
        threshold_flat_pace_sec_per_km,
    )
    (
        out_df_lt1,
        summary_df_lt1,
        metadata_df_lt1,
        grades_lt1,
        curve_lt1,
        _,
    ) = build_outputs_for_pace(
        athlete,
        calc,
        aerobic_threshold_flat_pace_sec_per_km,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = OUT_DIR / "gap_vertical_speed_cutoff_carlos.xlsx"
    plot_path = OUT_DIR / "gap_vertical_speed_cutoff_carlos.png"
    plot_path_lt1 = OUT_DIR / "gap_vertical_speed_cutoff_carlos_lt1.png"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # LT2 output
        out_df.to_excel(writer, sheet_name="grade_curves_LT2", index=False)
        summary_df.to_excel(writer, sheet_name="summary_LT2", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata_LT2", index=False)

        # LT1 output
        out_df_lt1.to_excel(writer, sheet_name="grade_curves_LT1", index=False)
        summary_df_lt1.to_excel(writer, sheet_name="summary_LT1", index=False)
        metadata_df_lt1.to_excel(writer, sheet_name="metadata_LT1", index=False)

    write_plot(
        plot_path,
        grades,
        threshold_flat_pace_sec_per_km,
        curve,
        pace_label="LT2 threshold",
    )
    write_plot(
        plot_path_lt1,
        grades_lt1,
        aerobic_threshold_flat_pace_sec_per_km,
        curve_lt1,
        pace_label="LT1 aerobic threshold",
    )

    print("Vertical-speed cutoff GAP analysis")
    print(f"Athlete: {athlete.get('name', 'unknown')}")
    print(
        f"Threshold flat pace (LT2): {seconds_per_km_to_pace(threshold_flat_pace_sec_per_km)}"
    )
    print(
        f"Aerobic threshold flat pace (LT1): "
        f"{seconds_per_km_to_pace(aerobic_threshold_flat_pace_sec_per_km)}"
    )
    print(
        f"Cutoffs: uphill={UPHILL_GRADE_CUTOFF * 100:.0f}%, "
        f"downhill={DOWNHILL_GRADE_CUTOFF * 100:.0f}%"
    )
    print(
        f"Uphill cutoff pace={seconds_per_km_to_pace(float(curve['uphill_cutoff_pace']))}, "
        f"vertical speed={float(curve['uphill_cutoff_vspeed']):.0f} m/h"
    )
    print(
        f"Downhill cutoff pace={seconds_per_km_to_pace(float(curve['downhill_cutoff_pace']))}, "
        f"vertical speed={float(curve['downhill_cutoff_vspeed']):.0f} m/h"
    )
    print(
        f"LT1 uphill cutoff pace={seconds_per_km_to_pace(float(curve_lt1['uphill_cutoff_pace']))}, "
        f"vertical speed={float(curve_lt1['uphill_cutoff_vspeed']):.0f} m/h"
    )
    print(
        f"LT1 downhill cutoff pace={seconds_per_km_to_pace(float(curve_lt1['downhill_cutoff_pace']))}, "
        f"vertical speed={float(curve_lt1['downhill_cutoff_vspeed']):.0f} m/h"
    )
    print(f"Wrote: {excel_path}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {plot_path_lt1}")


if __name__ == "__main__":
    main()
