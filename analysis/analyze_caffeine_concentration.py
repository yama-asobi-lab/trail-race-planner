"""
Caffeine concentration profile for race fueling planning.

Model assumptions (simplified):
- Each dose is fully absorbed after a fixed lag (default: 30 min).
- Elimination is first-order decay with configurable half-life (default: 5.5 h).
- Concentration is expressed as mg/kg body weight.

This script generates one figure only:
- analysis/results/analyze_caffeine_concentration/caffeine_concentration_profile.png
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from race_planner.models.nutrition import caffeine_concentration_mg_per_kg
from race_planner.models.tools import race_offset_to_clock_hhmm, hhmm_to_hours

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Athlete and race setup (edit here)
# ---------------------------------------------------------------------------

ATHLETE_WEIGHT_KG = 65.0
RACE_DURATION_H = 32.0
RACE_START_TIME = "16:00"

INGESTION_PLAN = [
    (5.0, 200.0),
    (8.0, 200.0),
    (13.0, 200.0),
    (22.0, 200.0),
    (27.0, 200.0),
]  # list of (time_hours, dose_mg) tuples

# PK-style parameters
ABSORPTION_LAG_H = 0.5
HALF_LIFE_H = 5.5

# Target range often discussed for endurance performance effects
TARGET_MIN_MG_PER_KG = 3.0
TARGET_MAX_MG_PER_KG = 6.0
RACE_TICK_STEP_H = 4.0
CLOCK_TICK_STEP_H = 4.0

# Output
OUTPUT_DIR = Path(__file__).parent / "results" / "analyze_caffeine_concentration"
OUTPUT_PATH = OUTPUT_DIR / "caffeine_concentration_profile.png"


def plot_caffeine_profile(
    time_h: np.ndarray,
    concentration_mg_per_kg: np.ndarray,
    ingestion_plan: list[tuple[float, float]],
    race_start_time: str,
    output_path: Path,
) -> None:
    """Plot caffeine concentration with target band and side table."""
    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.7, 1.1], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    # Main concentration curve
    ax.plot(time_h, concentration_mg_per_kg, color="#0b6e4f", linewidth=2.4)
    ax.fill_between(
        time_h,
        TARGET_MIN_MG_PER_KG,
        TARGET_MAX_MG_PER_KG,
        color="#ffd166",
        alpha=0.20,
        label="Suggested zone: 3-6 mg/kg",
    )

    # Mark ingestion times that occur during race timeline
    for dose_time_h, dose_mg in ingestion_plan:
        if 0.0 <= dose_time_h <= RACE_DURATION_H:
            ax.axvline(dose_time_h, linestyle=":", color="#6c757d", alpha=0.22, linewidth=0.9)

    ax.set_xlim(0.0, RACE_DURATION_H)
    upper_y = max(TARGET_MAX_MG_PER_KG + 1.0, float(np.max(concentration_mg_per_kg)) + 0.8)
    ax.set_ylim(0.0, upper_y)
    ax.set_xticks(np.arange(0.0, RACE_DURATION_H + 1e-9, RACE_TICK_STEP_H))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlabel("Race time (hours)")
    ax.set_ylabel("Caffeine concentration (mg/kg)")
    ax.set_title("Caffeine Concentration During Race", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)

    summary_text = (
        f"Athlete weight: {ATHLETE_WEIGHT_KG:.1f} kg\n"
        f"Race start: {race_start_time}\n"
        f"Absorption lag: {ABSORPTION_LAG_H:.1f} h\n"
        f"Half-life: {HALF_LIFE_H:.1f} h"
    )
    ax.text(
        0.015,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.0,
        bbox={"facecolor": "white", "alpha": 0.82, "boxstyle": "round,pad=0.25"},
    )

    # Secondary x-axis with time of day labels in 24-hour format.
    start_h = hhmm_to_hours(race_start_time)
    ax_top = ax.secondary_xaxis("top")
    ticks = np.arange(0.0, RACE_DURATION_H + 1e-9, CLOCK_TICK_STEP_H)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([race_offset_to_clock_hhmm(start_h, t) for t in ticks])
    ax_top.tick_params(axis="x", labelsize=9)
    ax_top.set_xlabel("Time of day (24h)")

    # Side table with ingestion plan
    ax_table.axis("off")
    table_rows = []
    for dose_time_h, dose_mg in ingestion_plan:
        race_label = f"T{dose_time_h:+.1f} h"
        clock_label = race_offset_to_clock_hhmm(start_h, dose_time_h)
        table_rows.append([race_label, clock_label, f"{dose_mg:.0f}"])

    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Race time", "Clock", "Dose (mg)"],
        loc="upper center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    # Light table borders improve readability without visual overload.
    for _, cell in table.get_celld().items():
        cell.set_edgecolor("#ced4da")
        cell.set_linewidth(0.5)

    ax_table.set_title("Ingestion Plan", fontsize=11, fontweight="bold", pad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    time_h = np.linspace(0.0, RACE_DURATION_H, 1201)
    concentration = caffeine_concentration_mg_per_kg(
        time_h=time_h,
        ingestion_plan=INGESTION_PLAN,
        weight_kg=ATHLETE_WEIGHT_KG,
        absorption_lag_h=ABSORPTION_LAG_H,
        half_life_h=HALF_LIFE_H,
    )

    plot_caffeine_profile(time_h, concentration, INGESTION_PLAN, RACE_START_TIME, OUTPUT_PATH)

    peak_value = float(np.max(concentration))
    mean_value = float(np.mean(concentration))
    logger.info(f"Saved caffeine concentration plot to {OUTPUT_PATH}")
    logger.info(f"Peak concentration: {peak_value:.2f} mg/kg")
    logger.info(f"Average concentration across race: {mean_value:.2f} mg/kg")


if __name__ == "__main__":
    main()
