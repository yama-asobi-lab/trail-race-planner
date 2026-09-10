"""Static pace-profile plots for race planning outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from race_planner.models.tools import seconds_to_hms


def plot_grade_adjusted_pace_profile(
    output_path: Path | str,
    pace_profile_data: dict,
) -> None:
    """Save a two-panel PNG using precomputed sea-level GAP pace-profile data."""
    required_keys = [
        "elapsed_time_h",
        "distance_km",
        "pace_min_per_km",
        "elevation_m",
        "aid_points",
        "break_indices",
        "total_time_h",
    ]
    missing = [key for key in required_keys if key not in pace_profile_data]
    if missing:
        raise ValueError(f"Missing required pace profile keys: {missing}")

    elapsed_time_h = np.asarray(pace_profile_data["elapsed_time_h"], dtype=float)
    distance_km = np.asarray(pace_profile_data["distance_km"], dtype=float)
    pace_min_per_km = np.asarray(pace_profile_data["pace_min_per_km"], dtype=float)
    elevation_m = np.asarray(pace_profile_data["elevation_m"], dtype=float)
    aid_points = list(pace_profile_data["aid_points"])
    break_indices = [int(index) for index in pace_profile_data["break_indices"]]
    total_time_h = float(pace_profile_data["total_time_h"])

    if elapsed_time_h.size == 0:
        raise ValueError("Pace profile data is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=True)
    pace_color = "#1d4ed8"
    elevation_color = "#475569"
    station_color = "#f59e0b"

    plot_specs = [
        (axes[0], elapsed_time_h, pace_min_per_km, "Elapsed time (h)", True),
        (axes[1], distance_km, pace_min_per_km, "Distance (km)", False),
    ]

    for axis, x_values, y_values, x_label, is_time_axis in plot_specs:
        twin_axis = axis.twinx()
        twin_axis.plot(
            x_values,
            elevation_m,
            color=elevation_color,
            alpha=0.16,
            linewidth=1.8,
            zorder=1,
        )
        twin_axis.fill_between(x_values, elevation_m, color=elevation_color, alpha=0.05, zorder=0)
        twin_axis.set_ylabel("Elevation (m)", color=elevation_color, fontsize=10)
        twin_axis.tick_params(axis="y", colors=elevation_color, labelsize=9)
        twin_axis.grid(False)

        plot_y = y_values.copy()
        if is_time_axis:
            for aid_index in break_indices:
                if 0 < aid_index < len(plot_y) - 1:
                    plot_y[aid_index] = np.nan

        axis.plot(
            x_values,
            plot_y,
            color=pace_color,
            linewidth=2.0,
            zorder=3,
            label="Sea-level back-converted GAP pace",
        )

        for aid in aid_points:
            aid_name = str(aid.get("name", "Aid station"))
            aid_y = float(aid.get("pace_min_per_km", np.nan))
            stop_time_s = float(aid.get("stop_time_s", 0.0))
            aid_x = (
                float(aid.get("elapsed_time_h", 0.0))
                if is_time_axis
                else float(aid.get("distance_km", 0.0))
            )
            axis.scatter(
                [aid_x],
                [aid_y],
                s=42,
                color=station_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            axis.axvline(aid_x, color=station_color, linestyle="--", linewidth=0.8, alpha=0.55)

            if is_time_axis and stop_time_s > 0:
                stop_hours = stop_time_s / 3600.0
                axis.axvspan(aid_x, aid_x + stop_hours, color=station_color, alpha=0.08, zorder=0)
                axis.text(
                    aid_x + stop_hours / 2.0,
                    0.98,
                    seconds_to_hms(int(round(stop_time_s))),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=station_color,
                    rotation=90,
                )

            axis.annotate(
                aid_name,
                xy=(aid_x, aid_y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="#8a5b00",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#fff7ed",
                    "edgecolor": "#fbbf24",
                    "linewidth": 0.5,
                    "alpha": 0.88,
                },
                zorder=5,
            )

        axis.set_xlabel(x_label)
        axis.set_ylabel("Sea-level back-converted GAP pace (min/km)")
        axis.grid(True, color="#cbd5e1", alpha=0.35, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)
        axis.spines["top"].set_visible(False)

    axes[0].set_title(
        "Sea-level back-converted GAP pace through the race", fontsize=15, fontweight="bold"
    )
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)

    axes[0].set_xlim(0.0, total_time_h)
    axes[1].set_xlim(0.0, float(distance_km[-1]))

    y_min = float(np.nanmin(pace_min_per_km))
    y_max = float(np.nanmax(pace_min_per_km))
    y_pad = max((y_max - y_min) * 0.10, 0.25)
    axes[0].set_ylim(y_min - y_pad, y_max + y_pad)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
