"""Analyze TOR330 2025 pacing strategies from an offline checkpoint snapshot.

This script reads a locally cached JSON snapshot extracted from:
    https://live.torxtrail.com/rankings/#/race/2025TOR330

For each finisher, it computes:
- Checkpoint elapsed times and checkpoint ranks for:
  Valgrisenche IN, Cogne IN, Donnas IN, Gressoney IN,
  Valtournenche IN, Ollomont IN, FINISH.
- Segment metrics across 7 macro sections:
  segment time, average pace, average GAP speed.
- Strategy indexes:
  donnas_time_ratio,
  gressoney_time_ratio,
  second_half_deceleration_ratio_donnas,
  second_half_deceleration_ratio_gressoney,
  rank_drift_donnas,
  rank_drift_gressoney,
  pace_variation_coefficient.

It also writes correlation-oriented plots to help identify robust pacing patterns.

Usage examples:
    python analysis/analyze_tor330_pacing_strategies.py
    python analysis/analyze_tor330_pacing_strategies.py --max-finish-hours 120
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from race_planner.course.course import Course
from race_planner.models.pacing_model import PacingModel


CHECKPOINT_ORDER = [
    "START",
    "Valgrisenche IN",
    "Cogne IN",
    "Donnas IN",
    "Gressoney IN",
    "Valtournenche IN",
    "Ollomont IN",
    "FINISH",
]

DISPLAY_CHECKPOINTS = [
    "Valgrisenche IN",
    "Cogne IN",
    "Donnas IN",
    "Gressoney IN",
    "Valtournenche IN",
    "Ollomont IN",
    "FINISH",
]


def _norm_station_name(name: str) -> str:
    text = re.sub(r"[*]+", "", str(name)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _checkpoint_to_yaml_station_name(checkpoint_name: str) -> str:
    mapping = {
        "START": "START",
        "Valgrisenche IN": "VALGRISENCHE",
        "Cogne IN": "COGNE",
        "Donnas IN": "DONNAS",
        "Gressoney IN": "GRESSONEY",
        "Valtournenche IN": "VALTOURNENCHE",
        "Ollomont IN": "OLLOMONT",
        "FINISH": "FINISH",
    }
    return mapping[checkpoint_name]


def _parse_iso(ts: str | None) -> pd.Timestamp:
    if not ts:
        return pd.NaT
    return pd.to_datetime(ts, utc=True, errors="coerce")


@dataclass
class SectionModel:
    start_checkpoint: str
    end_checkpoint: str
    distance_km: float
    avg_gap_correction: float

    @property
    def label(self) -> str:
        return f"{self.start_checkpoint} -> {self.end_checkpoint}"

    @property
    def gap_distance_km(self) -> float:
        return self.distance_km * self.avg_gap_correction


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    with snapshot_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _build_yaml_distance_map(race_cfg: dict[str, Any]) -> dict[str, float]:
    aid_stations = race_cfg.get("aid_stations", [])
    by_name_norm = {
        _norm_station_name(a.get("name", "")): float(a.get("distance_km", 0.0))
        for a in aid_stations
    }

    distances: dict[str, float] = {}
    for cp in CHECKPOINT_ORDER:
        yaml_key = _norm_station_name(_checkpoint_to_yaml_station_name(cp))
        matched = None
        for station_norm_name, station_distance in by_name_norm.items():
            if yaml_key in station_norm_name:
                matched = station_distance
                break
        if matched is None:
            raise ValueError(f"Could not map checkpoint '{cp}' to a station in race YAML")
        distances[cp] = float(matched)

    # Enforce non-decreasing distances.
    previous = -math.inf
    for cp in CHECKPOINT_ORDER:
        value = distances[cp]
        if value < previous:
            raise ValueError("Checkpoint distances are not non-decreasing in race config")
        previous = value

    return distances


def _build_section_models(
    course: Course, checkpoint_distances_km: dict[str, float]
) -> list[SectionModel]:
    model = PacingModel(ref_dist_km=42.195, ref_time_s=3 * 3600)
    sections: list[SectionModel] = []

    for idx in range(1, len(CHECKPOINT_ORDER)):
        start_cp = CHECKPOINT_ORDER[idx - 1]
        end_cp = CHECKPOINT_ORDER[idx]

        start_km = checkpoint_distances_km[start_cp]
        end_km = checkpoint_distances_km[end_cp]
        distance_km = max(end_km - start_km, 0.0)

        segment_df = course.get_segment(start_km=start_km, end_km=end_km)
        if segment_df.empty or distance_km <= 0.0:
            avg_corr = 1.0
        else:
            dist_km_points = segment_df["dist_m"].to_numpy(dtype=float) / 1000.0
            grade_decimal = segment_df["grade"].to_numpy(dtype=float) / 100.0
            corr = model.grade_correction(grade_decimal)
            weight = np.sum(dist_km_points)
            avg_corr = float(np.sum(corr * dist_km_points) / weight) if weight > 0 else 1.0

        sections.append(
            SectionModel(
                start_checkpoint=start_cp,
                end_checkpoint=end_cp,
                distance_km=distance_km,
                avg_gap_correction=avg_corr,
            )
        )

    return sections


def _build_checkpoint_rank_maps(runners: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    rank_maps: dict[str, dict[int, int]] = {}

    for cp in CHECKPOINT_ORDER:
        arrivals: list[tuple[pd.Timestamp, int]] = []
        for runner in runners:
            bib = int(runner.get("bib", -1))
            ts = _parse_iso(runner.get("checkpoint_times", {}).get(cp))
            if pd.notna(ts):
                arrivals.append((ts, bib))

        arrivals.sort(key=lambda x: (x[0], x[1]))
        rank_maps[cp] = {bib: idx + 1 for idx, (_, bib) in enumerate(arrivals)}

    return rank_maps


def _section_column_suffix(section: SectionModel) -> str:
    start = section.start_checkpoint.replace(" ", "_").lower().replace("->", "to")
    end = section.end_checkpoint.replace(" ", "_").lower().replace("->", "to")
    return f"{start}_to_{end}"


def _compute_runner_table(
    snapshot: dict[str, Any],
    sections: list[SectionModel],
    max_finish_hours: float | None,
) -> pd.DataFrame:
    runners: list[dict[str, Any]] = snapshot.get("runners", [])
    rank_maps = _build_checkpoint_rank_maps(runners)

    rows: list[dict[str, Any]] = []
    for runner in runners:
        checkpoint_times = runner.get("checkpoint_times", {})
        start_ts = _parse_iso(checkpoint_times.get("START"))
        finish_ts = _parse_iso(checkpoint_times.get("FINISH"))
        if pd.isna(start_ts) or pd.isna(finish_ts):
            continue

        total_h = (finish_ts - start_ts).total_seconds() / 3600.0
        if total_h <= 0:
            continue
        if max_finish_hours is not None and total_h > max_finish_hours:
            continue

        bib = int(runner.get("bib", -1))
        athlete = runner.get("athlete") or {}

        row: dict[str, Any] = {
            "bib": bib,
            "first_name": athlete.get("nome"),
            "last_name": athlete.get("cognome"),
            "nation": athlete.get("nazione"),
            "sex_is_male": athlete.get("sesso"),
            "team": athlete.get("team"),
            "total_time_h": total_h,
        }

        # Elapsed checkpoint times and checkpoint ranks
        elapsed_h_by_cp: dict[str, float] = {}
        for cp in DISPLAY_CHECKPOINTS:
            cp_ts = _parse_iso(checkpoint_times.get(cp))
            elapsed_h = np.nan
            if pd.notna(cp_ts):
                elapsed_h = (cp_ts - start_ts).total_seconds() / 3600.0
                if elapsed_h < 0:
                    elapsed_h = np.nan
            elapsed_h_by_cp[cp] = elapsed_h

            safe_cp = cp.lower().replace(" ", "_")
            row[f"elapsed_h_{safe_cp}"] = elapsed_h
            row[f"rank_{safe_cp}"] = rank_maps.get(cp, {}).get(bib)

        # Segment-level metrics
        section_gap_speeds: list[float] = []
        gap_distance_total = 0.0

        for section in sections:
            start_h = (
                0.0
                if section.start_checkpoint == "START"
                else elapsed_h_by_cp.get(section.start_checkpoint)
            )
            end_h = elapsed_h_by_cp.get(section.end_checkpoint)

            seg_h = np.nan
            if start_h is not None and end_h is not None and pd.notna(start_h) and pd.notna(end_h):
                seg_h = float(end_h - start_h)
                if seg_h <= 0:
                    seg_h = np.nan

            suffix = _section_column_suffix(section)
            row[f"segment_time_h_{suffix}"] = seg_h

            if pd.notna(seg_h) and section.distance_km > 0:
                pace_min_per_km = 60.0 * seg_h / section.distance_km
                actual_speed_kmh = section.distance_km / seg_h
                gap_speed_kmh = actual_speed_kmh * section.avg_gap_correction
            else:
                pace_min_per_km = np.nan
                gap_speed_kmh = np.nan

            row[f"avg_pace_min_per_km_{suffix}"] = pace_min_per_km
            row[f"avg_gap_speed_kmh_{suffix}"] = gap_speed_kmh
            row[f"avg_gap_correction_{suffix}"] = section.avg_gap_correction

            if pd.notna(gap_speed_kmh):
                section_gap_speeds.append(float(gap_speed_kmh))

            gap_distance_total += section.gap_distance_km

        # Summary indexes
        donnas_h = elapsed_h_by_cp.get("Donnas IN", np.nan)
        gressoney_h = elapsed_h_by_cp.get("Gressoney IN", np.nan)

        valg_h = elapsed_h_by_cp.get("Valgrisenche IN", np.nan)
        cogne_h = elapsed_h_by_cp.get("Cogne IN", np.nan)
        valtournenche_h = elapsed_h_by_cp.get("Valtournenche IN", np.nan)
        ollomont_h = elapsed_h_by_cp.get("Ollomont IN", np.nan)

        row["valgrisenche_time_ratio"] = (
            float(valg_h / total_h) if pd.notna(valg_h) and total_h > 0 else np.nan
        )
        row["cogne_time_ratio"] = (
            float(cogne_h / total_h) if pd.notna(cogne_h) and total_h > 0 else np.nan
        )
        row["donnas_time_ratio"] = (
            float(donnas_h / total_h) if pd.notna(donnas_h) and total_h > 0 else np.nan
        )
        row["gressoney_time_ratio"] = (
            float(gressoney_h / total_h) if pd.notna(gressoney_h) and total_h > 0 else np.nan
        )
        row["valtournenche_time_ratio"] = (
            float(valtournenche_h / total_h)
            if pd.notna(valtournenche_h) and total_h > 0
            else np.nan
        )
        row["ollomont_time_ratio"] = (
            float(ollomont_h / total_h) if pd.notna(ollomont_h) and total_h > 0 else np.nan
        )

        avg_gap_speed_total = gap_distance_total / total_h if total_h > 0 else np.nan
        row["avg_gap_speed_start_finish_kmh"] = avg_gap_speed_total

        gap_distance_donnas_finish = sum(
            s.gap_distance_km
            for s in sections
            if s.start_checkpoint
            in {"Donnas IN", "Gressoney IN", "Valtournenche IN", "Ollomont IN"}
        )
        gap_distance_start_donnas = sum(
            s.gap_distance_km
            for s in sections
            if s.start_checkpoint in {"START", "Valgrisenche IN", "Cogne IN"}
        )
        gap_distance_gressoney_finish = sum(
            s.gap_distance_km
            for s in sections
            if s.start_checkpoint in {"Gressoney IN", "Valtournenche IN", "Ollomont IN"}
        )
        gap_distance_start_gressoney = sum(
            s.gap_distance_km
            for s in sections
            if s.start_checkpoint in {"START", "Valgrisenche IN", "Cogne IN", "Donnas IN"}
        )

        if pd.notna(donnas_h) and total_h > donnas_h and avg_gap_speed_total > 0:
            donnas_first_half_gap_speed = gap_distance_start_donnas / donnas_h
            donnas_second_half_gap_speed = gap_distance_donnas_finish / (total_h - donnas_h)
            row["second_half_deceleration_ratio_donnas"] = (
                donnas_second_half_gap_speed / donnas_first_half_gap_speed
            )
        else:
            row["second_half_deceleration_ratio_donnas"] = np.nan

        if pd.notna(gressoney_h) and total_h > gressoney_h and avg_gap_speed_total > 0:
            gressoney_first_half_gap_speed = gap_distance_start_gressoney / gressoney_h
            gressoney_second_half_gap_speed = gap_distance_gressoney_finish / (
                total_h - gressoney_h
            )
            row["second_half_deceleration_ratio_gressoney"] = (
                gressoney_second_half_gap_speed / gressoney_first_half_gap_speed
            )
        else:
            row["second_half_deceleration_ratio_gressoney"] = np.nan

        rank_finish = row.get("rank_finish")
        rank_donnas = row.get("rank_donnas_in")
        rank_gressoney = row.get("rank_gressoney_in")

        row["rank_gain_donnas"] = (
            float(rank_donnas - rank_finish)
            if rank_donnas is not None and rank_finish is not None
            else np.nan
        )
        row["rank_gain_gressoney"] = (
            float(rank_gressoney - rank_finish)
            if rank_gressoney is not None and rank_finish is not None
            else np.nan
        )

        row["normalized_rank_gain_donnas"] = (
            row["rank_gain_donnas"] / rank_donnas
            if rank_donnas is not None and rank_donnas > 0 and pd.notna(row["rank_gain_donnas"])
            else np.nan
        )
        row["normalized_rank_gain_gressoney"] = (
            row["rank_gain_gressoney"] / rank_gressoney
            if rank_gressoney is not None
            and rank_gressoney > 0
            and pd.notna(row["rank_gain_gressoney"])
            else np.nan
        )

        decel_donnas = row.get("second_half_deceleration_ratio_donnas", np.nan)
        decel_gressoney = row.get("second_half_deceleration_ratio_gressoney", np.nan)
        nrg_donnas = row["normalized_rank_gain_donnas"]
        nrg_gressoney = row["normalized_rank_gain_gressoney"]

        row["execution_index_donnas"] = (
            float(nrg_donnas * decel_donnas)
            if pd.notna(nrg_donnas) and pd.notna(decel_donnas)
            else np.nan
        )
        row["execution_index_gressoney"] = (
            float(nrg_gressoney * decel_gressoney)
            if pd.notna(nrg_gressoney) and pd.notna(decel_gressoney)
            else np.nan
        )

        if section_gap_speeds:
            section_gap_speeds_np = np.asarray(section_gap_speeds, dtype=float)
            mean_speed = float(np.mean(section_gap_speeds_np))
            std_speed = float(np.std(section_gap_speeds_np))
            row["pace_variation_coefficient"] = std_speed / mean_speed if mean_speed > 0 else np.nan
        else:
            row["pace_variation_coefficient"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["total_time_h", "bib"]).reset_index(drop=True)
    return df


def _build_sections_export(sections: list[SectionModel]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": s.label,
                "start_checkpoint": s.start_checkpoint,
                "end_checkpoint": s.end_checkpoint,
                "distance_km": s.distance_km,
                "avg_gap_correction": s.avg_gap_correction,
                "gap_distance_km": s.gap_distance_km,
            }
            for s in sections
        ]
    )


def _plot_indexes(df: pd.DataFrame, output_dir: Path) -> None:
    # --- 1. Scatter Plot Matrix ---
    fig_scatter, axes_scatter = plt.subplots(2, 3, figsize=(18, 10))
    ax_scatter = axes_scatter.ravel()

    scatter_specs = [
        (
            "donnas_time_ratio",
            "rank_gain_donnas",
            "total_time_h",
            "Donnas Time Ratio",
            "Absolute Rank Gain",
            "Donnas ratio vs Absolute Rank Gain (color: Finish Time)",
        ),
        (
            "donnas_time_ratio",
            "pace_variation_coefficient",
            "total_time_h",
            "Donnas Time Ratio",
            "Pace Variation Coefficient",
            "Donnas ratio vs Pace Variation (color: Finish Time)",
        ),
        (
            "donnas_time_ratio",
            "total_time_h",
            "rank_gain_donnas",
            "Donnas Time Ratio",
            "Finish Time (h)",
            "Donnas ratio vs Finish Time (color: Absolute Rank Gain)",
        ),
        (
            "donnas_time_ratio",
            "normalized_rank_gain_donnas",
            "total_time_h",
            "Donnas Time Ratio",
            "Normalized Rank Gain (Donnas)",
            "Donnas ratio vs Normalized Rank Gain (color: Finish Time)",
        ),
        (
            "donnas_time_ratio",
            "second_half_deceleration_ratio_donnas",
            "total_time_h",
            "Donnas Time Ratio",
            "Deceleration from Donnas",
            "Donnas ratio vs Deceleration (color: Finish Time)",
        ),
        (
            "donnas_time_ratio",
            "execution_index_donnas",
            "total_time_h",
            "Donnas Time Ratio",
            "Execution Index (Donnas)",
            "Donnas ratio vs Execution Index (color: Finish Time)",
        ),
    ]

    for axis, spec in zip(ax_scatter, scatter_specs):
        x_col, y_col, c_col, x_label, y_label, title = spec
        mask = df[[x_col, y_col]].notna().all(axis=1)

        if c_col is not None:
            mask = mask & df[[c_col]].notna().all(axis=1)
            sc = axis.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                c=df.loc[mask, c_col],
                cmap="viridis",
                alpha=0.75,
                s=18,
            )
            plt.colorbar(sc, ax=axis, shrink=0.78)
        else:
            axis.scatter(df.loc[mask, x_col], df.loc[mask, y_col], color="C0", alpha=0.75, s=18)

        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)

    fig_scatter.suptitle(
        "TOR330 2025 Donnas-Relative Pacing Strategy Scatters", fontsize=14, fontweight="bold"
    )
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / "tor330_pacing_index_scatter_matrix.png", dpi=170)
    plt.close(fig_scatter)

    # --- 2. Histograms ---
    fig_hist, axes_hist = plt.subplots(2, 2, figsize=(12, 10))
    ax_hist = axes_hist.ravel()

    hist_specs = [
        ("donnas_time_ratio", "Donnas Time Ratio", "Histogram of Donnas Time Ratio"),
        ("execution_index_donnas", "Execution Index (Donnas)", "Histogram of Execution Index"),
        (
            "second_half_deceleration_ratio_donnas",
            "Deceleration from Donnas",
            "Histogram of Deceleration from Donnas",
        ),
        (
            "normalized_rank_gain_donnas",
            "Normalized Rank Gain",
            "Histogram of Normalized Rank Gain",
        ),
    ]

    for axis, (col, x_label, title) in zip(ax_hist, hist_specs):
        values = df[col].dropna()
        axis.hist(values, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Count")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)

    fig_hist.suptitle("TOR330 2025 Pacing Distribution Analysis", fontsize=14, fontweight="bold")
    fig_hist.tight_layout()
    fig_hist.savefig(output_dir / "tor330_pacing_index_histograms.png", dpi=170)
    plt.close(fig_hist)

    # --- 3. Correlation Heatmap ---
    idx_cols = [
        "donnas_time_ratio",
        "gressoney_time_ratio",
        "rank_gain_donnas",
        "normalized_rank_gain_donnas",
        "execution_index_donnas",
        "pace_variation_coefficient",
        "total_time_h",
        "second_half_deceleration_ratio_donnas",
    ]

    corr_df = df[idx_cols].dropna(how="all")
    corr = corr_df.corr(numeric_only=True)

    fig_corr, axis_corr = plt.subplots(figsize=(10, 8))
    im = axis_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axis_corr.set_xticks(np.arange(len(corr.columns)))
    axis_corr.set_yticks(np.arange(len(corr.index)))
    axis_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    axis_corr.set_yticklabels(corr.index)
    axis_corr.set_title("Correlation Heatmap of Pacing Indexes")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            axis_corr.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=axis_corr, fraction=0.046, pad=0.04)
    fig_corr.tight_layout()
    fig_corr.savefig(output_dir / "tor330_pacing_index_correlation_heatmap.png", dpi=170)
    plt.close(fig_corr)


def _build_index_table(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "bib",
        "first_name",
        "last_name",
        "nation",
        "total_time_h",
        "rank_finish",
        "valgrisenche_time_ratio",
        "cogne_time_ratio",
        "donnas_time_ratio",
        "gressoney_time_ratio",
        "valtournenche_time_ratio",
        "ollomont_time_ratio",
        "second_half_deceleration_ratio_donnas",
        "second_half_deceleration_ratio_gressoney",
        "rank_gain_donnas",
        "rank_gain_gressoney",
        "normalized_rank_gain_donnas",
        "normalized_rank_gain_gressoney",
        "execution_index_donnas",
        "execution_index_gressoney",
        "pace_variation_coefficient",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TOR330 2025 pacing strategies")
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        default=Path("analysis/data/tor330_2025_checkpoints_snapshot.json"),
        help="Offline JSON snapshot extracted from the TORX live page",
    )
    parser.add_argument(
        "--race-config",
        type=Path,
        default=Path("config/races/tor330.yaml"),
        help="Race YAML used to derive section distances and course profile",
    )
    parser.add_argument(
        "--max-finish-hours",
        type=float,
        default=None,
        help="Keep only finishers with total time <= this threshold (hours)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/analyze_tor330_pacing_strategies"),
        help="Directory where CSV/Excel and plots are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshot_path = args.snapshot_json
    race_config_path = args.race_config
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot JSON not found: {snapshot_path}. "
            "Create it once from the live site and keep it in analysis/data/."
        )
    if not race_config_path.exists():
        raise FileNotFoundError(f"Race config not found: {race_config_path}")

    snapshot = _load_snapshot(snapshot_path)

    with race_config_path.open("r", encoding="utf-8") as f:
        race_cfg = yaml.safe_load(f)

    race_section = race_cfg.get("race", {})
    gpx_file = race_section.get("gpx_file")
    if not gpx_file:
        raise ValueError("Race config must define race.gpx_file")

    checkpoint_distances_km = _build_yaml_distance_map(race_cfg)

    course = Course(Path(gpx_file), resample_m=float(race_section.get("resample_m", 5)))
    sections = _build_section_models(course, checkpoint_distances_km)

    runner_df = _compute_runner_table(snapshot, sections, args.max_finish_hours)
    if runner_df.empty:
        raise ValueError("No finishers left after filtering. Try a larger --max-finish-hours.")

    index_df = _build_index_table(runner_df)
    section_df = _build_sections_export(sections)

    runner_csv = output_dir / "tor330_2025_pacing_runner_table.csv"
    index_csv = output_dir / "tor330_2025_pacing_indexes.csv"
    section_csv = output_dir / "tor330_2025_section_model.csv"
    excel_path = output_dir / "tor330_2025_pacing_analysis.xlsx"

    runner_df.to_csv(runner_csv, index=False)
    index_df.to_csv(index_csv, index=False)
    section_df.to_csv(section_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        runner_df.to_excel(writer, sheet_name="runner_table", index=False)
        index_df.to_excel(writer, sheet_name="indexes", index=False)
        section_df.to_excel(writer, sheet_name="section_model", index=False)

    _plot_indexes(runner_df, output_dir)

    logger.info(f"Saved runner table to: {runner_csv}")
    logger.info(f"Saved index table to: {index_csv}")
    logger.info(f"Saved section model to: {section_csv}")
    logger.info(f"Saved Excel workbook to: {excel_path}")
    logger.info(f"Analyzed finishers: {len(runner_df)}")
    logger.info(
        "Median strategy profile: "
        f"donnas_ratio={index_df['donnas_time_ratio'].median():.3f}, "
        f"gressoney_ratio={index_df['gressoney_time_ratio'].median():.3f}, "
        f"decel_donnas={index_df['second_half_deceleration_ratio_donnas'].median():.3f}, "
        f"decel_gressoney={index_df['second_half_deceleration_ratio_gressoney'].median():.3f}, "
        f"rank_gain_donnas={index_df['rank_gain_donnas'].median():.1f}, "
        f"exec_index_donnas={index_df['execution_index_donnas'].median():.3f}, "
        f"pace_cv={index_df['pace_variation_coefficient'].median():.3f}"
    )


if __name__ == "__main__":
    main()
