"""Generate smartphone-friendly race plan HTML report with embedded elevation profile."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from race_planner.course import Course
from race_planner.models.tools import (
    clock_time_to_seconds,
    elapsed_hms_to_clock_time,
    hms_to_seconds,
    seconds_to_hms,
)
from race_planner.visualization.course_profile import CourseProfilePlotter


_KNOWN_AID_STATION_FIELDS = {
    "name",
    "jap_name",
    "distance_km",
    "elevation_m",
    "stop_time_s",
    "notes",
    "gmaps_link",
    "top_in_time",
    "cutoff_in_time",
    "cutoff_out_time",
    "reference_last_time",
    "cutoff_exempt",
    "cutoff_note",
}


_REPORT_CSS = """
    :root {
      --bg: #07131f;
      --bg-panel: #0d2031;
      --bg-panel-2: #132a3f;
      --bg-sticky: #102539;
      --bg-header: #183753;
      --line: #2d4b68;
      --line-strong: #47739d;
      --text: #f3e9c6;
      --text-soft: #cdbf93;
      --accent: #58d0d7;
      --accent-2: #ffc857;
      --accent-3: #88aee0;
      --shadow: rgba(0, 0, 0, 0.28);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background:
        radial-gradient(circle at top, rgba(88, 208, 215, 0.14), transparent 32%),
        linear-gradient(180deg, #05101a 0%, var(--bg) 100%);
      color: var(--text);
      font-family: "Segoe UI", "Aptos", "Noto Sans JP", sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    a {
      color: var(--accent);
    }

    .page {
      max-width: 1480px;
      margin: 0 auto;
      padding: 8px 4px 14px;
    }

    .hero {
      background: linear-gradient(180deg, rgba(19, 42, 63, 0.98), rgba(10, 23, 35, 0.96));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 10px 12px 10px;
      box-shadow: 0 16px 40px var(--shadow);
      margin-bottom: 8px;
    }

    .eyebrow {
      color: var(--accent);
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.45rem, 4.8vw, 2.2rem);
      line-height: 1.04;
      color: var(--text);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }

    .summary-card {
      background: rgba(7, 19, 31, 0.72);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 9px 11px;
      min-height: 62px;
    }

    .time-tune {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .time-tune-label {
      color: var(--text-soft);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .time-tune-input {
      background: rgba(7, 19, 31, 0.82);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 9px;
      padding: 6px 8px;
      min-width: 118px;
      font-variant-numeric: tabular-nums;
    }

    .time-tune-input.small {
      min-width: 84px;
    }

    .time-tune-button {
      background: var(--bg-header);
      border: 1px solid var(--line-strong);
      color: var(--text);
      border-radius: 9px;
      padding: 6px 10px;
      font-size: 0.78rem;
      cursor: pointer;
    }

    .time-tune-button:hover {
      background: #1f4567;
    }

    .time-tune-status {
      color: var(--text-soft);
      font-size: 0.75rem;
      min-height: 1em;
    }

    .summary-label {
      display: block;
      color: var(--text-soft);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 5px;
    }

    .summary-value {
      display: block;
      color: var(--text);
      font-size: 0.98rem;
      font-weight: 700;
    }

    .section-header {
      display: flex;
      align-items: baseline;
      gap: 10px;
      flex-wrap: wrap;
      margin: 10px 0 6px;
    }

    .section-title {
      margin: 0;
      font-size: 0.94rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent-2);
    }

    .section-meta {
      color: rgba(243, 233, 198, 0.68);
      font-size: 0.84rem;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }

    .table-shell {
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: auto;
      background: linear-gradient(180deg, rgba(13, 32, 49, 0.98), rgba(8, 20, 31, 0.98));
      box-shadow: 0 16px 40px var(--shadow);
      max-height: 68vh;
      -webkit-overflow-scrolling: touch;
    }

    table {
      width: max-content;
      min-width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: auto;
    }

    thead th {
      position: sticky;
      top: 0;
      z-index: 4;
      background: var(--bg-header);
      color: var(--text);
      text-align: left;
      padding: 8px 8px;
      font-size: 0.74rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line-strong);
      border-right: 1px solid var(--line);
    }

    thead th:first-child {
      left: 0;
      z-index: 6;
      background: var(--bg-sticky);
    }

    tbody th,
    tbody td {
      padding: 7px 7px;
      vertical-align: top;
      border-bottom: 1px solid rgba(71, 115, 157, 0.28);
      border-right: 1px solid rgba(71, 115, 157, 0.18);
      background: rgba(9, 22, 34, 0.92);
    }

    tbody tr:nth-child(even) th,
    tbody tr:nth-child(even) td {
      background: rgba(12, 28, 43, 0.95);
    }

    .sticky-col {
      position: sticky;
      left: 0;
      z-index: 3;
      background: var(--bg-sticky);
      min-width: 134px;
      box-shadow: 6px 0 14px rgba(3, 10, 18, 0.32);
    }

    tbody tr:nth-child(even) .sticky-col {
      background: #13293f;
    }

    .station-cell {
      width: 134px;
    }

    .station-name-wrap {
      display: block;
    }

    .station-name {
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--text);
      line-height: 1.12;
    }

    .station-japanese {
      margin-top: 2px;
      color: var(--accent-2);
      font-size: 0.74rem;
      line-height: 1.1;
    }

    .station-link {
      margin-top: 6px;
      font-size: 0.66rem;
      line-height: 1;
    }

    .cell-line {
      display: flex;
      justify-content: space-between;
      gap: 6px;
      align-items: baseline;
      margin-bottom: 3px;
    }

    .value-line {
      margin-bottom: 3px;
      white-space: nowrap;
    }

    .value-line:last-child,
    .cell-line:last-child {
      margin-bottom: 0;
    }

    .cell-label {
      color: var(--text-soft);
      font-size: 0.72rem;
      flex: 0 0 auto;
      line-height: 1;
    }

    .cell-value {
      color: var(--text);
      font-size: 0.8rem;
      line-height: 1.15;
      white-space: nowrap;
    }

    .cell-value.emphasized {
      color: #fff2b8;
      font-weight: 700;
      font-size: 0.86rem;
    }

    .cutoff-inline {
      color: var(--text-soft);
      font-size: 0.72rem;
    }

    .metric-cell {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .metric-cell .value-line {
      display: flex;
      justify-content: flex-end;
    }

    .timing-cell {
      min-width: 112px;
      font-variant-numeric: tabular-nums;
    }

    .comments-cell .cell-line {
      display: none;
    }

    .comments-empty {
      color: var(--text-soft);
      font-size: 0.88rem;
      white-space: nowrap;
    }

    .comments-cell {
      white-space: nowrap;
      width: 1%;
    }

    .comments-text {
      color: var(--text);
      font-size: 0.88rem;
      line-height: 1.12;
    }

    .comments-line {
      display: flex;
      gap: 7px;
      align-items: baseline;
      white-space: nowrap;
      margin-bottom: 4px;
    }

    .comments-line:last-child {
      margin-bottom: 0;
    }

    .comments-tag {
      color: var(--text-soft);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      flex: 0 0 auto;
    }

    .comments-value {
      color: var(--text);
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }

    .profile-shell {
      margin-top: 8px;
      border: none;
      border-radius: 12px;
      overflow: hidden;
      background: #1a1a1a;
      box-shadow: 0 8px 24px var(--shadow);
    }

    .profile-shell .plotly-graph-div,
    .profile-shell .js-plotly-plot,
    .profile-shell .plot-container {
      width: 100% !important;
      background: #1a1a1a !important;
    }

    @media (max-width: 720px) {
      .page {
        padding: 8px 6px 14px;
      }

      .hero {
        padding: 9px 9px 8px;
        border-radius: 16px;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }

      thead th,
      tbody th,
      tbody td {
        padding: 6px 6px;
      }

      .station-cell,
      .sticky-col {
        min-width: 118px;
        width: 118px;
      }

      .station-name {
        font-size: 0.78rem;
      }

      .station-japanese {
        font-size: 0.68rem;
      }

      .table-shell {
        max-height: 62vh;
      }
    }
"""


@dataclass(frozen=True)
class SummaryItemViewModel:
    label: str
    value: str


@dataclass(frozen=True)
class CommentsLineViewModel:
    tag: str
    value: str


@dataclass(frozen=True)
class TableRowViewModel:
    station_name: str
    station_japanese: str
    station_link: str
    distance_total: str
    distance_segment: str
    accum_gain: str
    accum_loss: str
    split_gain: str
    split_loss: str
    elapsed_time: str
    elapsed_seconds: int
    running_time: str
    running_seconds: int
    clock_time: str
    cutoff_in_time: str
    avg_pace: str
    avg_pace_seconds: int
    avg_gap: str
    avg_gap_seconds: int
    comments: tuple[CommentsLineViewModel, ...]


@dataclass(frozen=True)
class RacePlanReportViewModel:
    title: str
    race_name: str
    summary_items: tuple[SummaryItemViewModel, ...]
    profile_meta: str
    total_time_seconds: int
    race_start_time_s: int
    original_fatigue_decay_pct: float
    table_rows: tuple[TableRowViewModel, ...]
    plot_html: str


def _normalized_aid_stations(
    pacing_df: pd.DataFrame,
    aid_stations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return aid-station metadata aligned with pacing rows."""
    if aid_stations:
        return aid_stations

    return [
        {
            "name": str(row.get("Point Name", "Unknown")),
            "distance_km": float(row.get("Total Distance (km)", 0.0) or 0.0),
            "elevation_m": float(row.get("Elevation (m)", 0.0) or 0.0),
            "stop_time_s": 0,
        }
        for row in pacing_df.to_dict(orient="records")
    ]


def _summary_time_label(mode: str) -> str:
    """Return the correct lead summary label for the planning mode."""
    return "Target time" if mode == "target_time" else "Planned finish"


def _format_signed_meters(value: Any, sign: str) -> str:
    """Format elevation as signed meters."""
    amount = abs(float(value))
    return f"{sign}{amount:.0f} m"


def _format_extra_label(key: str) -> str:
    """Format extra field key as title case."""
    return key.replace("_", " ").strip().title()


def _format_extra_value(value: Any) -> str:
    """Format extra field value."""
    if isinstance(value, dict):
        return "; ".join(
            f"{_format_extra_label(str(key))}: {_format_extra_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return ", ".join(_format_extra_value(item) for item in value)
    return str(value)


def _pace_text_to_seconds(pace_text: str) -> int:
    """Convert pace text to seconds per km, returning -1 for non-numeric placeholders."""
    raw = pace_text.strip().removesuffix("/km")
    if raw in {"", "-"}:
        return -1

    parts = raw.split(":")
    if len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return -1


def _build_comments_view_model(
    aid_station: dict[str, Any],
) -> tuple[CommentsLineViewModel, ...]:
    """Build comment line view models for an aid station."""
    comments: list[CommentsLineViewModel] = []

    notes = aid_station.get("notes")
    if notes:
        comments.append(CommentsLineViewModel(tag="Notes", value=str(notes)))

    stop_time_s = float(aid_station.get("stop_time_s", 0) or 0)
    if stop_time_s > 0:
        comments.append(CommentsLineViewModel(tag="Rest", value=seconds_to_hms(stop_time_s)))

    for key, value in aid_station.items():
        if key in _KNOWN_AID_STATION_FIELDS or value in (None, "", []):
            continue
        comments.append(
            CommentsLineViewModel(
                tag=_format_extra_label(key),
                value=_format_extra_value(value),
            )
        )

    return tuple(comments)


def _build_table_row_view_models(
    pacing_df: pd.DataFrame,
    aid_stations: list[dict[str, Any]],
    race_start_time_s: int,
) -> tuple[TableRowViewModel, ...]:
    """Build row view models for the aid station table."""
    if len(pacing_df) != len(aid_stations):
        raise ValueError(
            "Pacing rows and aid stations must have the same length to build the race table report."
        )

    rows: list[TableRowViewModel] = []
    accum_loss_m = 0.0
    for row, aid_station in zip(pacing_df.to_dict(orient="records"), aid_stations, strict=True):
        split_loss_m = abs(float(row["Segment Elevation Loss (m)"]))
        accum_loss_m += split_loss_m
        station_name = str(aid_station.get("name", row.get("Point Name", "Unknown")))
        rows.append(
            TableRowViewModel(
                station_name=station_name,
                station_japanese=str(aid_station.get("jap_name", "") or ""),
                station_link=str(aid_station.get("gmaps_link", "") or ""),
                distance_total=f'{float(row["Total Distance (km)"]):.1f} km',
                distance_segment=f'{float(row["Segment Distance (km)"]):.1f} km',
                accum_gain=f'+{float(row["Accum. Elevation Gain (m)"]):.0f} m',
                accum_loss=f'-{accum_loss_m:.0f} m',
                split_gain=(
                    f'{_format_signed_meters(row["Segment Elevation Gain (m)"], "+")} '
                    f'({row["Segment Gain (%)"]}%)'
                ),
                split_loss=f'-{split_loss_m:.0f} m',
                elapsed_time=str(row["Elapsed Time"]),
                elapsed_seconds=int(hms_to_seconds(str(row["Elapsed Time"]))),
                running_time=str(row["Segment Running Time"]),
                running_seconds=int(hms_to_seconds(str(row["Segment Running Time"]))),
                clock_time=elapsed_hms_to_clock_time(str(row["Elapsed Time"]), race_start_time_s),
                cutoff_in_time=str(aid_station.get("cutoff_in_time", "") or ""),
                avg_pace=(
                    "-"
                    if str(row["Avg Pace (mm:ss/km)"]) == "-"
                    else f'{row["Avg Pace (mm:ss/km)"]}/km'
                ),
                avg_pace_seconds=_pace_text_to_seconds(str(row["Avg Pace (mm:ss/km)"])),
                avg_gap=(
                    "-"
                    if str(row["Avg Grade-Adjusted Pace (mm:ss/km)"]) == "-"
                    else f'{row["Avg Grade-Adjusted Pace (mm:ss/km)"]}/km'
                ),
                avg_gap_seconds=_pace_text_to_seconds(
                    str(row["Avg Grade-Adjusted Pace (mm:ss/km)"])
                ),
                comments=_build_comments_view_model(aid_station),
            )
        )

    return tuple(rows)


def _style_embedded_profile(fig: Any, course: Course) -> None:
    """Apply embedded profile styling for smartphone report."""
    total_distance = float(course.total_distance_km)
    elevation_series = course.df["ele_m"].dropna()

    if elevation_series.empty:
        y_min = 0.0
        y_max = 100.0
    else:
        min_elevation = float(elevation_series.min())
        max_elevation = float(elevation_series.max())
        y_min = max(0.0, math.floor((min_elevation - 40.0) / 50.0) * 50.0)
        y_max = math.ceil((max_elevation + 40.0) / 50.0) * 50.0

    grid_major = "rgba(232, 214, 150, 0.18)"
    grid_minor = "rgba(232, 214, 150, 0.08)"
    tick_color = "#d8c27a"

    fig.update_layout(
        title=None,
        height=430,
        margin=dict(l=16, r=8, t=18, b=30),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#e7dfc2"),
    )
    fig.update_xaxes(
        title_text="",
        range=[0, total_distance],
        showgrid=True,
        gridcolor=grid_major,
        gridwidth=0.35,
        tickfont=dict(color=tick_color, size=11),
        minor=dict(
            showgrid=True,
            gridcolor=grid_minor,
            gridwidth=0.2,
            griddash="dot",
            ticklen=3,
        ),
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="",
        range=[y_min, y_max],
        showgrid=True,
        gridcolor=grid_major,
        gridwidth=0.35,
        tickfont=dict(color=tick_color, size=11),
        minor=dict(
            showgrid=True,
            gridcolor=grid_minor,
            gridwidth=0.2,
            griddash="dot",
            ticklen=3,
        ),
        zeroline=False,
    )

    for trace in fig.data:
        if getattr(trace, "mode", "") == "markers":
            trace.marker.size = 11
            trace.marker.line.width = 1.3

    for annotation in fig.layout.annotations:
        if (
            getattr(annotation, "xref", None) == "paper"
            and getattr(annotation, "yref", None) == "paper"
        ):
            annotation.update(
                text="",
                bgcolor="rgba(0, 0, 0, 0)",
                bordercolor="rgba(0, 0, 0, 0)",
                borderwidth=0,
            )
            continue

        annotation.update(
            yshift=12,
            borderpad=2,
            font=dict(size=9, color="#7b5318"),
        )


def _build_report_view_model(
    course: Course,
    pacing_df: pd.DataFrame,
    aid_stations: list[dict[str, Any]],
    race_name: str,
    mode: str,
    race_start_time: str | None,
    title: str,
) -> RacePlanReportViewModel:
    """Build the report view model from course, pacing, and config data."""
    normalized_aid_stations = _normalized_aid_stations(pacing_df, aid_stations)
    race_start_time_s = clock_time_to_seconds(race_start_time)

    plotter = CourseProfilePlotter(course, normalized_aid_stations)
    fig = plotter.create_elevation_profile(title="")
    _style_embedded_profile(fig, course)

    summary_items = (
        SummaryItemViewModel(
            label=_summary_time_label(mode),
            value=seconds_to_hms(float(pacing_df.attrs.get("total_time_s", 0))),
        ),
        SummaryItemViewModel(
            label="Distance",
            value=f'{float(pacing_df["Total Distance (km)"].iloc[-1]):.1f} km',
        ),
        SummaryItemViewModel(
            label="Elevation gain",
            value=f'{float(pacing_df["Accum. Elevation Gain (m)"].iloc[-1]):.0f} m',
        ),
        SummaryItemViewModel(
            label="Start time",
            value=race_start_time or "00:00:00",
        ),
    )

    return RacePlanReportViewModel(
        title=title,
        race_name=race_name,
        summary_items=summary_items,
        profile_meta=(
            f'{float(course.total_distance_km):.1f} km'
            f' · +{float(course.total_elevation_gain_m):.0f} m'
            f' · -{float(course.total_elevation_loss_m):.0f} m'
        ),
        total_time_seconds=int(float(pacing_df.attrs.get("total_time_s", 0))),
        race_start_time_s=race_start_time_s,
        original_fatigue_decay_pct=float(pacing_df.attrs.get("fatigue_total_decay_pct", 0.0)),
        table_rows=_build_table_row_view_models(
            pacing_df=pacing_df,
            aid_stations=normalized_aid_stations,
            race_start_time_s=race_start_time_s,
        ),
        plot_html=fig.to_html(include_plotlyjs=True, full_html=False),
    )


def _render_summary_cards(summary_items: tuple[SummaryItemViewModel, ...]) -> str:
    """Render summary cards HTML."""
    cards: list[str] = []
    for index, item in enumerate(summary_items):
        value_id = ' id="summary-target-time"' if index == 0 else ""
        cards.append(
            f'<div class="summary-card"><span class="summary-label">{escape(item.label)}</span>'
            f'<span class="summary-value"{value_id}>{escape(item.value)}</span></div>'
        )
    return "".join(cards)


def _render_comments_html(comments: tuple[CommentsLineViewModel, ...]) -> str:
    """Render comment lines HTML."""
    if not comments:
        return '<div class="comments-empty">No special notes</div>'

    lines = "".join(
        '<div class="comments-line">'
        f'<span class="comments-tag">{escape(line.tag)}</span>'
        f'<span class="comments-value">{escape(line.value)}</span>'
        "</div>"
        for line in comments
    )
    return f'<div class="comments-text">{lines}</div>'


def _render_labeled_cell_line(label: str, value: str, emphasize: bool = False) -> str:
    """Render a single labeled line inside a table cell."""
    value_class = "cell-value emphasized" if emphasize else "cell-value"
    return (
        f'<div class="cell-line"><span class="cell-label">{escape(label)}</span>'
        f'<span class="{value_class}">{escape(value)}</span></div>'
    )


def _render_value_line(value: str, emphasize: bool = False) -> str:
    """Render a single value-only line inside a table cell."""
    value_class = "cell-value emphasized" if emphasize else "cell-value"
    return f'<div class="value-line"><span class="{value_class}">{escape(value)}</span></div>'


def _render_table_rows(table_rows: tuple[TableRowViewModel, ...]) -> str:
    """Render table row HTML."""
    rows_html: list[str] = []
    for row in table_rows:
        station_lines = [
            '<div class="station-name-wrap">'
            f'<div class="station-name">{escape(row.station_name)}</div>'
            "</div>"
        ]
        if row.station_japanese:
            station_lines.append(
                f'<div class="station-japanese">{escape(row.station_japanese)}</div>'
            )
        if row.station_link:
            station_lines.append(
                '<div class="station-link"><a href="'
                + escape(row.station_link, quote=True)
                + '" target="_blank" rel="noopener noreferrer">Map</a></div>'
            )

        timing_html = "".join(
            (
                f'<div class="cell-line"><span class="cell-label">⏱</span>'
                f'<span class="cell-value emphasized js-elapsed">{escape(row.elapsed_time)}</span></div>',
                f'<div class="cell-line"><span class="cell-label">🏃</span>'
                f'<span class="cell-value js-running">{escape(row.running_time)}</span></div>',
                f'<div class="cell-line"><span class="cell-label">🕒</span>'
                f'<span class="cell-value js-clock">{escape(row.clock_time)}</span>'
                f'{("<span class=\"cutoff-inline\"> (🚧 " + escape(row.cutoff_in_time) + ")</span>") if row.cutoff_in_time else ""}</div>',
            )
        )

        pace_html = "".join(
            (
                f'<div class="value-line"><span class="cell-value emphasized js-pace">{escape(row.avg_pace)}</span></div>',
                f'<div class="value-line"><span class="cell-value js-gap">{escape(row.avg_gap)}</span></div>',
            )
        )

        rows_html.append(
            "<tr>"
            f'<th scope="row" class="sticky-col station-cell">{"".join(station_lines)}</th>'
            f'<td class="metric-cell metric-distance">{_render_labeled_cell_line("Σ", row.distance_total, emphasize=True)}'
            f'{_render_labeled_cell_line("Δ", row.distance_segment)}</td>'
            f'<td class="metric-cell metric-gain">{_render_value_line(row.accum_gain, emphasize=True)}'
            f'{_render_value_line(row.split_gain)}</td>'
            f'<td class="metric-cell metric-loss">{_render_value_line(row.accum_loss, emphasize=True)}'
            f'{_render_value_line(row.split_loss)}</td>'
            f'<td class="timing-cell" data-elapsed-s="{row.elapsed_seconds}" data-running-s="{row.running_seconds}">{timing_html}</td>'
            f'<td class="metric-cell metric-pace" data-pace-s="{row.avg_pace_seconds}" data-gap-s="{row.avg_gap_seconds}">{pace_html}</td>'
            f'<td class="comments-cell">{_render_comments_html(row.comments)}</td>'
            "</tr>"
        )

    return "".join(rows_html)


def _render_report_html(view_model: RacePlanReportViewModel) -> str:
    """Render the complete report HTML from the view model."""
    summary_html = _render_summary_cards(view_model.summary_items)
    table_rows_html = _render_table_rows(view_model.table_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>{escape(view_model.title)} - {escape(view_model.race_name)}</title>
  <style>
{_REPORT_CSS}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">Smartphone Report</div>
      <h1>{escape(view_model.title)}</h1>
      <div class="summary-grid">{summary_html}</div>
      <div class="time-tune">
        <label class="time-tune-label" for="target-time-input">Target</label>
        <input id="target-time-input" class="time-tune-input" type="text" value="{escape(seconds_to_hms(view_model.total_time_seconds))}" placeholder="HH:MM:SS" />
        <label class="time-tune-label" for="fatigue-decay-input">Slowdown % (linear model)</label>
        <input id="fatigue-decay-input" class="time-tune-input small" type="number" min="0" max="100" step="0.1" value="{view_model.original_fatigue_decay_pct:.1f}" />
        <button id="target-time-apply" class="time-tune-button" type="button">Apply</button>
        <button id="target-time-reset" class="time-tune-button" type="button">Reset</button>
        <span id="target-time-status" class="time-tune-status"></span>
      </div>
    </section>

    <div class="section-header">
      <h2 class="section-title">Aid Station Table</h2>
    </div>
    <section class="table-shell">
      <table>
        <thead>
          <tr>
            <th>AS</th>
            <th>Dist</th>
            <th>Gain</th>
            <th>Loss</th>
            <th>Time</th>
            <th>Pace</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {table_rows_html}
        </tbody>
      </table>
    </section>

    <div class="section-header">
      <h2 class="section-title">Elevation Profile</h2>
      <span class="section-meta">{escape(view_model.profile_meta)}</span>
    </div>
    <section class="profile-shell">
      {view_model.plot_html}
    </section>
  </main>
  <script>
    (() => {{
      const originalTargetSeconds = {view_model.total_time_seconds};
      const startTimeSeconds = {view_model.race_start_time_s};
      const originalFatigueDecayPct = {view_model.original_fatigue_decay_pct};
      const targetInput = document.getElementById("target-time-input");
      const fatigueInput = document.getElementById("fatigue-decay-input");
      const applyButton = document.getElementById("target-time-apply");
      const resetButton = document.getElementById("target-time-reset");
      const status = document.getElementById("target-time-status");
      const summaryTarget = document.getElementById("summary-target-time");
      const timingCells = Array.from(document.querySelectorAll("td.timing-cell"));
      const paceCells = Array.from(document.querySelectorAll("td.metric-pace"));

      function toHms(totalSeconds) {{
        const safe = Math.max(0, Math.round(totalSeconds));
        const h = Math.floor(safe / 3600);
        const m = Math.floor((safe % 3600) / 60);
        const s = safe % 60;
        return `${{h}}:${{String(m).padStart(2, "0")}}:${{String(s).padStart(2, "0")}}`;
      }}

      function toClockWithDay(totalSeconds) {{
        const safe = Math.max(0, Math.floor(totalSeconds));
        const day = Math.floor(safe / 86400) + 1;
        const minutes = Math.floor((safe % 86400) / 60);
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${{String(hours).padStart(2, "0")}}:${{String(mins).padStart(2, "0")}} · D${{day}}`;
      }}

      function parseHms(value) {{
        const match = /^\\s*(\\d+):(\\d{{1,2}}):(\\d{{1,2}})\\s*$/.exec(value || "");
        if (!match) return null;
        const h = Number(match[1]);
        const m = Number(match[2]);
        const s = Number(match[3]);
        if (m > 59 || s > 59) return null;
        return h * 3600 + m * 60 + s;
      }}

      function toPace(totalSeconds) {{
        const safe = Math.max(0, Math.round(totalSeconds));
        const minutes = Math.floor(safe / 60);
        const seconds = safe % 60;
        return `${{minutes}}:${{String(seconds).padStart(2, "0")}}/km`;
      }}

      function fatigueRatio(progress, newDecayPct) {{
        const originalDecay = originalFatigueDecayPct / 100;
        const newDecay = newDecayPct / 100;
        const base = 1 + originalDecay * progress;
        if (base <= 0) return 1;
        return (1 + newDecay * progress) / base;
      }}

      function applyScale(targetSeconds, newDecayPct) {{
        const scale = targetSeconds / originalTargetSeconds;
        let cumulativeBaseRunning = 0;
        let cumulativeAdjustedRunning = 0;

        timingCells.forEach((cell) => {{
          const elapsedBase = Number(cell.dataset.elapsedS || "0");
          const runningBase = Number(cell.dataset.runningS || "0");
          cumulativeBaseRunning += runningBase;
          const progress = originalTargetSeconds > 0 ? elapsedBase / originalTargetSeconds : 0;
          const runningScaled = runningBase * scale * fatigueRatio(progress, newDecayPct);
          cumulativeAdjustedRunning += runningScaled;

          const baseStopCumulative = elapsedBase - cumulativeBaseRunning;
          const adjustedElapsed = cumulativeAdjustedRunning + baseStopCumulative * scale;
          const clockScaled = startTimeSeconds + adjustedElapsed;

          const elapsedNode = cell.querySelector(".js-elapsed");
          const runningNode = cell.querySelector(".js-running");
          const clockNode = cell.querySelector(".js-clock");
          if (elapsedNode) elapsedNode.textContent = toHms(adjustedElapsed);
          if (runningNode) runningNode.textContent = toHms(runningScaled);
          if (clockNode) clockNode.textContent = toClockWithDay(clockScaled);
        }});

        paceCells.forEach((cell, index) => {{
          const paceBase = Number(cell.dataset.paceS || "-1");
          const gapBase = Number(cell.dataset.gapS || "-1");
          const timingCell = timingCells[index];
          const elapsedBase = Number(timingCell?.dataset.elapsedS || "0");
          const progress = originalTargetSeconds > 0 ? elapsedBase / originalTargetSeconds : 0;
          const paceScale = scale * fatigueRatio(progress, newDecayPct);
          const paceNode = cell.querySelector(".js-pace");
          const gapNode = cell.querySelector(".js-gap");
          if (paceNode) paceNode.textContent = paceBase < 0 ? "-" : toPace(paceBase * paceScale);
          if (gapNode) gapNode.textContent = gapBase < 0 ? "-" : toPace(gapBase * paceScale);
        }});

        if (summaryTarget) summaryTarget.textContent = toHms(targetSeconds);
        status.textContent = `Scaled x${{scale.toFixed(3)}} · Slowdown ${{newDecayPct.toFixed(1)}}%`;
      }}

      applyButton?.addEventListener("click", () => {{
        const parsed = parseHms(targetInput?.value ?? "");
        const parsedDecay = Number(fatigueInput?.value ?? `${{originalFatigueDecayPct}}`);
        if (!parsed || parsed <= 0) {{
          status.textContent = "Use HH:MM:SS";
          return;
        }}
        if (!Number.isFinite(parsedDecay) || parsedDecay < 0 || parsedDecay > 100) {{
          status.textContent = "Slowdown must be 0-100";
          return;
        }}
        applyScale(parsed, parsedDecay);
      }});

      resetButton?.addEventListener("click", () => {{
        if (targetInput) targetInput.value = toHms(originalTargetSeconds);
        if (fatigueInput) fatigueInput.value = originalFatigueDecayPct.toFixed(1);
        applyScale(originalTargetSeconds, originalFatigueDecayPct);
        status.textContent = "Reset";
      }});
    }})();
  </script>
</body>
</html>
"""


def generate_race_plan_table_report(
    course: Course,
    aid_stations: list[dict[str, Any]],
    pacing_df: pd.DataFrame,
    output_path: Path | str,
    race_name: str,
    mode: str,
    race_start_time: str | None = None,
    title: str = "Race Plan Table",
) -> Path:
    """Generate a smartphone-oriented HTML report with pacing table and profile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    view_model = _build_report_view_model(
        course=course,
        pacing_df=pacing_df,
        aid_stations=aid_stations,
        race_name=race_name,
        mode=mode,
        race_start_time=race_start_time,
        title=title,
    )
    html = _render_report_html(view_model)

    output_path.write_text(html, encoding="utf-8")
    logger.success(f"Race plan table saved to: {output_path}")
    return output_path
