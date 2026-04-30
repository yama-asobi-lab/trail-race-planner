"""
Main entry point for trail race planner.

Usage:
    python -m race_planner.main <race_config.yaml> [options]

Options:
    --athlete NAME             Athlete name (default: yet_another_sato)
    --mode MODE                Planning mode (default: athlete_pb):
                                 athlete_pb   — derive pace from athlete reference PB
                                 target_time  — plan to a given total finish time
                                 target_itra  — plan to a given target ITRA score
    --target-time HH:MM:SS     Required for --mode target_time
    --target-itra-score N      Required for --mode target_itra

Notes:
    - For target_time, the provided time is the desired TOTAL finish time
      (running + all aid-station stops).
    - For target_itra, the race YAML must contain at least one entry in
      race.itra_reference_points.
    - The pacing plan is written as a new sheet in the existing
      segment-analysis Excel file defined by race.output_file.
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from race_planner.course import analyze_race
from race_planner.models.itra_predictor import ItraScorePredictor
from race_planner.models.tools import hours_to_hms, seconds_to_hms, time_to_seconds
from race_planner.planner import PaceCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_stop_time_s(aid_stations: list) -> float:
    return sum(float(aid.get('stop_time_s', 0)) for aid in aid_stations)


def _build_itra_predictor(race_config: dict) -> ItraScorePredictor | None:
    """Return an ItraScorePredictor from the first itra_reference_points entry, or None."""
    ref_points = race_config.get('race', {}).get('itra_reference_points', [])
    if not ref_points:
        logger.warning(
            "No 'itra_reference_points' in race config — " "ITRA score prediction unavailable."
        )
        return None
    ref = ref_points[0]
    try:
        return ItraScorePredictor(
            reference_time=ref['reference_time'],
            reference_score=int(ref['reference_score']),
        )
    except Exception as exc:
        logger.warning(f"Could not build ITRA predictor from race config: {exc}")
        return None


def _append_pacing_sheet(
    output_path: Path,
    pacing_df,
    sheet_name: str,
    mode: str,
    athlete_name: str,
    itra_score: int | None,
) -> None:
    """Add (or replace) a sheet with the pacing plan in an existing xlsx file."""
    wb = load_workbook(output_path)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    cols = list(pacing_df.columns)

    # Header
    for col_idx, col_name in enumerate(cols, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    # Data
    for row_idx, row in enumerate(pacing_df.itertuples(index=False), start=2):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    # Column widths
    col_widths = {
        'Point Name': 30,
        'Total Distance (km)': 18,
        'Elevation (m)': 14,
        'Accum. Elevation Gain (m)': 24,
        'Segment Distance (km)': 20,
        'Segment Elevation Gain (m)': 24,
        'Segment Elevation Loss (m)': 24,
        'Segment Running Time': 20,
        'Stop Time': 12,
        'Elapsed Time': 14,
    }
    for col_idx, col_name in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 16)

    # Summary block below the data table
    attrs = pacing_df.attrs
    summary_row = len(pacing_df) + 3
    summary = [
        ('Planning mode', mode),
        ('Athlete', athlete_name),
        ('Total running time', seconds_to_hms(attrs.get('total_running_time_s', 0))),
        ('Total stop time', seconds_to_hms(attrs.get('total_stop_time_s', 0))),
        ('Total finish time', seconds_to_hms(attrs.get('total_time_s', 0))),
    ]
    if itra_score is not None:
        summary.append(('Predicted ITRA score', itra_score))

    for offset, (label, value) in enumerate(summary):
        ws.cell(row=summary_row + offset, column=1, value=label)
        ws.cell(row=summary_row + offset, column=2, value=value)

    wb.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Trail race planner — segment analysis and pacing plan'
    )
    parser.add_argument('race_config', type=Path, help='Path to race YAML config')
    parser.add_argument(
        '--athlete',
        default='yet_another_sato',
        help='Athlete name (default: yet_another_sato)',
    )
    parser.add_argument(
        '--mode',
        choices=['athlete_pb', 'target_time', 'target_itra'],
        default='athlete_pb',
        help='Planning mode (default: athlete_pb)',
    )
    parser.add_argument(
        '--target-time',
        metavar='HH:MM:SS',
        help='Desired total finish time — required for --mode target_time',
    )
    parser.add_argument(
        '--target-itra-score',
        type=int,
        metavar='N',
        help='Target ITRA score — required for --mode target_itra',
    )
    cli_args = parser.parse_args()

    if cli_args.mode == 'target_time' and not cli_args.target_time:
        parser.error('--target-time HH:MM:SS is required when --mode target_time')
    if cli_args.mode == 'target_itra' and not cli_args.target_itra_score:
        parser.error('--target-itra-score N is required when --mode target_itra')

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    race_config_path = cli_args.race_config
    if not race_config_path.exists():
        logger.error(f"Race configuration file not found: {race_config_path}")
        sys.exit(1)

    project_root = Path.cwd()
    athlete_config_path = project_root / 'config' / 'athletes' / f'{cli_args.athlete}.yaml'
    if not athlete_config_path.exists():
        logger.error(f"Athlete configuration file not found: {athlete_config_path}")
        sys.exit(1)

    logger.info(f"Race config:    {race_config_path}")
    logger.info(f"Athlete config: {athlete_config_path}")
    logger.info(f"Planning mode:  {cli_args.mode}")

    with open(race_config_path, 'r', encoding='utf-8') as f:
        race_config = yaml.safe_load(f)
    with open(athlete_config_path, 'r', encoding='utf-8') as f:
        athlete_config = yaml.safe_load(f)

    race_info = race_config.get('race', {})
    gpx_file = race_info.get('gpx_file')
    output_file = race_info.get('output_file')
    resample_m = race_info.get('resample_m', 5)

    if not gpx_file:
        logger.error("'gpx_file' not specified in race config")
        sys.exit(1)
    if not output_file:
        logger.error("'output_file' not specified in race config")
        sys.exit(1)

    gpx_path = project_root / gpx_file
    output_path = project_root / output_file

    athlete_info = athlete_config.get('athlete', {})
    athlete_display_name = athlete_info.get('name', cli_args.athlete)
    logger.info(f"Athlete: {athlete_display_name}")

    # ------------------------------------------------------------------
    # 1. Segment analysis — always runs; creates / updates the xlsx file
    # ------------------------------------------------------------------
    try:
        analyzer = analyze_race(
            gpx_path=gpx_path,
            race_config_path=race_config_path,
            output_path=output_path,
            resample_m=resample_m,
            athlete_config=athlete_config,
        )
        logger.success(f"Segment analysis written to: {output_path}")
    except Exception as exc:
        logger.error(f"Segment analysis failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    course = analyzer.course
    aid_stations = analyzer.aid_stations

    # ------------------------------------------------------------------
    # 2. ITRA predictor — built here so it's available for both
    #    target_itra mode and score output at the end
    # ------------------------------------------------------------------
    itra_predictor = _build_itra_predictor(race_config)

    # ------------------------------------------------------------------
    # 3. Resolve target running time and build PaceCalculator
    # ------------------------------------------------------------------
    override_running_time_s: float | None = None

    if cli_args.mode == 'athlete_pb':
        calc = PaceCalculator.from_athlete_config(athlete_config)
        ref = athlete_info.get('reference_performance', {})
        logger.info(f"Reference performance: {ref.get('distance_km')} km in {ref.get('time')}")

    elif cli_args.mode == 'target_time':
        target_total_time_s = float(time_to_seconds(cli_args.target_time))
        total_stop_s = _total_stop_time_s(aid_stations)
        override_running_time_s = target_total_time_s - total_stop_s
        if override_running_time_s <= 0:
            logger.error(
                f"Target time {cli_args.target_time} is shorter than total stop time "
                f"({seconds_to_hms(total_stop_s)})"
            )
            sys.exit(1)
        calc = PaceCalculator.from_athlete_config(athlete_config)
        logger.info(
            f"Target finish time: {cli_args.target_time}  "
            f"(running: {seconds_to_hms(override_running_time_s)}, "
            f"stops: {seconds_to_hms(total_stop_s)})"
        )

    elif cli_args.mode == 'target_itra':
        if itra_predictor is None:
            logger.error(
                "Cannot use --mode target_itra: " "no 'itra_reference_points' in race config."
            )
            sys.exit(1)
        predicted_total_time_h = itra_predictor.predict_time(cli_args.target_itra_score)
        target_total_time_s = predicted_total_time_h * 3600
        total_stop_s = _total_stop_time_s(aid_stations)
        override_running_time_s = target_total_time_s - total_stop_s
        if override_running_time_s <= 0:
            logger.error(
                f"ITRA-predicted time {hours_to_hms(predicted_total_time_h)} is shorter "
                f"than total stop time ({seconds_to_hms(total_stop_s)})"
            )
            sys.exit(1)
        calc = PaceCalculator.from_athlete_config(athlete_config)
        logger.info(
            f"ITRA score {cli_args.target_itra_score} → "
            f"predicted finish time: {hours_to_hms(predicted_total_time_h)}  "
            f"(running: {seconds_to_hms(override_running_time_s)}, "
            f"stops: {seconds_to_hms(total_stop_s)})"
        )

    # ------------------------------------------------------------------
    # 4. Pacing plan
    # ------------------------------------------------------------------
    try:
        pacing_df = calc.calculate_pacing(
            course=course,
            aid_stations=aid_stations,
            use_fed=(cli_args.mode == 'athlete_pb'),
            override_total_running_time_s=override_running_time_s,
        )
    except Exception as exc:
        logger.error(f"Pacing calculation failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    total_time_s = pacing_df.attrs['total_time_s']

    # ------------------------------------------------------------------
    # 5. ITRA score for the computed finish time
    # ------------------------------------------------------------------
    itra_score_result: int | None = None
    if itra_predictor is not None:
        itra_score_result = itra_predictor.predict_score(total_time_s / 3600)
        logger.info(
            f"Expected ITRA score for finish time "
            f"{seconds_to_hms(total_time_s)}: {itra_score_result}"
        )
    else:
        logger.warning("ITRA score output skipped (no reference data in race config).")

    # ------------------------------------------------------------------
    # 6. Append pacing sheet to xlsx
    # ------------------------------------------------------------------
    sheet_name = f"Race Plan ({cli_args.mode})"
    _append_pacing_sheet(
        output_path=output_path,
        pacing_df=pacing_df,
        sheet_name=sheet_name,
        mode=cli_args.mode,
        athlete_name=athlete_display_name,
        itra_score=itra_score_result,
    )
    logger.success(f"Pacing plan written to sheet '{sheet_name}' in {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "=" * 52
    logger.info(sep)
    logger.info("RACE PLAN SUMMARY")
    logger.info(f"  Mode:          {cli_args.mode}")
    logger.info(f"  Athlete:       {athlete_display_name}")
    approx_running_s = pacing_df.attrs.get('riegel_running_time_approx_s')
    if approx_running_s is not None:
        logger.info(f"  Riegel approx running time: {seconds_to_hms(float(approx_running_s))}")
        logger.info(
            f"  Grade-adjusted running time: "
            f"{seconds_to_hms(pacing_df.attrs['total_running_time_s'])}"
        )
    else:
        logger.info(
            f"  Running time:  " f"{seconds_to_hms(pacing_df.attrs['total_running_time_s'])}"
        )
    logger.info(f"  Stop time:     " f"{seconds_to_hms(pacing_df.attrs['total_stop_time_s'])}")
    logger.info(f"  Finish time:   {seconds_to_hms(total_time_s)}")
    if itra_score_result is not None:
        logger.info(f"  ITRA score:    {itra_score_result}")
    logger.info(sep)


if __name__ == '__main__':
    main()
