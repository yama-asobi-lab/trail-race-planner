"""
Main entry point for trail race planner.

Usage:
    python -m race_planner.main <race_config.yaml> [--athlete <athlete_name>]

Examples:
    python -m race_planner.main config/races/tgt_2026.yaml
    python -m race_planner.main config/races/tgt_2026.yaml --athlete carlos
"""

import sys
from pathlib import Path
import yaml
from loguru import logger

from race_planner.course import analyze_race


def main():
    """Run race analysis from configuration file."""
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python -m race_planner.main <race_config.yaml> [--athlete <athlete_name>]"
        )
        logger.info(
            "Example: python -m race_planner.main races_config/tgt_2026.yaml --athlete carlos"
        )
        sys.exit(1)

    # Parse command line arguments
    race_config_path = Path(sys.argv[1])

    # Parse athlete argument (defaults to "yet_another_sato")
    athlete_name = "yet_another_sato"
    if len(sys.argv) >= 4 and sys.argv[2] == "--athlete":
        athlete_name = sys.argv[3]

    # Validate race config
    if not race_config_path.exists():
        logger.error(f"Race configuration file not found: {race_config_path}")
        sys.exit(1)

    # Load athlete config
    project_root = Path.cwd()
    athlete_config_path = project_root / "config" / "athletes" / f"{athlete_name}.yaml"

    if not athlete_config_path.exists():
        logger.error(f"Athlete configuration file not found: {athlete_config_path}")
        logger.info(f"Available athletes: yet_another_sato, carlos")
        sys.exit(1)

    logger.info(f"Loading race configuration from: {race_config_path}")
    logger.info(f"Loading athlete configuration from: {athlete_config_path}")

    # Load configurations
    with open(race_config_path, 'r', encoding='utf-8') as f:
        race_config = yaml.safe_load(f)

    with open(athlete_config_path, 'r', encoding='utf-8') as f:
        athlete_config = yaml.safe_load(f)

    # Get paths from race config
    race_info = race_config.get('race', {})
    gpx_file = race_info.get('gpx_file')
    output_file = race_info.get('output_file')
    resample_m = race_info.get('resample_m', 10)

    if not gpx_file:
        logger.error("'gpx_file' not specified in race configuration")
        sys.exit(1)

    if not output_file:
        logger.error("'output_file' not specified in race configuration")
        sys.exit(1)

    # Resolve paths relative to project root
    gpx_path = project_root / gpx_file
    output_path = project_root / output_file

    # Log athlete info
    athlete_info = athlete_config.get('athlete', {})
    athlete_display_name = athlete_info.get('name', athlete_name)
    logger.info(f"Athlete: {athlete_display_name}")

    # Run analysis
    try:
        analyzer = analyze_race(
            gpx_path=gpx_path,
            race_config_path=race_config_path,
            output_path=output_path,
            resample_m=resample_m,
            athlete_config=athlete_config,
        )
        logger.success("Analysis complete!")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
