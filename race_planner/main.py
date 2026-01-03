"""
Main entry point for trail race planner.

Usage:
    python -m race_planner.main races_config/tgt_2026.yaml
"""

import sys
from pathlib import Path
import yaml
from loguru import logger

from race_planner.course import analyze_race


def main():
    """Run race analysis from configuration file."""
    if len(sys.argv) < 2:
        logger.error("Usage: python -m race_planner.main <config.yaml>")
        logger.info("Example: python -m race_planner.main races_config/tgt_2026.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")

    # Load config to get paths
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get paths from config
    race_config = config.get('race', {})
    gpx_file = race_config.get('gpx_file')
    output_file = race_config.get('output_file')
    resample_m = race_config.get('resample_m', 10)

    if not gpx_file:
        logger.error("'gpx_file' not specified in race configuration")
        sys.exit(1)

    if not output_file:
        logger.error("'output_file' not specified in race configuration")
        sys.exit(1)

    # Resolve paths relative to project root
    project_root = Path.cwd()
    gpx_path = project_root / gpx_file
    output_path = project_root / output_file

    # Run analysis
    try:
        analyzer = analyze_race(
            gpx_path=gpx_path,
            config_path=config_path,
            output_path=output_path,
            resample_m=resample_m,
        )
        logger.success("Analysis complete!")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
