"""
Segment analyzer for race planning.

Analyzes race segments between aid stations, validates elevations,
and exports statistics to Excel.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from loguru import logger
import pandas as pd
import yaml

from race_planner.course.course import Course
from race_planner.visualization import plot_course_profile


class SegmentAnalyzer:
    """
    Analyzes race course segments between aid stations.

    Args:
        course: Course object with loaded GPX data.
        race_config_path: Path to race YAML configuration file.
        elevation_tolerance_m: Tolerance for elevation validation (default: 50m).
        athlete_config: Athlete configuration dictionary (defaults to Yet Another Sato,
            our reference baseline athlete).

    Attributes:
        course: The Course object.
        race_config: Loaded race configuration dictionary.
        aid_stations: List of aid station configurations.
        athlete_config: Athlete configuration for pacing calculations.
    """

    def __init__(
        self,
        course: Course,
        race_config_path: Path | str,
        elevation_tolerance_m: float = 50.0,
        athlete_config: Dict = None,
    ):
        """Initialize SegmentAnalyzer with course and configuration."""
        self.course = course
        self.elevation_tolerance_m = elevation_tolerance_m

        # Load race configuration
        race_config_path = Path(race_config_path)
        with open(race_config_path, 'r', encoding='utf-8') as f:
            self.race_config = yaml.safe_load(f)

        self.aid_stations = self.race_config.get('aid_stations', [])

        # Load default athlete config if not provided
        if athlete_config is None:
            default_athlete_path = Path('config/athletes/yet_another_sato.yaml')
            logger.info(
                f"No athlete config provided, using default: {default_athlete_path}"
            )
            with open(default_athlete_path, 'r', encoding='utf-8') as f:
                self.athlete_config = yaml.safe_load(f)
        else:
            self.athlete_config = athlete_config

        # Log athlete info
        athlete_info = self.athlete_config.get('athlete', {})
        athlete_name = athlete_info.get('name', 'Unknown')
        ref_perf = athlete_info.get('reference_performance', {})
        ref_time = ref_perf.get('time', 'N/A')
        ref_dist = ref_perf.get('distance_km', 'N/A')
        logger.info(
            f"Using athlete profile: {athlete_name} "
            f"(Reference: {ref_dist} km in {ref_time})"
        )

        # Run validations on initialization
        logger.info("Validating course...")
        self.validate_course_distance()
        self.validate_elevations()

    def validate_course_distance(self) -> None:
        """
        Validate that the last aid station distance matches the total course distance.

        Raises a warning if the difference exceeds 100m.
        """
        if not self.aid_stations:
            return

        last_aid = self.aid_stations[-1]
        last_distance_km = last_aid.get('distance_km', 0)
        last_distance_m = last_distance_km * 1000

        actual_distance_m = self.course.total_distance_m
        actual_distance_km = self.course.total_distance_km

        diff_m = abs(actual_distance_m - last_distance_m)
        tolerance_m = 100.0

        if diff_m > tolerance_m:
            warning_msg = (
                f"Course distance mismatch: "
                f"Last aid station at {last_distance_km:.1f} km, "
                f"actual course length {actual_distance_km:.1f} km "
                f"(diff={diff_m:.0f}m, tolerance=±{tolerance_m:.0f}m)"
            )
            warnings.warn(warning_msg, UserWarning)
        else:
            logger.success(f"Course distance validated: {actual_distance_km:.1f} km")

    def validate_elevations(self) -> None:
        """
        Validate that aid station elevations match course data.

        Logs warnings for any mismatches.
        """
        validation_count = 0
        mismatch_count = 0

        for aid in self.aid_stations:
            name = aid.get('name', 'Unknown')
            config_distance_km = aid.get('distance_km', 0)
            config_elevation_m = aid.get('elevation_m')

            if config_elevation_m is None:
                continue

            validation_count += 1

            # Get actual elevation from course
            distance_m = config_distance_km * 1000
            actual_elevation_m = self.course.get_elevation_at_distance(distance_m)

            # Check if within tolerance
            diff = abs(actual_elevation_m - config_elevation_m)
            is_valid = diff <= self.elevation_tolerance_m

            if not is_valid:
                mismatch_count += 1
                warning_msg = (
                    f"Elevation mismatch at {name}: "
                    f"config={config_elevation_m:.0f}m, "
                    f"actual={actual_elevation_m:.0f}m "
                    f"(diff={actual_elevation_m - config_elevation_m:.0f}m, "
                    f"tolerance=±{self.elevation_tolerance_m:.0f}m)"
                )
                warnings.warn(warning_msg, UserWarning)

        if mismatch_count == 0:
            logger.success(f"All {validation_count} aid station elevations validated")
        else:
            logger.warning(
                f"{mismatch_count}/{validation_count} aid stations have elevation mismatches"
            )

    def calculate_segment_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for each segment between aid stations.

        Returns:
            DataFrame with segment statistics.
        """
        segments = []

        for i, aid in enumerate(self.aid_stations):
            name = aid.get('name', 'Unknown')
            jap_name = aid.get('jap_name', '')

            # Format name with Japanese
            if jap_name:
                full_name = f"{name} ({jap_name})"
            else:
                full_name = name

            distance_km = aid.get('distance_km', 0)
            distance_m = distance_km * 1000

            # Get point data from course
            point = self.course.get_point_at_distance(distance_m)
            elevation_m = float(point['ele_m'])
            cum_ele_gain_m = float(point['cum_ele_gain_m'])

            # Calculate segment stats
            if i == 0:
                # First point (START)
                segment_distance_km = 0.0
                segment_ele_gain_m = 0.0
                segment_ele_loss_m = 0.0
                avg_pos_gradient_pct = 0.0
            else:
                # Calculate from previous aid station
                prev_distance_km = self.aid_stations[i - 1].get('distance_km', 0)
                prev_distance_m = prev_distance_km * 1000

                # Get segment DataFrame
                segment_df = self.course.get_segment(
                    start_m=prev_distance_m, end_m=distance_m
                )

                segment_distance_km = distance_km - prev_distance_km
                segment_ele_gain_m = segment_df['ele_gain_m'].sum()
                segment_ele_loss_m = segment_df['ele_loss_m'].sum()

                # Average gradient
                if segment_distance_km > 0:
                    # Net elevation change divided by horizontal distance
                    avg_pos_gradient_pct = (
                        segment_ele_gain_m / (segment_distance_km * 1000)
                    ) * 100
                else:
                    avg_pos_gradient_pct = 0.0

            segments.append(
                {
                    'Point Name': full_name,
                    'Total Distance (km)': distance_km,
                    'Elevation (m)': elevation_m,
                    'Accum. Elevation Gain (m)': cum_ele_gain_m,
                    'Segment Distance (km)': segment_distance_km,
                    'Segment Elevation Gain (m)': segment_ele_gain_m,
                    'Segment Elevation Loss (m)': segment_ele_loss_m,
                    'Average Gradient (%)': avg_pos_gradient_pct,
                }
            )

        return pd.DataFrame(segments)

    def generate_report(self, output_path: Path | str) -> None:
        """
        Generate segment analysis report and save to Excel.

        Args:
            output_path: Path to output Excel file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate segment stats
        logger.info("Calculating segment statistics...")
        segment_stats = self.calculate_segment_stats()

        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write segment stats
            segment_stats.to_excel(writer, sheet_name='Segment Statistics', index=False)

            # Format the segment stats sheet
            worksheet = writer.sheets['Segment Statistics']

            # Set column widths
            worksheet.column_dimensions['A'].width = 25  # Point Name
            worksheet.column_dimensions['B'].width = 20  # Distance from Start
            worksheet.column_dimensions['C'].width = 15  # Elevation
            worksheet.column_dimensions['D'].width = 20  # Accumulated Elevation Gain
            worksheet.column_dimensions['E'].width = 20  # Segment Distance
            worksheet.column_dimensions['F'].width = 20  # Segment Elevation Gain
            worksheet.column_dimensions['G'].width = 20  # Segment Elevation Loss
            worksheet.column_dimensions['H'].width = 20  # Average Gradient

        logger.success(f"Report saved to: {output_path}")

        # Print summary
        total_distance = segment_stats['Total Distance (km)'].iloc[-1]
        total_gain = segment_stats['Accum. Elevation Gain (m)'].iloc[-1]
        logger.info(f"Race Summary:")
        logger.info(f"  Total Distance: {total_distance:.1f} km")
        logger.info(f"  Total Elevation Gain: {total_gain:.0f} m")
        logger.info(f"  Number of Aid Stations: {len(self.aid_stations)}")

        # Generate elevation profile plot
        plot_path = output_path.parent / f"{output_path.stem}_elevation_profile.html"
        race_name = self.race_config.get('race', {}).get('name', 'Race Course')
        logger.info("Generating elevation profile plot...")
        plot_course_profile(
            course=self.course,
            aid_stations=self.aid_stations,
            output_path=plot_path,
            title=f"{race_name} - Elevation Profile",
        )


def analyze_race(
    gpx_path: Path | str,
    race_config_path: Path | str,
    output_path: Path | str,
    resample_m: Optional[float] = 5.0,
    athlete_config: Optional[Dict] = None,
) -> SegmentAnalyzer:
    """
    Convenience function to analyze a race course.

    Args:
        gpx_path: Path to GPX file.
        race_config_path: Path to race YAML configuration file.
        output_path: Path to output Excel file.
        resample_m: Resampling interval for course data (default: 5 sm).
        athlete_config: Optional athlete configuration dict (defaults to Yet Another Sato,
            the reference baseline athlete, if None).

    Returns:
        SegmentAnalyzer instance with analysis results.
    """
    # Load course
    logger.info(f"Loading course from: {gpx_path}")
    course = Course(gpx_path, resample_m=resample_m)

    # Create analyzer (will load default athlete config if None provided)
    analyzer = SegmentAnalyzer(course, race_config_path, athlete_config=athlete_config)

    # Generate report
    analyzer.generate_report(output_path)

    return analyzer
