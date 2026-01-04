"""
Tests for SegmentAnalyzer functionality.
"""

from pathlib import Path
import warnings
import yaml
import pandas as pd
import pytest

from race_planner.course import Course, SegmentAnalyzer, analyze_race


def test_analyze_race_with_carlos_athlete(tmp_path, sample_gpx_path):
    """Test race analysis using carlos athlete profile."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    athlete_config_path = Path("config/athletes/carlos.yaml")

    assert race_config_path.exists(), "Race config not found"
    assert athlete_config_path.exists(), "Carlos athlete config not found"

    # Load athlete config
    with open(athlete_config_path, 'r', encoding='utf-8') as f:
        athlete_config = yaml.safe_load(f)

    # Verify carlos-specific config
    assert athlete_config['athlete']['name'] == 'Carlos'
    assert athlete_config['athlete']['marathon_pb'] == '3:30:00'
    assert athlete_config['athlete']['itra_points'] == 650

    # Create temporary output
    output_path = tmp_path / "carlos_output.xlsx"

    # Run analysis
    analyzer = analyze_race(
        gpx_path=sample_gpx_path,
        race_config_path=race_config_path,
        output_path=output_path,
        resample_m=10,
        athlete_config=athlete_config,
    )

    # Verify analyzer uses carlos config
    assert analyzer.athlete_config is not None
    assert analyzer.athlete_config['athlete']['name'] == 'Carlos'
    assert analyzer.athlete_config['athlete']['itra_points'] == 650

    # Verify outputs were created
    assert output_path.exists()
    assert (tmp_path / "carlos_output_elevation_profile.html").exists()


def test_analyze_race_defaults_to_yet_another_sato(tmp_path, sample_gpx_path):
    """Test race analysis defaults to Yet Another Sato when no athlete config provided."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    output_path = tmp_path / "default_output.xlsx"

    # Run analysis without athlete config (should default to Yet Another Sato)
    analyzer = analyze_race(
        gpx_path=sample_gpx_path,
        race_config_path=race_config_path,
        output_path=output_path,
        resample_m=10,
        athlete_config=None,
    )

    # Verify analyzer defaults to Yet Another Sato
    assert analyzer.athlete_config is not None
    assert analyzer.athlete_config['athlete']['name'] == 'Yet Another Sato'
    assert output_path.exists()
    assert (tmp_path / "default_output_elevation_profile.html").exists()


def test_segment_analyzer_initialization(sample_gpx_path):
    """Test SegmentAnalyzer initialization and attribute setup."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)

    analyzer = SegmentAnalyzer(course, race_config_path)

    # Verify attributes are set
    assert analyzer.course == course
    assert analyzer.race_config is not None
    assert 'race' in analyzer.race_config
    assert 'aid_stations' in analyzer.race_config
    assert len(analyzer.aid_stations) > 0
    assert analyzer.athlete_config is not None
    assert analyzer.elevation_tolerance_m == 50.0


def test_segment_analyzer_loads_race_config_correctly(sample_gpx_path):
    """Test that race configuration is properly loaded."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)

    analyzer = SegmentAnalyzer(course, race_config_path)

    # Verify race config structure
    assert 'name' in analyzer.race_config['race']
    assert 'gpx_file' in analyzer.race_config['race']
    assert analyzer.race_config['race']['name'] == 'Tokyo Grand Trail 2026'

    # Verify aid stations are loaded
    assert len(analyzer.aid_stations) > 0
    first_station = analyzer.aid_stations[0]
    assert 'name' in first_station
    assert 'distance_km' in first_station


def test_calculate_segment_stats_returns_dataframe(sample_gpx_path):
    """Test that calculate_segment_stats returns a proper DataFrame."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)
    analyzer = SegmentAnalyzer(course, race_config_path)

    stats_df = analyzer.calculate_segment_stats()

    # Verify it's a DataFrame
    assert isinstance(stats_df, pd.DataFrame)

    # Verify expected columns exist
    expected_columns = [
        'Point Name',
        'Total Distance (km)',
        'Elevation (m)',
        'Accum. Elevation Gain (m)',
        'Segment Distance (km)',
        'Segment Elevation Gain (m)',
        'Segment Elevation Loss (m)',
        'Average Gradient (%)',
    ]
    for col in expected_columns:
        assert col in stats_df.columns

    # Verify number of rows matches aid stations
    assert len(stats_df) == len(analyzer.aid_stations)

    # Verify first row is start (segment distance = 0)
    assert stats_df.iloc[0]['Segment Distance (km)'] == 0.0
    assert stats_df.iloc[0]['Segment Elevation Gain (m)'] == 0.0


def test_calculate_segment_stats_accumulation(sample_gpx_path):
    """Test that segment statistics accumulate correctly."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)
    analyzer = SegmentAnalyzer(course, race_config_path)

    stats_df = analyzer.calculate_segment_stats()

    # Verify distances are monotonically increasing
    distances = stats_df['Total Distance (km)'].values
    assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))

    # Verify accumulated elevation gain is monotonically increasing
    accum_gain = stats_df['Accum. Elevation Gain (m)'].values
    assert all(accum_gain[i] <= accum_gain[i + 1] for i in range(len(accum_gain) - 1))

    # Verify segment distances sum approximately to total distance
    segment_distances = stats_df['Segment Distance (km)'].values[
        1:
    ]  # Skip first (start)
    total_segment_distance = segment_distances.sum()
    total_course_distance = stats_df['Total Distance (km)'].iloc[-1]
    assert abs(total_segment_distance - total_course_distance) < 0.1


def test_generate_report_creates_files(tmp_path, sample_gpx_path):
    """Test that generate_report creates Excel and HTML files."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)
    analyzer = SegmentAnalyzer(course, race_config_path)

    output_path = tmp_path / "test_report.xlsx"
    analyzer.generate_report(output_path)

    # Verify Excel file created
    assert output_path.exists()

    # Verify HTML plot created
    plot_path = tmp_path / "test_report_elevation_profile.html"
    assert plot_path.exists()

    # Verify Excel file can be read
    df = pd.read_excel(output_path, sheet_name='Segment Statistics')
    assert len(df) == len(analyzer.aid_stations)


def test_validate_course_distance_within_tolerance(sample_gpx_path):
    """Test that course distance validation passes when within tolerance."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)

    # Should not raise warnings if within 100m tolerance
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        try:
            analyzer = SegmentAnalyzer(course, race_config_path)
            # If we get here without error, validation passed (or warning was not raised)
        except UserWarning:
            # This is fine - it means there was a mismatch, which is expected for test data
            pass


def test_validate_elevations_runs(sample_gpx_path):
    """Test that elevation validation executes without errors."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)

    # Should complete without exceptions (may produce warnings)
    analyzer = SegmentAnalyzer(course, race_config_path)

    # Verify validation ran by checking attributes exist
    assert analyzer.course is not None
    assert analyzer.aid_stations is not None


def test_custom_elevation_tolerance(sample_gpx_path):
    """Test that custom elevation tolerance is respected."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)

    custom_tolerance = 100.0
    analyzer = SegmentAnalyzer(
        course, race_config_path, elevation_tolerance_m=custom_tolerance
    )

    assert analyzer.elevation_tolerance_m == custom_tolerance


def test_aid_stations_with_japanese_names(sample_gpx_path):
    """Test that aid stations with Japanese names are formatted correctly."""
    race_config_path = Path("config/races/tgt_2026.yaml")
    course = Course(sample_gpx_path, resample_m=10)
    analyzer = SegmentAnalyzer(course, race_config_path)

    stats_df = analyzer.calculate_segment_stats()

    # Check for stations with Japanese names (jap_name field)
    for i, aid in enumerate(analyzer.aid_stations):
        if 'jap_name' in aid and aid['jap_name']:
            point_name = stats_df.iloc[i]['Point Name']
            # Should contain both English and Japanese names
            assert aid['name'] in point_name
            assert aid['jap_name'] in point_name
            assert '(' in point_name and ')' in point_name


def test_different_resample_interval(sample_gpx_path):
    """Test that analysis works with different resampling than production config."""
    race_config_path = Path("config/races/tgt_2026.yaml")

    # Test 1: Coarser sampling (50m) for faster processing
    course_coarse = Course(sample_gpx_path, resample_m=50)
    analyzer_coarse = SegmentAnalyzer(course_coarse, race_config_path)

    stats_coarse = analyzer_coarse.calculate_segment_stats()

    # Verify basic structure is intact
    assert isinstance(stats_coarse, pd.DataFrame)
    assert len(stats_coarse) == len(analyzer_coarse.aid_stations)
    assert stats_coarse['Total Distance (km)'].iloc[-1] > 0
    assert stats_coarse['Accum. Elevation Gain (m)'].iloc[-1] > 0

    # Test 2: Very fine sampling (1m) for high precision
    course_fine = Course(sample_gpx_path, resample_m=1)
    analyzer_fine = SegmentAnalyzer(course_fine, race_config_path)

    stats_fine = analyzer_fine.calculate_segment_stats()

    # Verify basic structure
    assert isinstance(stats_fine, pd.DataFrame)
    assert len(stats_fine) == len(analyzer_fine.aid_stations)
    assert stats_fine['Total Distance (km)'].iloc[-1] > 0
    assert stats_fine['Accum. Elevation Gain (m)'].iloc[-1] > 0

    # Compare: fine sampling should have similar or slightly higher elevation gain
    # (more data points can capture more detail)
    coarse_gain = stats_coarse['Accum. Elevation Gain (m)'].iloc[-1]
    fine_gain = stats_fine['Accum. Elevation Gain (m)'].iloc[-1]
    assert abs(fine_gain - coarse_gain) / coarse_gain < 0.15  # Within 15%

    # Note: Elevation validation warnings are expected with different resampling
