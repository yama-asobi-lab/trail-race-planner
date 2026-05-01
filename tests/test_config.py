"""
Tests for configuration file structure validation.
"""

from pathlib import Path
import yaml


def test_athlete_config_structure():
    """Test that all athlete configs have consistent structure."""
    athlete_config_dir = Path("config/athletes")
    athlete_configs = list(athlete_config_dir.glob("*.yaml"))

    assert len(athlete_configs) > 0, "No athlete config files found"

    required_fields = ["name", "itra_points", "reference_performance", "gap_curve"]

    for config_path in athlete_configs:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check structure
        assert "athlete" in config, f"'athlete' key missing in {config_path}"
        athlete_info = config["athlete"]

        for field in required_fields:
            assert field in athlete_info, f"{field} missing in {config_path}"

        # Check reference_performance structure
        ref = athlete_info["reference_performance"]
        assert "distance_km" in ref, f"reference_performance.distance_km missing in {config_path}"
        assert "time" in ref, f"reference_performance.time missing in {config_path}"

        # Check gap_curve structure
        assert "points" in athlete_info["gap_curve"], f"gap_curve.points missing in {config_path}"


def test_race_config_structure():
    """Test that all race configs have correct structure."""
    race_config_dir = Path("config/races")
    race_configs = list(race_config_dir.glob("*.yaml"))

    assert len(race_configs) > 0, "No race config files found"

    for config_path in race_configs:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check top-level structure
        assert "race" in config, f"'race' key missing in {config_path}"
        assert "aid_stations" in config, f"'aid_stations' key missing in {config_path}"

        # Check race info
        race_info = config["race"]
        assert "name" in race_info, f"'name' missing in race info for {config_path}"
        assert "gpx_file" in race_info, f"'gpx_file' missing in race info for {config_path}"
        assert "output_file" in race_info, f"'output_file' missing in race info for {config_path}"

        # Check aid stations (empty list is valid — treated as start-to-finish)
        aid_stations = config["aid_stations"]
        assert isinstance(aid_stations, list), f"'aid_stations' must be a list in {config_path}"

        for station in aid_stations:
            assert "name" in station, f"Aid station missing 'name' in {config_path}"
            assert "distance_km" in station, f"Aid station missing 'distance_km' in {config_path}"
