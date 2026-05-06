import pathlib
import sys
import pytest
import yaml


TESTS_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
ATHLETES_FIXTURES_DIR = FIXTURES_DIR / "athletes"
RACES_FIXTURES_DIR = FIXTURES_DIR / "races"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_gpx_path():
    """Return path to TGT GPX file used in actual race config."""
    return pathlib.Path(__file__).parent.parent / "config" / "gpx_repo" / "TGT_2025.gpx"


@pytest.fixture
def carlos_config_path():
    return ATHLETES_FIXTURES_DIR / "carlos.yaml"


@pytest.fixture
def yas_config_path():
    return ATHLETES_FIXTURES_DIR / "yet_another_sato.yaml"


@pytest.fixture
def race_config_path():
    return RACES_FIXTURES_DIR / "tgt_2026.yaml"


@pytest.fixture
def carlos_config(carlos_config_path):
    with open(carlos_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def yas_config(yas_config_path):
    with open(yas_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def race_config(race_config_path):
    with open(race_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_course(sample_gpx_path):
    """Return a Course object loaded from sample GPX file."""
    from race_planner.course.course import Course

    return Course(sample_gpx_path, resample_m=5)


@pytest.fixture
def sample_pacing_df(sample_course, yas_config):
    """Return a sample pacing DataFrame for testing."""
    from race_planner.planner.pace_calculator import PaceCalculator

    calc = PaceCalculator.from_athlete_config(yas_config)

    # Use simple test aid stations at start and finish
    aid_stations = [
        {
            "name": "Start",
            "jap_name": "スタート",
            "distance_km": 0.0,
            "elevation_m": sample_course.df["ele_m"].iloc[0],
            "stop_time_s": 0,
            "notes": "Race start",
        },
        {
            "name": "Finish",
            "jap_name": "ゴール",
            "distance_km": sample_course.total_distance_km,
            "elevation_m": sample_course.df["ele_m"].iloc[-1],
            "stop_time_s": 300,
            "notes": "Race finish",
        },
    ]

    pacing_df = calc.calculate_pacing(
        course=sample_course,
        aid_stations=aid_stations,
        use_fed=True,
    )

    return pacing_df
