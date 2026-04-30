import pathlib
import pytest
import yaml


TESTS_DIR = pathlib.Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
ATHLETES_FIXTURES_DIR = FIXTURES_DIR / "athletes"
RACES_FIXTURES_DIR = FIXTURES_DIR / "races"


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
