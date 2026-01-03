import pathlib
import pytest


@pytest.fixture
def sample_gpx_path():
    return (
        pathlib.Path(__file__).parent.parent
        / "gpx_repo"
        / "Okumusashi_Long_Trail_Race_105K_2025.gpx"
    )
