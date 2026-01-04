import pathlib
import pytest


@pytest.fixture
def sample_gpx_path():
    """Return path to TGT GPX file used in actual race config."""
    return pathlib.Path(__file__).parent.parent / "config" / "gpx_repo" / "TGT_2025.gpx"
