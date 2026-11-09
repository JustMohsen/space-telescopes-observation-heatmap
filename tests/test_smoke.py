"""Basic smoke tests for telescope_heatmap package."""

from telescope_heatmap.config import SUPPORTED_TELESCOPES


def test_supported_telescopes() -> None:
    assert "HST" in SUPPORTED_TELESCOPES
    assert "JWST" in SUPPORTED_TELESCOPES
