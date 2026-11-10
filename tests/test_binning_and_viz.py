"""Unit tests for HEALPix spherical binning and Plotly/Matplotlib figure builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from telescope_heatmap.binning import compute_healpix_map, filter_observations, sample_healpix_on_grid
from telescope_heatmap.viz import build_2d_allsky_figure, build_3d_globe_figure, save_static_mollweide_png


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_id": ["1", "2", "3"],
            "obs_collection": ["HST", "JWST", "JWST"],
            "target_category": ["deep_space", "exoplanet", "solar_system"],
            "s_ra": [266.4168, 10.68, 42.32],
            "s_dec": [-29.0078, 41.26, 15.91],
            "gal_l": [0.0, -138.8, -160.2],
            "gal_b": [0.0, -21.5, -37.4],
            "t_exptime": [3600.0, 7200.0, 1800.0],
        }
    )


def test_filter_and_compute_healpix_map() -> None:
    df = _sample_df()
    filt = filter_observations(df, telescopes=["JWST"], categories=["exoplanet"])
    assert len(filt) == 1

    sky_log = compute_healpix_map(df, frame="galactic", metric="exposure_hours", scale="log10", nside=16)
    assert np.isclose(sky_log.raw_map.sum(), 3.5)  # (3600 + 7200 + 1800) / 3600 = 3.5 hrs
    assert np.isfinite(sky_log.display_map).sum() >= 1

    sky_cnt = compute_healpix_map(df, frame="equatorial", metric="pointing_count", scale="linear", nside=16)
    assert np.isclose(sky_cnt.raw_map.sum(), 3.0)


def test_3d_and_2d_figures_and_static_export(tmp_path: Path) -> None:
    df = _sample_df()
    sky_map = compute_healpix_map(df, frame="galactic", metric="exposure_hours", scale="log10", nside=16)
    mesh = sample_healpix_on_grid(sky_map, n_lon=72, n_lat=36)
    assert mesh["x"].shape == (36, 72)

    fig3d = build_3d_globe_figure(sky_map)
    assert len(fig3d.data) >= 3  # Surface + Galactic Plane + Galactic Center + Poles

    fig2d = build_2d_allsky_figure(sky_map)
    assert len(fig2d.data) >= 2

    out_png = tmp_path / "test_mollweide.png"
    save_static_mollweide_png(sky_map, out_png)
    assert out_png.exists() and out_png.stat().st_size > 1000
