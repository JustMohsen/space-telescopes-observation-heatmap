"""HEALPix spherical binning and scaling (exposure duration, pointing count, target count)."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from telescope_heatmap.config import DEFAULT_NSIDE


@dataclass(frozen=True)
class BinnedSkyMap:
    """Holds both raw physical pixel values, scaled display values, and top target labels on a HEALPix sphere."""

    nside: int
    frame: str  # "galactic", "ecliptic", or "equatorial"
    metric: str  # "exposure_hours", "pointing_count", or "target_count"
    scale: str  # "log10" or "linear"
    raw_map: np.ndarray  # Physical values per HEALPix pixel (hours or counts)
    display_map: np.ndarray  # Scaled values for color mapping
    target_labels: np.ndarray  # Top target names per HEALPix pixel (dtype=object/str)
    unit_label: str
    colorbar_title: str


def filter_observations(
    df: pd.DataFrame,
    telescopes: list[str] | None = None,
    categories: list[str] | None = None,
    mjd_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Filter an ETL table (`exposures`, `pointings`, or `targets`) by telescope, category, and MJD date range."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if telescopes is not None:
        mask &= df["obs_collection"].isin(telescopes)
    if categories is not None:
        mask &= df["target_category"].isin(categories)
    if mjd_range is not None and "t_min" in df.columns:
        mjd_start, mjd_end = mjd_range
        if "t_max" in df.columns:
            mask &= (df["t_max"].fillna(df["t_min"]) >= mjd_start) & (df["t_min"] <= mjd_end)
        else:
            mask &= (df["t_min"] >= mjd_start) & (df["t_min"] <= mjd_end)
    return df[mask]


def compute_healpix_map(
    df: pd.DataFrame,
    frame: str = "galactic",
    metric: str = "exposure_hours",
    scale: str = "log10",
    nside: int = DEFAULT_NSIDE,
) -> BinnedSkyMap:
    """Bin filtered observations into equal-area HEALPix pixels on the celestial sphere."""
    npix = hp.nside2npix(nside)
    raw_map = np.zeros(npix, dtype=np.float64)
    target_labels = np.full(npix, "None", dtype=object)

    metric_labels = {
        "exposure_hours": "Exposure Time (hrs)",
        "pointing_count": "Telescope Pointings",
        "target_count": "Unique Targets",
    }
    base_label = metric_labels.get(metric, metric)

    if not df.empty:
        if frame == "galactic":
            lon_deg = df["gal_l"].to_numpy(dtype=np.float64)
            lat_deg = df["gal_b"].to_numpy(dtype=np.float64)
        elif frame == "equatorial":
            lon_deg = df["s_ra"].to_numpy(dtype=np.float64)
            lat_deg = df["s_dec"].to_numpy(dtype=np.float64)
        elif frame == "ecliptic":
            if "ecl_lon" in df.columns and "ecl_lat" in df.columns:
                lon_deg = df["ecl_lon"].to_numpy(dtype=np.float64)
                lat_deg = df["ecl_lat"].to_numpy(dtype=np.float64)
            else:
                ra = df["s_ra"].to_numpy(dtype=np.float64)
                dec = df["s_dec"].to_numpy(dtype=np.float64)
                # Astropy creates several intermediate coordinate arrays. Transform
                # bounded chunks rather than all million exposures simultaneously.
                lon_deg = np.empty(len(df), dtype=np.float64)
                lat_deg = np.empty(len(df), dtype=np.float64)
                for start in range(0, len(df), 32_768):
                    end = start + 32_768
                    ecl = SkyCoord(
                        ra=ra[start:end] * u.deg, dec=dec[start:end] * u.deg, frame="icrs"
                    ).barycentrictrueecliptic
                    lon_deg[start:end] = ((ecl.lon.deg + 180.0) % 360.0) - 180.0
                    lat_deg[start:end] = ecl.lat.deg
        else:
            raise ValueError(f"Unsupported coordinate frame: {frame}")

        # HEALPix uses colatitude theta in [0, pi] (North Pole = 0) and longitude phi in [0, 2*pi)
        theta = np.radians(90.0 - np.clip(lat_deg, -90.0, 90.0))
        phi = np.radians(lon_deg % 360.0)
        pix_idx = hp.ang2pix(nside, theta, phi)

        if metric == "exposure_hours":
            weights = df["t_exptime"].to_numpy(dtype=np.float64) / 3600.0
        elif metric in ("pointing_count", "target_count"):
            weights = np.ones(len(df), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        raw_map = np.bincount(pix_idx, weights=weights, minlength=npix).astype(np.float64)

        # Build top target names per populated HEALPix pixel for hover tooltips
        if "target_name" in df.columns:
            tmp = pd.DataFrame(
                {
                    "pix": pix_idx,
                    "target_name": df["target_name"].astype("string[pyarrow]").fillna("Unknown").reset_index(drop=True),
                    "w": weights,
                }
            )
            agg_tgt = (
                tmp.groupby(["pix", "target_name"], as_index=False, observed=True)
                .agg({"w": "sum"})
                .sort_values(["pix", "w"], ascending=[True, False])
            )
            top_rows = agg_tgt.groupby("pix", sort=False).head(3)
            top3 = top_rows.groupby("pix", sort=False)["target_name"].agg(", ".join)
            target_labels[top3.index.to_numpy()] = top3.to_numpy()

    # Apply display scaling
    pos_mask = raw_map > 0.0
    if scale == "log10":
        display_map = np.full(npix, np.nan, dtype=np.float64)
        if np.any(pos_mask):
            display_map[pos_mask] = np.log10(raw_map[pos_mask])
        colorbar_title = f"log10({base_label})"
    elif scale == "linear":
        display_map = np.where(pos_mask, raw_map, np.nan)
        colorbar_title = base_label
    else:
        raise ValueError(f"Unsupported scale: {scale}")

    return BinnedSkyMap(
        nside=nside,
        frame=frame,
        metric=metric,
        scale=scale,
        raw_map=raw_map,
        display_map=display_map,
        target_labels=target_labels,
        unit_label=base_label,
        colorbar_title=colorbar_title,
    )


def sample_healpix_on_grid(
    sky_map: BinnedSkyMap,
    n_lon: int | None = None,
    n_lat: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample a BinnedSkyMap onto a structured (lon, lat) grid and 3D unit sphere (x, y, z).

    Caps interactive previews at 65,160 vertices, independently of the scientific
    HEALPix resolution. Explicit grid dimensions (e.g. for PNG export) override the cap.
    Populated cells are forward-splatted onto their enclosing 2x2 grid quad so
    isolated pointings remain visible between vertices. Brighter cells take
    precedence when several scientific pixels share a preview quad.
    """
    if n_lon is None:
        n_lon = min(max(180, sky_map.nside * 4), 360)
    if n_lat is None:
        n_lat = n_lon // 2 + 1

    lon_1d = np.linspace(-180.0, 180.0, n_lon, dtype=np.float32)
    lat_1d = np.linspace(-90.0, 90.0, n_lat, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # Use float64 angles at the poles to stay within healpy's [0, pi] domain.
    theta_grid = np.radians(90.0 - lat_grid.astype(np.float64))
    phi_grid = np.radians(lon_grid % 360.0)
    pix_grid = hp.ang2pix(sky_map.nside, theta_grid, phi_grid)

    raw_grid = sky_map.raw_map[pix_grid]
    display_grid = sky_map.display_map[pix_grid]
    target_grid = sky_map.target_labels[pix_grid]

    # Forward-splat populated HEALPix cells onto their enclosing 2x2 mesh quad so every populated
    # sky cell forms at least one complete, un-diluted quad face on the 3D surface and 2D heatmap.
    pop_pix = np.where(np.isfinite(sky_map.display_map))[0]
    if pop_pix.size > 0:
        pop_pix = pop_pix[np.argsort(sky_map.display_map[pop_pix])]
        theta_p, phi_p = hp.pix2ang(sky_map.nside, pop_pix)
        lat_p = 90.0 - np.degrees(theta_p)
        lon_p = ((np.degrees(phi_p) + 180.0) % 360.0) - 180.0
        i0 = np.clip(np.searchsorted(lat_1d, lat_p) - 1, 0, n_lat - 2)
        j0 = np.clip(np.searchsorted(lon_1d, lon_p) - 1, 0, n_lon - 2)
        new_disp = sky_map.display_map[pop_pix]
        new_raw = sky_map.raw_map[pop_pix]
        new_lbl = sky_map.target_labels[pop_pix]
        for di in (0, 1):
            for dj in (0, 1):
                ii = i0 + di
                jj = j0 + dj
                cur = display_grid[ii, jj]
                overwrite = np.isnan(cur) | (new_disp >= cur)
                display_grid[ii[overwrite], jj[overwrite]] = new_disp[overwrite]
                raw_grid[ii[overwrite], jj[overwrite]] = new_raw[overwrite]
                target_grid[ii[overwrite], jj[overwrite]] = new_lbl[overwrite]

    # 3D unit sphere Cartesian coordinates with (lon=0, lat=0) pointing along +X
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return {
        "lon_1d": lon_1d,
        "lat_1d": lat_1d,
        "lon_grid": lon_grid,
        "lat_grid": lat_grid,
        "x": x,
        "y": y,
        "z": z,
        "raw_grid": raw_grid,
        "display_grid": display_grid,
        "target_grid": target_grid,
    }
