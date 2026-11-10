"""Interactive 3D Plotly globe and 2D all-sky projection builders with Light & Dark mode support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord

from telescope_heatmap.binning import BinnedSkyMap, sample_healpix_on_grid


def _skycoord_to_frame_lonlat(coords: SkyCoord, target_frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert any SkyCoord array into wrapped (lon in [-180, 180], lat in [-90, 90]) for target_frame."""
    if target_frame == "galactic":
        c = coords.galactic
        lon = ((c.l.deg + 180.0) % 360.0) - 180.0
        lat = c.b.deg
    elif target_frame == "ecliptic":
        c = coords.barycentrictrueecliptic
        lon = ((c.lon.deg + 180.0) % 360.0) - 180.0
        lat = c.lat.deg
    elif target_frame == "equatorial":
        c = coords.icrs
        lon = ((c.ra.deg + 180.0) % 360.0) - 180.0
        lat = c.dec.deg
    else:
        raise ValueError(f"Unsupported frame: {target_frame}")
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _galactic_plane_and_center_coords(frame: str, n_points: int = 361) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (plane_lon, plane_lat, center_lon, center_lat) in degrees for the Galactic Plane in `frame`."""
    l_deg = np.linspace(0.0, 360.0, n_points)
    plane_sc = SkyCoord(l=l_deg * u.deg, b=np.zeros_like(l_deg) * u.deg, frame="galactic")
    plane_lon, plane_lat = _skycoord_to_frame_lonlat(plane_sc, frame)

    center_sc = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")
    c_lon, c_lat = _skycoord_to_frame_lonlat(center_sc, frame)
    return plane_lon, plane_lat, float(c_lon), float(c_lat)


def _ecliptic_plane_and_origin_coords(frame: str, n_points: int = 361) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (plane_lon, plane_lat, origin_lon, origin_lat) for the Solar System Ecliptic Plane in `frame`."""
    lam_deg = np.linspace(0.0, 360.0, n_points)
    plane_sc = SkyCoord(lon=lam_deg * u.deg, lat=np.zeros_like(lam_deg) * u.deg, frame="barycentrictrueecliptic")
    plane_lon, plane_lat = _skycoord_to_frame_lonlat(plane_sc, frame)

    origin_sc = SkyCoord(lon=0.0 * u.deg, lat=0.0 * u.deg, frame="barycentrictrueecliptic")
    o_lon, o_lat = _skycoord_to_frame_lonlat(origin_sc, frame)
    return plane_lon, plane_lat, float(o_lon), float(o_lat)


def _frame_axis_labels(frame: str) -> tuple[str, str, str, str]:
    """Return (short_lon, short_lat, long_lon, long_lat) labels for the active reference frame."""
    if frame == "galactic":
        return "Galactic l", "Galactic b", "Galactic Longitude l (deg)", "Galactic Latitude b (deg)"
    if frame == "ecliptic":
        return "Ecliptic λ", "Ecliptic β", "Ecliptic Longitude λ (deg)", "Ecliptic Latitude β (deg)"
    return "RA (wrapped)", "Dec", "Right Ascension (wrapped deg)", "Declination (deg)"


def _theme_palette(theme: str) -> dict[str, Any]:
    """Return color palette settings tailored for 'dark' or 'light' mode."""
    if theme == "light":
        return {
            "scene_bg": "#f8fafc",
            "plot_2d_bg": "#f1f5f9",
            "title_color": "#0f172a",
            "text_color": "#1e293b",
            "pole_color": "#334155",
            "grid_color": "rgba(71, 85, 105, 0.22)",
            "zero_color": "rgba(2, 132, 199, 0.45)",
            "legend_bg": "rgba(255, 255, 255, 0.88)",
            "legend_border": "rgba(100, 116, 139, 0.35)",
            "gal_line_color": "#0284c7",
            "gal_center_color": "#0369a1",
            "ecl_line_color": "#d97706",
            "ecl_origin_color": "#b45309",
            "surface_colorscale": [
                [0.0, "#e2e8f0"],
                [0.08, "#cbd5e1"],
                [0.12, "#fde68a"],
                [0.32, "#f59e0b"],
                [0.55, "#ea580c"],
                [0.78, "#be123c"],
                [0.92, "#6d28d9"],
                [1.0, "#1e1b4b"],
            ],
            "heatmap_2d_colorscale": [
                [0.0, "#fef3c7"],
                [0.25, "#f59e0b"],
                [0.50, "#ea580c"],
                [0.75, "#be123c"],
                [1.0, "#311042"],
            ],
        }
    return {
        "scene_bg": "#050811",
        "plot_2d_bg": "#0b0f19",
        "title_color": "#f8fafc",
        "text_color": "#e2e8f0",
        "pole_color": "#cbd5e1",
        "grid_color": "rgba(148, 163, 184, 0.18)",
        "zero_color": "rgba(56, 189, 248, 0.4)",
        "legend_bg": "rgba(15, 23, 42, 0.78)",
        "legend_border": "rgba(148, 163, 184, 0.3)",
        "gal_line_color": "#38bdf8",
        "gal_center_color": "#00f5d4",
        "ecl_line_color": "#fbbf24",
        "ecl_origin_color": "#fbbf24",
        "surface_colorscale": [
            [0.0, "#0b0f19"],
            [0.08, "#111827"],
            [0.12, "#1e1b4b"],
            [0.30, "#4c1d95"],
            [0.50, "#9d174d"],
            [0.70, "#ea580c"],
            [0.88, "#facc15"],
            [1.0, "#fef08a"],
        ],
        "heatmap_2d_colorscale": "Inferno",
    }


def build_3d_globe_figure(
    sky_map: BinnedSkyMap,
    title: str = "3D Celestial Observation Heatmap",
    show_galactic_plane: bool = True,
    show_ecliptic_plane: bool = True,
    theme: str = "dark",
) -> go.Figure:
    """Build an interactive 3D rotatable celestial globe in Plotly with Light & Dark mode support."""
    pal = _theme_palette(theme)
    mesh = sample_healpix_on_grid(sky_map)
    display_grid = mesh["display_grid"]
    raw_grid = mesh["raw_grid"]

    finite_vals = display_grid[np.isfinite(display_grid)]
    if finite_vals.size > 0:
        vmin = float(np.percentile(finite_vals, 2))
        vmax = float(np.percentile(finite_vals, 99.5))
        if vmax <= vmin:
            vmax = float(np.max(finite_vals))
            vmin = float(np.min(finite_vals)) - 1.0
        bg_val = vmin - 0.18 * max(vmax - vmin, 1.0)
        surface_color = np.where(np.isfinite(display_grid), display_grid, bg_val)
    else:
        bg_val, vmin, vmax = 0.0, 0.0, 1.0
        surface_color = np.zeros_like(display_grid)

    lon_label, lat_label, _, _ = _frame_axis_labels(sky_map.frame)

    # Keep numeric arrays numeric so Plotly can use compact typed-array encoding.
    # Mixing target strings into customdata creates one Python object per number.
    customdata = np.dstack((mesh["lon_grid"], mesh["lat_grid"], raw_grid))

    fig = go.Figure()

    # 1. Main 3D spherical heatmap surface
    fig.add_trace(
        go.Surface(
            x=mesh["x"],
            y=mesh["y"],
            z=mesh["z"],
            surfacecolor=surface_color,
            cmin=bg_val,
            cmax=vmax,
            colorscale=pal["surface_colorscale"],
            customdata=customdata,
            text=mesh["target_grid"],
            colorbar=dict(
                title=dict(text=sky_map.colorbar_title, side="right", font=dict(color=pal["text_color"])),
                tickfont=dict(color=pal["text_color"]),
                thickness=18,
                len=0.75,
            ),
            hovertemplate=(
                f"<b>{lon_label}:</b> %{{customdata[0]:.1f}}°<br>"
                f"<b>{lat_label}:</b> %{{customdata[1]:.1f}}°<br>"
                f"<b>{sky_map.unit_label}:</b> %{{customdata[2]:.3f}}<br>"
                "<b>Top Targets:</b> %{text}<extra></extra>"
            ),
            lighting=dict(ambient=0.88, diffuse=0.22, specular=0.05, roughness=0.9),
            showscale=True,
        )
    )

    r_overlay = 1.012

    # 2. Galactic Plane (b=0°) ring & Galactic Center (l=0°, b=0°) spike
    if show_galactic_plane:
        plane_lon, plane_lat, gc_lon, gc_lat = _galactic_plane_and_center_coords(sky_map.frame)
        plat_rad = np.radians(plane_lat)
        plon_rad = np.radians(plane_lon)
        px = r_overlay * np.cos(plat_rad) * np.cos(plon_rad)
        py = r_overlay * np.cos(plat_rad) * np.sin(plon_rad)
        pz = r_overlay * np.sin(plat_rad)

        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="lines",
                line=dict(color=pal["gal_line_color"], width=4, dash="dot"),
                name="Milky Way Galactic Plane (b = 0°)",
                hoverinfo="name",
            )
        )

        gc_lat_rad = np.radians(gc_lat)
        gc_lon_rad = np.radians(gc_lon)
        gc_x = np.cos(gc_lat_rad) * np.cos(gc_lon_rad)
        gc_y = np.cos(gc_lat_rad) * np.sin(gc_lon_rad)
        gc_z = np.sin(gc_lat_rad)

        gc_col = pal["gal_center_color"]
        fig.add_trace(
            go.Scatter3d(
                x=[1.01 * gc_x, 1.28 * gc_x],
                y=[1.01 * gc_y, 1.28 * gc_y],
                z=[1.01 * gc_z, 1.28 * gc_z],
                mode="lines+markers+text",
                line=dict(color=gc_col, width=7),
                marker=dict(size=[7, 11], color=[gc_col, gc_col], symbol=["circle", "diamond"]),
                text=["", "Galactic Center (Sgr A*)"],
                textposition="top center",
                textfont=dict(color=gc_col, size=12),
                name="Galactic Center (l=0°, b=0°)",
                hovertemplate="<b>Galactic Center (Sgr A*)</b><br>l = 0.0°, b = 0.0°<extra></extra>",
            )
        )

    # 3. Solar System / Ecliptic Plane (β=0°) ring & Vernal Equinox (λ=0°, β=0°) spike
    if show_ecliptic_plane:
        e_lon, e_lat, eq_lon, eq_lat = _ecliptic_plane_and_origin_coords(sky_map.frame)
        elat_rad = np.radians(e_lat)
        elon_rad = np.radians(e_lon)
        ex = r_overlay * np.cos(elat_rad) * np.cos(elon_rad)
        ey = r_overlay * np.cos(elat_rad) * np.sin(elon_rad)
        ez = r_overlay * np.sin(elat_rad)

        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=pal["ecl_line_color"], width=4, dash="dash"),
                name="Solar System / Ecliptic Plane (β = 0°)",
                hoverinfo="name",
            )
        )

        eq_lat_rad = np.radians(eq_lat)
        eq_lon_rad = np.radians(eq_lon)
        eq_x = np.cos(eq_lat_rad) * np.cos(eq_lon_rad)
        eq_y = np.cos(eq_lat_rad) * np.sin(eq_lon_rad)
        eq_z = np.sin(eq_lat_rad)

        ec_col = pal["ecl_origin_color"]
        fig.add_trace(
            go.Scatter3d(
                x=[1.01 * eq_x, 1.28 * eq_x],
                y=[1.01 * eq_y, 1.28 * eq_y],
                z=[1.01 * eq_z, 1.28 * eq_z],
                mode="lines+markers+text",
                line=dict(color=ec_col, width=7),
                marker=dict(size=[7, 11], color=[ec_col, ec_col], symbol=["circle", "diamond"]),
                text=["", "Vernal Equinox (λ=0°, β=0°)"],
                textposition="top center",
                textfont=dict(color=ec_col, size=12),
                name="Solar System Origin / Vernal Equinox (0°, 0°)",
                hovertemplate="<b>Solar System / Ecliptic Origin (Vernal Equinox)</b><br>λ = 0.0°, β = 0.0° (RA=0°, Dec=0°)<extra></extra>",
            )
        )

    # 4. North & South Pole labels for the active reference frame
    pole_names = {
        "galactic": ("North Galactic Pole (+90°)", "South Galactic Pole (-90°)"),
        "ecliptic": ("North Ecliptic Pole (+90° / JWST CVZ)", "South Ecliptic Pole (-90° / JWST CVZ)"),
        "equatorial": ("North Celestial Pole (+90°)", "South Celestial Pole (-90°)"),
    }
    north_label, south_label = pole_names.get(sky_map.frame, ("North Pole (+90°)", "South Pole (-90°)"))
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[1.15, -1.15],
            mode="markers+text",
            marker=dict(size=5, color=pal["pole_color"]),
            text=[north_label, south_label],
            textfont=dict(color=pal["pole_color"], size=11),
            name="Reference Frame Poles (±90°)",
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=18, color=pal["title_color"])),
        paper_bgcolor=pal["scene_bg"],
        plot_bgcolor=pal["scene_bg"],
        font=dict(color=pal["text_color"]),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor=pal["legend_bg"],
            font=dict(color=pal["text_color"]),
            bordercolor=pal["legend_border"],
            borderwidth=1,
        ),
        scene=dict(
            bgcolor=pal["scene_bg"],
            xaxis=dict(visible=False, range=[-1.35, 1.35]),
            yaxis=dict(visible=False, range=[-1.35, 1.35]),
            zaxis=dict(visible=False, range=[-1.35, 1.35]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.65, y=0.65, z=0.55)),
        ),
    )
    return fig


def build_2d_allsky_figure(
    sky_map: BinnedSkyMap,
    title: str = "2D All-Sky Observation Heatmap",
    show_galactic_plane: bool = True,
    show_ecliptic_plane: bool = True,
    theme: str = "dark",
) -> go.Figure:
    """Build an interactive 2D all-sky heatmap in Plotly with Light & Dark mode support."""
    pal = _theme_palette(theme)
    mesh = sample_healpix_on_grid(sky_map)
    display_grid = mesh["display_grid"]
    raw_grid = mesh["raw_grid"]

    _, _, lon_label, lat_label = _frame_axis_labels(sky_map.frame)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=mesh["lon_1d"],
            y=mesh["lat_1d"],
            z=display_grid,
            customdata=raw_grid,
            text=mesh["target_grid"],
            colorscale=pal["heatmap_2d_colorscale"],
            zsmooth=False,
            colorbar=dict(
                title=dict(text=sky_map.colorbar_title, side="right", font=dict(color=pal["text_color"])),
                tickfont=dict(color=pal["text_color"]),
            ),
            hovertemplate=(
                f"<b>{lon_label}:</b> %{{x:.1f}}°<br>"
                f"<b>{lat_label}:</b> %{{y:.1f}}°<br>"
                f"<b>{sky_map.unit_label}:</b> %{{customdata:.3f}}<br>"
                "<b>Top Targets:</b> %{text}<extra></extra>"
            ),
        )
    )

    if show_galactic_plane:
        plane_lon, plane_lat, gc_lon, gc_lat = _galactic_plane_and_center_coords(sky_map.frame)
        order = np.argsort(plane_lon)
        fig.add_trace(
            go.Scatter(
                x=plane_lon[order],
                y=plane_lat[order],
                mode="lines",
                line=dict(color=pal["gal_line_color"], width=2.5, dash="dot"),
                name="Milky Way Galactic Plane (b = 0°)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[gc_lon],
                y=[gc_lat],
                mode="markers+text",
                marker=dict(
                    size=12, color=pal["gal_center_color"], symbol="star", line=dict(color=pal["plot_2d_bg"], width=1)
                ),
                text=["Galactic Center (Sgr A*)"],
                textposition="top center",
                textfont=dict(color=pal["gal_center_color"], size=12),
                name="Galactic Center (l=0°, b=0°)",
            )
        )

    if show_ecliptic_plane:
        e_lon, e_lat, eq_lon, eq_lat = _ecliptic_plane_and_origin_coords(sky_map.frame)
        order = np.argsort(e_lon)
        fig.add_trace(
            go.Scatter(
                x=e_lon[order],
                y=e_lat[order],
                mode="lines",
                line=dict(color=pal["ecl_line_color"], width=2.5, dash="dash"),
                name="Solar System / Ecliptic Plane (β = 0°)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[eq_lon],
                y=[eq_lat],
                mode="markers+text",
                marker=dict(
                    size=11,
                    color=pal["ecl_origin_color"],
                    symbol="diamond",
                    line=dict(color=pal["plot_2d_bg"], width=1),
                ),
                text=["Vernal Equinox (λ=0°, β=0°)"],
                textposition="bottom center",
                textfont=dict(color=pal["ecl_origin_color"], size=12),
                name="Solar System / Ecliptic Origin (0°, 0°)",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=18, color=pal["title_color"])),
        paper_bgcolor=pal["scene_bg"],
        plot_bgcolor=pal["plot_2d_bg"],
        font=dict(color=pal["text_color"]),
        height=560,
        margin=dict(l=60, r=40, t=55, b=55),
        xaxis=dict(
            title=dict(text=lon_label, font=dict(color=pal["text_color"])),
            tickfont=dict(color=pal["text_color"]),
            range=[180.0, -180.0],
            tickvals=[180, 120, 60, 0, -60, -120, -180],
            gridcolor=pal["grid_color"],
            zerolinecolor=pal["zero_color"],
        ),
        yaxis=dict(
            title=dict(text=lat_label, font=dict(color=pal["text_color"])),
            tickfont=dict(color=pal["text_color"]),
            range=[-90.0, 90.0],
            tickvals=[-90, -60, -30, 0, 30, 60, 90],
            gridcolor=pal["grid_color"],
            zerolinecolor=pal["zero_color"],
        ),
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor=pal["legend_bg"],
            font=dict(color=pal["text_color"]),
            bordercolor=pal["legend_border"],
            borderwidth=1,
        ),
    )
    return fig


def save_static_mollweide_png(
    sky_map: BinnedSkyMap,
    output_path: Path,
    title: str = "Space Telescope All-Sky Heatmap (Mollweide Projection)",
    theme: str = "dark",
) -> Path:
    """Render a static publication-quality 2D Mollweide projection using Matplotlib."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    pal = _theme_palette(theme)
    mesh = sample_healpix_on_grid(sky_map, n_lon=720, n_lat=360)
    lon_rad = np.radians(-mesh["lon_grid"])
    lat_rad = np.radians(mesh["lat_grid"])

    # An explicit Agg canvas avoids pyplot's process-wide figure registry.
    fig = Figure(figsize=(12, 6.5), dpi=160, facecolor=pal["scene_bg"])
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="mollweide", facecolor=pal["plot_2d_bg"])

    cmap_name = "YlOrRd" if theme == "light" else "inferno"
    cmap = colormaps[cmap_name].with_extremes(bad=pal["plot_2d_bg"])

    masked_z = np.ma.masked_invalid(mesh["display_grid"])
    pcm = ax.pcolormesh(lon_rad, lat_rad, masked_z, cmap=cmap, shading="auto")

    ax.plot(0.0, 0.0, marker="*", markersize=12, color=pal["gal_center_color"], label="Origin (0°, 0°)")
    ax.grid(True, color="#64748b", linestyle=":", alpha=0.5)
    ax.tick_params(colors=pal["text_color"], labelsize=9)
    ax.set_title(title, color=pal["title_color"], fontsize=14, pad=16, fontweight="bold")

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.07, shrink=0.65)
    cbar.set_label(sky_map.colorbar_title, color=pal["text_color"], fontsize=10)
    cbar.ax.xaxis.set_tick_params(color=pal["text_color"], labelcolor=pal["text_color"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        fig.clear()
    return output_path
