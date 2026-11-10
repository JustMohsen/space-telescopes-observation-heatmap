"""Streamlit web application entrypoint for the Space Telescopes Observation Heatmap."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from telescope_heatmap.binning import compute_healpix_map, filter_observations
from telescope_heatmap.config import (
    CALIBRATIONS_PARQUET_PATH,
    DATA_DIR,
    DEFAULT_NSIDE,
    EXPOSURES_PARQUET_PATH,
    HEATMAP_TARGET_CATEGORIES,
    NSIDE_OPTIONS,
    POINTINGS_PARQUET_PATH,
    SUPPORTED_TELESCOPES,
    TARGET_CATEGORIES,
    TARGETS_PARQUET_PATH,
)
from telescope_heatmap.viz import (
    build_2d_allsky_figure,
    build_3d_globe_figure,
    save_static_mollweide_png,
)


def _inject_apple_ui_styles() -> None:
    """Apply a clean, minimal macOS / Apple Human Interface Guidelines aesthetic."""
    st.markdown(
        """
        <style>
        /* Apple System Typography */
        html, body, [class*="css"], .stMarkdown, .stText, .stRadio, .stCheckbox {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
                         "Helvetica Neue", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        /* Main container spacing — prevents top header bar from clipping KPI cards */
        .block-container {
            padding-top: 3.25rem !important;
            padding-bottom: 3rem !important;
            max-width: 1440px;
        }

        /* Headings with tight Apple-style tracking */
        h1, h2, h3, h4 {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif !important;
            letter-spacing: -0.024em !important;
            font-weight: 600 !important;
        }

        h1 {
            font-size: 2.1rem !important;
            margin-bottom: 0.15rem !important;
        }

        /* Subtle uppercase section labels in sidebar and panels */
        .apple-section-label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.58;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
        }

        /* Frosted macOS-style KPI Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.045);
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 14px;
            padding: 14px 18px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: border-color 0.18s ease, transform 0.18s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: rgba(0, 113, 227, 0.45);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.76rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.03em !important;
            text-transform: uppercase;
            opacity: 0.68;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.65rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            font-variant-numeric: tabular-nums;
        }

        /* macOS Segmented-Control Tab Bar */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: rgba(128, 128, 128, 0.09);
            border: 1px solid rgba(128, 128, 128, 0.14);
            border-radius: 11px;
            padding: 4px;
            gap: 4px;
            margin-top: 0.65rem;
            margin-bottom: 0.85rem;
            width: fit-content;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 7px 16px !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em;
            border: none !important;
            background: transparent !important;
            transition: all 0.16s ease;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: rgba(0, 113, 227, 0.16) !important;
            color: #0071e3 !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }

        /* Hide default Streamlit tab bottom underline bar */
        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Clean Apple-style pill buttons */
        .stButton > button, .stDownloadButton > button {
            border-radius: 980px !important;
            padding: 0.42rem 1.15rem !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            border: 1px solid rgba(128, 128, 128, 0.24) !important;
            transition: all 0.16s ease !important;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #0071e3 !important;
            color: #0071e3 !important;
        }

        /* Subtle hairline dividers */
        hr {
            margin: 1.1rem 0 !important;
            opacity: 0.22 !important;
        }

        /* Story card in About tab */
        .apple-story-card {
            background: rgba(128, 128, 128, 0.04);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 16px;
            padding: 24px 28px;
            line-height: 1.72;
            font-size: 1.02rem;
            margin-bottom: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_parquet_tables(mtime_key: float) -> dict[str, pd.DataFrame]:
    """Load the 4 pre-computed Parquet tables into memory (cached across UI interactions)."""
    del mtime_key  # Used solely to bust cache when ETL updates the Parquet files
    return {
        "exposures": pd.read_parquet(EXPOSURES_PARQUET_PATH) if EXPOSURES_PARQUET_PATH.exists() else pd.DataFrame(),
        "pointings": pd.read_parquet(POINTINGS_PARQUET_PATH) if POINTINGS_PARQUET_PATH.exists() else pd.DataFrame(),
        "targets": pd.read_parquet(TARGETS_PARQUET_PATH) if TARGETS_PARQUET_PATH.exists() else pd.DataFrame(),
        "calibrations": (
            pd.read_parquet(CALIBRATIONS_PARQUET_PATH) if CALIBRATIONS_PARQUET_PATH.exists() else pd.DataFrame()
        ),
    }


def main() -> None:
    st.set_page_config(
        page_title="Space Telescopes Observation Heatmap",
        page_icon="🔭",
        layout="wide",
    )
    _inject_apple_ui_styles()

    st.title("Space Telescopes Observation Heatmap")
    st.caption(
        "Interactive 3D celestial sphere and 2D all-sky heatmaps of where space observatories "
        "have pointed across their mission lifetimes (1978–2026)."
    )

    if not EXPOSURES_PARQUET_PATH.exists():
        st.warning(
            "No pre-computed Parquet tables found in `data/`. "
            "Run `uv run python scripts/run_etl.py` first to populate observation metadata."
        )
        return

    mtime_key = EXPOSURES_PARQUET_PATH.stat().st_mtime
    tables = load_parquet_tables(mtime_key)
    exposures_df = tables["exposures"]
    pointings_df = tables["pointings"]
    targets_df = tables["targets"]
    calibrations_df = tables["calibrations"]

    available_missions = sorted(exposures_df["obs_collection"].dropna().unique().tolist())

    # --- Main Left Sidebar (Controls 3D & 2D Fixed-Sky Heatmaps) ---
    with st.sidebar:
        st.markdown('<div class="apple-section-label">Observatories</div>', unsafe_allow_html=True)
        selected_missions: list[str] = []
        default_missions = {"HST": True, "JWST": True}
        for m in available_missions:
            m_label = f"**{m}** — {SUPPORTED_TELESCOPES.get(m, {}).get('name', m)}"
            if st.checkbox(m_label, value=default_missions.get(m, False), key=f"mission_{m}"):
                selected_missions.append(m)

        if len(selected_missions) == len(available_missions) and len(available_missions) > 1:
            mission_title = "All Loaded Telescopes (" + ", ".join(selected_missions) + ")"
        elif len(selected_missions) == 1:
            code = selected_missions[0]
            mission_title = str(SUPPORTED_TELESCOPES.get(code, {}).get("name", code))
        elif selected_missions:
            mission_title = "Selected Telescopes (" + ", ".join(selected_missions) + ")"
        else:
            mission_title = "No Telescope Selected"

        st.divider()

        # Two-Knob Observation Time Period Slider
        st.markdown('<div class="apple-section-label">Observation Time Period</div>', unsafe_allow_html=True)
        mjd_epoch = date(1858, 11, 17)
        mission_exp_sub = (
            exposures_df[exposures_df["obs_collection"].isin(selected_missions)]
            if selected_missions and not exposures_df.empty
            else exposures_df
        )
        if not mission_exp_sub.empty and "t_min" in mission_exp_sub.columns:
            min_mjd_val = float(mission_exp_sub["t_min"].min())
            max_mjd_val = float(mission_exp_sub["t_max"].fillna(mission_exp_sub["t_min"]).max())
            min_obs_date = mjd_epoch + timedelta(days=int(min_mjd_val))
            max_obs_date = mjd_epoch + timedelta(days=int(max_mjd_val) + 1)
        else:
            min_obs_date = date(1978, 1, 1)
            max_obs_date = date(2026, 12, 31)

        if min_obs_date >= max_obs_date:
            max_obs_date = min_obs_date + timedelta(days=1)

        selected_date_range = st.slider(
            "Start & End Date",
            min_value=min_obs_date,
            max_value=max_obs_date,
            value=(min_obs_date, max_obs_date),
            format="YYYY-MM-DD",
            help="Drag both knobs to filter observations to a specific historical date range.",
        )
        sel_start_date, sel_end_date = selected_date_range
        mjd_range: tuple[float, float] = (
            float((sel_start_date - mjd_epoch).days),
            float((sel_end_date - mjd_epoch).days + 1),
        )

        st.divider()

        st.markdown('<div class="apple-section-label">Map Display & Frame</div>', unsafe_allow_html=True)
        metric_choice = st.radio(
            "Heatmap Metric",
            options=[
                "Exposure Duration (Hours)",
                "Telescope Pointings (Visits)",
                "Unique Targets Count",
            ],
            index=0,
            help=(
                "**Exposure Duration** sums the total time the telescope's shutter was open. "
                "**Telescope Pointings** counts distinct visits/slews to a sky position. "
                "**Unique Targets** counts distinct astronomical targets observed in each sky cell."
            ),
        )
        metric_map = {
            "Exposure Duration (Hours)": ("exposure_hours", exposures_df),
            "Telescope Pointings (Visits)": ("pointing_count", pointings_df),
            "Unique Targets Count": ("target_count", targets_df),
        }
        metric_key, active_source_df = metric_map[metric_choice]

        scale_choice = st.radio(
            "Color Scale",
            options=["Logarithmic (log10)", "Linear"],
            index=0,
            horizontal=True,
        )
        scale_key = "log10" if scale_choice.startswith("Logarithmic") else "linear"

        frame_choice = st.radio(
            "Coordinate Reference Frame",
            options=[
                "Galactic (Milky Way Center: l, b)",
                "Solar System / Ecliptic (Planetary Plane: λ, β)",
                "Equatorial (Earth Equator: RA / Dec)",
            ],
            index=0,
            help=(
                "**Galactic** aligns the sphere's equator with the Milky Way disk (centered on Sgr A*). "
                "**Solar System / Ecliptic** aligns the sphere's equator with the orbital plane of Earth & the planets. "
                "**Equatorial** aligns with Earth's tilted celestial equator (RA/Dec)."
            ),
        )
        if frame_choice.startswith("Galactic"):
            frame_key = "galactic"
        elif frame_choice.startswith("Solar System"):
            frame_key = "ecliptic"
        else:
            frame_key = "equatorial"

        st.divider()

        st.markdown('<div class="apple-section-label">Target Categories</div>', unsafe_allow_html=True)
        st.caption("Moving Solar System objects are excluded from the fixed-sky heatmaps.")
        selected_heatmap_categories: list[str] = []
        for cat_key, cat_label in HEATMAP_TARGET_CATEGORIES.items():
            if st.checkbox(cat_label, value=True, key=f"hm_cat_{cat_key}"):
                selected_heatmap_categories.append(cat_key)

        st.divider()

        st.markdown('<div class="apple-section-label">Resolution & Overlays</div>', unsafe_allow_html=True)
        nside_labels = {
            16: "Coarse (3.7° cells)",
            32: "Low (1.8° cells)",
            64: "Standard (0.9° cells)",
            128: "Fine (0.46° cells)",
            256: "Ultra-Fine (0.23° cells)",
        }
        nside = st.select_slider(
            "Sky Grid Detail (5 Levels)",
            options=NSIDE_OPTIONS,
            value=DEFAULT_NSIDE,
            format_func=lambda v: nside_labels.get(v, str(v)),
        )
        show_galactic_plane = st.checkbox(
            "Show Milky Way Galactic Plane & Center",
            value=(frame_key == "galactic"),
        )
        show_ecliptic_plane = st.checkbox(
            "Show Solar System / Ecliptic Plane",
            value=(frame_key in ("ecliptic", "equatorial")),
        )

    # Detect active browser / Streamlit theme ("light" or "dark") automatically
    detected_theme = getattr(getattr(st, "context", None), "theme", None)
    active_theme = getattr(detected_theme, "type", None) or "dark"
    if active_theme not in ("light", "dark"):
        active_theme = "dark"

    # Filter fixed-sky science tables for KPI cards and active heatmap
    filt_exp = filter_observations(exposures_df, selected_missions, selected_heatmap_categories, mjd_range=mjd_range)
    filt_ptg = filter_observations(pointings_df, selected_missions, selected_heatmap_categories, mjd_range=mjd_range)
    filt_tgt = filter_observations(targets_df, selected_missions, selected_heatmap_categories, mjd_range=mjd_range)
    filt_active = filter_observations(
        active_source_df, selected_missions, selected_heatmap_categories, mjd_range=mjd_range
    )

    # Top KPI Summary Metrics (Fixed-Sky Heatmap Selection)
    col1, col2, col3, col4 = st.columns(4)
    total_hours = filt_exp["t_exptime"].sum() / 3600.0 if not filt_exp.empty else 0.0
    col1.metric("Fixed-Sky Exposure Time", f"{total_hours:,.1f} hrs")
    col2.metric("Telescope Pointings (Visits)", f"{len(filt_ptg):,}")
    col3.metric("Physical Exposures", f"{len(filt_exp):,}")
    col4.metric("Unique Fixed-Sky Targets", f"{len(filt_tgt):,}")

    # --- Visualization & Data Tabs ---
    tab_3d, tab_2d, tab_data, tab_about = st.tabs(
        [
            "3D Celestial Sphere",
            "2D All-Sky Map",
            "Target & Observation Explorer",
            "About & Methodology",
        ]
    )

    map_title = f"{mission_title} — {metric_choice} [{scale_choice}, {frame_key.capitalize()} Frame]"

    with tab_3d:
        if filt_active.empty:
            st.info("No fixed-sky observations match the selected sidebar filters.")
        else:
            sky_map = compute_healpix_map(
                filt_active,
                frame=frame_key,
                metric=metric_key,
                scale=scale_key,
                nside=nside,
            )
            fig_3d = build_3d_globe_figure(
                sky_map,
                title=map_title,
                show_galactic_plane=show_galactic_plane,
                show_ecliptic_plane=show_ecliptic_plane,
                theme=active_theme,
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption(
                "Drag to rotate the 3D sphere, scroll to zoom, and hover over any region to inspect its "
                "coordinates, cumulative exposure or pointings, and top target names. Toggle the Milky Way "
                "and Ecliptic plane overlays in the sidebar to see how the two planes tilt at 60.2° to each other."
            )

    with tab_2d:
        if filt_active.empty:
            st.info("No fixed-sky observations match the selected sidebar filters.")
        else:
            sky_map = compute_healpix_map(
                filt_active,
                frame=frame_key,
                metric=metric_key,
                scale=scale_key,
                nside=nside,
            )
            fig_2d = build_2d_allsky_figure(
                sky_map,
                title=map_title,
                show_galactic_plane=show_galactic_plane,
                show_ecliptic_plane=show_ecliptic_plane,
                theme=active_theme,
            )
            st.plotly_chart(fig_2d, use_container_width=True)

            if st.button("Generate High-Res Mollweide Sky Map (.png)"):
                out_png = DATA_DIR / "mollweide_preview.png"
                save_static_mollweide_png(sky_map, out_png, title=map_title, theme=active_theme)
                st.image(str(out_png), caption=map_title)
                with open(out_png, "rb") as png_file:
                    st.download_button(
                        label="Download High-Res PNG",
                        data=png_file.read(),
                        file_name="space_telescopes_heatmap.png",
                        mime="image/png",
                    )

    with tab_data:
        filter_col, table_col = st.columns([1.05, 3.45], gap="large")

        with filter_col:
            st.markdown('<div class="apple-section-label">Explorer Filters</div>', unsafe_allow_html=True)
            st.caption("Filter observations across all catalog levels, including Solar System and Calibrations.")

            table_view = st.radio(
                "Catalog Granularity",
                options=[
                    "Unique Targets",
                    "Telescope Pointings (Visits)",
                    "Individual Exposures",
                    "Instrument Calibrations",
                ],
                index=0,
                key="tab3_table_view",
            )

            st.markdown('<div class="apple-section-label">Target Categories</div>', unsafe_allow_html=True)
            tab3_categories: list[str] = []
            is_cal_only = table_view.startswith("Instrument Calibrations")
            for cat_key, cat_label in TARGET_CATEGORIES.items():
                checked = st.checkbox(
                    cat_label,
                    value=True,
                    key=f"tab3_cat_{cat_key}",
                    disabled=is_cal_only,
                )
                if checked:
                    tab3_categories.append(cat_key)

            st.markdown('<div class="apple-section-label">Observation Time Period</div>', unsafe_allow_html=True)
            tab3_date_range = st.slider(
                "Table Start & End Date",
                min_value=min_obs_date,
                max_value=max_obs_date,
                value=(sel_start_date, sel_end_date),
                format="YYYY-MM-DD",
                key="tab3_date_slider",
                help="Drag both knobs to filter the explorer tables to a specific date range.",
            )
            tab3_start_date, tab3_end_date = tab3_date_range
            tab3_mjd_range: tuple[float, float] = (
                float((tab3_start_date - mjd_epoch).days),
                float((tab3_end_date - mjd_epoch).days + 1),
            )

            search_query = st.text_input(
                "Search Target / Description / Program",
                value="",
                placeholder="e.g. Saturn, WASP, Andromeda, Quasar...",
                key="tab3_search",
            )

            max_rows = st.select_slider(
                "Rows to Display",
                options=[50, 100, 250, 500, 1000],
                value=100,
                key="tab3_max_rows",
            )

        with table_col:
            if table_view == "Unique Targets":
                base_df = filter_observations(targets_df, selected_missions, tab3_categories, mjd_range=tab3_mjd_range)
                header_label = "Unique Targets"
            elif table_view == "Telescope Pointings (Visits)":
                base_df = filter_observations(
                    pointings_df, selected_missions, tab3_categories, mjd_range=tab3_mjd_range
                )
                header_label = "Telescope Pointings (Visits)"
            elif table_view == "Individual Exposures":
                base_df = filter_observations(
                    exposures_df, selected_missions, tab3_categories, mjd_range=tab3_mjd_range
                )
                header_label = "Individual Exposures"
            else:
                base_df = filter_observations(calibrations_df, selected_missions, None, mjd_range=tab3_mjd_range)
                header_label = "Instrument Calibration Observations"

            display_df = base_df.copy()
            if not display_df.empty and search_query.strip():
                q = search_query.strip()
                t_match = (
                    display_df.get("target_name", pd.Series("", index=display_df.index))
                    .fillna("")
                    .astype(str)
                    .str.contains(q, case=False, regex=False)
                )
                s_match = (
                    display_df.get("target_system", pd.Series("", index=display_df.index))
                    .fillna("")
                    .astype(str)
                    .str.contains(q, case=False, regex=False)
                )
                d_match = (
                    display_df.get("target_description", pd.Series("", index=display_df.index))
                    .fillna("")
                    .astype(str)
                    .str.contains(q, case=False, regex=False)
                )
                c_match = (
                    display_df.get("target_classification", pd.Series("", index=display_df.index))
                    .fillna("")
                    .astype(str)
                    .str.contains(q, case=False, regex=False)
                )
                o_match = (
                    display_df.get("obs_title", pd.Series("", index=display_df.index))
                    .fillna("")
                    .astype(str)
                    .str.contains(q, case=False, regex=False)
                )
                display_df = display_df[t_match | s_match | d_match | c_match | o_match]

            cat_display_map = {
                "solar_system": "Solar System",
                "exoplanet": "Exoplanets & Disks",
                "milky_way": "Milky Way & Local Group",
                "deep_space": "Deep Space",
            }

            # 1. Aggregated Table by Target / System
            if not is_cal_only and not display_df.empty and "target_system" in display_df.columns:
                ptg_col = (
                    display_df["pointing_count"]
                    if "pointing_count" in display_df.columns
                    else pd.Series(1, index=display_df.index)
                )
                exp_col = (
                    display_df["exposure_count"]
                    if "exposure_count" in display_df.columns
                    else pd.Series(1, index=display_df.index)
                )
                tmp_sys = display_df.assign(_ptg_cnt=ptg_col, _exp_cnt=exp_col)

                sys_df = (
                    tmp_sys.groupby("target_system", dropna=False, as_index=False)
                    .agg(
                        target_category=(
                            "target_category",
                            lambda s: cat_display_map.get(str(s.iloc[0]), str(s.iloc[0])),
                        ),
                        what_is_it=("target_description", "first"),
                        telescopes=("obs_collection", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
                        exposure_hours=("t_exptime", lambda s: round(float(s.sum()) / 3600.0, 2)),
                        pointing_count=("_ptg_cnt", "sum"),
                        exposure_count=("_exp_cnt", "sum"),
                        sub_target_count=("target_name", "nunique"),
                        sub_targets=(
                            "target_name",
                            lambda s: ", ".join(list(dict.fromkeys(s.dropna().astype(str)))[:6]),
                        ),
                        science_program_title=("obs_title", "first"),
                        gal_l=("gal_l", lambda s: round(float(s.median()), 2)),
                        gal_b=("gal_b", lambda s: round(float(s.median()), 2)),
                        s_ra=("s_ra", lambda s: round(float(s.median()), 4)),
                        s_dec=("s_dec", lambda s: round(float(s.median()), 4)),
                    )
                    .sort_values("exposure_hours", ascending=False)
                    .reset_index(drop=True)
                )

                st.subheader(f"Aggregated by Target System ({len(sys_df):,} systems)")
                st.caption(
                    "Groups observations of the same physical system—such as a planet and its moons "
                    "(Saturn + Titan, Jupiter + Io/Europa), survey mosaic tiles (M31, HUDF, GOODS-N), "
                    "or catalog naming aliases—into a single summary row."
                )
                st.dataframe(
                    sys_df.head(max_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=350,
                    column_config={
                        "target_system": st.column_config.TextColumn("Target System"),
                        "target_category": st.column_config.TextColumn("Category"),
                        "what_is_it": st.column_config.TextColumn("What Is It?"),
                        "telescopes": st.column_config.TextColumn("Telescopes"),
                        "exposure_hours": st.column_config.NumberColumn("Exposure (hrs)", format="%.2f"),
                        "pointing_count": st.column_config.NumberColumn("Visits"),
                        "exposure_count": st.column_config.NumberColumn("Exposures"),
                        "sub_target_count": st.column_config.NumberColumn("Sub-Targets"),
                        "sub_targets": st.column_config.TextColumn("Included Target Names"),
                        "science_program_title": st.column_config.TextColumn("Science Program"),
                        "gal_l": st.column_config.NumberColumn("Galactic l (°)", format="%.2f"),
                        "gal_b": st.column_config.NumberColumn("Galactic b (°)", format="%.2f"),
                        "s_ra": st.column_config.NumberColumn("RA (°)", format="%.4f"),
                        "s_dec": st.column_config.NumberColumn("Dec (°)", format="%.4f"),
                    },
                )
                st.divider()

            # 2. Detailed Row-Level Table
            st.subheader(f"{header_label} — Detailed Records ({len(display_df):,} matching rows)")

            if display_df.empty:
                st.info("No records match the current filters.")
            else:
                if "t_exptime" in display_df.columns:
                    display_df["exposure_hours"] = (display_df["t_exptime"] / 3600.0).round(2)
                if "target_category" in display_df.columns:
                    display_df["target_category"] = display_df["target_category"].map(
                        lambda c: cat_display_map.get(str(c), str(c))
                    )
                for deg_col, decimals in [("gal_l", 2), ("gal_b", 2), ("s_ra", 4), ("s_dec", 4)]:
                    if deg_col in display_df.columns:
                        display_df[deg_col] = display_df[deg_col].round(decimals)

                preferred_cols = [
                    "obs_collection",
                    "target_system",
                    "target_description",
                    "target_name",
                    "target_category",
                    "exposure_hours",
                    "pointing_count",
                    "exposure_count",
                    "instrument_name",
                    "filters",
                    "obs_title",
                    "target_classification",
                    "gal_l",
                    "gal_b",
                    "s_ra",
                    "s_dec",
                ]
                cols_to_show = [c for c in preferred_cols if c in display_df.columns]
                st.dataframe(
                    display_df[cols_to_show].head(max_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=440,
                    column_config={
                        "obs_collection": st.column_config.TextColumn("Telescope"),
                        "target_system": st.column_config.TextColumn("Target System"),
                        "target_description": st.column_config.TextColumn("What Is It?"),
                        "target_name": st.column_config.TextColumn("Raw Target Name"),
                        "target_category": st.column_config.TextColumn("Category"),
                        "exposure_hours": st.column_config.NumberColumn("Exposure (hrs)", format="%.2f"),
                        "pointing_count": st.column_config.NumberColumn("Visits"),
                        "exposure_count": st.column_config.NumberColumn("Exposures"),
                        "instrument_name": st.column_config.TextColumn("Instrument"),
                        "filters": st.column_config.TextColumn("Filters / Gratings"),
                        "obs_title": st.column_config.TextColumn("Science Program"),
                        "target_classification": st.column_config.TextColumn("MAST Classification"),
                        "gal_l": st.column_config.NumberColumn("Galactic l (°)", format="%.2f"),
                        "gal_b": st.column_config.NumberColumn("Galactic b (°)", format="%.2f"),
                        "s_ra": st.column_config.NumberColumn("RA (°)", format="%.4f"),
                        "s_dec": st.column_config.NumberColumn("Dec (°)", format="%.4f"),
                    },
                )

    with tab_about:
        st.subheader("Motivation & Story")
        st.markdown(
            """
            <div class="apple-story-card">
            I love maps, graphs, and visualizations, like many other space and science enthusiasts. One day, the
            Hubble Ultra Deep Field image came across my feed, and I thought: I really want to see a heatmap of where
            we have been pointing our space telescopes. The story I remember about the first Deep Field image (which I
            was not sure how accurate it was) is that a proposal came up to point the Hubble Space Telescope at an empty
            patch of sky that we hadn't studied before to see what was out there, and there was resistance to wasting
            precious telescope time. Then that first Deep Field image came back and changed our estimate of the number
            of galaxies in the universe by a factor of 10—aside from the visual beauty and the sense of awe it brought.
            <br><br>
            So I thought: I want to see where in the sky we have been exploring and pointing our telescopes. I searched
            a lot and couldn't find anything resembling what I was looking for, so I decided to take on the project of
            building a heatmap to show where we have been pointing our telescopes. I planned the project and thought
            about how to represent it, the coordinate systems, and the different targets. Solar System targets are moving
            targets that don't make sense on a fixed-sky map; we expect galactic targets to be denser along the Galactic
            Plane, whereas deep-space targets should be denser away from the Galactic Plane to avoid an obstructed view.
            <br><br>
            A few months later, I finally started working on this. I built it in a few days, which was possible thanks
            to AI. It is not the most polished, but I am really happy with it and went through many iterations. I also
            learned that there is a bias toward the Northern Galactic Pole—partly because most historical ground-based
            sky surveys were built in Earth's northern hemisphere (which faces the North Galactic Pole and the Virgo
            Supercluster), and we try to point our space telescopes at the same targets for follow-up and comparison,
            and partly because the Southern Galactic Hemisphere contains the Large and Small Magellanic Clouds, which
            obstruct a chunk of the southern extragalactic sky. Pointing space telescopes at the same targets observed
            by ground-based telescopes within their observable regions is also the reason we see a dense belt of
            observations along the equator in the Equatorial reference frame.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Source Code & Documentation:** "
            "[github.com/JustMohsen/space-telescopes-observation-heatmap]"
            "(https://github.com/JustMohsen/space-telescopes-observation-heatmap)"
        )

        st.divider()

        abt_col1, abt_col2 = st.columns(2, gap="large")
        with abt_col1:
            st.subheader("10 Space Observatories (1978–2026)")
            st.markdown("""
                Rather than downloading petabytes of raw `.fits` image files, this application queries **1,837,561 science
                observation records** from the **Mikulski Archive for Space Telescopes (MAST)** at the Space Telescope
                Science Institute (**STScI**), spanning ten missions:

                - **Hubble Space Telescope (`HST`, 1990–2026):** 36 years of UV, optical, and near-IR deep fields, nebulae, and galaxies.
                - **James Webb Space Telescope (`JWST`, 2022–2026):** Infrared flagship peering into early galaxies and exoplanet atmospheres.
                - **International Ultraviolet Explorer (`IUE`, 1978–1996):** An 18-year ultraviolet pioneer of space spectroscopy.
                - **Neil Gehrels Swift Observatory (`SWIFT`, 2004–2026):** Rapid-response observatory studying gamma-ray bursts, supernovae, and active galaxies.
                - **GALEX (`2003–2013`), EUVE (`1992–2001`), & FUSE (`1999–2007`):** All-sky and high-resolution ultraviolet explorers.
                - **Kepler (`KeplerFFI`), K2 (`K2FFI`), & TESS (`2009–2026`):** Wide-field photometers staring at hundreds of thousands of stars to detect exoplanet transits.
                """)

        with abt_col2:
            st.subheader("Patterns You Can Spot on the Map")
            st.markdown("""
                1. **The 60.2° Plane Tilt:** Enable both the **Milky Way Galactic Plane** and the **Solar System Ecliptic Plane** in the sidebar. Our Solar System orbits tilted at **60.2°** relative to the Milky Way disk.
                2. **The Galactic Zone of Avoidance:** Filter to **Deep Space / Extragalactic** in the Galactic frame. A clean band appears along the Galactic Equator ($b = 0^\\circ$) where Milky Way dust blocks distant galaxies, pushing deep fields toward high galactic latitudes.
                3. **The Celestial Equator Belt (SDSS Stripe 82):** In the Southern Galactic Hemisphere ($b \\approx -43^\\circ$ to $-63^\\circ$), a curved arc traces Earth's Celestial Equator ($\\text{Dec} \\approx 0^\\circ$), where both Northern and Southern Hemisphere ground telescopes can jointly follow up space-telescope targets.
                4. **Kepler's Cygnus Field & K2's Ecliptic Stepping Stones:** Filter to `KeplerFFI` to see its four-year stare just above the Milky Way disk in Cygnus, or `K2FFI` to see how engineers balanced the spacecraft with solar radiation pressure along the Ecliptic plane.
                """)

        st.divider()
        st.subheader("Methodology: Processing 1.84 Million Archive Records")

        meth1, meth2, meth3 = st.columns(3, gap="medium")
        with meth1:
            st.markdown("""
                #### 1. Physics-Aware Deduplication
                Raw archive catalogs contain structural duplicates that would distort a naive heatmap:
                - **Hubble Mosaic Co-Adds (`HAP-SVM` / `HAP-MVM`):** MAST stores both the individual camera exposures and synthetic multi-filter stacks. We exclude synthetic co-adds while preserving every physical mosaic tile so exposure time is never double-counted.
                - **JWST Micro-Shutters & Dichroics:** A single **NIRSpec MSA** pointing opens ~100 micro-shutters simultaneously (creating 100 catalog rows for one exposure), and **MIRI's IFU** splits one beam into 12 sub-bands at once. We collapse simultaneous multi-channel readouts by `(mission, proposal, instrument, timestamp)` so one hour of telescope time equals one hour.
                """)

        with meth2:
            st.markdown("""
                #### 2. Equal-Area Spherical Math (`HEALPix`)
                Standard latitude-longitude grids pinch together at the poles, making polar regions look artificially small or over-dense.
                - We use **Astropy (`SkyCoord`)** to transform native Equatorial coordinates $(\\text{RA}, \\text{Dec})$ into **Galactic** $(l, b)$ and **Ecliptic** $(\\lambda, \\beta)$ frames.
                - We tessellate the sphere using **HEALPix (`healpy`)**—the equal-area spherical pixelation developed for the *WMAP* and *Planck* Cosmic Microwave Background missions—so every cell covers the exact same solid angle ($\\Omega$).
                """)

        with meth3:
            st.markdown("""
                #### 3. Four-Table Normalization & Taxonomy
                Our offline ETL pipeline compiles four `zstd`-compressed Parquet tables:
                - **`exposures` (`982,227` rows):** Every physical camera exposure and mosaic tile.
                - **`pointings` (`572,766` rows):** Distinct telescope slews and visits.
                - **`targets` (`171,473` rows):** Unique targets rolled up into canonical **Target Systems** (grouping *Saturn*, *Titan*, and *Enceladus* under `Saturn System`, or `NGC 224` under `Andromeda Galaxy (M31)`).
                - **`calibrations` (`110,463` rows):** Dark frames, flats, and sky backgrounds (`ECLIPTIC-RA12H`), isolated so they never bias science maps.
                """)


if __name__ == "__main__":
    main()
