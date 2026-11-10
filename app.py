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

    st.title("🔭 Space Telescopes Observation Heatmap")
    st.caption(
        "Interactive 3D Celestial Sphere & 2D All-Sky Heatmaps of where humanity's space telescopes "
        "have pointed and stared."
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
        st.header("🎛️ Sky Heatmap Controls")

        # 1. Telescope Checkbox List
        st.subheader("🛰️ Space Telescopes")
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

        # 1b. Two-Knob Observation Time Period Slider
        st.subheader("📅 Observation Time Period")
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

        # 2. Heatmap Metric Toggle
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

        # 3. Color Scale Toggle
        scale_choice = st.radio(
            "Color Scale",
            options=["Logarithmic (log10)", "Linear"],
            index=0,
            horizontal=True,
        )
        scale_key = "log10" if scale_choice.startswith("Logarithmic") else "linear"

        # 4. Reference Frame Toggle
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

        # 5. Fixed-Sky Target Category Toggles (Solar System excluded from sky heatmaps)
        st.subheader("🌌 Sky Map Target Categories")
        st.caption("Moving Solar System objects are excluded from the fixed-sky heatmaps.")
        selected_heatmap_categories: list[str] = []
        for cat_key, cat_label in HEATMAP_TARGET_CATEGORIES.items():
            if st.checkbox(cat_label, value=True, key=f"hm_cat_{cat_key}"):
                selected_heatmap_categories.append(cat_key)

        st.divider()

        # 6. Resolution & Overlays (5 positions)
        st.subheader("🧭 Grid Resolution & Overlays")
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
            "Show Milky Way Galactic Plane & Center (Cyan)",
            value=(frame_key == "galactic"),
        )
        show_ecliptic_plane = st.checkbox(
            "Show Solar System / Ecliptic Plane (Gold)",
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
    tab_3d, tab_2d, tab_data = st.tabs(
        [
            "🌐 3D Interactive Celestial Sphere",
            "🗺️ 2D All-Sky Projection (Mollweide / Rectangular)",
            "🎯 Targets & Observations Explorer",
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
                "💡 **Tip:** Drag to spin the 3D sphere, scroll to zoom, and hover over any glowing region to inspect its "
                "exact coordinates, exposure/pointings, and **top target names**. Toggle the **Cyan (Milky Way)** and "
                "**Gold (Solar System / Ecliptic)** rings in the sidebar to compare how the two planes tilt at **60.2°** to each other!"
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

            if st.button("📸 Generate High-Res Mollweide Sky Map (.png)"):
                out_png = DATA_DIR / "mollweide_preview.png"
                save_static_mollweide_png(sky_map, out_png, title=map_title, theme=active_theme)
                st.image(str(out_png), caption=map_title)
                with open(out_png, "rb") as png_file:
                    st.download_button(
                        label="⬇️ Download High-Res PNG",
                        data=png_file.read(),
                        file_name="space_telescopes_heatmap.png",
                        mime="image/png",
                    )

    with tab_data:
        # Option 3: Tab 3 has its own dedicated left-side filter panel!
        filter_col, table_col = st.columns([1.05, 3.45], gap="large")

        with filter_col:
            st.subheader("🔍 Explorer Filters")
            st.caption("Filter observations across all catalog levels (including Solar System & Calibrations).")

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

            st.markdown("**Target Categories**")
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

            st.markdown("**📅 Observation Time Period**")
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
                header_label = "🎯 Unique Targets"
            elif table_view == "Telescope Pointings (Visits)":
                base_df = filter_observations(
                    pointings_df, selected_missions, tab3_categories, mjd_range=tab3_mjd_range
                )
                header_label = "🧭 Telescope Pointings (Visits)"
            elif table_view == "Individual Exposures":
                base_df = filter_observations(
                    exposures_df, selected_missions, tab3_categories, mjd_range=tab3_mjd_range
                )
                header_label = "📸 Individual Exposures"
            else:
                base_df = filter_observations(calibrations_df, selected_missions, None, mjd_range=tab3_mjd_range)
                header_label = "🔧 Instrument Calibration Observations"

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
                "solar_system": "☀️ Solar System",
                "exoplanet": "🪐 Exoplanets & Disks",
                "milky_way": "🌠 Milky Way & Local Group",
                "deep_space": "🌌 Deep Space",
            }

            # 1. Aggregated Table by Target / System (groups e.g. Saturn + Titan + moons, or M31 tiles, or HUDF tiles)
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

                st.subheader(f"📊 Aggregated by Target System ({len(sys_df):,} systems)")
                st.caption(
                    "Groups observations of the same physical system—such as a planet and its moons "
                    "(e.g., **Saturn + Titan**, **Jupiter + Io/Europa**), survey mosaic tiles (**M31**, **HUDF**, **GOODS-N**), "
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


if __name__ == "__main__":
    main()
