"""Offline ETL pipeline for fetching, cleaning, and normalizing MAST observation metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.mast import Observations

from telescope_heatmap.config import (
    CALIBRATIONS_PARQUET_PATH,
    CHUNKS_DIR,
    EXPOSURES_PARQUET_PATH,
    POINTINGS_PARQUET_PATH,
    SYNTHETIC_MOSAIC_PROVENANCES,
    TARGETS_PARQUET_PATH,
)
from telescope_heatmap.coordinates import (
    add_galactic_coordinates,
    classify_targets,
    is_calibration_row,
    normalize_target_system,
)

CORE_COLUMNS = [
    "obs_id",
    "obs_collection",
    "proposal_id",
    "instrument_name",
    "filters",
    "dataproduct_type",
    "calib_level",
    "provenance_name",
    "target_name",
    "target_classification",
    "obs_title",
    "mtFlag",
    "s_ra",
    "s_dec",
    "t_min",
    "t_max",
    "t_exptime",
]


@dataclass
class ETLResult:
    """Container holding the 4 normalized tables produced by the ETL pipeline."""

    exposures: pd.DataFrame
    pointings: pd.DataFrame
    targets: pd.DataFrame
    calibrations: pd.DataFrame


def fetch_mast_chunk(
    obs_collection: str,
    start_date: str,
    end_date: str,
    chunks_dir: Path = CHUNKS_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch science observations from MAST for a given mission and [start_date, end_date] window.

    Caches raw chunks as Parquet files in `chunks_dir` so repeated runs are instant and resumable.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    cache_file = chunks_dir / f"{obs_collection}_{safe_start}_{safe_end}.parquet"

    if use_cache and cache_file.exists():
        print(f"  [Cache Hit] Loading {cache_file.name}")
        return pd.read_parquet(cache_file)

    t_min_mjd = Time(start_date).mjd
    t_max_mjd = Time(end_date).mjd

    print(
        f"  [MAST Query] {obs_collection}: {start_date} -> {end_date} (MJD {t_min_mjd:.1f}..{t_max_mjd:.1f})...",
        flush=True,
    )
    t0 = time.perf_counter()
    last_err: Exception | None = None
    table = None
    extra_criteria = {"dataproduct_type": "image"} if obs_collection == "TESS" else {}
    for attempt in range(4):
        try:
            table = Observations.query_criteria(
                obs_collection=obs_collection,
                intentType="science",
                t_min=[t_min_mjd, t_max_mjd],
                **extra_criteria,
            )
            break
        except Exception as err:
            last_err = err
            wait_s = 2.0 * (2**attempt)
            print(
                f"  [MAST Retry {attempt + 1}/4] {obs_collection} {start_date}..{end_date}: {err} (sleeping {wait_s:.0f}s)",
                flush=True,
            )
            time.sleep(wait_s)
    if table is None and last_err is not None:
        raise last_err
    elapsed = time.perf_counter() - t0
    df = table.to_pandas() if table is not None and len(table) > 0 else pd.DataFrame(columns=CORE_COLUMNS)

    # Keep only relevant columns that exist in the response and normalize types for Parquet
    available_cols = [c for c in CORE_COLUMNS if c in df.columns]
    df = df[available_cols].copy()
    for missing_col in CORE_COLUMNS:
        if missing_col not in df.columns:
            df[missing_col] = pd.NA

    for str_col in [
        "obs_id",
        "obs_collection",
        "proposal_id",
        "instrument_name",
        "filters",
        "dataproduct_type",
        "provenance_name",
        "target_name",
        "target_classification",
        "obs_title",
    ]:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype("string")

    for num_col in ["s_ra", "s_dec", "t_min", "t_max", "t_exptime"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    if "calib_level" in df.columns:
        df["calib_level"] = pd.to_numeric(df["calib_level"], errors="coerce").astype("Int64")

    if "mtFlag" in df.columns:
        df["mtFlag"] = df["mtFlag"].fillna(False).astype(bool)

    if use_cache:
        df.to_parquet(cache_file, index=False)

    print(f"  [MAST Done]  {obs_collection}: {len(df):,} raw rows in {elapsed:.2f}s -> saved {cache_file.name}")
    return df


def process_raw_observations(raw_df: pd.DataFrame) -> ETLResult:
    """Transform raw MAST rows into the 4 clean tables: exposures, pointings, targets, calibrations."""
    if raw_df.empty:
        empty = pd.DataFrame()
        return ETLResult(exposures=empty, pointings=empty, targets=empty, calibrations=empty)

    df = raw_df.copy()

    # 1. Basic validity filter: valid sky coordinates and positive exposure time
    valid_mask = (
        df["s_ra"].notna()
        & df["s_dec"].notna()
        & (df["s_ra"] >= 0.0)
        & (df["s_ra"] <= 360.0)
        & (df["s_dec"] >= -90.0)
        & (df["s_dec"] <= 90.0)
        & df["t_exptime"].notna()
        & (df["t_exptime"] > 0.0)
    )
    df = df[valid_mask].copy()

    # 2. Separate residual calibration observations into `calibrations` table
    cal_mask = is_calibration_row(df)
    calibrations_df = add_galactic_coordinates(df[cal_mask].copy()).reset_index(drop=True)
    science_df = df[~cal_mask].copy()

    if science_df.empty:
        empty = pd.DataFrame()
        return ETLResult(
            exposures=empty,
            pointings=empty,
            targets=empty,
            calibrations=calibrations_df,
        )

    # 3. Drop HST synthetic mosaic stacks (HAP-SVM, HAP-MVM, HASP, HSLA, and filters='detection')
    # and JWST multi-visit composite associations (obs_id containing '-c1\d{3}_')
    # so individual instrument exposures represent each mosaic tile once.
    prov = science_df.get("provenance_name", pd.Series("", index=science_df.index)).fillna("").astype(str).str.upper()
    filt = science_df.get("filters", pd.Series("", index=science_df.index)).fillna("").astype(str).str.lower()
    obs_id_str = science_df.get("obs_id", pd.Series("", index=science_df.index)).fillna("").astype(str).str.lower()

    is_synthetic_mosaic = (
        prov.isin(SYNTHETIC_MOSAIC_PROVENANCES)
        | (filt == "detection")
        | obs_id_str.str.contains(r"^jw\d+-c\d+_", regex=True)
    )
    science_df = science_df[~is_synthetic_mosaic].copy()

    # Cap single-exposure t_exptime at wall-clock span (t_max - t_min)*86400 when t_max > t_min
    # (fixes CALJWST Level-3 timeseries segment-multiplier bug on multi-segment transit observations).
    if "t_max" in science_df.columns and "t_min" in science_df.columns:
        span_sec = (science_df["t_max"] - science_df["t_min"]) * 86400.0
        valid_span = span_sec.notna() & (span_sec > 60.0) & (science_df["t_exptime"] > span_sec * 1.05)
        science_df.loc[valid_span, "t_exptime"] = span_sec.loc[valid_span]

    # 4. Collapse simultaneous multi-channel / multi-shutter rows (e.g. JWST MIRI/IFU 12 sub-bands
    # or NIRSpec/MSA 100 slitlets) that come from the exact same physical exposure timestamp & pointing.
    # Rounding t_min to 4 decimal places (~8.6 seconds) and coordinates to 1 decimal place (~6 arcmin,
    # covering the 3.6x3.4 arcmin NIRSpec MSA field of view) groups simultaneous channel/shutter outputs
    # while keeping distinct mosaic step tiles and sequential filter exposures separate.
    science_df["_time_key"] = science_df["t_min"].round(4)
    science_df["_ra_key"] = science_df["s_ra"].round(1)
    science_df["_dec_key"] = science_df["s_dec"].round(1)

    # Prefer rows that have a non-null target_classification when collapsing simultaneous readouts
    science_df["_has_class"] = science_df["target_classification"].notna() & (
        science_df["target_classification"].astype(str).str.strip() != ""
    )
    science_df = science_df.sort_values(
        by=["_has_class", "calib_level", "t_exptime"],
        ascending=[False, False, False],
    )

    group_cols = ["obs_collection", "proposal_id", "instrument_name", "_time_key", "_ra_key", "_dec_key"]
    exposures_df = (
        science_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            obs_id=("obs_id", "first"),
            filters=("filters", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
            dataproduct_type=("dataproduct_type", "first"),
            calib_level=("calib_level", "max"),
            provenance_name=("provenance_name", "first"),
            target_name=("target_name", "first"),
            target_classification=("target_classification", "first"),
            obs_title=("obs_title", "first"),
            mtFlag=("mtFlag", "any"),
            s_ra=("s_ra", "median"),
            s_dec=("s_dec", "median"),
            t_min=("t_min", "min"),
            t_max=("t_max", "max"),
            t_exptime=("t_exptime", "max"),
            simultaneous_channels=("obs_id", "count"),
        )
        .drop(columns=["_time_key", "_ra_key", "_dec_key"])
    )

    # 5. Backfill missing `target_classification` across all telescopes sharing the same `target_name`
    # or canonical `target_system` so targets like ECLIPTIC-RA12H, WASP-107, or M31 sub-tiles that have
    # classification metadata on HST (or one visit) but NaN on JWST/FUSE/EUVE are consistently classified.
    exposures_df["_tmp_system"] = normalize_target_system(exposures_df)
    generic_systems = {
        "Unidentified / Parallel Fields",
        "HST Pure Parallel Fields (ANY)",
        "Unknown Target",
        "TARGETS",
        "NIRCAM",
    }
    non_empty_rows = exposures_df.dropna(subset=["target_classification"]).loc[
        lambda d: d["target_classification"].astype(str).str.strip() != ""
    ]
    if not non_empty_rows.empty:
        name_lookup = non_empty_rows.groupby("target_name")["target_classification"].first()
        sys_lookup = (
            non_empty_rows.loc[~non_empty_rows["_tmp_system"].isin(generic_systems)]
            .groupby("_tmp_system")["target_classification"]
            .first()
        )
        missing_mask = exposures_df["target_classification"].isna() | (
            exposures_df["target_classification"].astype(str).str.strip() == ""
        )
        if missing_mask.any():
            mapped = exposures_df.loc[missing_mask, "target_name"].map(name_lookup)
            still_missing = mapped.isna()
            if still_missing.any():
                mapped.loc[still_missing] = (
                    exposures_df.loc[missing_mask, "_tmp_system"].loc[still_missing].map(sys_lookup)
                )
            exposures_df.loc[missing_mask, "target_classification"] = pd.Series(
                mapped, index=exposures_df.loc[missing_mask].index, dtype="string"
            )
    exposures_df = exposures_df.drop(columns=["_tmp_system"])

    # Disambiguate generic parallel-field or missing target names ("ANY", "PARALLEL", "UNIDENTIFIED", "BLANK", <NA>)
    # by appending their rounded sky coordinates so unrelated parallel fields across the sky
    # don't collapse into a single fake target in `targets.parquet`.
    generic_names = {
        "",
        "ANY",
        "PARALLEL",
        "PARALLEL-FIELD",
        "PAR-TARGET",
        "PAR",
        "HIGH",
        "HI-LATITUDE-FIELD",
        "HI-LAT",
        "HIGH-GALACTIC-LATITUDE",
        "LOW",
        "LOW-LATITUDE-FIELD",
        "UNKNOWN-TARGET-1",
        "LIST-1",
        "UNIDENTIFIED",
        "BLANK",
        "BLANK FIELD",
        "UNKNOWN",
        "<NA>",
        "NAN",
        "SKY",
        "SKY BKGD",
        "TESS FFI",
    }
    t_name_clean = exposures_df["target_name"].fillna("").astype(str).str.strip()
    is_generic = t_name_clean.str.upper().isin(generic_names)
    if is_generic.any():
        prefix = np.where(t_name_clean == "", exposures_df["obs_collection"].astype(str) + "-FIELD", t_name_clean)
        exposures_df.loc[is_generic, "target_name"] = pd.Series(
            prefix[is_generic]
            + " ("
            + exposures_df.loc[is_generic, "s_ra"].round(1).astype(str)
            + ", "
            + exposures_df.loc[is_generic, "s_dec"].round(1).astype(str)
            + ")",
            index=exposures_df.loc[is_generic].index,
            dtype="string",
        )

    # Compute Galactic coordinates (gal_l, gal_b) and target classifications on exposures
    exposures_df = add_galactic_coordinates(exposures_df)
    exposures_df = classify_targets(exposures_df)
    exposures_df = exposures_df.sort_values(["obs_collection", "t_min"]).reset_index(drop=True)

    # 6. Build `pointings` table: group sequential filter exposures during the same visit/slew
    # at the same sky tile (~36 arcsec = 0.01 deg, within ~0.1 MJD = 2.4 hours) into 1 pointing event.
    ptg = exposures_df.copy()
    ptg["_visit_time_key"] = ptg["t_min"].round(1)
    ptg["_tile_ra_key"] = ptg["s_ra"].round(2)
    ptg["_tile_dec_key"] = ptg["s_dec"].round(2)

    pointings_df = (
        ptg.groupby(
            [
                "obs_collection",
                "proposal_id",
                "instrument_name",
                "target_name",
                "_visit_time_key",
                "_tile_ra_key",
                "_tile_dec_key",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            obs_id=("obs_id", "first"),
            filters=("filters", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
            target_classification=("target_classification", "first"),
            obs_title=("obs_title", "first"),
            mtFlag=("mtFlag", "any"),
            s_ra=("s_ra", "mean"),
            s_dec=("s_dec", "mean"),
            gal_l=("gal_l", "mean"),
            gal_b=("gal_b", "mean"),
            t_min=("t_min", "min"),
            t_max=("t_max", "max"),
            t_exptime=("t_exptime", "sum"),
            exposure_count=("obs_id", "count"),
            target_system=("target_system", "first"),
            target_description=("target_description", "first"),
            target_category=("target_category", "first"),
            is_solar_system=("is_solar_system", "any"),
            is_exoplanet=("is_exoplanet", "any"),
            is_milky_way=("is_milky_way", "any"),
            is_deep_space=("is_deep_space", "any"),
        )
        .drop(columns=["_visit_time_key", "_tile_ra_key", "_tile_dec_key"])
        .sort_values(["obs_collection", "t_min"])
        .reset_index(drop=True)
    )

    # 7. Build `targets` table: aggregate by unique (obs_collection, target_name)
    targets_df = (
        pointings_df.groupby(["obs_collection", "target_name"], dropna=False, as_index=False)
        .agg(
            target_classification=("target_classification", "first"),
            obs_title=("obs_title", "first"),
            mtFlag=("mtFlag", "any"),
            s_ra=("s_ra", "median"),
            s_dec=("s_dec", "median"),
            gal_l=("gal_l", "median"),
            gal_b=("gal_b", "median"),
            t_min=("t_min", "min"),
            t_max=("t_max", "max"),
            t_exptime=("t_exptime", "sum"),
            pointing_count=("obs_id", "count"),
            exposure_count=("exposure_count", "sum"),
            target_system=("target_system", "first"),
            target_description=("target_description", "first"),
            target_category=("target_category", "first"),
            is_solar_system=("is_solar_system", "any"),
            is_exoplanet=("is_exoplanet", "any"),
            is_milky_way=("is_milky_way", "any"),
            is_deep_space=("is_deep_space", "any"),
        )
        .sort_values(["obs_collection", "t_exptime"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return ETLResult(
        exposures=exposures_df,
        pointings=pointings_df,
        targets=targets_df,
        calibrations=calibrations_df,
    )


def save_etl_tables(
    result: ETLResult,
    exposures_path: Path = EXPOSURES_PARQUET_PATH,
    pointings_path: Path = POINTINGS_PARQUET_PATH,
    targets_path: Path = TARGETS_PARQUET_PATH,
    calibrations_path: Path = CALIBRATIONS_PARQUET_PATH,
) -> None:
    """Persist all 4 ETL tables to zstd-compressed Parquet files."""
    exposures_path.parent.mkdir(parents=True, exist_ok=True)
    result.exposures.to_parquet(exposures_path, index=False, compression="zstd")
    result.pointings.to_parquet(pointings_path, index=False, compression="zstd")
    result.targets.to_parquet(targets_path, index=False, compression="zstd")
    result.calibrations.to_parquet(calibrations_path, index=False, compression="zstd")
