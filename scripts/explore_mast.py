"""Step 1: Explore MAST observation metadata columns, target classifications, and duplicate patterns."""

from __future__ import annotations

import time

import pandas as pd
from astropy.time import Time
from astroquery.mast import Observations


def explore_mission_sample(obs_collection: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a narrow time window of science observations from MAST to inspect schema & values."""
    t_min_mjd = Time(start_date).mjd
    t_max_mjd = Time(end_date).mjd

    print(f"\n{'=' * 70}")
    print(f"Exploring {obs_collection} science observations ({start_date} to {end_date})")
    print(f"MJD range: [{t_min_mjd:.1f}, {t_max_mjd:.1f}]")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    table = Observations.query_criteria(
        obs_collection=obs_collection,
        intentType="science",
        t_min=[t_min_mjd, t_max_mjd],
    )
    elapsed = time.perf_counter() - t0
    df = table.to_pandas()

    print(f"Fetched {len(df):,} rows in {elapsed:.2f}s")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    if df.empty:
        return df

    # 1. Check nulls and exposure time summary
    print("\n--- Core Coordinate & Exposure Summary ---")
    for col in ["s_ra", "s_dec", "t_exptime", "target_name", "target_classification", "mtFlag"]:
        if col in df.columns:
            null_cnt = df[col].isna().sum()
            print(f"  {col:24s}: {len(df) - null_cnt:,} non-null ({null_cnt:,} null), dtype={df[col].dtype}")

    # 2. Check dataproduct_type & calib_level distribution
    if "dataproduct_type" in df.columns:
        print("\n--- dataproduct_type counts ---")
        print(df["dataproduct_type"].value_counts(dropna=False).to_string())

    if "calib_level" in df.columns:
        print("\n--- calib_level counts ---")
        print(df["calib_level"].value_counts(dropna=False).to_string())

    # 3. Check duplicate patterns (obs_id vs coordinate+time)
    print("\n--- Duplicate / Grouping Analysis ---")
    unique_obs_id = df["obs_id"].nunique()
    unique_pointing = df.groupby(["s_ra", "s_dec", "t_min", "instrument_name"], dropna=False).ngroups
    unique_prop_time = df.groupby(["proposal_id", "target_name", "t_min"], dropna=False).ngroups
    print(f"  Total rows                           : {len(df):,}")
    print(f"  Unique obs_id                        : {unique_obs_id:,}")
    print(f"  Unique (s_ra, s_dec, t_min, instr)   : {unique_pointing:,}")
    print(f"  Unique (proposal_id, target, t_min)  : {unique_prop_time:,}")

    # Show a sample where multiple rows share the same (proposal_id, target_name, t_min)
    dup_groups = df.groupby(["proposal_id", "target_name", "t_min"], dropna=False).filter(lambda g: len(g) > 1)
    if not dup_groups.empty:
        sample_cols = [
            c
            for c in [
                "obs_id",
                "proposal_id",
                "instrument_name",
                "filters",
                "calib_level",
                "dataproduct_type",
                "s_ra",
                "s_dec",
                "t_exptime",
                "t_min",
                "target_name",
            ]
            if c in df.columns
        ]
        print("\n  Sample multi-row group for same (proposal_id, target_name, t_min):")
        print(dup_groups.sort_values(["t_min", "obs_id"])[sample_cols].head(8).to_string(index=False))

    # 4. Target classification & Moving Target flag (mtFlag)
    print("\n--- Top 20 target_classification values ---")
    if "target_classification" in df.columns:
        print(df["target_classification"].value_counts(dropna=False).head(20).to_string())

    if "mtFlag" in df.columns:
        print("\n--- Moving Target (mtFlag) breakdown ---")
        print(df["mtFlag"].value_counts(dropna=False).to_string())
        mt_rows = df[df["mtFlag"] == True]  # noqa: E712
        if not mt_rows.empty:
            print("  Sample targets with mtFlag=True:")
            print(
                mt_rows[["target_name", "target_classification", "s_ra", "s_dec", "t_exptime"]]
                .drop_duplicates(subset=["target_name"])
                .head(15)
                .to_string(index=False)
            )

    return df


def main() -> None:
    # Sample 30 days of HST and 30 days of JWST
    explore_mission_sample("HST", "2023-01-01", "2023-02-01")
    explore_mission_sample("JWST", "2023-01-01", "2023-02-01")


if __name__ == "__main__":
    main()
