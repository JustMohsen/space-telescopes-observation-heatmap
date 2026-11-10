"""Step 2: CLI entrypoint for running the offline ETL pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from telescope_heatmap.config import (
    CALIBRATIONS_PARQUET_PATH,
    CHUNKS_DIR,
    EXPOSURES_PARQUET_PATH,
    POINTINGS_PARQUET_PATH,
    SUPPORTED_TELESCOPES,
    TARGETS_PARQUET_PATH,
)
from telescope_heatmap.etl import (
    fetch_mast_chunk,
    process_raw_observations,
    save_etl_tables,
)

# Default chunk window size (in months) per mission when running --full-mission
FULL_MISSION_CHUNK_MONTHS = {
    "JWST": 1,
    "HST": 3,
    "EUVE": 120,
    "FUSE": 120,
    "KeplerFFI": 120,
    "K2FFI": 120,
    "IUE": 36,
    "GALEX": 24,
    "SWIFT": 24,
    "TESS": 120,
}

FULL_MISSION_START_DATES = {
    "JWST": "2022-05-01",
    "HST": "1990-04-01",
    "EUVE": "1992-01-01",
    "FUSE": "1999-01-01",
    "KeplerFFI": "2009-01-01",
    "K2FFI": "2014-01-01",
    "IUE": "1978-01-01",
    "GALEX": "2003-04-01",
    "SWIFT": "2004-11-01",
    "TESS": "2018-07-01",
}


def generate_date_windows(start_date: str, end_date: str, months_per_chunk: int = 3) -> list[tuple[str, str]]:
    """Split [start_date, end_date] into manageable sub-windows to prevent MAST API timeouts."""
    s_ts = pd.Timestamp(start_date)
    e_ts = pd.Timestamp(end_date)
    windows: list[tuple[str, str]] = []
    cur = s_ts
    while cur < e_ts:
        nxt = min(cur + pd.DateOffset(months=months_per_chunk), e_ts)
        windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return windows


def print_summary(result) -> None:
    """Print a detailed summary of the 4 generated Parquet tables."""
    print(f"\n{'=' * 72}")
    print("ETL PIPELINE SUMMARY")
    print(f"{'=' * 72}")
    print(f"  1. Exposures Table    ({EXPOSURES_PARQUET_PATH.name:20s}) : {len(result.exposures):8,} rows")
    print(f"  2. Pointings Table    ({POINTINGS_PARQUET_PATH.name:20s}) : {len(result.pointings):8,} rows")
    print(f"  3. Targets Table      ({TARGETS_PARQUET_PATH.name:20s}) : {len(result.targets):8,} rows")
    print(f"  4. Calibrations Table ({CALIBRATIONS_PARQUET_PATH.name:20s}) : {len(result.calibrations):8,} rows")

    if result.exposures.empty:
        return

    print("\n--- Breakdown by Telescope & Target Category (Exposures / Total Hours) ---")
    summary = result.exposures.groupby(["obs_collection", "target_category"], as_index=False).agg(
        exposures=("obs_id", "count"),
        exposure_hours=("t_exptime", lambda s: s.sum() / 3600.0),
    )
    print(summary.to_string(index=False))

    print("\n--- Breakdown by Telescope & Target Category (Pointings / Visits) ---")
    ptg_summary = result.pointings.groupby(["obs_collection", "target_category"], as_index=False).agg(
        pointings=("obs_id", "count"),
        visit_hours=("t_exptime", lambda s: s.sum() / 3600.0),
    )
    print(ptg_summary.to_string(index=False))

    print("\n--- Top 5 Most-Observed Targets per Telescope (by Exposure Hours) ---")
    for mission, group in result.targets.groupby("obs_collection"):
        print(f"\n  [{mission}]")
        top5 = group.head(5)[
            ["target_name", "target_category", "pointing_count", "exposure_count", "t_exptime", "gal_l", "gal_b"]
        ].copy()
        top5["exposure_hours"] = (top5["t_exptime"] / 3600.0).round(2)
        top5["gal_l"] = top5["gal_l"].round(2)
        top5["gal_b"] = top5["gal_b"].round(2)
        print(
            top5[
                [
                    "target_name",
                    "target_category",
                    "pointing_count",
                    "exposure_count",
                    "exposure_hours",
                    "gal_l",
                    "gal_b",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Space Telescopes Observation Heatmap ETL pipeline.")
    parser.add_argument(
        "--missions",
        nargs="+",
        default=["HST", "JWST"],
        choices=list(SUPPORTED_TELESCOPES.keys()),
        help="Telescope missions to fetch (default: HST JWST).",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date (YYYY-MM-DD). Default: 2023-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default="2023-04-01",
        help=f"End date (YYYY-MM-DD). Default: 2023-04-01 (today is {date.today().isoformat()}).",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=1,
        help="Months per query chunk (default: 1, or mission-specific when --full-mission is used).",
    )
    parser.add_argument(
        "--full-mission",
        action="store_true",
        help="Fetch the entire mission lifetime for each specified telescope.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel MAST query workers (default: 6).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached chunk files and re-query MAST.",
    )
    args = parser.parse_args()

    for mission in args.missions:
        m_cfg = SUPPORTED_TELESCOPES[mission]
        if args.full_mission:
            mission_start = FULL_MISSION_START_DATES.get(mission, f"{m_cfg['start_year']}-01-01")
            mission_end = f"{m_cfg.get('end_year', 2026)}-12-31"
            chunk_months = FULL_MISSION_CHUNK_MONTHS.get(mission, args.chunk_months)
        else:
            mission_start = max(args.start_date, f"{m_cfg['start_year']}-01-01")
            mission_end = min(args.end_date, f"{m_cfg.get('end_year', 2026)}-12-31")
            if mission_start >= mission_end:
                mission_start = f"{m_cfg['start_year']}-01-01"
                mission_end = f"{m_cfg.get('end_year', 2026)}-01-01"
            chunk_months = args.chunk_months

        windows = generate_date_windows(mission_start, mission_end, months_per_chunk=chunk_months)
        print(
            f"\nFetching {mission} across {len(windows)} time chunk(s) ({mission_start} -> {mission_end}, "
            f"chunk={chunk_months}m, workers={args.workers})...",
            flush=True,
        )
        if args.workers > 1 and len(windows) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        fetch_mast_chunk,
                        obs_collection=mission,
                        start_date=w_start,
                        end_date=w_end,
                        use_cache=not args.no_cache,
                    ): (w_start, w_end)
                    for w_start, w_end in windows
                }
                done_cnt = 0
                for fut in as_completed(futures):
                    w_start, w_end = futures[fut]
                    fut.result()
                    done_cnt += 1
                    if done_cnt % 10 == 0 or done_cnt == len(windows):
                        print(f"  [{mission} Progress] {done_cnt}/{len(windows)} chunks complete.", flush=True)
        else:
            for w_start, w_end in windows:
                fetch_mast_chunk(
                    obs_collection=mission,
                    start_date=w_start,
                    end_date=w_end,
                    use_cache=not args.no_cache,
                )

    # Load all cached chunk files in CHUNKS_DIR so all downloaded missions/windows are merged together
    all_chunks: list[pd.DataFrame] = []
    cached_files = sorted(CHUNKS_DIR.glob("*.parquet"))
    for fpath in cached_files:
        cdf = pd.read_parquet(fpath)
        if not cdf.empty:
            all_chunks.append(cdf)

    raw_df = (
        pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["obs_collection", "obs_id"])
        if all_chunks
        else pd.DataFrame()
    )
    print(f"\nProcessing {len(raw_df):,} combined raw rows into normalized tables...", flush=True)
    result = process_raw_observations(raw_df)
    save_etl_tables(result)
    print_summary(result)


if __name__ == "__main__":
    main()
