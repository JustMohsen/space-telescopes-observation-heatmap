"""Projected, filtered catalog reads without caching entire observation tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SEARCH_COLUMNS = ("target_name", "target_system", "target_description", "target_classification", "obs_title")


def read_catalog(
    path: Path,
    columns: Sequence[str] | None = None,
    telescopes: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    mjd_range: tuple[float, float] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Filter in Arrow before materializing only the requested columns in pandas.

    Dictionary-encoded Arrow strings avoid expanding repeated descriptions and
    titles for every row. Small scan batches and limited read-ahead bound decoding memory,
    including for the existing single-row-group release files.
    """
    schema = pq.read_schema(path)
    text_columns = {
        field.name for field in schema if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }
    parquet_format = ds.ParquetFileFormat(read_options=ds.ParquetReadOptions(dictionary_columns=text_columns))
    dataset = ds.dataset(path, format=parquet_format)
    names = dataset.schema.names
    predicate = ds.scalar(True)
    if telescopes is not None:
        predicate = predicate & (ds.field("obs_collection").isin(list(telescopes)) if telescopes else ds.scalar(False))
    if categories is not None and "target_category" in names:
        predicate = predicate & (ds.field("target_category").isin(list(categories)) if categories else ds.scalar(False))
    if mjd_range is not None and "t_min" in names:
        start, end = mjd_range
        t_min = ds.field("t_min")
        t_max = pc.coalesce(ds.field("t_max"), t_min) if "t_max" in names else t_min
        predicate = predicate & (t_max >= start) & (t_min <= end)
    if search.strip():
        matches = ds.scalar(False)
        for name in SEARCH_COLUMNS:
            if name in names:
                matches = matches | pc.coalesce(
                    pc.match_substring(ds.field(name).cast(pa.string()), search.strip(), ignore_case=True), False
                )
        predicate = predicate & matches

    projection = [name for name in columns if name in names] if columns is not None else None
    table = dataset.scanner(
        columns=projection,
        filter=predicate,
        batch_size=32_768,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
        # Return temporary decoding buffers to the OS instead of retaining them
        # in Arrow's default mimalloc pool across successive filter changes.
        memory_pool=pa.system_memory_pool(),
    ).to_table()
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def summarize_target_systems(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, int]:
    """Rank every system with native reductions; join text only for displayed rows."""
    if df.empty or "target_system" not in df.columns:
        return pd.DataFrame(), 0

    grouped = df.groupby("target_system", dropna=False, observed=True, sort=False)
    summary = grouped.agg(
        target_category=("target_category", "first"),
        what_is_it=("target_description", "first"),
        exposure_hours=("t_exptime", "sum"),
        pointing_count=("pointing_count", "sum") if "pointing_count" in df else ("target_name", "size"),
        exposure_count=("exposure_count", "sum") if "exposure_count" in df else ("target_name", "size"),
        sub_target_count=("target_name", "nunique"),
        science_program_title=("obs_title", "first"),
        gal_l=("gal_l", "median"),
        gal_b=("gal_b", "median"),
        s_ra=("s_ra", "median"),
        s_dec=("s_dec", "median"),
    )
    system_count = len(summary)
    summary = summary.sort_values("exposure_hours", ascending=False).head(max_rows).copy()
    summary["exposure_hours"] = (summary["exposure_hours"] / 3600.0).round(2)
    for col, digits in (("gal_l", 2), ("gal_b", 2), ("s_ra", 4), ("s_dec", 4)):
        summary[col] = summary[col].round(digits)

    visible = df.loc[
        df["target_system"].isin(summary.index.tolist()), ["target_system", "obs_collection", "target_name"]
    ]
    visible_groups = visible.groupby("target_system", dropna=False, observed=True, sort=False)
    summary["telescopes"] = visible_groups["obs_collection"].agg(lambda s: ", ".join(sorted(s.dropna().unique())))
    summary["sub_targets"] = visible_groups["target_name"].agg(lambda s: ", ".join(s.dropna().unique()[:6]))
    return summary.reset_index(), system_count
