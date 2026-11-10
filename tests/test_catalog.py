"""Regression tests for projected queries and bounded Explorer summaries."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from telescope_heatmap.binning import filter_observations
from telescope_heatmap.catalog import read_catalog, summarize_target_systems


def search_query(url: str) -> dict:
    return json.loads(parse_qs(urlsplit(url).query)["searchQuery"][0])


@pytest.fixture
def catalog_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_collection": ["HST", "JWST", "HST", "JWST", "HST"],
            "target_category": ["deep_space", "deep_space", "milky_way", "solar_system", "deep_space"],
            "t_min": [10.0, 20.0, 30.0, 40.0, 50.0],
            "t_max": [25.0, np.nan, 35.0, 45.0, 55.0],
            "t_exptime": [3600.0, 7200.0, 1800.0, 900.0, 3600.0],
            "target_name": ["M31-A", "M31-B", "Star[1]", "Saturn", None],
            "target_system": ["M31", "M31", "Star", "Saturn", None],
            "target_description": ["Galaxy", "Galaxy", None, "Planet", None],
            "target_classification": ["GALAXY", None, "STAR", "PLANET", None],
            "obs_title": ["Survey", "Survey", "Stars", "Rings", None],
            "s_ra": [10.0, 12.0, 40.0, 60.0, 80.0],
            "s_dec": [20.0, 22.0, 10.0, 0.0, -10.0],
            "gal_l": [30.0, 32.0, 50.0, 70.0, 90.0],
            "gal_b": [40.0, 42.0, 0.0, -10.0, -20.0],
            "pointing_count": [2, 3, 1, 1, 1],
            "exposure_count": [4, 6, 2, 1, 1],
        }
    )


def test_projected_query_preserves_filters_and_interval_overlap(tmp_path: Path, catalog_df: pd.DataFrame) -> None:
    path = tmp_path / "catalog.parquet"
    catalog_df.to_parquet(path, index=False)
    columns = ["target_name", "t_exptime"]
    actual = read_catalog(path, columns, ["HST", "JWST"], ["deep_space"], (20.0, 30.0))
    expected = filter_observations(catalog_df, ["HST", "JWST"], ["deep_space"], (20.0, 30.0))[columns]
    assert actual["target_name"].tolist() == expected["target_name"].tolist()
    assert actual["t_exptime"].tolist() == expected["t_exptime"].tolist()
    assert actual.columns.tolist() == columns
    assert all(isinstance(dtype, pd.ArrowDtype) for dtype in actual.dtypes)
    target_dtype = actual["target_name"].dtype
    assert isinstance(target_dtype, pd.ArrowDtype)
    assert pa.types.is_dictionary(target_dtype.pyarrow_dtype)
    assert read_catalog(path, telescopes=[]).empty
    assert read_catalog(path, categories=[]).empty


def test_search_is_literal_case_insensitive_and_null_safe(tmp_path: Path, catalog_df: pd.DataFrame) -> None:
    path = tmp_path / "catalog.parquet"
    catalog_df.to_parquet(path, index=False)
    assert read_catalog(path, search="  m31  ")["target_name"].tolist() == ["M31-A", "M31-B"]
    assert read_catalog(path, search="[1]")["target_name"].tolist() == ["Star[1]"]
    assert read_catalog(path, search="rings")["target_name"].tolist() == ["Saturn"]
    assert read_catalog(path, search="absent").empty


def test_missing_end_time_and_optional_columns(tmp_path: Path, catalog_df: pd.DataFrame) -> None:
    path = tmp_path / "catalog.parquet"
    catalog_df.drop(columns=["t_max", "target_description"]).to_parquet(path, index=False)
    actual = read_catalog(path, ["target_name", "target_description"], mjd_range=(20.0, 20.0))
    assert actual["target_name"].tolist() == ["M31-B"]
    assert actual.columns.tolist() == ["target_name"]


def test_system_summary_keeps_totals_and_limits_text_work(tmp_path: Path, catalog_df: pd.DataFrame) -> None:
    path = tmp_path / "catalog.parquet"
    catalog_df.to_parquet(path, index=False)
    source = read_catalog(path)
    summary, count = summarize_target_systems(source, 1)
    assert count == 4  # Missing system names form one group, as in the original Explorer.
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["target_system"] == "M31"
    assert row["exposure_hours"] == 3.0
    assert row["pointing_count"] == 5
    assert row["exposure_count"] == 10
    assert row["sub_target_count"] == 2
    assert row["sub_targets"] == "M31-A, M31-B"
    assert row["telescopes"] == "HST, JWST"
    assert row["s_ra"] == 11.0
    filters = {item["paramName"]: item["values"] for item in search_query(row["mast_url"])["inputText"]}
    assert filters == {"obs_collection": ["HST", "JWST"], "target_name": ["M31-A", "M31-B"]}
    detailed = source.drop(columns=["pointing_count", "exposure_count"])
    summary, _ = summarize_target_systems(detailed, 1)
    assert summary.iloc[0]["pointing_count"] == summary.iloc[0]["exposure_count"] == 2
    all_systems, _ = summarize_target_systems(source, 100)
    assert all_systems.loc[all_systems["target_system"].isna(), "mast_url"].isna().all()


def test_system_links_use_bounded_raw_names_instead_of_canonical_labels(catalog_df: pd.DataFrame) -> None:
    source = pd.concat([catalog_df.iloc[[0]]] * 8, ignore_index=True)
    names = ["Tile, A", *(f"Tile-{i}" for i in range(7))]
    source["target_name"] = names
    source["target_system"] = "Descriptive group label absent from MAST"
    summary, _ = summarize_target_systems(source, 1)
    query = search_query(summary.iloc[0]["mast_url"])
    filters = {item["paramName"]: item["values"] for item in query["inputText"]}
    assert filters["target_name"] == names[:6]
    assert summary.iloc[0]["sub_target_count"] == 8
