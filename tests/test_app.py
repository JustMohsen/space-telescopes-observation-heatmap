"""Exercise lazy views using a tiny local catalog rather than live downloads."""

import json
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

import app


def table_search_query(data: bytes) -> dict:
    # Inspect Arrow directly: AppTest's pandas conversion does not support the
    # dictionary-encoded strings used to keep our catalog queries small.
    table = pa.ipc.open_stream(data).read_all()
    url = table.column("mast_url")[0].as_py()
    return json.loads(parse_qs(urlsplit(url).query)["searchQuery"][0])


@pytest.fixture
def local_catalogs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    source = pd.DataFrame(
        {
            "obs_collection": ["HST", "JWST"],
            "obs_id": ["n4qf18090", "jw01442-o001_t001_nircam_f322w2-grismr-subgrism128"],
            "target_category": ["deep_space", "deep_space"],
            "target_name": ["M31-A", "M31-B"],
            "target_system": ["M31", "M31"],
            "target_description": ["Galaxy", "Galaxy"],
            "target_classification": ["GALAXY", "GALAXY"],
            "obs_title": ["Survey", "Survey"],
            "t_min": [60_000.0, 60_001.0],
            "t_max": [60_000.5, 60_001.5],
            "t_exptime": [3600.0, 7200.0],
            "pointing_count": [1, 2],
            "exposure_count": [2, 3],
            "s_ra": [10.0, 12.0],
            "s_dec": [20.0, 22.0],
            "gal_l": [30.0, 32.0],
            "gal_b": [40.0, 42.0],
        }
    )
    paths = {}
    for name in ("targets", "exposures", "pointings", "calibrations"):
        path = tmp_path / f"{name}.parquet"
        table = source.drop(columns="obs_id") if name == "targets" else source
        table.to_parquet(path, index=False)
        paths[name] = path
    monkeypatch.setattr(app, "_TABLE_PATHS", paths)
    monkeypatch.setattr(app, "TARGETS_PARQUET_PATH", paths["targets"])
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    app.cached_sky_map.clear()
    app.load_target_summary.clear()
    yield
    app.cached_sky_map.clear()
    app.load_target_summary.clear()


def test_hidden_views_skip_charts_and_queries(local_catalogs: None, monkeypatch: pytest.MonkeyPatch) -> None:
    binning = Mock(wraps=app.compute_healpix_map)
    summary = Mock(wraps=app.summarize_target_systems)
    monkeypatch.setattr(app, "compute_healpix_map", binning)
    monkeypatch.setattr(app, "summarize_target_systems", summary)
    at = AppTest.from_string("import app; app.main()", default_timeout=10).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    assert not at.dataframe
    assert binning.call_count == 1

    assert summary.call_count == 0

    at.session_state["main_view"] = "2D All-Sky Map"
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    assert binning.call_count == 1  # Reuse the same shared sky map.
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get("download_button")) == 1

    at.session_state["main_view"] = "Target & Observation Explorer"
    at.run()
    assert not at.exception
    assert not at.get("plotly_chart")
    assert len(at.dataframe) == 2
    assert all(table_search_query(table.proto.arrow_data.data) for table in at.dataframe)
    unique_target_query = table_search_query(at.dataframe[-1].proto.arrow_data.data)
    assert unique_target_query["inputText"][-1]["paramName"] == "target_name"
    assert summary.call_count == 1
    assert binning.call_count == 1
    for catalog_view in ("Telescope Pointings (Visits)", "Individual Exposures", "Instrument Calibrations"):
        at.radio(key="tab3_table_view").set_value(catalog_view).run()
        assert not at.exception
        assert not at.get("plotly_chart")
        query = table_search_query(at.dataframe[-1].proto.arrow_data.data)
        assert query["inputText"][-1]["paramName"] == "obs_id"
        assert query["inputText"][-1]["values"] == ["n4qf18090"]
    at.text_input(key="tab3_search").set_value("missing").run()
    assert not at.exception
    assert not at.dataframe

    at.session_state["main_view"] = "About & Methodology"
    at.run()
    assert not at.exception
    assert not at.get("plotly_chart")
    assert not at.dataframe
    assert binning.call_count == 1

    at.session_state["main_view"] = "Target & Observation Explorer"
    at.run()
    assert not at.exception
    assert at.text_input(key="tab3_search").value == "missing"
    assert at.radio(key="tab3_table_view").value == "Instrument Calibrations"


def test_cache_refreshes_when_selected_table_changes(local_catalogs: None) -> None:
    def get_map():
        return app.cached_sky_map(
            "exposures",
            ("HST", "JWST"),
            ("deep_space",),
            (0.0, 100_000.0),
            "galactic",
            "exposure_hours",
            "linear",
            16,
            app.table_version("exposures"),
        )

    original = get_map()
    assert original is not None
    assert get_map() is original
    assert original.raw_map.sum() == 3.0
    assert not original.raw_map.flags.writeable
    assert not original.display_map.flags.writeable
    assert not original.target_labels.flags.writeable
    path = app._TABLE_PATHS["exposures"]
    source = pd.read_parquet(path)
    source["t_exptime"] *= 2
    source.to_parquet(path, index=False)
    updated = get_map()
    assert updated is not None
    assert updated is not original
    assert updated.raw_map.sum() == 6.0
