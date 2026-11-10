"""MAST links must identify real archive metadata, including moving targets."""

import json
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from telescope_heatmap.archive_links import MAST_PORTAL, mast_search_url


def search_query(url: str | None) -> dict:
    assert url is not None and url.startswith(MAST_PORTAL + "?")
    return json.loads(parse_qs(urlsplit(url).query)["searchQuery"][0])


@pytest.mark.parametrize(
    ("mission", "identifier"),
    [("HST", "n4qf18090"), ("JWST", "jw01442-o001_t001_nircam_f322w2-grismr-subgrism128"), ("IUE", "lwr01024")],
)
def test_dataset_link_uses_mission_and_original_identifier(mission: str, identifier: str) -> None:
    query = search_query(mast_search_url([mission], obs_id=identifier, target_names=["ANY (1.0, 2.0)"]))
    assert query["service"] == "CAOMFILTERED"
    filters = {item["paramName"]: item["values"] for item in query["inputText"]}
    assert filters == {"obs_collection": [mission], "obs_id": [identifier]}


def test_raw_names_survive_encoding_without_resolving_solar_system_positions() -> None:
    names = ["Titan", "A, B & C/+?#%", "Titan"]
    query = search_query(mast_search_url(["HST", "JWST", "HST"], target_names=names, ra=1, dec=2))
    filters = {item["paramName"]: item["values"] for item in query["inputText"]}
    assert filters == {"obs_collection": ["HST", "JWST"], "target_name": names[:2]}


def test_generated_field_labels_search_their_recorded_position() -> None:
    query = search_query(mast_search_url(["HST"], target_names=["ANY (53.2, -27.8)"], ra=100, dec=50))
    assert query == {"service": "CAOMDB", "inputText": "53.200000 -27.800000 r=0.1d"}
    query = search_query(mast_search_url(["TESS"], target_names=["TESS FFI (360.0, 90.0)"]))
    assert query["inputText"] == "0.000000 90.000000 r=0.1d"


def test_missing_metadata_never_links_to_an_unfiltered_archive() -> None:
    assert mast_search_url(["HST"], obs_id=pd.NA, target_names=[pd.NA]) is None
    assert mast_search_url([], obs_id="n4qf18090") is None
    assert mast_search_url(["HST"], ra=float("nan"), dec=2) is None
    assert mast_search_url(["HST"], ra=20, dec=91) is None
    query = search_query(mast_search_url(["HST"], ra=20, dec=-90))
    assert query["inputText"] == "20.000000 -90.000000 r=0.1d"
