"""Unit tests for coordinate transformations, target classification, and ETL deduplication."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telescope_heatmap.coordinates import (
    add_galactic_coordinates,
    classify_targets,
    is_calibration_row,
    normalize_target_system,
)
from telescope_heatmap.etl import process_raw_observations


def test_galactic_center_coordinates() -> None:
    """Sagittarius A* (RA ~ 266.4168°, Dec ~ -29.0078°) should map to (l ~ 0°, b ~ 0°)."""
    df = pd.DataFrame({"s_ra": [266.41683], "s_dec": [-29.00781]})
    out = add_galactic_coordinates(df)
    assert abs(float(out["gal_l"].iloc[0])) < 0.1
    assert abs(float(out["gal_b"].iloc[0])) < 0.1


def test_exoplanet_not_confused_with_solar_system() -> None:
    """Targets tagged EXTRA-SOLAR PLANET or PROTOPLANETARY DISK must be classified as exoplanet, not solar_system."""
    df = pd.DataFrame(
        {
            "target_name": ["WASP-39B", "TW-HYA", "JUPITER", "COMET-P-2020", "M31-HALO", "NGC-104"],
            "target_classification": [
                "STAR;BROWN DWARF;EXTRA-SOLAR PLANET",
                "STAR;PRE-MAIN SEQUENCE STAR;PROTOPLANETARY DISK",
                "PLANET; JUPITER",
                "Solar System; Comet",
                "GALAXY;SPIRAL",
                "GLOBULAR CLUSTER",
            ],
            "obs_title": ["", "", "", "", "", ""],
            "mtFlag": [False, False, True, True, False, False],
            # Put all at high galactic latitude except NGC-104
            "s_ra": [217.3, 165.4, 4.1, 346.4, 10.68, 6.02],
            "s_dec": [-3.4, -34.7, 0.4, -6.7, 41.26, -72.08],
        }
    )
    df = add_galactic_coordinates(df)
    out = classify_targets(df)

    cats = dict(zip(out["target_name"], out["target_category"]))
    assert cats["WASP-39B"] == "exoplanet"
    assert cats["TW-HYA"] == "exoplanet"
    assert cats["JUPITER"] == "solar_system"
    assert cats["COMET-P-2020"] == "solar_system"
    assert cats["M31-HALO"] == "deep_space"
    assert cats["NGC-104"] == "milky_way"


def test_calibration_separation_and_deduplication() -> None:
    """Verify HAP synthetic mosaics are dropped, MIRI 12-channel IFU rows collapse to 1 exposure,
    mosaic tiles at different coordinates stay separate, and calibration rows go to calibrations table.
    """
    raw = pd.DataFrame(
        [
            # 1. Primary HST exposure Tile 1 (keep)
            {
                "obs_id": "hst_1_f336w",
                "obs_collection": "HST",
                "proposal_id": "16778",
                "instrument_name": "WFC3/UVIS",
                "filters": "F336W",
                "dataproduct_type": "image",
                "calib_level": 3,
                "provenance_name": "CALWF3",
                "target_name": "M31",
                "target_classification": "GALAXY;SPIRAL",
                "obs_title": "M31 Survey",
                "mtFlag": False,
                "s_ra": 10.69,
                "s_dec": 41.18,
                "t_min": 59945.1047,
                "t_max": 59945.1147,
                "t_exptime": 878.0,
            },
            # 2. Synthetic HAP detection stack at Tile 1 (must be dropped)
            {
                "obs_id": "hst_1_total",
                "obs_collection": "HST",
                "proposal_id": "16778",
                "instrument_name": "WFC3/UVIS",
                "filters": "detection",
                "dataproduct_type": "image",
                "calib_level": 3,
                "provenance_name": "HAP-SVM",
                "target_name": "M31",
                "target_classification": "GALAXY;SPIRAL",
                "obs_title": "M31 Survey",
                "mtFlag": False,
                "s_ra": 10.69,
                "s_dec": 41.18,
                "t_min": 59945.1047,
                "t_max": 59945.1147,
                "t_exptime": 2060.0,
            },
            # 3. Primary HST exposure Tile 2 at different coordinate (must be kept!)
            {
                "obs_id": "hst_2_f336w",
                "obs_collection": "HST",
                "proposal_id": "16778",
                "instrument_name": "WFC3/UVIS",
                "filters": "F336W",
                "dataproduct_type": "image",
                "calib_level": 3,
                "provenance_name": "CALWF3",
                "target_name": "M31",
                "target_classification": "GALAXY;SPIRAL",
                "obs_title": "M31 Survey",
                "mtFlag": False,
                "s_ra": 11.25,
                "s_dec": 41.85,
                "t_min": 59945.1708,
                "t_max": 59945.1808,
                "t_exptime": 878.0,
            },
            # 4 & 5. Two simultaneous JWST MIRI/IFU channels (must collapse into 1 exposure & 1 pointing)
            {
                "obs_id": "jw_ch1_short",
                "obs_collection": "JWST",
                "proposal_id": "2361",
                "instrument_name": "MIRI/IFU",
                "filters": "CH1-SHORT",
                "dataproduct_type": "cube",
                "calib_level": 3,
                "provenance_name": "CALJWST",
                "target_name": "2085HENAN",
                "target_classification": "Solar System; Asteroid",
                "obs_title": "Asteroid",
                "mtFlag": True,
                "s_ra": 16.3377,
                "s_dec": 2.4153,
                "t_min": 59945.5326,
                "t_max": 59945.5460,
                "t_exptime": 1187.7,
            },
            {
                "obs_id": "jw_ch1_long",
                "obs_collection": "JWST",
                "proposal_id": "2361",
                "instrument_name": "MIRI/IFU",
                "filters": "CH1-LONG",
                "dataproduct_type": "cube",
                "calib_level": 3,
                "provenance_name": "CALJWST",
                "target_name": "2085HENAN",
                "target_classification": np.nan,
                "obs_title": "Asteroid",
                "mtFlag": True,
                "s_ra": 16.3377,
                "s_dec": 2.4153,
                "t_min": 59945.5326,
                "t_max": 59945.5460,
                "t_exptime": 1187.7,
            },
            # 6. Calibration row (must be routed to calibrations table)
            {
                "obs_id": "hst_cal_1",
                "obs_collection": "HST",
                "proposal_id": "16000",
                "instrument_name": "WFC3/UVIS",
                "filters": "F606W",
                "dataproduct_type": "image",
                "calib_level": 2,
                "provenance_name": "CALWF3",
                "target_name": "GD153-CAL",
                "target_classification": "CALIBRATION;PHOTOMETRIC",
                "obs_title": "Photometric Cal",
                "mtFlag": False,
                "s_ra": 194.25,
                "s_dec": 22.03,
                "t_min": 59945.9,
                "t_max": 59945.91,
                "t_exptime": 60.0,
            },
        ]
    )

    res = process_raw_observations(raw)
    assert len(res.calibrations) == 1
    assert len(res.exposures) == 3  # 2 distinct M31 tiles + 1 collapsed MIRI IFU exposure
    assert len(res.pointings) == 3  # 2 distinct M31 tile pointings + 1 MIRI pointing
    assert len(res.targets) == 2  # M31 (HST) and 2085HENAN (JWST)


def test_normalize_target_system_aggregates_planets_moons_and_mosaics() -> None:
    df = pd.DataFrame(
        {
            "target_name": [
                "Saturn",
                "Titan",
                "ENCELADUS-PLUME",
                "Jupiter",
                "JUPITER-NPOLE",
                "IOTORUS-NHFLYBY",
                "M31-B25-F01-UVIS",
                "NGC224",
                "Mrk 421",
                "MRK421",
            ]
        }
    )
    systems = normalize_target_system(df).tolist()
    assert systems[0] == "Saturn System (Saturn & Moons)"
    assert systems[1] == "Saturn System (Saturn & Moons)"
    assert systems[2] == "Saturn System (Saturn & Moons)"
    assert systems[3] == "Jupiter System (Jupiter & Moons)"
    assert systems[4] == "Jupiter System (Jupiter & Moons)"
    assert systems[5] == "Jupiter System (Jupiter & Moons)"
    assert systems[6] == "M31 (Andromeda Galaxy)"
    assert systems[7] == "M31 (Andromeda Galaxy)"
    assert systems[8] == "MRK-421"
    assert systems[9] == "MRK-421"


def test_edge_case_classifications_and_missing_metadata() -> None:
    """Ensure Centaurus A, Alpha Centauri, iotaCep, protostars with 'comets' in obs_title,
    Type II supernovae, and EUVE stars lacking target_classification are never misclassified.
    """
    df = pd.DataFrame(
        {
            "obs_collection": ["JWST", "FUSE", "FUSE", "JWST", "JWST", "EUVE", "EUVE"],
            "target_name": [
                "CENA-NUC",
                "HD128621",
                "iotaCep",
                "HOPS68-THERM",
                "2022ACKO",
                "GJ 205",
                "PG 1658+440",
            ],
            "target_classification": [
                "Galaxy; Active galactic nuclei",
                "STAR;CORONA;K V-IV",
                "",
                "ISM; Protostars",
                "Star; Type II supernovae",
                "",
                "",
            ],
            "obs_title": [
                "NIRSpec Observations of Centaurus-A: a GTO program proposal",
                "FUSE Observations of alpha Centauri A",
                "Activity and Fluorescence Among Nearby Clump Giants",
                "Disentangling complex organic molecules from molecular clouds to comets.",
                "Dust, Mass Loss and Explosions of Massive Stars in the MIR",
                "",
                "",
            ],
            "mtFlag": [False, False, False, False, False, False, False],
            "s_ra": [201.36, 219.90, 342.42, 83.85, 49.91, 82.86, 254.99],
            "s_dec": [-43.02, -60.83, 66.20, -5.09, -15.02, -3.68, 44.02],
            "t_exptime": [1000.0] * 7,
        }
    )
    df = add_galactic_coordinates(df)
    out = classify_targets(df)
    cats = dict(zip(out["target_name"], out["target_category"]))

    assert cats["CENA-NUC"] == "deep_space"
    assert cats["HD128621"] == "milky_way"
    assert cats["iotaCep"] == "milky_way"
    assert cats["HOPS68-THERM"] == "milky_way"
    assert cats["2022ACKO"] == "deep_space"
    assert cats["GJ 205"] == "milky_way"
    assert cats["PG 1658+440"] == "milky_way"
