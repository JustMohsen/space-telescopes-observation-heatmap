"""Configuration, paths, and telescope catalog definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"

EXPOSURES_PARQUET_PATH = DATA_DIR / "exposures.parquet"
POINTINGS_PARQUET_PATH = DATA_DIR / "pointings.parquet"
TARGETS_PARQUET_PATH = DATA_DIR / "targets.parquet"
CALIBRATIONS_PARQUET_PATH = DATA_DIR / "calibrations.parquet"

SUPPORTED_TELESCOPES = {
    "HST": {
        "name": "Hubble Space Telescope (HST)",
        "obs_collection": "HST",
        "start_year": 1990,
        "end_year": 2026,
    },
    "JWST": {
        "name": "James Webb Space Telescope (JWST)",
        "obs_collection": "JWST",
        "start_year": 2021,
        "end_year": 2026,
    },
    "FUSE": {
        "name": "Far Ultraviolet Spectroscopic Explorer (FUSE)",
        "obs_collection": "FUSE",
        "start_year": 1999,
        "end_year": 2008,
    },
    "EUVE": {
        "name": "Extreme Ultraviolet Explorer (EUVE)",
        "obs_collection": "EUVE",
        "start_year": 1992,
        "end_year": 2002,
    },
    "GALEX": {
        "name": "Galaxy Evolution Explorer (GALEX)",
        "obs_collection": "GALEX",
        "start_year": 2003,
        "end_year": 2014,
    },
    "SWIFT": {
        "name": "Neil Gehrels Swift Observatory (SWIFT)",
        "obs_collection": "SWIFT",
        "start_year": 2004,
        "end_year": 2026,
    },
    "IUE": {
        "name": "International Ultraviolet Explorer (IUE)",
        "obs_collection": "IUE",
        "start_year": 1978,
        "end_year": 1997,
    },
    "KeplerFFI": {
        "name": "Kepler Space Telescope (Full-Frame Pointings)",
        "obs_collection": "KeplerFFI",
        "start_year": 2009,
        "end_year": 2014,
    },
    "K2FFI": {
        "name": "K2 Mission (Ecliptic Full-Frame Pointings)",
        "obs_collection": "K2FFI",
        "start_year": 2014,
        "end_year": 2019,
    },
    "TESS": {
        "name": "Transiting Exoplanet Survey Satellite (TESS)",
        "obs_collection": "TESS",
        "start_year": 2018,
        "end_year": 2026,
    },
}

# Fixed-sky categories shown on the 3D/2D Celestial Heatmaps (excludes moving Solar System objects)
HEATMAP_TARGET_CATEGORIES = {
    "deep_space": "Deep Space / Extragalactic",
    "milky_way": "Milky Way & Local Group",
    "exoplanet": "Exoplanets & Disks",
}

# Full set of categories available in the Tab 3 Observations & Targets Explorer
TARGET_CATEGORIES = {
    "deep_space": "Deep Space / Extragalactic",
    "milky_way": "Milky Way & Local Group",
    "exoplanet": "Exoplanets & Disks",
    "solar_system": "Solar System (Moving Targets)",
}

# Provenance pipelines that generate synthetic multi-filter or multi-visit mosaic co-adds
# on top of existing CALWF3/CALACS/CALSTIS/CALCOS exposures.
SYNTHETIC_MOSAIC_PROVENANCES = frozenset({"HAP-SVM", "HAP-MVM", "HASP", "HSLA"})

DEFAULT_NSIDE = 64
NSIDE_OPTIONS = [16, 32, 64, 128, 256]
GALACTIC_DISK_LAT_DEG = 10.0

# Approximate Galactic coordinate bounding boxes for the Large & Small Magellanic Clouds
# (l in [-180, 180] deg, b in [-90, 90] deg)
LMC_L_RANGE = (-90.0, -65.0)
LMC_B_RANGE = (-40.0, -25.0)
SMC_L_RANGE = (-65.0, -50.0)
SMC_B_RANGE = (-50.0, -38.0)
