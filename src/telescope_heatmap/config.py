"""Configuration, paths, and telescope catalog definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
PARQUET_PATH = DATA_DIR / "telescope_metadata.parquet"

SUPPORTED_TELESCOPES = {
    "HST": {
        "name": "Hubble Space Telescope (HST)",
        "obs_collection": "HST",
        "start_year": 1990,
    },
    "JWST": {
        "name": "James Webb Space Telescope (JWST)",
        "obs_collection": "JWST",
        "start_year": 2021,
    },
}

DEFAULT_NSIDE = 64
GALACTIC_DISK_LAT_DEG = 10.0
