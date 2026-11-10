# 🔭 Space Telescopes Observation Heatmap

Interactive **3D Celestial Sphere**, **2D Mollweide All-Sky Heatmaps**, and **Target & Observation Explorer** showing where **10 major space observatories** have pointed and stared across their entire mission lifetimes (**1978–2026**).

See [telescope_heatmap_blueprint.md](./telescope_heatmap_blueprint.md) for the full system architecture, technical stack, classification methodology, and implementation roadmap.

---

## ✨ Supported Space Observatories (Full Mission Catalog: 1978–2026)

| Code | Observatory | Wavelength / Regime | Mission Coverage |
| :--- | :--- | :--- | :--- |
| **`HST`** | Hubble Space Telescope | UV / Optical / Near-IR | 1990–2026 |
| **`JWST`** | James Webb Space Telescope | Near-IR / Mid-IR | 2022–2026 |
| **`IUE`** | International Ultraviolet Explorer | Ultraviolet | 1978–1996 |
| **`SWIFT`** | Neil Gehrels Swift Observatory | UV / Optical / X-Ray | 2004–2026 |
| **`GALEX`** | Galaxy Evolution Explorer | Far-UV / Near-UV | 2003–2013 |
| **`EUVE`** | Extreme Ultraviolet Explorer | Extreme UV | 1992–2001 |
| **`FUSE`** | Far Ultraviolet Spectroscopic Explorer | Far-UV | 1999–2007 |
| **`KeplerFFI`** | Kepler Full-Frame Images | Optical Transit Survey | 2009–2013 |
| **`TESS`** | Transiting Exoplanet Survey Satellite (FFIs) | Optical All-Sky Survey | 2018–2026 |
| **`K2FFI`** | K2 Full-Frame Images | Ecliptic Optical Survey | 2014–2018 |

---

## 🚀 Key Features

- **🌐 3D Celestial Sphere:** Interactive rotatable 3D globe with explicit **Galactic Center** $(l=0^\circ, b=0^\circ)$ and **Galactic Plane** $(b=0^\circ)$ overlays, hover coordinates, and top observed targets per sky cell.
- **🗺️ 2D All-Sky Projection:** Interactive Plotly projection + publication-quality **Matplotlib Mollweide** all-sky heatmap with one-click browser `.png` download.
- **📋 Target & Observation Explorer:**
  - **Aggregated by Target System (`target_system`):** Groups observations of planetary systems (e.g., *Jupiter System*, *Saturn System*, *TRAPPIST-1*), galaxies, and calibration fields into canonical parent systems.
  - **Detailed Target & Pointing Breakdown:** Searchable by target name, proposal ID, instrument, category, telescope, and observation date window.
- **📅 Observation Time Period Sliders:** Filter both the 3D/2D heatmaps and the Target Explorer to any custom date window (`1978–2026`).
- **🌓 Theme Support:** Full **Dark**, **Light**, and **System** theme adaptation across all 3D globes, 2D Plotly charts, and Matplotlib Mollweide exports.

---

## 🛠️ Quickstart

### 1. Environment Setup (`uv` + Python 3.12)
```bash
uv sync --extra dev
uv run pre-commit install
```

### 2. Run the Offline MAST ETL Pipeline
Fetch and build the 4 normalized Parquet tables (`exposures.parquet`, `pointings.parquet`, `targets.parquet`, `calibrations.parquet`):
```bash
# Full mission backfill across all 10 space telescopes (resumable, cached in data/chunks/)
uv run python scripts/run_etl.py --full-mission --workers 6

# Or fetch a specific date range / subset of telescopes
uv run python scripts/run_etl.py --telescopes HST JWST --start-date 2023-01-01 --end-date 2023-02-01
```

### 3. Launch the Interactive Web App
```bash
uv run streamlit run app.py
```

### 4. Code Quality & Tests (`pre-commit`, `isort`, `black`, `mypy`, `pytest`)
```bash
uv run pre-commit run --all-files
uv run pytest
```
