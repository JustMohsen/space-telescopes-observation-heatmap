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

## Performance and Streamlit deployment

The app renders only the selected tab. Explorer searches and view changes no longer rebuild hidden charts or aggregate hidden tables. Explorer filters persist when switching tabs. This requires Streamlit 1.67 or later; the deployment requirements already pin that version.

Catalog queries project only needed columns and apply mission, category, date, and literal text filters in Arrow before creating pandas frames. Repeated text stays dictionary encoded, and temporary scan buffers use the system allocator. The shared cache holds a small target summary and at most two read-only sky maps, rather than full exposure/pointing tables and twelve serialized maps. Cache invalidation uses each selected file's own modification time and size. Catalogs download on demand, with atomic writes protected across sessions.

Interactive meshes contain at most 65,160 vertices. Numeric hover values stay in numeric arrays, separate from target strings, so Plotly can encode them efficiently. HEALPix calculations still use the full selected resolution; the interactive preview can merge nearby cells visually, with brighter cells taking precedence. The 720 × 360 PNG export provides a denser preview. Ecliptic transformations run in bounded chunks, and Explorer summaries use native reductions with text joins limited to displayed systems. PNG exports use isolated temporary files and serialized rendering across sessions.

Local measurements with the existing release catalogs, Python 3.12, and Streamlit's app test runner:

| Default HST + JWST view | Before | After |
| :--- | ---: | ---: |
| First app execution | 11.12 s | 0.90 s |
| Cached refresh | 10.11 s | 0.058 s |
| Chart JSON sent per refresh | 21.71 MiB | 3.16 MiB |
| Process peak memory during default startup | 1,064 MiB | 448 MiB |

A 65-run stress test covering all ten missions, all three metrics and frames, repeated resolution changes through NSIDE 256, PNG export, and every Explorer table peaked at approximately 791 MiB. Scientific pixel values, scaled maps, and target labels matched the original implementation for all nine metric/frame combinations on the real HST/JWST catalogs. These measurements exclude downloads, network transfer, and browser rendering; cloud performance and concurrent-session memory usage will vary.

Reproduce the measurements in a fresh process with local catalogs available:

```bash
uv run python scripts/profile_app.py
uv run python scripts/profile_app.py --stress
```

After pushing changes to the repository connected to Streamlit Community Cloud, reboot the app from its management menu to discard the previous process's caches. Inspect the cloud logs and repeat the interactions that previously crashed to verify the deployed environment.
