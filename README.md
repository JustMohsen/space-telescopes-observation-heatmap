# Space Telescopes Observation Heatmap

Interactive 3D globe and 2D Mollweide all-sky heatmaps showing where major space observatories (**Hubble**, **JWST**, and more) have pointed throughout their mission lifetimes.

See [telescope_heatmap_blueprint.md](./telescope_heatmap_blueprint.md) for the full system architecture, technical stack, and implementation roadmap.

## Quickstart

### 1. Environment Setup (`uv` + Python 3.12)
```bash
uv sync --extra dev
```

### 2. Explore MAST Metadata (Step 1)
```bash
uv run python scripts/explore_mast.py
```

### 3. Run the Offline ETL Pipeline (Step 2)
```bash
uv run python scripts/run_etl.py
```

### 4. Launch the Interactive Web App (Step 4)
```bash
uv run streamlit run app.py
```
