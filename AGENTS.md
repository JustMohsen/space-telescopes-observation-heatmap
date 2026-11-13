# Working on this repository

This is a Python 3.12 / Streamlit app for exploring space-telescope observation metadata from MAST. Its maps show recorded pointing locations and aggregate exposure time or catalog counts. The catalogs are release snapshots, not a live archive feed.

## Find the right context

- [README.md](./README.md): current features, personal motivation, setup, licensing, and future work.
- [Performance notes](./docs/performance.md): read before changing rendering, catalog queries, caching, or exports. These explain the memory fixes and profiling workflow.
- [JWST audit](./docs/jwst-catalog-audit.md): read when investigating sparse maps, row counts, or archive completeness. Its measurements apply to the dated local snapshot.
- [Original blueprint](./telescope_heatmap_blueprint.md): design rationale and implementation history. Parts predate the current architecture; check implementation and tests for current behavior, and primary mission documentation for scientific claims.

## Code map

| Area | Start here |
| --- | --- |
| UI, lazy tabs, downloads, and shared caches | [app.py](./app.py) |
| Projected/filtered Arrow reads and system summaries | [catalog.py](./src/telescope_heatmap/catalog.py) |
| Offline MAST retrieval, deduplication, and table construction | [etl.py](./src/telescope_heatmap/etl.py), [run_etl.py](./scripts/run_etl.py) |
| Coordinate transformations, categories, and target aliases | [coordinates.py](./src/telescope_heatmap/coordinates.py) |
| Scientific HEALPix values and bounded display sampling | [binning.py](./src/telescope_heatmap/binning.py) |
| Plotly maps and static PNG exports | [viz.py](./src/telescope_heatmap/viz.py) |
| External MAST search URLs | [archive_links.py](./src/telescope_heatmap/archive_links.py) |
| Shared story, disclosures, notes, and related links | [content.py](./src/telescope_heatmap/content.py) |
| Mission definitions, paths, and resolution options | [config.py](./src/telescope_heatmap/config.py) |

## Scientific constraints

- Preserve the metric-to-table mapping: exposure hours use `exposures`, visit counts use `pointings`, and catalog target counts use `targets`. Exposure durations are stored in seconds and converted to hours for binning. Observation dates use MJD.
- RA/Dec are equatorial coordinates in degrees. Use the existing Astropy transformations for Galactic and barycentric true ecliptic frames; verify orientation and coordinate changes with the coordinate/binning tests.
- Moving Solar System targets belong in the Explorer, but are excluded from fixed-sky heatmaps. Calibrations remain separate from science maps. Exoplanets, disks, and planetary nebulae must not be mistaken for moving Solar System objects.
- ETL excludes synthetic composites and groups simultaneous instrument outputs using time and position keys. These are scientific heuristics, not independently proven completeness guarantees. When changing them, check both exposure-time accounting and retention of distinct mosaic tiles using [ETL tests](./tests/test_etl_and_coordinates.py).
- Targets are catalog entries grouped by mission and raw target name; system summaries combine aliases and related targets. Do not interpret either as an exact census of distinct physical objects.
- Occupied HEALPix cells are not instrument footprints or the fraction of sky imaged. Keep scientific pixel values separate from display scaling and mesh sampling. Distinguish snapshot validation from live archive completeness.

## Memory and interaction constraints

- Keep work conditional on the active tab (`st.tabs(..., on_change="rerun")` and each tab's `.open`). Hidden views must not build charts or aggregate tables.
- Use projected, filtered `read_catalog` queries before materializing pandas frames. Preserve dictionary-encoded repeated strings and bound formatting/text joins to displayed rows. Avoid caching full catalogs or unbounded filter combinations; shared cached maps and summaries are read-only, and invalidation uses the selected table's version.
- The interactive mesh is capped at 360 × 181 vertices; PNG exports use a denser 720 × 360 grid. Fine and Ultra-Fine still change scientific HEALPix bins even when previews look similar. The cap is an intentional responsiveness tradeoff: measure memory, payload size, and refresh time before changing it.
- Keep Plotly numeric hover arrays separate from string labels, and keep coordinate transforms chunked. Preserve atomic/locked downloads and isolated temporary export files with serialized Matplotlib rendering across sessions.
- Build MAST links locally for visible rows. Avoid per-row archive requests or raw image downloads during interactive reruns. `obs_id` is a mission-specific identifier; MAST's numeric product-group `obsid` is different. Use the existing link builder: canonical system names and generated field labels are not necessarily valid archive target names.

## Development and verification

Run from the repository root using relative paths:

```bash
uv sync --extra dev
uv run streamlit run app.py
uv run pytest
uv run pre-commit run --all-files
```

Use [tests/](./tests/) fixtures for ordinary development; a full mission backfill is a separate data-refresh task. For code changes, run relevant tests, including [app tests](./tests/test_app.py) for tab/filter behavior, [catalog tests](./tests/test_catalog.py) for query changes, and [binning/visualization tests](./tests/test_binning_and_viz.py) for map changes. For documentation-only edits, check links and consistency; runtime tests are unnecessary unless behavior changes.

For performance-sensitive changes, use `uv run python scripts/profile_app.py` in a fresh process with local catalogs; add `--stress` when broader scenarios are relevant. Report results with their measurement conditions, not as a cloud performance guarantee. See [performance notes](./docs/performance.md).

Run pre-commit before a requested commit/push. Keep downloaded catalogs and generated outputs out of Git. Virtual environments and installed hooks are path-dependent; after moving a checkout, recreate the local environment as needed and reinstall hooks with `uv run pre-commit install`.

## Public content and documentation

Keep shared public wording in [content.py](./src/telescope_heatmap/content.py) consistent with matching README sections. Preserve the author's personal motivation and voice unless asked to edit it. Retain the AI disclosure and the distinction between the MIT-licensed project and third-party data/image terms.

Update this guide when architecture or scientific assumptions change; keep detailed procedures in the linked docs. The image browser and image-curation project are future work, not current app capabilities.
