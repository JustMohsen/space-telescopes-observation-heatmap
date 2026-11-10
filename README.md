# Space Telescopes Observation Heatmap

Explore where ten space observatories have pointed across a catalog spanning 1978–2026, through an interactive celestial globe, all-sky maps, and an observation explorer.

**[Open the live app](https://space-telescopes-observations-heatmap.streamlit.app/)**

## Why I built this

I love maps, graphs, and visualizations, like many other space and science enthusiasts. One day, the Hubble Ultra Deep Field image came across my feed, and I thought: I really want to see a heatmap of where we have been pointing our space telescopes. The story I remember about the first Deep Field image (which I was not sure how accurate it was) is that a proposal came up to point the Hubble Space Telescope at an empty patch of sky that we hadn't studied before to see what was out there, and there was resistance to wasting precious telescope time. Then that first Deep Field image came back and changed our estimate of the number of galaxies in the universe by a factor of 10—aside from the visual beauty and the sense of awe it brought.

So I thought: I want to see where in the sky we have been exploring and pointing our telescopes. I searched a lot and couldn't find anything resembling what I was looking for, so I decided to take on the project of building a heatmap to show where we have been pointing our telescopes. I planned the project and thought about how to represent it, the coordinate systems, and the different targets. Solar System targets are moving targets that don't make sense on a fixed-sky map; we expect galactic targets to be denser along the Galactic Plane, whereas deep-space targets should be denser away from the Galactic Plane to avoid an obstructed view.

A few months later, I finally started working on this. I built it in a few days, which was possible thanks to AI. It is not the most polished, but I am really happy with it and went through many iterations. I also learned that there is a bias toward the Northern Galactic Pole—partly because most historical ground-based sky surveys were built in Earth's northern hemisphere (which faces the North Galactic Pole and the Virgo Supercluster), and we try to point our space telescopes at the same targets for follow-up and comparison, and partly because the Southern Galactic Hemisphere contains the Large and Small Magellanic Clouds, which obstruct a chunk of the southern extragalactic sky. Pointing space telescopes at the same targets observed by ground-based telescopes within their observable regions is also the reason we see a dense belt of observations along the equator in the Equatorial reference frame.

*I built this project with AI assistance: AI wrote most of the text (except for this motivation section) and created the design, while I focused on the technical and scientific structure, iterative cleanup, functional testing, and checking the accuracy of the coordinate systems, telescope information, observation categories, and target classifications.*

## Explore the sky

- **3D celestial sphere:** Rotate and zoom the heatmap, inspect coordinates and target names, and overlay the Galactic and ecliptic planes.
- **2D all-sky maps:** Explore an interactive map and export a Mollweide PNG.
- **Three coordinate frames:** Switch between Galactic, equatorial, and ecliptic coordinates.
- **Three metrics:** Compare exposure duration, telescope visits, and unique catalog targets.
- **Observation explorer:** Search target names, descriptions, classifications, and program titles; inspect systems, visits, exposures, and calibration observations; open their images and data in MAST.
- **Filters and display:** Choose observatories, observation dates, target categories, grid detail, and light or dark appearance.

Moving Solar System targets appear in the observation explorer and are excluded from the fixed-sky heatmaps. The maps bin recorded pointing locations; colored cells do not represent the exact instrument image footprints.

**MAST links:** “View in MAST” opens a retained archive dataset for visits, exposures, and calibrations, or a raw target-name search for unique targets. “Search MAST” in the system table searches the listed example target names (up to six), rather than every observation in that system. Generic field labels use a nearby-sky search within 0.1° across missions. Archive results may extend beyond the app's filters; image previews and public downloads depend on the dataset. Links are built locally for displayed rows, without additional archive requests or image downloads in the app.

> **Map detail:** To keep memory use low and updates responsive, interactive maps use a capped display grid. Fine and Ultra-Fine change the underlying HEALPix bins but may look similar. Calculations still use the selected binning; PNG exports use a denser display grid.

## Observatory coverage

| Code | Observatory | Wavelength / Regime | Mission Coverage |
| :--- | :--- | :--- | :--- |
| `HST` | Hubble Space Telescope | UV / Optical / Near-IR | 1990–2026 |
| `JWST` | James Webb Space Telescope | Near-IR / Mid-IR | 2022–2026 |
| `IUE` | International Ultraviolet Explorer | Ultraviolet | 1978–1996 |
| `SWIFT` | Neil Gehrels Swift Observatory | UV / Optical / X-Ray | 2004–2026 |
| `GALEX` | Galaxy Evolution Explorer | Far-UV / Near-UV | 2003–2013 |
| `EUVE` | Extreme Ultraviolet Explorer | Extreme UV | 1992–2001 |
| `FUSE` | Far Ultraviolet Spectroscopic Explorer | Far-UV | 1999–2007 |
| `KeplerFFI` | Kepler Full-Frame Images | Optical Transit Survey | 2009–2013 |
| `TESS` | Transiting Exoplanet Survey Satellite (FFIs) | Optical All-Sky Survey | 2018–2026 |
| `K2FFI` | K2 Full-Frame Images | Ecliptic Optical Survey | 2014–2018 |

Coverage reflects the bundled catalog snapshot and each mission's observation history. It is not a live archive feed. The app uses observation metadata from [MAST](https://archive.stsci.edu/), rather than downloading astronomical image files.

## Run locally

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run streamlit run app.py
```

The app downloads the required precomputed Parquet tables from the project's GitHub release on demand. You can explore it without rebuilding the archive catalogs.

<details>
<summary>Rebuild the catalogs from MAST</summary>

Run the offline ETL pipeline to fetch, classify, and deduplicate archive metadata. Downloads are cached and resumable.

```bash
# Full mission backfill for all ten supported observatories
uv run python scripts/run_etl.py --full-mission --workers 6 \
  --missions HST JWST IUE SWIFT GALEX EUVE FUSE KeplerFFI K2FFI TESS

# A smaller mission and date selection
uv run python scripts/run_etl.py --missions HST JWST \
  --start-date 2023-01-01 --end-date 2023-02-01
```

The pipeline produces exposure, pointing, target, and calibration Parquet tables.

</details>

## Development

```bash
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
```

## More about the project

- [Architecture and scientific methodology](./telescope_heatmap_blueprint.md)
- [Performance measurements and deployment notes](./docs/performance.md)
- [JWST catalog and sky sparsity audit](./docs/jwst-catalog-audit.md)
