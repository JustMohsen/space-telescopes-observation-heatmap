# Performance and Streamlit deployment

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
