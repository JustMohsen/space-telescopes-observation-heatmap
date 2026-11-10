"""Measure app rerun times, chart payloads and process peak memory with local data.

Run in a fresh process for each comparison: uv run python scripts/profile_app.py
Add --stress to exercise all missions, metrics, frames and resolutions repeatedly.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)

    def measure(label: str) -> None:
        start = time.perf_counter()
        at.run()
        if at.exception:
            raise RuntimeError([exc.message for exc in at.exception])
        payload_bytes = sum(len(chart.proto.spec.encode()) for chart in at.get("plotly_chart"))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = rss / (2**20 if sys.platform == "darwin" else 1024)
        print(
            json.dumps(
                {
                    "scenario": label,
                    "seconds": round(time.perf_counter() - start, 3),
                    "peak_rss_mib": round(peak_mib, 1),
                    "chart_mib": round(payload_bytes / 2**20, 2),
                }
            ),
            flush=True,
        )

    measure("default cold")
    measure("default warm")
    for view in ("2D All-Sky Map", "Target & Observation Explorer", "About & Methodology"):
        at.session_state["main_view"] = view
        measure(view)

    if args.stress:
        for widget in at.sidebar.checkbox:
            if widget.key and str(widget.key).startswith("mission_"):
                widget.set_value(True)
        at.session_state["main_view"] = "3D Celestial Sphere"
        for repeat in range(2):
            for frame in at.sidebar.radio[2].options:
                at.sidebar.radio[2].set_value(frame)
                for metric in at.sidebar.radio[0].options:
                    at.sidebar.radio[0].set_value(metric)
                    for nside in (16, 64, 256):
                        at.sidebar.select_slider[0].set_value(nside)
                        measure(f"stress {repeat + 1}: {frame}, {metric}, nside={nside}")
        at.session_state["main_view"] = "2D All-Sky Map"
        measure("all missions 2D at nside=256")
        at.button[0].click()
        measure("high-resolution PNG export")
        at.session_state["main_view"] = "Target & Observation Explorer"
        measure("all missions explorer")
        for view in ("Telescope Pointings (Visits)", "Individual Exposures", "Instrument Calibrations"):
            at.radio(key="tab3_table_view").set_value(view)
            measure(f"all missions explorer: {view}")


if __name__ == "__main__":
    main()
