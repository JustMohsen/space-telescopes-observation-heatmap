"""Streamlit web application entrypoint for the Space Telescopes Observation Heatmap."""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Space Telescopes Observation Heatmap",
        page_icon="🔭",
        layout="wide",
    )
    st.title("🔭 Space Telescopes Observation Heatmap")
    st.info("Run the offline ETL pipeline (`uv run python scripts/run_etl.py`) to populate observation metadata.")


if __name__ == "__main__":
    main()
