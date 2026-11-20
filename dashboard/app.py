"""UK Economy from Space - interactive dashboard.

Run from the repository root:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `dashboard.*` / `src.*`

st.set_page_config(page_title="UK Economy from Space", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

pages = {
    "Overview": [st.Page("views/home.py", title="The story in 60 seconds", icon="🏠", default=True)],
    "Explore": [
        st.Page("views/outlook.py", title="Regional outlook", icon="🗺️"),
        st.Page("views/region_card.py", title="Region report card", icon="📍"),
        st.Page("views/satellite.py", title="Satellite explorer", icon="🛰️"),
    ],
    "Evidence": [
        st.Page("views/verdict.py", title="Does satellite data help?", icon="⚖️"),
        st.Page("views/track_record.py", title="Forecast track record", icon="📈"),
        st.Page("views/drivers.py", title="What drives the forecasts", icon="🔍"),
    ],
    "Learn": [
        st.Page("views/data_quality.py", title="Data quality checks", icon="🧪"),
        st.Page("views/how_it_works.py", title="How it works & glossary", icon="📘"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🛰️ UK Economy from Space")
    st.caption("Testing whether night-time satellite images can give early estimates "
               "of UK regional economic growth.")
    st.divider()
    st.caption("**Data:** ONS & Nomis (Open Government Licence) · HKU/NOAA VIIRS night lights "
               "(CC BY 4.0)  \n**Code:** [GitHub](https://github.com/manichitumala/uk-economy-space-nowcasting)")

nav.run()
