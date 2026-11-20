import streamlit as st

from dashboard.common import page_header

page_header("How it works", "The method in plain English, plus a glossary of every term used.")

st.subheader("The pipeline")
steps = [
    ("🛰️", "Collect satellite images", "165 monthly night-time light images of the UK (Apr 2012–Dec 2025), "
     "each pixel about 500 m across."),
    ("🧹", "Clean them", "Detect and drop months that are copies of December; dampen cloud noise; "
     "measure each pixel's true area."),
    ("🗺️", "Measure each area", "Total brightness, lit-up area and concentration of light for 10 regions "
     "and 361 local authorities, every month."),
    ("🏛️", "Add official data", "ONS regional GDP, UK monthly GDP and benefit-claimant counts, each "
     "tagged with the date it was actually published."),
    ("⏪", "Replay history", "For every quarter since 2016, build a forecast using only what was public "
     "5 weeks after the quarter ended, exactly as an analyst would have."),
    ("⚖️", "Compare fairly", "Run the same methods with and without satellite data and test whether "
     "the difference is bigger than chance."),
]
cols = st.columns(3)
for i, (icon, title, text) in enumerate(steps):
    with cols[i % 3].container(border=True, height=170):
        st.markdown(f"#### {icon} {i + 1}. {title}")
        st.caption(text)

st.subheader("Glossary")
GLOSSARY = {
    "Nowcast": "An estimate of the economy *right now* or the recent past, made before official "
               "figures are published.",
    "GDP / GVA": "Gross domestic product / gross value added: the total value of goods and services "
                 "an area produces. Regional quarterly figures here are GVA, which ONS uses as a proxy for GDP.",
    "Growth rate": "Percentage change from the previous quarter (or year).",
    "Growth vs. England & Wales average": "A region's growth minus the overall growth, which shows "
                                          "which regions are pulling ahead or falling behind.",
    "Percentage points (pts)": "The difference between two percentages. A forecast of 1.0% when the "
                               "truth is 1.5% is off by 0.5 pts.",
    "Average forecast error (MAE)": "On average, how far forecasts were from the official figure. Lower is better.",
    "80% range": "The band we expect the true figure to fall inside 8 times out of 10, learned from "
                 "our own past mistakes.",
    "Economic data": "Last published regional growth, UK monthly GDP and benefit-claimant counts.",
    "Satellite data": "Changes in night-time brightness, lit-up area and concentration of light.",
    "Long-run average": "A simple benchmark: always predict the region's historical average growth.",
    "Ridge / Elastic Net": "Linear statistical models that weigh many inputs while avoiding over-fitting.",
    "XGBoost / LightGBM": "Machine-learning models built from many small decision trees.",
    "Bridge equation": "A classic economists' method linking monthly indicators to quarterly GDP.",
    "Point-in-time / no look-ahead": "Every forecast uses only data that had actually been published "
                                     "by its forecast date. Our automated tests check this.",
    "Statistically significant": "A difference large enough that it is unlikely to be luck. A 95% "
                                 "range that crosses zero is *not* significant.",
    "SHAP": "A method that shows how much each input contributed to a model's prediction.",
    "Imputed / copied month": "A satellite month with no real observation, filled in by the data "
                              "provider with a scaled copy of December.",
    "ITL1 region": "The 12 UK statistical regions (we use the 9 English regions plus Wales).",
}
q = st.text_input("🔎 Search the glossary", placeholder="e.g. range, SHAP, nowcast")
for term, text in GLOSSARY.items():
    if not q or q.lower() in (term + text).lower():
        with st.expander(term):
            st.markdown(text)

st.subheader("Learn more")
st.markdown("- 📄 [Full research report](https://github.com/manichitumala/uk-economy-space-nowcasting/blob/main/reports/research_report.md)\n"
            "- 🔎 [Data audit](https://github.com/manichitumala/uk-economy-space-nowcasting/blob/main/docs/data_availability_audit.md)\n"
            "- 🧪 [Forecasting protocol](https://github.com/manichitumala/uk-economy-space-nowcasting/blob/main/docs/forecasting_protocol.md)")
