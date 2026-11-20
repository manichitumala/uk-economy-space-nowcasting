import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GREY, INK2, ORANGE, callout, how_to_read, model_label,
                              page_header, results, style, table)

R = results()
page_header("Does satellite data help?",
            "A fair head-to-head: the same forecasting methods, with and without satellite data.")
callout("<b>Verdict: no measurable improvement.</b> For every forecasting method we tried, adding "
        "satellite data either made no proven difference (bar crosses zero) or made forecasts "
        "slightly <i>worse</i>. In no case did it make them reliably better.")

PAIRS = {"ridge_A_econ": "Ridge regression", "enet_A_econ": "Elastic Net", "xgb_A_econ": "XGBoost",
         "lgbm_A_econ": "LightGBM", "bridge_conventional": "Bridge equation"}


def forest(tests: list[dict], title: str) -> go.Figure:
    t = pd.DataFrame(tests)
    t = t[t["benchmark"].isin(PAIRS)].copy()
    t["method"] = t["benchmark"].map(PAIRS)
    sig = (t["ci_low"] > 0) | (t["ci_high"] < 0)
    t["verdict"] = np.where(~sig, "No proven difference",
                            np.where(t["improvement_pct"] > 0, "Satellite helps", "Satellite hurts"))
    colours = {"No proven difference": GREY, "Satellite helps": BLUE, "Satellite hurts": ORANGE}
    fig = go.Figure()
    for v, g in t.groupby("verdict"):
        fig.add_trace(go.Scatter(
            x=g["improvement_pct"], y=g["method"], mode="markers", name=v,
            marker=dict(size=14, color=colours[v]),
            error_x=dict(type="data", symmetric=False, array=g["ci_high"] - g["improvement_pct"],
                         arrayminus=g["improvement_pct"] - g["ci_low"], color=colours[v], thickness=3, width=0),
            customdata=np.c_[g["ci_low"], g["ci_high"], g["dm_p_abs"]],
            hovertemplate="%{y}<br>Change in accuracy: %{x:+.1f}%<br>95% range: %{customdata[0]:+.1f}% to "
                          "%{customdata[1]:+.1f}%<br>p-value: %{customdata[2]:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK2, line_width=1.5)
    fig.update_layout(title=title, xaxis_title="← satellite makes it worse · accuracy change (%) · satellite makes it better →")
    return style(fig, 360, legend_bottom=True)


tab_q, tab_a = st.tabs(["📅 Quarterly growth · 10 regions", "🏘️ Yearly growth · 361 local areas"])
with tab_q:
    covid = st.toggle("Include the COVID period (2020–mid 2021)", value=False, key="cq")
    res = R["track_b"]["target_all" if covid else "target_ex_covid"]
    st.plotly_chart(forest(res["tests"], "Effect of adding satellite data, by forecasting method"), width="stretch")
    if covid:
        st.caption("With COVID included, satellite data makes the bridge equation clearly *worse*: "
                   "the spring-2020 crash happened in months the satellite cannot see.")
with tab_a:
    res_a = R["track_a"]["target_all"]
    st.plotly_chart(forest(res_a["tests"], "Effect of adding satellite data, by forecasting method"), width="stretch")
    st.caption("Ridge appears to gain ~3%, but the whole gain comes from one year (2016) when the model "
               "had only two years of history. Once that is accounted for, the difference disappears.")
how_to_read("each dot is the change in accuracy from adding satellite data; the bar is the range "
            "the true effect plausibly lies in (95%). If the bar crosses the zero line, we cannot "
            "say satellite data helps.")

st.divider()
st.subheader("So what *can* night lights tell us?")
lev = next(r for r in R["econometrics"]["levels"] if r["year"] == 2019)
ld = next(r for r in R["econometrics"]["long_differences"] if r["from"] == 2015)
tw = R["econometrics"]["twfe"]["light_only"]
c = st.columns(3)
with c[0].container(border=True):
    st.markdown("#### ✅ Location")
    st.metric("Explained by brightness", f"{lev['r2']:.0%}")
    st.caption("Brighter local areas have bigger economies. Strong, consistent every year.")
with c[1].container(border=True):
    st.markdown("#### 〰️ Long-run trends")
    st.metric("Explained by brightness", f"{ld['r2']:.0%}")
    st.caption("Areas that got brighter grew a little faster over 2015–2024. Real, but weak.")
with c[2].container(border=True):
    st.markdown("#### ❌ Short-run growth")
    st.metric("Explained by brightness", f"{tw['within_r2']:.0%}")
    st.caption("Year-to-year changes in brightness say nothing about year-to-year growth.")

fa = table("fa")
L = table("lights")
lad = L[(L["level"] == "lad") & (L["valid_share"] >= 0.5) & (L["month"].dt.year == 2019)]
lv = lad.groupby("code")["total_radiance"].mean().rename("light")
x = fa[fa["year"] == 2019].set_index("code").join(lv).dropna(subset=["light"])
fig = px.scatter(x.reset_index(), x="light", y="gdp_cp_level", hover_name="name", log_x=True, log_y=True,
                 trendline="ols", trendline_options=dict(log_x=True, log_y=True),
                 trendline_color_override=INK2,
                 labels={"light": "Total night-time light, 2019 (log scale)", "gdp_cp_level": "GDP 2019, £m (log scale)"})
fig.update_traces(marker=dict(color=BLUE, size=8, opacity=0.6), selector=dict(mode="markers"))
fig.update_layout(title="Brighter places have bigger economies (each dot = one local authority)")
st.plotly_chart(style(fig, 460), width="stretch")

with st.expander("🔬 Technical details: full results table and statistical tests"):
    track = st.radio("Track", ["Quarterly", "Yearly"], horizontal=True)
    key = "track_b" if track == "Quarterly" else "track_a"
    sample = st.radio("Sample", ["target_ex_covid", "target_all", "target_rel_ex_covid", "target_rel_all"],
                      horizontal=True, format_func=lambda s: {
                          "target_ex_covid": "Growth, excl. COVID", "target_all": "Growth, all",
                          "target_rel_ex_covid": "vs. average, excl. COVID", "target_rel_all": "vs. average, all"}[s])
    mt = pd.DataFrame(R[key][sample]["metrics"])
    mt.insert(0, "Model", mt["model"].map(model_label))
    st.dataframe(mt.drop(columns=["model", "group"]).round(3), hide_index=True, width="stretch")
    tt = pd.DataFrame(R[key][sample]["tests"])
    tt.insert(0, "Comparison", tt["benchmark"].map(model_label) + "  →  " + tt["challenger"].map(model_label))
    st.dataframe(tt.drop(columns=["benchmark", "challenger"]).round(3), hide_index=True, width="stretch")
    st.caption("mae = mean absolute error (percentage points); cov80 = share of outcomes inside the 80% "
               "range; dm_p = Diebold–Mariano test p-value; CI = 95% block-bootstrap interval.")
