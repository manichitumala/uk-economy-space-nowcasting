import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (AQUA, BLUE, GREY, INK, ORANGE, REGION_CODE, REGIONS, SEQ,
                              how_to_read, page_header, quarter_axis, style, table)

page_header("Forecast track record",
            "How our forecasts would have performed if we had made them in real time since 2016.")
st.caption("Every past forecast below was produced using only the data that was public at that moment "
           "(about 5 weeks after each quarter ended), then compared with the official figure.")

APPROACH = {"historical_mean": ("Long-run average", GREY), "ridge_A_econ": ("Economic data", BLUE),
            "ridge_C_econ_sat": ("Economic + satellite", ORANGE),
            "lgbm_C_econ_sat": ("Machine learning + satellite", AQUA)}
c1, c2 = st.columns([2, 1])
pick = c1.pills("Approaches to compare", [v[0] for v in APPROACH.values()],
                default=["Economic data", "Economic + satellite"], selection_mode="multi")
covid = c2.toggle("Include COVID period", value=False)
chosen = [k for k, v in APPROACH.items() if v[0] in (pick or [])] or ["ridge_A_econ"]

bt = table("bt_q")
bt = bt[(bt["target_name"] == "target") & bt["y"].notna() & bt["model"].isin(APPROACH)].copy()
if not covid:
    bt = bt[~bt["quarter"].between("2020Q1", "2021Q2")]
bt["approach"] = bt["model"].map(lambda m: APPROACH[m][0])
bt["abs_error"] = (bt["y"] - bt["pred"]).abs()

# ----------------------------------------------------------------------------- headline KPIs
k = st.columns(len(chosen))
for col, m in zip(k, chosen):
    d = bt[bt["model"] == m]
    inside = ((d["y"] >= d["lo80"]) & (d["y"] <= d["hi80"])).mean()
    col.metric(APPROACH[m][0], f"±{d['abs_error'].mean():.2f} pts", border=True,
               delta=f"range held the truth {inside:.0%} of the time (target 80%)", delta_color="off")

# ----------------------------------------------------------------------------- region drill-down
region = st.selectbox("Region", sorted(REGIONS.values()), index=sorted(REGIONS.values()).index("London"))
s = bt[bt["code"] == REGION_CODE[region]]
fig = go.Figure()
act = s[s["model"] == chosen[0]].sort_values("quarter")
fig.add_trace(go.Scatter(x=act["quarter"], y=act["y"], name="Official figure", line=dict(color=INK, width=3)))
for m in chosen:
    g = s[s["model"] == m].sort_values("quarter")
    fig.add_trace(go.Scatter(x=g["quarter"], y=g["pred"], name=APPROACH[m][0],
                             line=dict(color=APPROACH[m][1], width=2)))
fig.update_layout(title=f"{region}: forecast vs. official quarterly growth", yaxis_title="Growth (%)",
                  hovermode="x unified")
quarter_axis(fig, act["quarter"])
st.plotly_chart(style(fig, 420, legend_bottom=True), width="stretch")

# ----------------------------------------------------------------------------- heatmap by region
st.subheader("Which regions are hardest to forecast?")
h = bt[bt["model"].isin(chosen)].groupby(["code", "approach"])["abs_error"].mean().unstack()
h.index = h.index.map(REGIONS)
h = h.sort_values(h.columns[0], ascending=False)
fig = px.imshow(h.round(2), text_auto=".2f", color_continuous_scale=SEQ, aspect="auto",
                labels=dict(color="Avg error (pts)", x="", y=""))
fig.update_layout(title="Average forecast error by region (darker = harder)")
st.plotly_chart(style(fig, 460), width="stretch")
how_to_read("compare across a row: if the satellite column isn't lighter than the economic column, "
            "satellite data didn't help for that region.")

# ----------------------------------------------------------------------------- error over time
st.subheader("Has accuracy changed over time?")
r = (bt[bt["model"].isin(chosen)].groupby(["quarter", "approach"])["abs_error"].mean()
     .unstack().rolling(4, min_periods=2).mean())
fig = go.Figure([go.Scatter(x=r.index, y=r[c], name=c, line=dict(width=2.5,
                 color=next(v[1] for v in APPROACH.values() if v[0] == c))) for c in r.columns])
fig.update_layout(title="Average error across all regions (rolling 4 quarters)", yaxis_title="pts",
                  hovermode="x unified")
quarter_axis(fig, r.index)
st.plotly_chart(style(fig, 380, legend_bottom=True), width="stretch")
how_to_read("lower is better. Errors jumped in 2021–2023 (post-lockdown reopening, then high "
            "inflation made cash-terms growth volatile) and fell back to about half a point in 2024–2025.")
