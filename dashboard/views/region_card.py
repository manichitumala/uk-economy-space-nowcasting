import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GREY, INK, ORANGE, REGION_CODE, REGIONS, download,
                              how_to_read, page_header, quarter_axis, style, table)

page_header("Region report card", "Everything about one region on a single page.")
region = st.selectbox("Choose a region", sorted(REGIONS.values()), index=sorted(REGIONS.values()).index("London"))
code = REGION_CODE[region]

bt = table("bt_q")
hist = bt[(bt["code"] == code) & (bt["target_name"] == "target") & bt["y"].notna()]
live = table("live_q")
live = live[(live["code"] == code) & (live["target_name"] == "target")].set_index("model")
fq = table("fq")
fr = fq[fq["code"] == code].sort_values("quarter")

calm = hist[~hist["quarter"].between("2020Q1", "2021Q2")]
mae_a = (calm[calm.model == "ridge_A_econ"]["y"] - calm[calm.model == "ridge_A_econ"]["pred"]).abs().mean()
mae_c = (calm[calm.model == "ridge_C_econ_sat"]["y"] - calm[calm.model == "ridge_C_econ_sat"]["pred"]).abs().mean()
last = hist[hist.model == "ridge_A_econ"].iloc[-1]
est = live.loc["ridge_A_econ"]
light = fr.dropna(subset=["light_yoy"]).iloc[-1]

c = st.columns(4)
c[0].metric("Estimate for Oct–Dec 2025", f"{est['pred']:+.1f}%", border=True,
            help=f"80% range {est['lo80']:+.1f}% to {est['hi80']:+.1f}%. Economic-data model. Not yet published by ONS.")
c[1].metric(f"Last official figure ({last['quarter']})", f"{last['y']:+.1f}%", border=True,
            help="Quarter-on-quarter growth, current prices, seasonally adjusted (ONS, July 2026).")
c[2].metric("Typical forecast miss", f"±{mae_a:.1f} pts", border=True,
            delta=f"{mae_c - mae_a:+.2f} pts with satellite", delta_color="inverse",
            help="Average absolute error of our forecasts for this region, 2016–2025 excluding COVID.")
c[3].metric(f"Night-light change ({light['quarter']})", f"{light['light_yoy']:+.0f}%", border=True,
            delta="likely cloud noise", delta_color="off",
            help="Year-on-year change in total brightness over the usable months of that quarter. "
                 "Swings this large usually reflect cloud cover (e.g. the very cloudy winter of 2024).")

better = "slightly better" if mae_c < mae_a - 0.01 else ("slightly worse" if mae_c > mae_a + 0.01 else "no different")
st.info(f"**In plain English:** we expect {region}'s economy to grow by about **{est['pred']:.1f}%** in "
        f"Oct–Dec 2025 (likely between {est['lo80']:.1f}% and {est['hi80']:.1f}%). Our forecasts for "
        f"{region} have typically been off by about **{mae_a:.1f} percentage points**. Adding satellite "
        f"data made them **{better}** ({mae_c:.2f} vs {mae_a:.2f}).")

# ----------------------------------------------------------------------------- forecast history
a = hist[hist.model == "ridge_A_econ"].set_index("quarter")
s = hist[hist.model == "ridge_C_econ_sat"].set_index("quarter").reindex(a.index)
fig = go.Figure()
fig.add_trace(go.Scatter(x=a.index, y=a["hi80"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=a.index, y=a["lo80"], fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                         line=dict(width=0), name="Expected range (80%)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=a.index, y=a["y"], name="Official figure", line=dict(color=INK, width=3)))
fig.add_trace(go.Scatter(x=a.index, y=a["pred"], name="Forecast: economic data", line=dict(color=BLUE, width=2)))
fig.add_trace(go.Scatter(x=s.index, y=s["pred"], name="Forecast: + satellite",
                         line=dict(color=ORANGE, width=2, dash="dot")))
fig.add_trace(go.Scatter(x=[est.get("quarter", "2025Q4")], y=[est["pred"]], mode="markers",
                         marker=dict(size=12, color=BLUE, symbol="diamond"), name="Live estimate",
                         error_y=dict(type="data", symmetric=False, array=[est["hi80"] - est["pred"]],
                                      arrayminus=[est["pred"] - est["lo80"]], color=BLUE)))
fig.update_layout(title=f"{region}: our forecasts vs. what actually happened",
                  yaxis_title="Quarterly growth (%)", hovermode="x unified")
fig.update_yaxes(range=[-6, 7])
quarter_axis(fig, list(a.index) + ["2025Q4"])
st.plotly_chart(style(fig, 440, legend_bottom=True), width="stretch")
how_to_read("the black line is the official figure. The closer the blue and orange lines sit to it, "
            "the better the forecast. The 2020 COVID crash goes off the chart (about −20%).")

# ----------------------------------------------------------------------------- lights history
L = table("lights")
L = L[(L["level"] == "itl1") & L["code"].isin(REGIONS)]
L = L.assign(ok=L["valid_share"] >= 0.5)
reg = L[L["code"] == code].set_index("month")
base = reg[reg["ok"] & (reg.index.year == 2019)]["mean_radiance"].mean()
ew = L[L["ok"]].groupby("month")["total_radiance"].sum() / L.groupby("month")["zone_area_km2"].sum()
ew_base = ew[ew.index.year == 2019].mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=ew.index, y=100 * ew / ew_base, mode="markers", name="England & Wales",
                         marker=dict(color=GREY, size=5)))
fig.add_trace(go.Scatter(x=reg.index[reg["ok"]], y=100 * reg.loc[reg["ok"], "mean_radiance"] / base,
                         mode="markers+lines", name=region, line=dict(color=ORANGE, width=1),
                         marker=dict(color=ORANGE, size=6)))
fig.update_layout(title=f"How bright is {region} at night? (2019 average = 100, usable months only)",
                  yaxis_title="Brightness index")
st.plotly_chart(style(fig, 400, legend_bottom=True), width="stretch")
how_to_read("each dot is one month. Summer months are missing because they are not real observations. "
            "The month-to-month jumps are mostly cloud, which is why lights struggle to track growth.")

out = a[["y", "pred", "lo80", "hi80"]].rename(columns={"y": "official", "pred": "forecast_economic"})
out["forecast_with_satellite"] = s["pred"]
download(out.reset_index().round(3), f"Download {region} forecast history", f"{code}_history")
