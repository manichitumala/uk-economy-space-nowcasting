import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (AQUA, BLUE, ORANGE, REGIONS, SEQ, geojson, how_to_read,
                              page_header, style, style_map, table)

page_header("Satellite explorer", "What the UK looks like from space at night, month by month.")

METRICS = {
    "Brightness": ("mean_radiance", "Average night-time brightness (nW/cm²/sr)"),
    "Lit-up area": ("lit_area_km2", "Area brighter than a dim rural road (km²)"),
    "Very bright areas": ("bright_share", "Share of the lit area that is very bright (city centres)"),
    "Data quality": ("valid_share", "Share of the image that is a real observation"),
}
L = table("lights")
c1, c2 = st.columns([1, 2])
geo = c1.segmented_control("Show by", ["Regions", "Local areas"], default="Local areas") or "Local areas"
metric = c2.pills("Measure", list(METRICS), default="Brightness") or "Brightness"
col, desc = METRICS[metric]
level = "itl1" if geo == "Regions" else "lad"

months = sorted(L["month"].dt.strftime("%Y-%m").unique())
month = st.select_slider("Month", months, value="2025-01",
                         help="Tip: try a summer month to see the data-quality warning.")
sub = L[(L["level"] == level) & (L["month"].dt.strftime("%Y-%m") == month)].copy()
if level == "itl1":
    sub = sub[sub["code"].isin(REGIONS)]
    sub["area"] = sub["code"].map(REGIONS)
else:
    names = {f["properties"]["code"]: f["properties"]["name"] for f in geojson("lad")["features"]}
    sub = sub[sub["code"].str[0].isin(["E", "W"])]
    sub["area"] = sub["code"].map(names)

quality = sub["valid_share"].mean()
if quality < 0.5:
    st.warning(f"⚠️ **{month} is not a real observation for most of the UK** (only {quality:.0%} "
               "genuine). UK summer nights are too short for the satellite, so the data provider "
               "filled this month with a scaled copy of December. We exclude such months from all "
               "forecasts.")
else:
    st.success(f"✅ {month}: {quality:.0%} of the image is a genuine observation.")

sub["shown"] = np.log1p(sub[col]) if col == "mean_radiance" else sub[col]
fig = px.choropleth(sub, geojson=geojson(level), locations="code", featureidkey="properties.code",
                    color="shown", color_continuous_scale=SEQ, hover_name="area",
                    hover_data={"code": False, "shown": False, col: ":.3f", "valid_share": ":.0%"},
                    labels={col: metric, "valid_share": "Genuine data"})
fig.update_layout(title=desc)
left, right = st.columns([3, 2])
left.plotly_chart(style_map(fig, 640, "log scale" if col == "mean_radiance" else ""), width="stretch")
with right:
    st.markdown(f"**Top 10 by {metric.lower()}**")
    top = sub.nlargest(10, col)[["area", col]].rename(columns={"area": "Area", col: metric})
    st.dataframe(top, hide_index=True, width="stretch",
                 column_config={metric: st.column_config.ProgressColumn(
                     metric, format="%.2f" if col != "valid_share" else "percent",
                     min_value=0, max_value=float(sub[col].max()))})

st.subheader("Compare areas over time")
opts = sorted(sub["area"].dropna().unique())
default = [o for o in ["City of London", "Manchester", "Cardiff", "London", "North West", "Wales"] if o in opts][:3]
pick = st.multiselect("Areas", opts, default=default, max_selections=3)
codes = dict(zip(sub["area"], sub["code"]))
fig = go.Figure()
for i, a in enumerate(pick):
    g = L[(L["level"] == level) & (L["code"] == codes[a])].sort_values("month")
    ok = g["valid_share"] >= 0.5
    colr = [ORANGE, BLUE, AQUA][i]
    fig.add_trace(go.Scatter(x=g["month"], y=g[col].where(ok), mode="lines+markers", name=a,
                             line=dict(color=colr, width=2), marker=dict(size=5), connectgaps=False))
    fig.add_trace(go.Scatter(x=g["month"][~ok], y=g[col][~ok], mode="markers", name=f"{a} (copied month)",
                             marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color=colr, width=1.5)),
                             legendgroup=a, showlegend=i == 0))
fig.update_layout(title=f"{metric} over time", yaxis_title=metric, hovermode="x unified")
st.plotly_chart(style(fig, 420, legend_bottom=True), width="stretch")
how_to_read("filled dots are real observations; hollow dots are summer months copied from December "
            "and ignored by the models. Gaps in the line mark those months.")
