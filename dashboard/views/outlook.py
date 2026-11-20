import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, DIVERGING, ORANGE, SEQ, callout, download, geojson,
                              how_to_read, page_header, style, style_map, table)

page_header("Regional outlook",
            "Our early estimates for periods the Office for National Statistics hasn't published yet.")
callout("🔶 <b>Live estimates, not yet checked.</b> These figures cover <b>Oct–Dec 2025</b> "
        "(regions) and <b>2025</b> (local areas). Official figures are not out yet, so treat them "
        "as informed guesses with the uncertainty ranges shown.", "live")

c1, c2 = st.columns([1, 1])
basis = c1.segmented_control("Forecast based on", ["Economic data", "Economic + satellite"],
                             default="Economic data")
measure = c2.segmented_control("Show", ["Growth rate", "vs. England & Wales average"],
                               default="Growth rate")
basis = basis or "Economic data"
measure = measure or "Growth rate"
model = "ridge_A_econ" if basis == "Economic data" else "ridge_C_econ_sat"
tgt = "target" if measure == "Growth rate" else "target_rel"

lq = table("live_q")
d = lq[(lq["model"] == model) & (lq["target_name"] == tgt)].copy()
d = d.rename(columns={"pred": "estimate", "lo80": "low", "hi80": "high"}).sort_values("estimate")
unit = "%" if tgt == "target" else " pts"

top, bottom = d.iloc[-1], d.iloc[0]
st.markdown(f"#### Fastest expected: **{top['region']}** ({top['estimate']:+.1f}{unit}) · "
            f"slowest: **{bottom['region']}** ({bottom['estimate']:+.1f}{unit})")

m1, m2 = st.columns([3, 2])
with m1:
    fig = px.choropleth(d, geojson=geojson("itl1"), locations="code", featureidkey="properties.code",
                        color="estimate", hover_name="region",
                        color_continuous_scale=SEQ if tgt == "target" else DIVERGING,
                        color_continuous_midpoint=None if tgt == "target" else 0,
                        hover_data={"code": False, "estimate": ":.2f", "low": ":.2f", "high": ":.2f"})
    st.plotly_chart(style_map(fig, 560, f"Growth ({unit.strip() or '%'})"), width="stretch")
with m2:
    colour = BLUE if model == "ridge_A_econ" else ORANGE
    fig = go.Figure(go.Scatter(
        x=d["estimate"], y=d["region"], mode="markers", marker=dict(size=13, color=colour),
        error_x=dict(type="data", symmetric=False, array=d["high"] - d["estimate"],
                     arrayminus=d["estimate"] - d["low"], color=colour, thickness=2, width=0),
        hovertemplate="%{y}: %{x:.2f}" + unit + "<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_width=1)
    fig.update_layout(title="Quarterly growth estimate, Oct–Dec 2025",
                      xaxis_title=f"Growth ({unit.strip()})")
    st.plotly_chart(style(fig, 520), width="stretch")
how_to_read("dots are our best estimate; the lines show the range we expect the official figure to "
            "fall in 8 times out of 10. Overlapping lines mean we can't confidently rank those regions.")
download(d[["region", "estimate", "low", "high"]].round(3), "Download regional estimates",
         "regional_estimates_2025Q4")

st.divider()
st.subheader("Local areas: estimated real growth in 2025")
st.caption("Official local-authority GDP for 2025 is due around September 2027. Estimates use UK "
           "monthly GDP and local benefit-claimant trends (the most accurate model in our tests).")
la = table("live_a")
la = la[(la["model"] == "bridge_conventional") & (la["target_name"] == "target_rel")].copy()
la["estimate"] = la["pred"]
fig = px.choropleth(la, geojson=geojson("lad"), locations="code", featureidkey="properties.code",
                    color="estimate", color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                    hover_name="name", hover_data={"code": False, "estimate": ":.2f",
                                                   "itl1_region": True})
st.plotly_chart(style_map(fig, 680, "Growth vs UK average (pts)"), width="stretch")
how_to_read("blue areas are expected to grow faster than the UK average, red areas slower. "
            "Differences are small relative to typical error (about 3 points), so read this as a "
            "rough guide, not a ranking.")
download(la[["code", "name", "itl1_region", "estimate", "lo80", "hi80"]].round(3),
         "Download local-area estimates", "local_area_estimates_2025")
