import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, INK2, SEQ, callout, how_to_read, page_header, report_table,
                              results, style)

R = results()
au = R["audit"]
page_header("Data quality checks", "Before trusting any model, we checked what the data really contains.")
callout(f"<b>Only {au['months_genuine_ew']} of {au['months_total']} monthly satellite images are real "
        "observations for England & Wales.</b> UK summer nights are too short for the satellite, so the "
        "data provider filled those months with a scaled copy of the same year's <i>December</i>. "
        "Using them would let a forecast 'see the future', so we detect and exclude them.")

mat = pd.DataFrame(au["genuine_share_matrix"]).set_index("y")
mat.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig = px.imshow(mat, color_continuous_scale=SEQ, zmin=0, zmax=1, aspect="auto",
                text_auto=".0%", labels=dict(color="Genuine"))
fig.update_traces(hovertemplate="%{x} %{y}: %{z:.0%} genuine<extra></extra>")
fig.update_layout(title="Share of each month's image that is a real observation (dark = real)",
                  coloraxis_colorbar_tickformat=".0%")
fig.update_yaxes(title=None, type="category", dtick=1)
fig.update_xaxes(title=None, side="top")
st.plotly_chart(style(fig, 520), width="stretch")
how_to_read("dark squares are usable. May–August are never usable; April rarely. In 2013 most of the "
            "country (north of Birmingham) was also filled in.")

q = au["mean_genuine_months_by_quarter"]
c = st.columns(4)
for col, (k, lab) in zip(c, [("1", "Jan–Mar"), ("2", "Apr–Jun"), ("3", "Jul–Sep"), ("4", "Oct–Dec")]):
    col.metric(f"Usable months, {lab}", f"{q[k]:.1f} of 3", border=True)

st.divider()
st.subheader("Official figures change a lot after publication")
rv = au["annual_revisions"]
callout(f"Regional growth figures were revised by <b>{rv['mean_abs_revision_pp']:.1f} percentage points</b> "
        f"on average between first publication and today (largest: {rv['max_abs_revision_pp']:.1f} pts; "
        f"{rv['sign_changes']} times growth flipped to decline or vice versa). Typical growth is only 1–3%, "
        "so the 'truth' we forecast is itself uncertain.")
r = report_table("annual_itl1_revisions")
from dashboard.common import REGIONS  # noqa: E402
r["region"] = r["code"].map(REGIONS)
lim = [min(r["first"].min(), r["latest"].min()) - 1, max(r["first"].max(), r["latest"].max()) + 1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=INK2, dash="dash"),
                         name="No revision", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=r["first"], y=r["latest"], mode="markers", marker=dict(size=10, color=BLUE, opacity=0.75),
                         text=r["region"] + " " + r["year"].astype(str), name="Region-year",
                         hovertemplate="%{text}<br>First published: %{x:.1f}%<br>Now: %{y:.1f}%<extra></extra>"))
fig.update_layout(title="Real growth: first published vs. latest estimate",
                  xaxis_title="First published (%)", yaxis_title="Latest estimate (%)")
st.plotly_chart(style(fig, 460, legend_bottom=True), width="stretch")
how_to_read("dots on the dashed line were never revised. The further away, the bigger the revision.")

st.divider()
st.subheader("Other checks built into the pipeline")
st.markdown("""
| Check | What it guards against |
|---|---|
| ⛅ Cloud-noise dampening (±30% cap on monthly changes) | e.g. London looked 45% darker in Nov–Dec 2024 purely because of persistent cloud |
| 🌍 True pixel area | pixels shrink ~22% from southern England to northern Scotland |
| 🗺️ Exact boundary matching | all 361 local authorities match across GDP, claimant and map data |
| ⏱️ Publication-date filter | no forecast uses any figure released after its forecast date (tested automatically) |
""")
