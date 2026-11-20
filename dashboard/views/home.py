import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (DATA, GROUP_COLOUR, best_by_group, callout, model_label,
                              page_header, results, style)

R = results()
page_header("Can satellites see the UK economy?",
            "We tested whether night-time light seen from space can estimate regional economic "
            "growth months before the official statistics are published.")

callout("<b>Short answer: not for short-term growth.</b> Night lights show clearly <i>where</i> "
        "economic activity is, and they can spot big shocks such as COVID lockdowns. But adding "
        "them to a forecast of <i>this quarter's growth</i> does not make the forecast more accurate.")

# ----------------------------------------------------------------------------- KPIs
lev = next(r for r in R["econometrics"]["levels"] if r["year"] == 2019)
q = R["track_b"]["target_ex_covid"]
best = best_by_group(q["metrics"]).set_index("group")
gain = 100 * (best.at["A_econ", "mae"] - best.at["C_econ_sat", "mae"]) / best.at["A_econ", "mae"]
test = next(t for t in q["tests"] if t["challenger"] == best.at["C_econ_sat", "model"])
unusable = 1 - R["audit"]["share_genuine"]
shock = 100 * (np.exp(R["shock"]["lockdown_jan_mar_2021_mean_pp"] / 100) - 1)

c = st.columns(4)
c[0].metric("Local GDP explained by lights", f"{lev['r2']:.0%}", border=True,
            help="Across 361 local authorities, brighter places have bigger economies.")
c[1].metric("Gain from satellite data", f"{gain:+.1f}%", border=True,
            delta=f"not significant (p = {test['dm_p_abs']:.2f})", delta_color="off",
            help="Quarterly regional forecasts, best model with vs. without satellite features.")
c[2].metric("Satellite months unusable", f"{unusable:.0%}", border=True,
            help="UK summer nights are too short; the source fills them with copies of December.")
c[3].metric("London lockdown dimming", f"{shock:.0f}%", border=True,
            help="City of London + Westminster, Jan-Mar 2021, compared with the national trend and the same months of 2019.")

st.write("")
# ----------------------------------------------------------------------------- main chart
st.subheader("Adding satellite data barely changes forecast error")
cols = st.columns(2)
for col, key, title, period in [
    (cols[0], ("track_b", "target_ex_covid"), "Quarterly growth, 10 regions", "2016–2025, excl. COVID"),
    (cols[1], ("track_a", "target_all"), "Yearly growth, 361 local areas", "2016–2024"),
]:
    b = best_by_group(R[key[0]][key[1]]["metrics"])
    b = b.set_index("group").loc[["baseline", "B_sat", "A_econ", "C_econ_sat"]].reset_index()
    fig = go.Figure(go.Bar(
        x=b["mae"], y=[DATA[g] for g in b["group"]], orientation="h",
        marker_color=[GROUP_COLOUR[g] for g in b["group"]],
        text=[f"{v:.2f} pts" for v in b["mae"]], textposition="outside",
        customdata=[model_label(m) for m in b["model"]],
        hovertemplate="%{y}<br>Best model: %{customdata}<br>Average error: %{x:.3f} pts<extra></extra>"))
    fig.update_layout(title=f"{title}<br><sup>{period} · lower is better</sup>",
                      xaxis_title="Average forecast error (percentage points)", showlegend=False)
    fig.update_xaxes(range=[0, b["mae"].max() * 1.25])
    col.plotly_chart(style(fig, 330), width="stretch")
st.caption("💡 **How to read this:** each bar is the best model using that kind of data. The orange "
           "bar (economic + satellite) is no shorter than the blue bar (economic data alone), so the "
           "satellite data adds nothing measurable.")

# ----------------------------------------------------------------------------- how it works
st.subheader("How the test works")
s = st.columns(3)
s[0].markdown("<div class='step'><b>1 · Look from space</b><br>165 monthly night-light images "
              "(2012–2025), measured for each of 10 regions and 361 local areas.</div>",
              unsafe_allow_html=True)
s[1].markdown("<div class='step'><b>2 · Match to official data</b><br>Office for National "
              "Statistics GDP, monthly UK GDP and benefit-claimant counts, with their real "
              "publication dates.</div>", unsafe_allow_html=True)
s[2].markdown("<div class='step'><b>3 · Run a fair race</b><br>Re-run history quarter by quarter, "
              "using only data that was public at the time, and compare forecasts with and without "
              "satellite data.</div>", unsafe_allow_html=True)

st.subheader("What we learned")
st.markdown(
    "- ✅ **Lights locate the economy.** Brighter places have bigger economies.\n"
    "- ✅ **Lights detect big local shocks.** Central London's offices visibly went dark in the 2021 lockdown.\n"
    "- ❌ **Lights don't track short-term growth.** Cloud, LED street lighting and a service-based "
    "economy swamp the signal.\n"
    "- ⚠️ **Half the satellite record is not real data for the UK.** Summer months are copies of "
    "December, so we detect them and leave them out.")

st.subheader("Explore next")
n = st.columns(3)
n[0].page_link("views/outlook.py", label="See the latest regional estimates", icon="🗺️")
n[1].page_link("views/verdict.py", label="See the full evidence", icon="⚖️")
n[2].page_link("views/how_it_works.py", label="Plain-English glossary", icon="📘")
