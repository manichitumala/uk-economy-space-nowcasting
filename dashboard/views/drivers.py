import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, FEATURE, INK2, ORANGE, SAT_FEATURES, GRIDC, callout,
                              how_to_read, page_header, report_table, results, style)

R = results()
ex = R["explainability"]
page_header("What drives the forecasts?",
            "Which pieces of information the models lean on, and what satellites can genuinely detect.")

shp = pd.Series(ex["shap_xgb_C_trackB"]).sort_values()
sat_share = ex["shap_satellite_share"]
callout(f"The machine-learning model gives satellite inputs about <b>{sat_share:.0%}</b> of its "
        "attention, yet its forecasts are no more accurate. It is using the satellite data to chase "
        "noise. <b>A model 'using' an input is not proof that the input helps.</b>")

fig = go.Figure(go.Bar(
    x=shp.values, y=[FEATURE.get(c, c) for c in shp.index], orientation="h",
    marker_color=[ORANGE if c in SAT_FEATURES else BLUE for c in shp.index],
    hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
fig.update_layout(title="Share of the model's attention (SHAP), XGBoost with all inputs",
                  xaxis_tickformat=".0%")
fig.add_annotation(x=shp.max() * 0.75, y=1, text="🟧 satellite   🟦 economic", showarrow=False,
                   font=dict(color=INK2))
st.plotly_chart(style(fig, 560), width="stretch")
how_to_read("longer bars = the model relied more on that input. The top inputs are recent official "
            "growth and UK monthly GDP, i.e. ordinary economic data.")

st.divider()
st.subheader("What satellites *can* see: COVID lockdowns in central London")
ev = report_table("shock_event_study")
ev["month"] = pd.to_datetime(ev["month"])
ev = ev[(ev["month"] >= "2020-01-01") & (ev["month"] <= "2022-12-01")]
ev["pct"] = 100 * (np.exp(ev["cbd_minus_national"] / 100) - 1)
sd = R["shock"]["placebo_sd_pp"]
band = 100 * (np.exp(1.96 * sd / 100) - 1)
fig = go.Figure()
fig.add_hrect(y0=-band, y1=band, fillcolor=GRIDC, opacity=0.6, line_width=0,
              annotation_text="normal month-to-month noise", annotation_position="top left")
for s, e, lab in [("2020-03-23", "2020-06-15", "Lockdown 1"), ("2020-11-05", "2020-12-02", "L2"),
                  ("2021-01-06", "2021-04-12", "Lockdown 3")]:
    fig.add_vrect(x0=s, x1=e, fillcolor=BLUE, opacity=0.08, line_width=0, annotation_text=lab,
                  annotation_position="bottom left")
fig.add_trace(go.Scatter(x=ev["month"], y=ev["pct"], mode="lines+markers", connectgaps=False,
                         line=dict(color=ORANGE, width=2), marker=dict(size=8),
                         name="City of London + Westminster vs national",
                         hovertemplate="%{x|%b %Y}: %{y:+.0f}%<extra></extra>"))
fig.add_hline(y=0, line_color=INK2)
fig.update_layout(title="Central London brightness vs. the national trend (same month of 2019 = 0)",
                  yaxis_title="% difference", showlegend=False)
st.plotly_chart(style(fig, 440), width="stretch")
sh = R["shock"]
c = st.columns(3)
c[0].metric("Jan–Mar 2021 lockdown", f"{100 * (np.exp(sh['lockdown_jan_mar_2021_mean_pp'] / 100) - 1):.0f}%",
            border=True, help="Average over the three months, relative to national trend.")
c[1].metric("After reopening (Sep–Dec 2021)", f"{100 * (np.exp(sh['reopening_sep_dec_2021_mean_pp'] / 100) - 1):+.0f}%",
            border=True)
c[2].metric("Strength of signal", f"{abs(sh['lockdown_z']):.1f}× noise", border=True,
            help="How many standard errors the lockdown dip sits below zero (z-score).")
how_to_read("when offices emptied, the City and West End went visibly darker than the rest of the "
            "country, then recovered. The first lockdown (spring 2020) is invisible: it fell in the "
            "summer months the satellite cannot observe.")
