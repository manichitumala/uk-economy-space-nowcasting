"""UK Economy from Space - interactive dashboard.

Run:  streamlit run dashboard/app.py
Reads only the small processed artefacts in data/processed, data/interim and reports/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import regions  # noqa: E402

PROC, INTERIM, REPORTS = ROOT / "data" / "processed", ROOT / "data" / "interim", ROOT / "reports"
C_ECON, C_SAT, C_THIRD, C_BASE, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#52514e"
GROUP_COLORS = {"baseline": C_BASE, "A_econ": C_ECON, "B_sat": C_THIRD, "C_econ_sat": C_SAT}
GROUP_LABELS = {"baseline": "Naive baseline", "A_econ": "A: economic only",
                "B_sat": "B: satellite only", "C_econ_sat": "C: economic + satellite"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#c2410c"], [0.5, "#f1f0ec"], [1, "#1c5cab"]]

st.set_page_config(page_title="UK Economy from Space", page_icon="🛰️", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    d = {
        "results": json.loads((REPORTS / "results.json").read_text(encoding="utf-8")),
        "bt_q": pd.read_parquet(PROC / "backtest_quarterly.parquet"),
        "bt_a": pd.read_parquet(PROC / "backtest_annual_la.parquet"),
        "fq": pd.read_parquet(PROC / "features_quarterly.parquet"),
        "lights": pd.read_parquet(INTERIM / "lights_monthly.parquet"),
        "live_q": pd.read_parquet(PROC / "live_nowcasts_quarterly.parquet"),
        "live_a": pd.read_parquet(PROC / "live_nowcasts_annual_la.parquet"),
    }
    return d


@st.cache_data
def geo(level: str) -> dict:
    p = PROC / f"{level}_simplified.geojson"
    if not p.exists():
        g = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / f"{level}.geojson")
        g["geometry"] = g.geometry.simplify(0.005 if level == "itl1" else 0.002, preserve_topology=True)
        g.to_file(p, driver="GeoJSON")
    return json.loads(p.read_text(encoding="utf-8"))


def ew_map(df, value, label, color_scale, midpoint=None, hover=None):
    fig = px.choropleth(df, geojson=geo("itl1"), locations="code", featureidkey="properties.code",
                        color=value, color_continuous_scale=color_scale,
                        color_continuous_midpoint=midpoint, hover_name="region",
                        hover_data=hover or {value: ":.2f"}, labels={value: label})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0))
    return fig


D = load()
R = D["results"]

st.sidebar.title("🛰️ UK Economy from Space")
page = st.sidebar.radio("Page", ["1 · Economic intelligence", "2 · Satellite explorer",
                                  "3 · Forecast performance", "4 · Alternative-data research",
                                  "5 · Model explainability", "About the data"])
st.sidebar.caption("Satellite: HKU monthly VIIRS-like NTL (CC BY 4.0). Economic: ONS (OGL v3). "
                   "Historical backtests and live nowcasts are labelled separately.")

# ----------------------------------------------------------------------------- page 1
if page.startswith("1"):
    st.title("Regional economic intelligence")
    st.warning("**LIVE NOWCAST, not a backtest.** Target: 2025 Q4 quarter-on-quarter growth of "
               "regional GDP (current prices, SA). Forecast origin: 4 Feb 2026. As of Nov 2026 ONS "
               "has not published 2025 Q4 regional estimates, so these numbers are unverified.")
    lq = D["live_q"]
    tgt = st.radio("Target", ["target", "target_rel"], horizontal=True,
                   format_func=lambda t: "Growth (%)" if t == "target" else "Growth relative to E&W (pp)")
    model = st.selectbox("Model", sorted(lq["model"].unique()), index=sorted(lq["model"].unique()).index("ridge_C_econ_sat"))
    sub = lq[(lq["target_name"] == tgt) & (lq["model"] == model)].copy()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(ew_map(sub, "pred", "Nowcast", SEQ if tgt == "target" else DIVERGING,
                               midpoint=None if tgt == "target" else 0,
                               hover={"pred": ":.2f", "lo80": ":.2f", "hi80": ":.2f"}),
                        use_container_width=True)
    with c2:
        tbl = sub[["region", "pred", "lo80", "hi80"]].sort_values("pred", ascending=False)
        tbl.columns = ["Region", "Nowcast", "80% low", "80% high"]
        st.dataframe(tbl.round(2), hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tbl["Nowcast"], y=tbl["Region"], mode="markers",
                                 marker=dict(size=10, color=C_SAT),
                                 error_x=dict(type="data", symmetric=False,
                                              array=tbl["80% high"] - tbl["Nowcast"],
                                              arrayminus=tbl["Nowcast"] - tbl["80% low"],
                                              color=C_SAT, thickness=1.5), name="nowcast ± 80% PI"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="%",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    st.info("Interpretation guardrail: the research (page 4) finds **no statistically significant gain** "
            "from satellite features, and the interval widths show the irreducible uncertainty.")

# ----------------------------------------------------------------------------- page 2
elif page.startswith("2"):
    st.title("Satellite explorer: night-time lights")
    L = D["lights"]
    level = st.radio("Geography", ["itl1", "lad"], horizontal=True,
                     format_func=lambda s: "ITL1 regions" if s == "itl1" else "Local authorities")
    months = sorted(L["month"].dt.strftime("%Y-%m").unique())
    m = st.select_slider("Month", months, value="2025-01")
    metric = st.selectbox("Metric", ["mean_radiance", "total_radiance", "lit_area_km2", "bright_share",
                                     "top10_share", "valid_share"])
    sub = L[(L["level"] == level) & (L["month"].dt.strftime("%Y-%m") == m)].copy()
    if level == "itl1":
        sub = sub[sub["code"].isin(regions())]
        sub["region"] = sub["code"].map(regions())
    else:
        names = {f["properties"]["code"]: f["properties"]["name"] for f in geo("lad")["features"]}
        sub["region"] = sub["code"].map(names)
        sub = sub[sub["code"].str[0].isin(["E", "W"])]
    vs = sub["valid_share"].mean()
    if vs < 0.5:
        st.error(f"⚠️ {m}: only {vs:.0%} of lit pixels are genuine observations - this month is mostly "
                 "a scaled copy of the same year's December in the source data and is excluded from modelling.")
    color = np.log1p(sub[metric]) if metric in ("total_radiance", "mean_radiance") else sub[metric]
    sub["colour"] = color
    fig = px.choropleth(sub, geojson=geo(level), locations="code", featureidkey="properties.code",
                        color="colour", color_continuous_scale=SEQ, hover_name="region",
                        hover_data={metric: ":.3f", "valid_share": ":.0%", "colour": False})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar_title=("log(1+x)" if metric in ("total_radiance", "mean_radiance") else metric))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Time series (imputed months hollow)")
    opts = sorted(sub["region"].dropna().unique())
    pick = st.multiselect("Areas", opts, default=[o for o in ["London", "North West", "Wales"] if o in opts][:3] or opts[:2])
    codes = sub[sub["region"].isin(pick)]["code"].tolist()
    ts = L[(L["level"] == level) & L["code"].isin(codes)].copy()
    ts["area"] = ts["code"].map(dict(zip(sub["code"], sub["region"])))
    fig = go.Figure()
    for i, (a, g) in enumerate(ts.groupby("area")):
        col = [C_ECON, C_SAT, C_THIRD][i % 3]
        ok = g["valid_share"] >= 0.5
        fig.add_trace(go.Scatter(x=g["month"], y=g[metric].where(ok), mode="lines+markers", name=a,
                                 line=dict(color=col, width=2), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=g["month"][~ok], y=g[metric][~ok], mode="markers", name=f"{a} (imputed)",
                                 marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color=col, width=1.5)),
                                 showlegend=i == 0))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=metric, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- page 3
elif page.startswith("3"):
    st.title("Forecast performance (historical walk-forward backtest)")
    st.caption("Expanding-window, point-in-time backtest. Each nowcast uses only data public at its origin "
               "(quarter end + 35 days). Training uses only targets published by then (≤ t-2).")
    bt = D["bt_q"][D["bt_q"]["y"].notna()].copy()
    tgt = st.radio("Target", ["target", "target_rel"], horizontal=True,
                   format_func=lambda t: "Nominal growth" if t == "target" else "Growth relative to E&W")
    ex = st.checkbox("Exclude COVID quarters (2020Q1-2021Q2) from evaluation", value=True)
    bt = bt[bt["target_name"] == tgt]
    if ex:
        bt = bt[~bt["quarter"].between("2020Q1", "2021Q2")]
    reg = st.selectbox("Region", list(regions().values()), index=6)
    code = {v: k for k, v in regions().items()}[reg]
    models = st.multiselect("Models", sorted(bt["model"].unique()),
                            default=["historical_mean", "ridge_A_econ", "ridge_C_econ_sat"])
    s = bt[(bt["code"] == code)]
    fig = go.Figure()
    act = s[s["model"] == s["model"].iloc[0]]
    fig.add_trace(go.Scatter(x=act["quarter"], y=act["y"], name="Actual", line=dict(color="#0b0b0b", width=2.5)))
    for i, mname in enumerate(models):
        g = s[s["model"] == mname]
        fig.add_trace(go.Scatter(x=g["quarter"], y=g["pred"], name=mname,
                                 line=dict(color=[C_BASE, C_ECON, C_SAT, C_THIRD][i % 4], width=2)))
    fig.update_layout(height=420, hovermode="x unified", yaxis_title="%", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MAE by region")
        mm = bt[bt["model"].isin(models)].groupby(["code", "model"]).apply(
            lambda d: np.mean(np.abs(d["y"] - d["pred"]))).unstack()
        mm.index = mm.index.map(regions())
        st.dataframe(mm.round(3), use_container_width=True)
    with c2:
        st.subheader("Rolling 4-quarter MAE (all regions)")
        rr = (bt[bt["model"].isin(models)].assign(ae=lambda d: (d["y"] - d["pred"]).abs())
              .groupby(["quarter", "model"])["ae"].mean().unstack().rolling(4, min_periods=2).mean())
        fig = go.Figure([go.Scatter(x=rr.index, y=rr[c], name=c,
                                    line=dict(color=[C_BASE, C_ECON, C_SAT, C_THIRD][i % 4]))
                         for i, c in enumerate(rr.columns)])
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="pp")
        st.plotly_chart(fig, use_container_width=True)
    cov = bt[bt["model"].isin(models)].groupby("model").apply(
        lambda d: np.mean((d["y"] >= d["lo80"]) & (d["y"] <= d["hi80"])))
    st.write("**Empirical coverage of nominal 80% conformal intervals:**",
             ", ".join(f"{k}: {v:.0%}" for k, v in cov.items()))

# ----------------------------------------------------------------------------- page 4
elif page.startswith("4"):
    st.title("Does satellite data improve the nowcast?")
    st.markdown("The core experiment compares **A** (economic indicators only), **B** (satellite only) "
                "and **C** (economic + satellite) under identical walk-forward protocols.")
    track = st.radio("Track", ["B: quarterly regional GDP (10 regions)", "A: annual local-authority GDP (361 LAs)"])
    key = "track_b" if track.startswith("B") else "track_a"
    tgt = st.radio("Target", ["target", "target_rel"], horizontal=True,
                   format_func=lambda t: "Growth" if t == "target" else "Relative growth")
    sample = st.radio("Sample", ["ex_covid", "all"], horizontal=True,
                      format_func=lambda s: "Excluding COVID" if s == "ex_covid" else "All periods")
    res = R[key][f"{tgt}_{sample}"]
    mt = pd.DataFrame(res["metrics"])
    best_a = mt[mt["group"] == "A_econ"].iloc[0]
    best_c = mt[mt["group"] == "C_econ_sat"].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Best economic-only (A)", f"{best_a['mae']:.3f} pp", best_a["model"], delta_color="off")
    k2.metric("Best economic + satellite (C)", f"{best_c['mae']:.3f} pp", best_c["model"], delta_color="off")
    k3.metric("Relative improvement", f"{100 * (best_a['mae'] - best_c['mae']) / best_a['mae']:+.1f}%")
    mt = mt.sort_values("mae")
    fig = px.bar(mt, x="mae", y="model", color="group", orientation="h",
                 color_discrete_map=GROUP_COLORS, labels={"mae": "MAE (pp)", "group": ""},
                 category_orders={"model": mt["model"].tolist()})
    fig.for_each_trace(lambda t: t.update(name=GROUP_LABELS[t.name]))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Paired significance tests (C vs A within each model family)")
    tt = pd.DataFrame(res["tests"])[["benchmark", "challenger", "mae_benchmark", "mae_challenger",
                                     "improvement_pct", "ci_low", "ci_high", "dm_p_abs", "T"]]
    st.dataframe(tt.round(3), hide_index=True, use_container_width=True)
    st.caption("improvement_pct: MAE reduction of challenger vs benchmark; ci: 95% period-block bootstrap; "
               "dm_p_abs: Diebold-Mariano (HLN) p-value on period-average absolute errors.")
    st.success("**Verdict:** across both tracks and all model families, adding satellite features does not "
               "produce a statistically significant improvement. Lights strongly track the *level* of local "
               "economic activity (R² ≈ 0.55) but carry no detectable signal for short-run *growth* in the UK.")

# ----------------------------------------------------------------------------- page 5
elif page.startswith("5"):
    st.title("Model explainability")
    ex = R["explainability"]
    shp = pd.Series(ex["shap_xgb_C_trackB"]).sort_values()
    sat_cols = {"light_yoy", "light_yoy_rel", "lit_area_yoy", "std_yoy", "bright_share_yoy_d",
                "top10_share_yoy_d", "light_yoy_d", "light_obs_months"}
    fig = go.Figure(go.Bar(x=shp.values, y=shp.index, orientation="h",
                           marker_color=[C_SAT if c in sat_cols else C_ECON for c in shp.index]))
    fig.update_layout(height=520, xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=30, b=0),
                      title="XGBoost (C), mean |SHAP| share - orange = satellite features")
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Satellite features account for **{ex['shap_satellite_share']:.0%}** of mean |SHAP| - but the "
             "out-of-sample tests show this attribution does not translate into accuracy: the model uses "
             "them to fit noise. SHAP explains a fitted model, not causation.")
    co = pd.Series(ex["ridge_C_trackB_std_coefs"]).sort_values(key=np.abs)
    fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h",
                           marker_color=[C_SAT if c in sat_cols else C_ECON for c in co.index]))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      title=f"Ridge (C) standardised coefficients (alpha = {ex['ridge_C_trackB_alpha']:.0f})")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Shock detection: COVID-19 in central London")
    sh = R["shock"]
    st.write(f"City of London + Westminster lights fell **{sh['lockdown_jan_mar_2021_mean_pp']:.1f} log points** "
             f"relative to the national trend in the Jan-Mar 2021 lockdown (pre-COVID placebo s.d. "
             f"{sh['placebo_sd_pp']:.1f}; z ≈ {sh['lockdown_z']:.1f}) and recovered to "
             f"{sh['reopening_sep_dec_2021_mean_pp']:+.1f} by Sep-Dec 2021. The April-June 2020 lockdown "
             "falls entirely inside the imputed summer window and cannot be observed.")
    st.image(str(REPORTS / "figures" / "07_covid_cbd_event_study.png"))

# ----------------------------------------------------------------------------- about
else:
    st.title("About the data")
    au = R["audit"]
    st.markdown(f"""
- **{au['months_genuine_ew']} of {au['months_total']}** monthly composites (Apr 2012-Dec 2025) contain genuine
  observations for England & Wales; the rest are December-scaled imputations (short summer nights at 50-56°N).
- Mean genuine months per quarter: Q1 {au['mean_genuine_months_by_quarter']['1']:.1f},
  Q2 {au['mean_genuine_months_by_quarter']['2']:.1f}, Q3 {au['mean_genuine_months_by_quarter']['3']:.1f},
  Q4 {au['mean_genuine_months_by_quarter']['4']:.1f}.
- Annual ITL1 real-growth revisions average **{au['annual_revisions']['mean_abs_revision_pp']:.2f} pp**
  (max {au['annual_revisions']['max_abs_revision_pp']:.1f} pp; {au['annual_revisions']['sign_changes']} sign changes in {au['annual_revisions']['n']} region-years).
""")
    st.image(str(REPORTS / "figures" / "01_genuine_observation_share.png"))
