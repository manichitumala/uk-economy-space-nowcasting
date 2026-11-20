"""Shared data loading, plain-English labels, styling and chart helpers for the dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.config import regions  # noqa: E402
from src.utils.labels import FEATURE, SAT_FEATURES  # noqa: E402,F401

PROC, INTERIM, REPORTS = ROOT / "data" / "processed", ROOT / "data" / "interim", ROOT / "reports"

# Validated reference palette
BLUE, ORANGE, AQUA, GREY, INK, INK2, GRIDC = ("#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2",
                                             "#0b0b0b", "#52514e", "#e4e3df")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#c2410c"], [0.5, "#f1f0ec"], [1, "#1c5cab"]]

REGIONS = regions()
REGION_CODE = {v: k for k, v in REGIONS.items()}

# ----------------------------------------------------------------------------- plain-English labels
METHOD = {"zero": "Always predict zero", "historical_mean": "Long-run average",
          "last_published": "Last official figure", "bridge_conventional": "Bridge equation",
          "ridge": "Ridge regression", "enet": "Elastic Net", "xgb": "XGBoost", "lgbm": "LightGBM"}
DATA = {"A_econ": "Economic data", "B_sat": "Satellite only", "C_econ_sat": "Economic + satellite",
        "baseline": "Simple benchmark"}
GROUP_COLOUR = {"baseline": GREY, "A_econ": BLUE, "B_sat": AQUA, "C_econ_sat": ORANGE}
TARGET = {"target": "Growth rate", "target_rel": "Growth vs. England & Wales average"}



def model_label(m: str) -> str:
    if m in METHOD:
        return METHOD[m]
    if m == "ridge_B_sat_plus_bridge":
        return "Bridge equation + satellite"
    fam, _, grp = m.partition("_")
    return f"{METHOD.get(fam, fam)} · {DATA.get(grp, grp)}"


def model_group(m: str) -> str:
    for g in ("A_econ", "B_sat", "C_econ_sat"):
        if m.endswith(g):
            return g
    if m == "ridge_B_sat_plus_bridge":
        return "C_econ_sat"
    if m == "bridge_conventional":
        return "A_econ"
    return "baseline"


# ----------------------------------------------------------------------------- data
@st.cache_data
def results() -> dict:
    return json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))


@st.cache_data
def table(name: str) -> pd.DataFrame:
    files = {"bt_q": "backtest_quarterly", "bt_a": "backtest_annual_la", "fq": "features_quarterly",
             "live_q": "live_nowcasts_quarterly", "live_a": "live_nowcasts_annual_la",
             "fa": "features_annual_la"}
    if name == "lights":
        return pd.read_parquet(INTERIM / "lights_monthly.parquet")
    return pd.read_parquet(PROC / f"{files[name]}.parquet")


@st.cache_data
def geojson(level: str) -> dict:
    return json.loads((PROC / f"{level}_simplified.geojson").read_text(encoding="utf-8"))


@st.cache_data
def report_table(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / "tables" / f"{name}.csv")


# ----------------------------------------------------------------------------- UI helpers
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
.hero h1 {font-size: 2.3rem; line-height: 1.15; margin-bottom: .3rem;}
.hero p {font-size: 1.1rem; color: #52514e; margin-top: 0;}
.answer {border-left: 5px solid #2a78d6; background: #eef4fc; padding: 1rem 1.2rem;
         border-radius: 8px; font-size: 1.08rem; margin: .6rem 0 1.2rem 0;}
.live {border-left: 5px solid #eb6834; background: #fdf1eb; padding: .8rem 1.1rem;
       border-radius: 8px; margin-bottom: 1rem;}
.step {font-size: .95rem; color: #52514e;}
.step b {color: #0b0b0b; font-size: 1.02rem;}
.small {font-size: .85rem; color: #52514e;}
div[data-testid="stMetricValue"] {font-size: 2rem;}
</style>
"""


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "answer") -> None:
    st.markdown(f"<div class='{kind}'>{text}</div>", unsafe_allow_html=True)


def how_to_read(text: str) -> None:
    st.caption(f"💡 **How to read this:** {text}")


def style(fig: go.Figure, height: int = 420, legend_bottom: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, font=dict(size=13, color=INK),
        margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(font_size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor=GRIDC, zerolinecolor=INK2)
    fig.update_yaxes(gridcolor=GRIDC, zerolinecolor=INK2)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    return fig


def download(df: pd.DataFrame, label: str, name: str) -> None:
    st.download_button(f"⬇️ {label}", df.to_csv(index=False).encode("utf-8"), f"{name}.csv",
                       "text/csv")


def best_by_group(metrics: list[dict]) -> pd.DataFrame:
    """Lowest-error model within each data group."""
    mt = pd.DataFrame(metrics)
    mt["group"] = mt["model"].map(model_group)
    return mt.sort_values("mae").groupby("group", as_index=False).first()


def style_map(fig: go.Figure, height: int = 560, colorbar_title: str = "") -> go.Figure:
    """Large, frameless UK map with a horizontal colour bar underneath."""
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(coloraxis_colorbar=dict(orientation="h", y=-0.02, yanchor="top", x=0.5,
                                              xanchor="center", len=0.7, thickness=12,
                                              title=dict(text=colorbar_title, side="top")))
    style(fig, height)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=40))
    return fig


def quarter_axis(fig: go.Figure, quarters) -> go.Figure:
    """One tick per year on a categorical quarter axis (labels like 2016Q1)."""
    qs = sorted(set(map(str, quarters)))
    ticks = [q for q in qs if q.endswith("Q1")]
    fig.update_xaxes(tickvals=ticks, ticktext=[q[:4] for q in ticks], tickangle=0)
    return fig
