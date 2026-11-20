"""Point-in-time feature construction.

For every target period a *forecast origin* is defined. Each source is first filtered to
observations whose public-availability date is on or before that origin, and only then are
features computed. This makes look-ahead leakage structurally impossible and testable
(see tests/test_no_leakage.py).

Track B (quarterly): region x quarter, origin = quarter end + ORIGIN_LAG_Q days.
Track A (annual):    local authority x year, origin = year end + ORIGIN_LAG_A days.

Usage:  python -m src.features.build_features
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.processing import economic as econ
from src.utils.config import load_config, path

log = logging.getLogger(__name__)

ORIGIN_LAG_Q = 35   # nowcast made 5 weeks after the quarter ends
ORIGIN_LAG_A = 35   # annual nowcast made 5 weeks after the year ends
MIN_VALID_SHARE = 0.5
# Residual cloud contamination (no cloud-free-count band is distributed) produces month-level
# swings of +/-50% (e.g. London, Nov-Dec 2024 "anticyclonic gloom"); monthly log changes are
# winsorised at this many log points before aggregation.
YOY_CLIP = 30.0

LIGHT_COLS = ["light_yoy", "lit_area_yoy", "std_yoy", "bright_share_yoy_d", "top10_share_yoy_d",
              "light_yoy_d", "light_obs_months"]
ECON_COLS = ["g_lag2", "g_lag3", "g_lag4", "ew_lag2", "rel_lag2", "g_hist_mean",
             "uk_gdp_nowcast", "claim_qoq", "claim_rel", "production_share"]


# --------------------------------------------------------------------------- lights
def lights_with_availability(lights: pd.DataFrame) -> pd.DataFrame:
    lag = load_config()["satellite"]["availability_lag_days"]
    out = lights.copy()
    out["available_date"] = out["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    out["genuine"] = out["valid_share"] >= MIN_VALID_SHARE
    return out


def monthly_light_yoy(lights: pd.DataFrame) -> pd.DataFrame:
    """Month-level year-on-year log changes, defined only when both months are genuine."""
    df = lights.sort_values(["code", "month"]).copy()
    prev = df[["code", "month", "total_radiance", "lit_area_km2", "std_radiance",
               "bright_share", "top10_share", "genuine"]].copy()
    prev["month"] = prev["month"] + pd.DateOffset(years=1)
    df = df.merge(prev, on=["code", "month"], how="left", suffixes=("", "_ly"))
    both = df["genuine"] & df["genuine_ly"].fillna(False).astype(bool)

    def ldiff(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 100 * np.log(df[a] / df[b])
        return v.where(both & np.isfinite(v)).clip(-YOY_CLIP, YOY_CLIP)

    df["light_yoy"] = ldiff("total_radiance", "total_radiance_ly")
    df["lit_area_yoy"] = ldiff("lit_area_km2", "lit_area_km2_ly")
    df["std_yoy"] = ldiff("std_radiance", "std_radiance_ly")
    df["bright_share_yoy_d"] = (100 * (df["bright_share"] - df["bright_share_ly"])).where(both)
    df["top10_share_yoy_d"] = (100 * (df["top10_share"] - df["top10_share_ly"])).where(both)
    return df


def _agg_light_period(m: pd.DataFrame, period_col: str) -> pd.DataFrame:
    cols = ["light_yoy", "lit_area_yoy", "std_yoy", "bright_share_yoy_d", "top10_share_yoy_d"]
    g = m.groupby(["code", period_col])
    out = g[cols].mean()
    out["light_obs_months"] = g["light_yoy"].count()
    out["valid_share_mean"] = g["valid_share"].mean()
    return out.reset_index()


# --------------------------------------------------------------------------- Track B
def quarterly_origin(q: pd.Period) -> pd.Timestamp:
    return q.end_time.normalize() + pd.Timedelta(days=ORIGIN_LAG_Q)


def quarterly_features_at(target_q: pd.Period, gdp: pd.DataFrame, lights_m: pd.DataFrame,
                          ukm: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    """All features for every region for one target quarter, using only data public at origin."""
    origin = quarterly_origin(target_q)
    lag_days = load_config()["ons"]["quarterly_regional_gdp_lag_days"]

    # --- regional GDP: published lag_days after the reference quarter ends
    gdp = gdp.assign(available_date=gdp["quarter"].map(lambda p: p.end_time.normalize())
                     + pd.Timedelta(days=lag_days))
    known = gdp[gdp["available_date"] <= origin]
    ew = econ.england_wales_total(known)
    rows = []
    for code, g in known.groupby("code"):
        g = g.set_index("quarter").sort_index()
        last = g.index.max()
        r = {"code": code, "quarter": target_q, "origin": origin,
             "gdp_last_known": last, "gdp_steps_ahead": (target_q - last).n}
        for k, lag in [(2, 0), (3, 1), (4, 2)]:
            q = last - lag
            r[f"g_lag{k}"] = g["growth_qoq"].get(q, np.nan)
        r["g_hist_mean"] = g["growth_qoq"].mean()
        r["production_share"] = g["production_share"].iloc[-1]
        e = ew.set_index("quarter")["ew_growth_qoq"]
        r["ew_lag2"] = e.get(last, np.nan)
        r["rel_lag2"] = r["g_lag2"] - r["ew_lag2"]
        rows.append(r)
    feats = pd.DataFrame(rows)

    # --- UK monthly GDP (national): available months of the target quarter vs previous quarter
    u = ukm[ukm["available_date"] <= origin].set_index("month")["uk_gdp_index"]
    q_months = pd.period_range(target_q.start_time, target_q.end_time, freq="M").to_timestamp()
    p_months = pd.period_range((target_q - 1).start_time, (target_q - 1).end_time,
                               freq="M").to_timestamp()
    avail = [m for m in q_months if m in u.index]
    if avail and all(m in u.index for m in p_months):
        feats["uk_gdp_nowcast"] = 100 * np.log(u[avail].mean() / u[p_months].mean())
    else:
        feats["uk_gdp_nowcast"] = np.nan
    feats["uk_gdp_months"] = len(avail)

    # --- regional claimant count: quarter average vs previous quarter (log change)
    c = claim[claim["available_date"] <= origin]
    cq = c[c["month"].isin(q_months)].groupby("code")["claimants"].mean()
    cp = c[c["month"].isin(p_months)].groupby("code")["claimants"].mean()
    claim_qoq = 100 * np.log(cq / cp)
    feats["claim_qoq"] = feats["code"].map(claim_qoq)
    feats["claim_rel"] = feats["claim_qoq"] - claim_qoq.mean()

    # --- satellite: genuine months of the target quarter that are public by the origin
    lm = lights_m[(lights_m["available_date"] <= origin)]
    cur = lm[lm["month"].isin(q_months)].copy()
    cur["quarter"] = target_q
    agg = _agg_light_period(cur, "quarter")
    prev = lm[lm["month"].isin(p_months)].copy()
    prev["quarter"] = target_q
    prev_agg = _agg_light_period(prev, "quarter")[["code", "light_yoy"]]
    agg = agg.merge(prev_agg.rename(columns={"light_yoy": "light_yoy_prev"}), on="code", how="left")
    agg["light_yoy_d"] = agg["light_yoy"] - agg["light_yoy_prev"]
    feats = feats.merge(agg.drop(columns=["quarter", "light_yoy_prev"]), on="code", how="left")
    feats["light_obs_months"] = feats["light_obs_months"].fillna(0)
    ew_light = feats["light_yoy"].mean()
    feats["light_yoy_rel"] = feats["light_yoy"] - ew_light
    return feats


def build_quarterly() -> pd.DataFrame:
    gdp = econ.quarterly_regional_gdp()
    lights = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    lights_m = monthly_light_yoy(lights_with_availability(lights[lights["level"] == "itl1"]))
    ukm, claim = econ.uk_monthly_gdp(), econ.claimant_count()

    quarters = sorted(gdp["quarter"].unique())
    # one genuinely live quarter beyond the last published target
    live = [quarters[-1] + 1]
    frames = [quarterly_features_at(q, gdp, lights_m, ukm, claim) for q in quarters[4:] + live]
    feats = pd.concat(frames, ignore_index=True)

    tgt = gdp[["code", "region", "quarter", "growth_qoq"]].merge(
        econ.england_wales_total(gdp), on="quarter")
    tgt["target_rel"] = tgt["growth_qoq"] - tgt["ew_growth_qoq"]
    out = feats.merge(tgt.rename(columns={"growth_qoq": "target"}), on=["code", "quarter"],
                      how="left")
    out["region"] = out["code"].map(load_config()["regions"])
    out["is_live"] = out["target"].isna()
    out["covid"] = out["quarter"].between(pd.Period("2020Q1"), pd.Period("2021Q2"))
    out["qtr"] = out["quarter"].map(lambda p: p.quarter)
    return out.sort_values(["quarter", "code"]).reset_index(drop=True)


# --------------------------------------------------------------------------- Track A
def annual_origin(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=ORIGIN_LAG_A)


def build_annual() -> pd.DataFrame:
    """Local-authority panel: real GDP growth vs satellite and conventional indicators.

    Annual target data are published ~21 months after the year ends, so at the origin
    (5 weeks after year end) the latest known LA growth is for year Y-2.
    """
    la = econ.annual_la()
    la = la.sort_values(["code", "year"])
    la["growth"] = la.groupby("code")["gdp_cvm"].transform(lambda s: 100 * np.log(s).diff())
    la_cp = econ.annual_la("Table 5", "gdp_cp")
    lights = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    lm = monthly_light_yoy(lights_with_availability(lights[lights["level"] == "lad"]))
    ukm = econ.uk_monthly_gdp()
    claim = econ.claimant_count_la()

    rows = []
    for y in range(2013, 2026):  # 2025 is a live nowcast (published ~Sept 2027)
        origin = annual_origin(y)
        cur = lm[(lm["month"].dt.year == y) & (lm["available_date"] <= origin)].copy()
        cur["year"] = y
        a = _agg_light_period(cur, "year")
        lev = cur.groupby("code")["total_radiance"].mean().rename("light_level")
        a = a.merge(lev, on="code", how="left")
        # national GDP growth from monthly index: available months of y vs same months y-1
        u = ukm[ukm["available_date"] <= origin].set_index("month")["uk_gdp_index"]
        ms = [m for m in u.index if m.year == y]
        uk = 100 * np.log(u[ms].mean() / u[[m - pd.DateOffset(years=1) for m in ms]].mean())
        a["uk_gdp_nowcast"] = uk
        c = claim[claim["available_date"] <= origin]
        cy = c[c["month"].dt.year == y].groupby("code")["claimants"].mean()
        cl = c[c["month"].dt.year == y - 1].groupby("code")["claimants"].mean()
        a["claim_yoy"] = a["code"].map(100 * np.log(cy / cl))
        a["origin"] = origin
        rows.append(a)
    panel = pd.concat(rows, ignore_index=True)

    g = la.pivot(index="code", columns="year", values="growth")
    panel["target"] = [g.at[c, y] if (c in g.index and y in g.columns) else np.nan
                       for c, y in zip(panel["code"], panel["year"])]
    # pseudo-real-time lags: annual regional GDP for Y-2 is published by the origin for Y
    panel["g_lag2"] = [g.at[c, y - 2] if c in g.index else np.nan
                       for c, y in zip(panel["code"], panel["year"])]
    panel["g_lag3"] = [g.at[c, y - 3] if c in g.index else np.nan
                       for c, y in zip(panel["code"], panel["year"])]
    panel = panel.merge(la[["code", "name", "itl1_region"]].drop_duplicates("code"), on="code")
    lvl = la_cp.pivot(index="code", columns="year", values="gdp_cp")
    panel["gdp_cp_level"] = [lvl.at[c, y] if y in lvl.columns else np.nan
                             for c, y in zip(panel["code"], panel["year"])]
    panel["is_live"] = panel["target"].isna()
    panel["uk_growth"] = panel.groupby("year")["target"].transform("mean")
    panel["target_rel"] = panel["target"] - panel["uk_growth"]
    panel["light_yoy_rel"] = panel["light_yoy"] - panel.groupby("year")["light_yoy"].transform("mean")
    panel["claim_rel"] = panel["claim_yoy"] - panel.groupby("year")["claim_yoy"].transform("mean")
    return panel.sort_values(["year", "code"]).reset_index(drop=True)


def main() -> None:
    import duckdb
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    q = build_quarterly()
    a = build_annual()
    pdir = path("processed")
    qq = q.assign(quarter=q["quarter"].astype(str), gdp_last_known=q["gdp_last_known"].astype(str))
    qq.to_parquet(pdir / "features_quarterly.parquet", index=False)
    a.to_parquet(pdir / "features_annual_la.parquet", index=False)
    lights = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    with duckdb.connect(str(path("warehouse"))) as con:
        for name, df in [("features_quarterly", qq), ("features_annual_la", a), ("lights_monthly", lights)]:
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
    log.info("quarterly features %s, annual LA panel %s", q.shape, a.shape)


if __name__ == "__main__":
    main()
