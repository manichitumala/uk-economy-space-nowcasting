"""Parse ONS workbooks into tidy, point-in-time-aware economic tables.

Every table keeps both the *reference period* and the *publication date* of the
statistic, so downstream feature builders can enforce point-in-time integrity.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.ingestion.ons import ons_dir, read_ons_timeseries_csv
from src.utils.config import load_config, regions

# Official release dates of the annual regional GDP editions (ONS previous-releases page).
ANNUAL_RELEASE_DATES = {
    2018: "2019-12-19", 2019: "2021-05-26", 2020: "2022-05-30", 2021: "2023-04-25",
    2022: "2024-04-24", 2023: "2025-04-17", 2024: "2026-09-23",
}
QUARTERLY_RELEASE_DATE = "2026-07-31"

# Nomis region (GSS) codes -> ITL1 codes
GSS_TO_ITL1 = {
    "E12000001": "TLC", "E12000002": "TLD", "E12000003": "TLE", "E12000004": "TLF",
    "E12000005": "TLG", "E12000006": "TLH", "E12000007": "TLI", "E12000008": "TLJ",
    "E12000009": "TLK", "W92000004": "TLL",
}
_Q = {"Jan to Mar": 1, "Apr to Jun": 2, "Jul to Sep": 3, "Oct to Dec": 4}


def _read_table(xlsx, sheet: str) -> pd.DataFrame:
    """Read an ONS table whose header row is the first row starting with 'ITL'."""
    raw = pd.read_excel(xlsx, sheet, header=None)
    hdr = next(i for i in range(15) if str(raw.iloc[i, 0]).strip().startswith("ITL")
               and "code" in str(raw.iloc[i, 1]).lower())
    df = raw.iloc[hdr + 1:].copy()
    cols = []
    for c in raw.iloc[hdr]:
        s = str(c).split("\n")[0].strip()
        m = re.match(r"^(\d{4})", s)
        cols.append(int(m.group(1)) if m else s)
    df.columns = cols
    return df.dropna(how="all")


# --------------------------------------------------------------------------- quarterly
def quarterly_regional_gdp() -> pd.DataFrame:
    """Tidy quarterly regional GVA (current prices, SA, £m) with q/q growth.

    Columns: code, region, quarter (Period Q), gva_cp, growth_qoq (% log-diff x100),
             production_share (production industries / all industries).
    """
    raw = pd.read_excel(ons_dir() / "quarterly_regional_gdp_2025q3.xlsx", "Table 1", header=None)
    hdr = next(i for i in range(15) if str(raw.iloc[i, 1]).startswith("ITL code"))
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).split("\n")[0].strip() for c in raw.iloc[hdr]]
    df = df.rename(columns={"ITL code": "code", "Region name": "region",
                            "SIC Industry Section Description": "industry"})
    qcols = [c for c in df.columns if any(c.startswith(k) for k in _Q)]
    long = df.melt(id_vars=["code", "region", "industry"], value_vars=qcols,
                   var_name="label", value_name="gva_cp")
    long["quarter"] = [pd.Period(year=int(l[-4:]), quarter=_Q[l[:10]], freq="Q")
                       for l in long["label"]]
    long["gva_cp"] = pd.to_numeric(long["gva_cp"], errors="coerce")
    long["industry"] = long["industry"].str.strip()

    tot = long[long["industry"] == "All industries"][["code", "region", "quarter", "gva_cp"]]
    prod = long[long["industry"] == "Total production industries"][["code", "quarter", "gva_cp"]]
    out = tot.merge(prod.rename(columns={"gva_cp": "gva_production"}), on=["code", "quarter"])
    out["production_share"] = out["gva_production"] / out["gva_cp"]
    out = out[out["code"].isin(regions())].sort_values(["code", "quarter"]).reset_index(drop=True)
    out["growth_qoq"] = out.groupby("code")["gva_cp"].transform(lambda s: 100 * np.log(s).diff())
    out["release_date"] = pd.Timestamp(QUARTERLY_RELEASE_DATE)
    return out.drop(columns="gva_production")


def england_wales_total(q: pd.DataFrame) -> pd.DataFrame:
    """Aggregate growth of the ten modelled regions (common/national component)."""
    agg = q.groupby("quarter", as_index=False)["gva_cp"].sum().sort_values("quarter")
    agg["ew_growth_qoq"] = 100 * np.log(agg["gva_cp"]).diff()
    return agg[["quarter", "ew_growth_qoq"]]


# --------------------------------------------------------------------------- annual
def annual_itl(vintage: int = 2024, table: str = "Table 10", value: str = "gdp_cvm") -> pd.DataFrame:
    """Annual regional series for every ITL level from one vintage (long format)."""
    df = _read_table(ons_dir() / f"annual_gdp_itl_vintage{vintage}.xlsx", table)
    df = df.rename(columns={"ITL": "itl_level", "ITL code": "code", "Region name": "region"})
    years = [c for c in df.columns if isinstance(c, int)]
    long = df.melt(id_vars=["itl_level", "code", "region"], value_vars=years,
                   var_name="year", value_name=value)
    long[value] = pd.to_numeric(long[value], errors="coerce")
    long["vintage"] = vintage
    long["release_date"] = pd.Timestamp(ANNUAL_RELEASE_DATES[vintage])
    return long.dropna(subset=[value])


def annual_growth_vintages() -> pd.DataFrame:
    """Real (CVM) annual GDP growth for ITL1 regions in every available vintage.

    The 'first release' of year Y is the vintage published with Y as its latest year.
    """
    frames = [annual_itl(v, "Table 12", "growth_cvm") for v in sorted(ANNUAL_RELEASE_DATES)
              if (ons_dir() / f"annual_gdp_itl_vintage{v}.xlsx").exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["itl_level"].astype(str).str.strip().isin(["ITL1", "UK"])]
    return df[["code", "region", "year", "vintage", "release_date", "growth_cvm"]]


def annual_la(table: str = "Table 10", value: str = "gdp_cvm") -> pd.DataFrame:
    """Local-authority annual GDP (2024 vintage). Default: CVM, 2023 money value, £m."""
    df = _read_table(ons_dir() / "annual_gdp_la_vintage2024.xlsx", table)
    df = df.rename(columns={"ITL1 Region": "itl1_region", "LA code": "code", "LA name": "name"})
    years = [c for c in df.columns if isinstance(c, int)]
    long = df.melt(id_vars=["itl1_region", "code", "name"], value_vars=years,
                   var_name="year", value_name=value)
    long[value] = pd.to_numeric(long[value], errors="coerce")
    long["release_date"] = pd.Timestamp(ANNUAL_RELEASE_DATES[2024])
    return long.dropna(subset=[value])


# --------------------------------------------------------------------------- monthly
def uk_monthly_gdp() -> pd.DataFrame:
    """UK monthly GDP index (ECY2, CVM SA) with m/m growth and assumed publication date."""
    df = read_ons_timeseries_csv(ons_dir() / "uk_monthly_gdp_ecy2.csv")
    df = df[df["period"].str.len() == 8].copy()
    df["month"] = pd.to_datetime(df["period"], format="%Y %b")
    df = df.rename(columns={"value": "uk_gdp_index"}).sort_values("month")
    df["uk_gdp_mom"] = 100 * np.log(df["uk_gdp_index"]).diff()
    lag = load_config()["ons"]["monthly_gdp_lag_days"]
    df["available_date"] = df["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    return df[["month", "uk_gdp_index", "uk_gdp_mom", "available_date"]].reset_index(drop=True)


def claimant_count() -> pd.DataFrame:
    """Regional claimant count (Nomis NM_162_1) mapped to ITL1, with publication date."""
    df = pd.read_csv(ons_dir() / "claimant_count_itl1.csv")
    df = df.rename(columns=str.lower)
    df["code"] = df["geography_code"].map(GSS_TO_ITL1)
    df["month"] = pd.to_datetime(df["date"], format="%Y-%m")
    lag = load_config()["ons"]["claimant_count_lag_days"]
    df["available_date"] = df["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    df = df.rename(columns={"obs_value": "claimants"})
    return df.dropna(subset=["code"])[["code", "month", "claimants", "available_date"]]


def claimant_count_la() -> pd.DataFrame:
    """Local-authority claimant count (Nomis TYPE424 = December 2024 LADs)."""
    df = pd.read_csv(ons_dir() / "claimant_count_la.csv").rename(columns=str.lower)
    df = df.rename(columns={"geography_code": "code", "obs_value": "claimants"})
    df["month"] = pd.to_datetime(df["date"], format="%Y-%m")
    lag = load_config()["ons"]["claimant_count_lag_days"]
    df["available_date"] = df["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    return df[["code", "month", "claimants", "available_date"]]
