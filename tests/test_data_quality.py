"""Integrity checks on the processed datasets (skipped when the pipeline has not been run)."""
import pandas as pd
import pytest

from src.processing.quality_checks import check_panel
from src.utils.config import ROOT, regions

PROC = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return pd.read_parquet(p)


def test_lights_panel_integrity():
    df = _load(INTERIM / "lights_monthly.parquet")
    probs = check_panel(df, ["level", "code", "month"], ["total_radiance", "lit_area_km2"])
    assert not probs, probs
    assert df["valid_share"].between(0, 1).all()
    itl = df[df["level"] == "itl1"]
    assert set(regions()) <= set(itl["code"])
    # every month present for every ITL1 region
    assert itl.groupby("code")["month"].nunique().nunique() == 1


def test_summer_months_flagged_as_imputed():
    df = _load(INTERIM / "lights_monthly.parquet")
    itl = df[(df["level"] == "itl1") & df["code"].isin(regions())]
    june = itl[itl["month"].dt.month == 6]["valid_share"]
    # 2013 is a documented exception: Jan-Nov north of ~52N is December-scaled in the source
    jan = itl[(itl["month"].dt.month == 1) & (itl["month"].dt.year != 2013)]["valid_share"]
    assert june.max() < 0.1, "June should be December-imputed at UK latitudes"
    assert jan.min() > 0.9


def test_quarterly_features_integrity():
    df = _load(PROC / "features_quarterly.parquet")
    assert not df.duplicated(["code", "quarter"]).any()
    assert set(df["code"]) == set(regions())
    hist = df[~df["is_live"]]
    assert hist["target"].notna().all()
    assert hist["target"].abs().lt(40).all(), "implausible quarterly growth"
    assert (df["gdp_steps_ahead"] >= 2).all()


def test_annual_panel_integrity():
    df = _load(PROC / "features_annual_la.parquet")
    assert not df.duplicated(["code", "year"]).any()
    assert df["code"].nunique() == 361
    hist = df[~df["is_live"]]
    assert hist["target"].notna().mean() > 0.99
