"""Point-in-time integrity tests.

The strongest check: corrupt every observation that became public *after* a forecast
origin and confirm that the features computed for that origin do not change.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.models import baselines as B
from src.models.train import Spec, walk_forward


def _synthetic_sources(seed=0):
    rng = np.random.default_rng(seed)
    codes = ["TLC", "TLI"]
    quarters = pd.period_range("2014Q1", "2019Q4", freq="Q")
    gdp = pd.DataFrame([(c, c, q, 1000 + rng.normal(), rng.uniform(.1, .3))
                        for c in codes for q in quarters],
                       columns=["code", "region", "quarter", "gva_cp", "production_share"])
    gdp["growth_qoq"] = gdp.groupby("code")["gva_cp"].transform(lambda s: 100 * np.log(s).diff())
    months = pd.date_range("2013-01-01", "2020-06-01", freq="MS")
    lights = pd.DataFrame([(c, m, rng.uniform(50, 60), rng.uniform(10, 20), rng.uniform(1, 2),
                            rng.uniform(0, .1), rng.uniform(.3, .5), 1.0)
                           for c in codes for m in months],
                          columns=["code", "month", "total_radiance", "lit_area_km2", "std_radiance",
                                   "bright_share", "top10_share", "valid_share"])
    lights_m = bf.monthly_light_yoy(bf.lights_with_availability(lights))
    ukm = pd.DataFrame({"month": months, "uk_gdp_index": 100 + rng.normal(size=len(months)).cumsum()})
    ukm["available_date"] = ukm["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=42)
    claim = pd.DataFrame([(c, m, 1000 + rng.integers(0, 100)) for c in codes for m in months],
                         columns=["code", "month", "claimants"])
    claim["available_date"] = claim["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=18)
    return gdp, lights_m, ukm, claim


def _corrupt_after(origin, gdp, lights_m, ukm, claim, lag_days=130):
    rng = np.random.default_rng(99)
    gdp, lights_m, ukm, claim = gdp.copy(), lights_m.copy(), ukm.copy(), claim.copy()
    g_avail = gdp["quarter"].map(lambda p: p.end_time.normalize()) + pd.Timedelta(days=lag_days)
    gdp.loc[g_avail > origin, ["growth_qoq", "production_share"]] = rng.normal(size=(int((g_avail > origin).sum()), 2)) * 1e3
    fut = lights_m["available_date"] > origin
    lights_m.loc[fut, ["light_yoy", "lit_area_yoy", "std_yoy"]] = 1e6
    ukm.loc[ukm["available_date"] > origin, "uk_gdp_index"] = 1e6
    claim.loc[claim["available_date"] > origin, "claimants"] = 1e9
    return gdp, lights_m, ukm, claim


@pytest.mark.parametrize("q", ["2016Q2", "2017Q4", "2019Q1"])
def test_features_invariant_to_future_data(q):
    q = pd.Period(q, freq="Q")
    src = _synthetic_sources()
    origin = bf.quarterly_origin(q)
    f1 = bf.quarterly_features_at(q, *src)
    f2 = bf.quarterly_features_at(q, *_corrupt_after(origin, *src))
    pd.testing.assert_frame_equal(f1, f2)


def test_last_known_gdp_is_two_quarters_back():
    src = _synthetic_sources()
    f = bf.quarterly_features_at(pd.Period("2018Q3", freq="Q"), *src)
    assert (f["gdp_steps_ahead"] == 2).all()


def test_satellite_months_respect_availability_lag():
    # Origin 5 weeks after quarter end and a 30-day lag: all three months usable, none later.
    src = _synthetic_sources()
    f = bf.quarterly_features_at(pd.Period("2018Q1", freq="Q"), *src)
    assert (f["light_obs_months"] == 3).all()


class _Recorder(B.ConstantModel):
    seen: list = []

    def fit(self, X, y):
        _Recorder.seen.append(np.asarray(X)[:, 0].max())
        return super().fit(X, y)


def test_walk_forward_trains_only_on_published_targets():
    df = pd.DataFrame({"p": np.repeat(np.arange(20), 3), "code": list("abc") * 20})
    df["period_copy"] = df["p"].astype(float)
    df["y"] = np.random.default_rng(0).normal(size=len(df))
    _Recorder.seen = []
    res = walk_forward(df, "p", "y", [Spec("rec", ["period_copy"], _Recorder, "baseline")],
                       first_test=10, gap=2)
    tests = sorted(res["p"].unique())
    assert all(max_seen <= P - 2 for max_seen, P in zip(_Recorder.seen, tests))
