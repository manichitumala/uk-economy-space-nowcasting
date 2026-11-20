"""Walk-forward (expanding-window) backtests for both research tracks.

At each test period P the training set contains only rows whose *target* had been
published by P's forecast origin:
  * Track B (quarterly regional GDP, ~130-day lag): quarters <= P-2
  * Track A (annual local-authority GDP, ~21-month lag): years <= P-2

80% prediction intervals are split-conformal: the 10th/90th percentiles of the model's own
*past out-of-sample* errors (only errors whose targets were already published), falling
back to in-sample residuals until 30 such errors exist.

Usage:  python -m src.models.train
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.models import baselines as B
from src.utils.config import path

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- feature sets
Q_ECON = ["g_lag2", "g_lag3", "g_lag4", "ew_lag2", "rel_lag2", "g_hist_mean",
          "uk_gdp_nowcast", "claim_qoq", "claim_rel", "production_share"]
Q_SAT = ["light_yoy", "light_yoy_rel", "lit_area_yoy", "std_yoy", "bright_share_yoy_d",
         "top10_share_yoy_d", "light_yoy_d", "light_obs_months"]
A_ECON = ["g_lag2", "g_lag3", "uk_gdp_nowcast", "claim_yoy", "claim_rel"]
A_SAT = ["light_yoy", "light_yoy_rel", "lit_area_yoy", "std_yoy", "bright_share_yoy_d",
         "top10_share_yoy_d", "light_obs_months"]


@dataclass
class Spec:
    name: str
    features: list[str]
    make: Callable
    group: str  # baseline | A_econ | B_sat | C_econ_sat


def specs(track: str, target: str) -> list[Spec]:
    econ, sat = (Q_ECON, Q_SAT) if track == "B" else (A_ECON, A_SAT)
    lag_col = "rel_lag2" if (track == "B" and target == "target_rel") else "g_lag2"
    bridge_cols = (["uk_gdp_nowcast", "claim_qoq"] if track == "B" else ["uk_gdp_nowcast", "claim_yoy"])
    if target.endswith("_rel"):
        bridge_cols = ["claim_rel"]
    out = [
        Spec("zero", [lag_col], lambda: B.ConstantModel("zero"), "baseline"),
        Spec("historical_mean", [lag_col], lambda: B.ConstantModel("mean"), "baseline"),
        Spec("last_published", [lag_col], lambda: B.ColumnModel(0), "baseline"),
        Spec("bridge_conventional", bridge_cols, B.bridge, "A_econ"),
    ]
    for set_name, cols in [("A_econ", econ), ("B_sat", sat), ("C_econ_sat", econ + sat)]:
        out.append(Spec(f"ridge_{set_name}", cols, B.ridge, set_name))
        out.append(Spec(f"enet_{set_name}", cols, B.elastic_net, set_name))
        if set_name != "B_sat":
            out.append(Spec(f"xgb_{set_name}", cols, B.xgboost, set_name))
            out.append(Spec(f"lgbm_{set_name}", cols, B.lightgbm, set_name))
    out.append(Spec("ridge_B_sat_plus_bridge", bridge_cols + sat, B.ridge, "C_econ_sat"))
    return out


# ----------------------------------------------------------------------------- engine
def walk_forward(df: pd.DataFrame, period: str, target: str, spec_list: list[Spec],
                 first_test, gap: int = 2, exclude_train: pd.Series | None = None,
                 clip_quantile: float | None = None, min_oos_resid: int = 30) -> pd.DataFrame:
    """Expanding-window backtest. Returns long frame of predictions with 80% intervals."""
    periods = sorted(df[period].unique())
    tests = [p for p in periods if p >= first_test]
    preds, resid_store = [], {s.name: [] for s in spec_list}
    for P in tests:
        test = df[df[period] == P]
        train_mask = (df[period] <= P - gap) & df[target].notna()
        if exclude_train is not None:
            train_mask &= ~exclude_train
        train = df[train_mask]
        if train.empty:
            continue
        y = train[target].to_numpy(float)
        if clip_quantile:
            lo, hi = np.quantile(y, [clip_quantile, 1 - clip_quantile])
            y = np.clip(y, lo, hi)
        for s in spec_list:
            m = s.make()
            Xtr, Xte = train[s.features].to_numpy(float), test[s.features].to_numpy(float)
            m.fit(Xtr, y)
            p = m.predict(Xte)
            # conformal interval from published past OOS errors, else in-sample residuals
            past = [e for (pp, e) in resid_store[s.name] if pp <= P - gap]
            errs = np.concatenate(past) if past and sum(map(len, past)) >= min_oos_resid \
                else y - m.predict(Xtr)
            lo_q, hi_q = np.quantile(errs, [0.10, 0.90])
            yt = test[target].to_numpy(float)
            preds.append(pd.DataFrame({
                period: P, "code": test["code"].to_numpy(), "model": s.name, "group": s.group,
                "y": yt, "pred": p, "lo80": p + lo_q, "hi80": p + hi_q,
                "n_train": len(train),
            }))
            if np.isfinite(yt).all():
                resid_store[s.name].append((P, yt - p))
    return pd.concat(preds, ignore_index=True)


def fit_final(df: pd.DataFrame, target: str, spec: Spec, exclude: pd.Series | None = None):
    mask = df[target].notna() if exclude is None else df[target].notna() & ~exclude
    train = df[mask]
    m = spec.make().fit(train[spec.features].to_numpy(float), train[target].to_numpy(float))
    return m, train


# ----------------------------------------------------------------------------- runners
def run_track_b() -> pd.DataFrame:
    q = pd.read_parquet(path("processed") / "features_quarterly.parquet")
    q["quarter"] = pd.PeriodIndex(q["quarter"], freq="Q")
    q["qidx"] = q["quarter"].map(lambda p: p.ordinal)
    out = []
    for target in ["target", "target_rel"]:
        res = walk_forward(q, "qidx", target, specs("B", target),
                           first_test=pd.Period("2016Q1").ordinal, gap=2,
                           exclude_train=q["covid"])
        res["target_name"] = target
        out.append(res)
    res = pd.concat(out, ignore_index=True)
    res["quarter"] = res["qidx"].map(lambda o: str(pd.Period(ordinal=o, freq="Q")))
    return res


def run_track_a() -> pd.DataFrame:
    a = pd.read_parquet(path("processed") / "features_annual_la.parquet")
    out = []
    for target in ["target", "target_rel"]:
        res = walk_forward(a, "year", target, specs("A", target), first_test=2016, gap=2,
                           clip_quantile=0.01)
        res["target_name"] = target
        out.append(res)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rb = run_track_b()
    rb.to_parquet(path("processed") / "backtest_quarterly.parquet", index=False)
    log.info("Track B backtest rows: %d", len(rb))
    ra = run_track_a()
    ra.to_parquet(path("processed") / "backtest_annual_la.parquet", index=False)
    log.info("Track A backtest rows: %d", len(ra))


if __name__ == "__main__":
    main()
