"""Extract genuinely live nowcasts (targets not yet published by ONS) from the walk-forward run.

* Track B: 2025 Q4 quarterly regional GDP growth (origin 4 Feb 2026; ONS indicative series
  ends at 2025 Q3 as of November 2026).
* Track A: 2025 annual local-authority real GDP growth (origin 4 Feb 2026; due ~Sept 2027).

These are clearly separated from historical backtests everywhere they are displayed.

Usage:  python -m src.models.predict
"""
from __future__ import annotations

import logging

import pandas as pd

from src.utils.config import ROOT, path, regions

log = logging.getLogger(__name__)
LIVE_MODELS_B = ["historical_mean", "ridge_A_econ", "ridge_C_econ_sat", "lgbm_C_econ_sat"]
LIVE_MODELS_A = ["bridge_conventional", "ridge_A_econ", "ridge_C_econ_sat"]


def live_nowcasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_parquet(path("processed") / "backtest_quarterly.parquet")
    b = b[b["y"].isna() & b["model"].isin(LIVE_MODELS_B)].copy()
    b["region"] = b["code"].map(regions())
    a = pd.read_parquet(path("processed") / "backtest_annual_la.parquet")
    a = a[a["y"].isna() & a["model"].isin(LIVE_MODELS_A)].copy()
    names = pd.read_parquet(path("processed") / "features_annual_la.parquet")[["code", "name", "itl1_region"]]
    a = a.merge(names.drop_duplicates("code"), on="code", how="left")
    return b, a


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    b, a = live_nowcasts()
    b.to_parquet(path("processed") / "live_nowcasts_quarterly.parquet", index=False)
    a.to_parquet(path("processed") / "live_nowcasts_annual_la.parquet", index=False)
    cols = ["quarter", "region", "target_name", "model", "pred", "lo80", "hi80"]
    b[cols].round(3).to_csv(ROOT / "reports" / "tables" / "live_nowcast_2025Q4.csv", index=False)
    log.info("live quarterly rows %d, live annual rows %d", len(b), len(a))
    print(b[b["target_name"] == "target"].pivot(index="region", columns="model", values="pred").round(2))


if __name__ == "__main__":
    main()
