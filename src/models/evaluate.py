"""Forecast evaluation: point metrics, interval coverage and forecast-comparison tests.

Panel forecasts are cross-sectionally dependent (all regions share the same macro shock in
a given period), so comparison tests operate on the *time series of period-average loss
differentials*, and bootstrap confidence intervals resample whole periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def directional_accuracy(y, p, centre: float | np.ndarray = 0.0) -> float:
    """Share of forecasts on the correct side of `centre` (0 for growth, the mean for deviations)."""
    y, p = np.asarray(y) - centre, np.asarray(p) - centre
    return float(np.mean(np.sign(y) == np.sign(p)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def summarise(df: pd.DataFrame, y: str = "y", p: str = "pred") -> dict:
    return {"n": len(df), "mae": mae(df[y], df[p]), "rmse": rmse(df[y], df[p]),
            "dir_acc": directional_accuracy(df[y], df[p])}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test with the Harvey, Leybourne & Newbold (1997) correction.

    d_t = loss_a - loss_b; positive mean => model b is more accurate.
    Returns the statistic and two-sided p-value (Student-t, T-1 df).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 5:
        return {"dm_stat": np.nan, "p_value": np.nan, "T": T, "mean_diff": np.nan}
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: T - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "T": T, "mean_diff": dbar}
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"dm_stat": float(stat), "p_value": float(p), "T": T, "mean_diff": float(dbar)}


def period_losses(df: pd.DataFrame, period: str, models: list[str], y: str = "y",
                  loss: str = "abs") -> pd.DataFrame:
    """Period-average loss per model (columns) from a long prediction frame."""
    f = (lambda e: np.abs(e)) if loss == "abs" else (lambda e: e ** 2)
    out = {}
    for m in models:
        sub = df[df["model"] == m]
        out[m] = sub.assign(_l=f(sub[y] - sub["pred"])).groupby(period)["_l"].mean()
    return pd.DataFrame(out)


def block_bootstrap_mae_diff(df: pd.DataFrame, period: str, model_a: str, model_b: str,
                             y: str = "y", n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI for relative MAE improvement of b over a, resampling whole periods."""
    L = period_losses(df, period, [model_a, model_b], y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(L))
    a, b = L[model_a].to_numpy(), L[model_b].to_numpy()
    imp = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        imp.append(100 * (a[s].mean() - b[s].mean()) / a[s].mean())
    point = 100 * (a.mean() - b.mean()) / a.mean()
    lo, hi = np.percentile(imp, [2.5, 97.5])
    return {"improvement_pct": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_improve": float(np.mean(np.asarray(imp) > 0))}
