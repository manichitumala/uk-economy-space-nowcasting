"""Quality control for the monthly night-time light rasters.

Key issue (documented in docs/data_availability_audit.md): at UK latitudes the short
summer nights leave the EOG monthly composites with no cloud- and stray-light-free
observations. The HKU product fills those pixels with ``k * December radiance`` of the
*same year*. Such pixels therefore carry no information about the month itself and,
worse, embed information from up to eight months in the future. They must be masked
before any point-in-time modelling.

Detection: a pixel is "imputed" when its ratio to the same-year December value equals the
month's modal ratio to within a relative tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_TOL = 1e-3
MIN_DEC_RADIANCE = 0.5


def modal_ratio(month: np.ndarray, december: np.ndarray) -> float:
    """Most common month/December ratio among lit pixels (rounded histogram mode)."""
    m = (december > MIN_DEC_RADIANCE) & (month > 0) & np.isfinite(month)
    if m.sum() < 100:
        return np.nan
    r = np.round(month[m] / december[m], 4)
    vals, counts = np.unique(r, return_counts=True)
    return float(vals[np.argmax(counts)])


def imputed_mask(month: np.ndarray, december: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that are scalar copies of the same-year December field."""
    k = modal_ratio(month, december)
    out = np.zeros(month.shape, dtype=bool)
    if not np.isfinite(k) or k == 0:
        return out
    m = (december > MIN_DEC_RADIANCE) & np.isfinite(month)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(month / (k * december) - 1)
    out[m] = rel[m] < RATIO_TOL
    # Only treat as imputation if a substantial share of lit pixels follows the rule;
    # isolated coincidences in a genuine month are ignored.
    if out[m].mean() < 0.05:
        out[:] = False
    return out


def check_panel(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> list[str]:
    """Generic integrity checks for a tidy panel. Returns a list of problems (empty = OK)."""
    problems = []
    dup = df.duplicated(keys).sum()
    if dup:
        problems.append(f"{dup} duplicate {keys} rows")
    for c in value_cols:
        n = df[c].isna().sum()
        if n:
            problems.append(f"{n} missing values in {c}")
        if (df[c] < 0).any():
            problems.append(f"negative values in {c}")
    return problems
