"""Aggregate monthly night-time light rasters to ITL1 regions and local authorities.

Output: data/interim/lights_monthly.parquet with one row per (level, code, month):

    total_radiance   sum of radiance x pixel area (nW cm-2 sr-1 km2)
    mean_radiance    area-weighted mean radiance over the zone
    median_radiance  median pixel radiance
    std_radiance     pixel standard deviation
    lit_area_km2     area with radiance > lit_threshold
    bright_share     share of lit area with radiance > bright_threshold
    top10_share      share of total radiance emitted by the brightest 10% of pixels
    valid_share      share of reference-lit pixels that are genuine (not December-imputed)
    zone_area_km2    zone area on the raster grid

Usage:  python -m src.processing.regional_aggregation
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion import boundaries
from src.processing import raster_processing as rp
from src.processing.quality_checks import MIN_DEC_RADIANCE, imputed_mask
from src.utils.config import load_config, path

log = logging.getLogger(__name__)


def _zone_index(labels: np.ndarray, n: int) -> list[np.ndarray]:
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(1, n + 2))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n)]


def zone_stats(rad: np.ndarray, area: np.ndarray, imputed: np.ndarray, ref_lit: np.ndarray,
               lit_thr: float, bright_thr: float) -> dict:
    """Statistics for one zone given flattened pixel arrays."""
    ok = np.isfinite(rad)
    rad, area, imputed, ref_lit = rad[ok], area[ok], imputed[ok], ref_lit[ok]
    zone_area = area.sum()
    total = float((rad * area).sum())
    lit = rad > lit_thr
    lit_area = float(area[lit].sum())
    if rad.size:
        k = max(1, int(np.ceil(0.10 * rad.size)))
        top = np.partition(rad * area, rad.size - k)[rad.size - k:].sum()
    else:
        top = np.nan
    n_ref = ref_lit.sum()
    return {
        "total_radiance": total,
        "mean_radiance": total / zone_area if zone_area else np.nan,
        "median_radiance": float(np.median(rad)) if rad.size else np.nan,
        "std_radiance": float(rad.std()) if rad.size else np.nan,
        "lit_area_km2": lit_area,
        "bright_share": float(area[rad > bright_thr].sum() / lit_area) if lit_area else 0.0,
        "top10_share": float(top / total) if total > 0 else np.nan,
        "valid_share": float(1 - (imputed & ref_lit).sum() / n_ref) if n_ref else np.nan,
        "zone_area_km2": float(zone_area),
    }


def aggregate() -> pd.DataFrame:
    cfg = load_config()["satellite"]
    files = rp.monthly_files()
    months = [m for m in files if m >= cfg["first_month"] and m <= cfg["last_month"]]
    grid = rp.grid_profile(files[months[0]])
    area = rp.pixel_area_km2(grid["transform"], grid["shape"]).ravel()

    layers = {}
    for level in ("itl1", "lad"):
        gdf = boundaries.load(level)
        lab, codes = rp.zone_raster(gdf, grid["transform"], grid["shape"])
        layers[level] = (codes, _zone_index(lab, len(codes)))

    rows, dec_cache = [], {}
    for m in months:
        year = m[:4]
        a = rp.read_month(files[m])
        dec_key = f"{year}-12"
        if dec_key not in dec_cache and dec_key in files:
            dec_cache = {dec_key: rp.read_month(files[dec_key])}
        dec = dec_cache.get(dec_key)
        if dec is not None and m != dec_key:
            imp = imputed_mask(a, dec)
            ref_lit = dec > MIN_DEC_RADIANCE
        else:  # December itself (or no reference): genuine by construction
            imp = np.zeros(a.shape, bool)
            ref_lit = a > MIN_DEC_RADIANCE
        a_f, imp_f, ref_f = a.ravel(), imp.ravel(), ref_lit.ravel()
        for level, (codes, idx) in layers.items():
            for code, ix in zip(codes, idx):
                s = zone_stats(a_f[ix], area[ix], imp_f[ix], ref_f[ix],
                               cfg["lit_threshold"], cfg["bright_threshold"])
                rows.append({"level": level, "code": code, "month": m, **s})
        log.info("%s  UK imputed share among lit pixels: %.3f", m, (imp & ref_lit).sum() / max(ref_lit.sum(), 1))
    df = pd.DataFrame(rows)
    df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = aggregate()
    out = path("interim") / "lights_monthly.parquet"
    df.to_parquet(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
