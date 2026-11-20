"""Raster helpers: monthly file discovery, zone rasterisation and pixel areas.

The VIIRS grid is geographic (15 arc-seconds), so pixel area shrinks with latitude.
Every aggregate that sums radiance over space is weighted by true pixel area (km2).
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from src.utils.config import path

EARTH_RADIUS_KM = 6371.0088


def monthly_files() -> dict[str, Path]:
    """{'YYYY-MM': path} for all cropped monthly rasters, sorted."""
    out = {}
    for p in sorted((path("raw") / "viirs").glob("uk_*.tif")):
        m = re.fullmatch(r"uk_(\d{4})(\d{2})\.tif", p.name)
        if m:
            out[f"{m.group(1)}-{m.group(2)}"] = p
    return out


def read_month(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        a = src.read(1).astype("float64")
    a[~np.isfinite(a)] = np.nan
    a[a < 0] = 0.0
    return a


def grid_profile(reference: Path) -> dict:
    with rasterio.open(reference) as src:
        return {"transform": src.transform, "shape": (src.height, src.width), "crs": src.crs}


def pixel_area_km2(transform, shape) -> np.ndarray:
    """Exact spherical area of each pixel row, broadcast to the grid shape."""
    h, w = shape
    lat_top = transform.f + transform.e * np.arange(h)
    lat_bot = lat_top + transform.e
    dlon = np.radians(abs(transform.a))
    row_area = EARTH_RADIUS_KM**2 * dlon * np.abs(
        np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))
    return np.repeat(row_area[:, None], w, axis=1)


def zone_raster(zones: gpd.GeoDataFrame, transform, shape) -> tuple[np.ndarray, list[str]]:
    """Rasterise polygons to integer labels (0 = outside every zone).

    Pixel-centre rule (all_touched=False) so each pixel belongs to at most one zone.
    """
    codes = list(zones["code"])
    shapes = ((geom, i + 1) for i, geom in enumerate(zones.geometry))
    lab = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32",
                    all_touched=False)
    return lab, codes
