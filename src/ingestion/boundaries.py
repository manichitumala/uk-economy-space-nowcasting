"""Download UK statistical boundaries from the ONS Open Geography Portal.

* ITL1 regions (January 2021, generalised clipped)    -> data/raw/boundaries/itl1.geojson
* Local authority districts (December 2024, BGC)      -> data/raw/boundaries/lad.geojson

Contains public sector information licensed under the Open Government Licence v3.0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import requests

from src.utils.config import load_config, path

log = logging.getLogger(__name__)


def _query_all(url: str) -> gpd.GeoDataFrame:
    """Page through an ArcGIS FeatureServer layer and return all features in EPSG:4326."""
    frames, offset = [], 0
    while True:
        params = {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                  "resultOffset": offset, "resultRecordCount": 500}
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        gdf = gpd.read_file(r.text) if r.json().get("features") else None
        if gdf is None or gdf.empty:
            break
        frames.append(gdf)
        offset += len(gdf)
        if len(gdf) < 500:
            break
    out = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=4326)
    return out


def boundary_dir() -> Path:
    d = path("raw") / "boundaries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(force: bool = False) -> dict[str, Path]:
    cfg = load_config()["boundaries"]
    out = {}
    for name, key, code_col, name_col in [("itl1", "itl1_url", "ITL121CD", "ITL121NM"),
                                          ("lad", "lad_url", "LAD24CD", "LAD24NM")]:
        dst = boundary_dir() / f"{name}.geojson"
        if force or not dst.exists():
            gdf = _query_all(cfg[key])
            gdf = gdf.rename(columns={code_col: "code", name_col: "name"})[["code", "name", "geometry"]]
            gdf.to_file(dst, driver="GeoJSON")
            log.info("wrote %s (%d features)", dst.name, len(gdf))
        out[name] = dst
    return out


def load(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(boundary_dir() / f"{name}.geojson")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    download()
