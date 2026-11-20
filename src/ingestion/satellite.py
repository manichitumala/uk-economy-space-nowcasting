"""Download monthly VIIRS-like night-time lights and crop them to the UK.

Source: Cheng, Geng, Li, ... Zhou (2026), "A temporally consistent global 500 m-resolution
monthly VIIRS-like nighttime light dataset (1992-2024)", ESSD 18, 3449-3479.
Data: https://doi.org/10.25442/hku.31321315 (CC BY 4.0).

For 2012-2024 the files are distributed as one zip per year (~2 GB, deflate-compressed),
so each archive is streamed to a temporary file, every monthly raster is read through
GDAL's /vsizip/ handler with a UK window, and the archive is deleted. 2025 months are
standalone tiled GeoTIFFs and are cropped the same way after download.

Output: data/raw/viirs/uk_YYYYMM.tif (float32, EPSG:4326, 15 arc-second grid).

Usage:  python -m src.ingestion.satellite [--years 2012 2013 ...] [--workers 3]
"""
from __future__ import annotations

import argparse
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rasterio
import requests
from rasterio.windows import from_bounds

from src.utils.config import load_config, path

log = logging.getLogger(__name__)
FIGSHARE_API = "https://api.figshare.com/v2/articles/{id}"


def list_files() -> dict[str, str]:
    """Return {file name: download url} for the figshare article."""
    cfg = load_config()["satellite"]
    r = requests.get(FIGSHARE_API.format(id=cfg["figshare_article_id"]), timeout=60)
    r.raise_for_status()
    return {f["name"]: f["download_url"] for f in r.json()["files"]}


def out_dir() -> Path:
    d = path("raw") / "viirs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def crop_to_uk(src_path: str, dst: Path) -> None:
    """Read the UK window of a global raster and write it as a compressed GeoTIFF."""
    bbox = load_config()["satellite"]["uk_bbox"]
    with rasterio.open(src_path) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        profile = src.profile.copy()
        profile.update(
            width=data.shape[1], height=data.shape[0],
            transform=src.window_transform(win),
            compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
            crs="EPSG:4326",
        )
    tmp = dst.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(data, 1)
    tmp.replace(dst)


def _download(url: str, dst: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 << 20):
                fh.write(chunk)


def fetch_year(year: int, files: dict[str, str]) -> list[Path]:
    """Fetch and crop all months of one year. Idempotent: existing outputs are skipped."""
    od = out_dir()
    wanted = [od / f"uk_{year}{m:02d}.tif" for m in range(1, 13)]
    if all(p.exists() for p in wanted):
        return wanted
    tmp_dir = path("raw") / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if f"{year}.zip" in files:
        zpath = tmp_dir / f"{year}.zip"
        log.info("downloading %s.zip", year)
        _download(files[f"{year}.zip"], zpath)
        with zipfile.ZipFile(zpath) as z:
            members = [n for n in z.namelist() if re.search(r"SVDNB_npp_\d{6}\.tif$", n)]
        for m in members:
            ym = re.search(r"(\d{6})\.tif$", m).group(1)
            dst = od / f"uk_{ym}.tif"
            if not dst.exists():
                crop_to_uk(f"/vsizip/{zpath.as_posix()}/{m}", dst)
                log.info("wrote %s", dst.name)
        zpath.unlink()
    else:
        for mth in range(1, 13):
            name = f"SVDNB_npp_{year}{mth:02d}.tif"
            dst = od / f"uk_{year}{mth:02d}.tif"
            if dst.exists() or name not in files:
                continue
            tpath = tmp_dir / name
            _download(files[name], tpath)
            crop_to_uk(str(tpath), dst)
            tpath.unlink()
            log.info("wrote %s", dst.name)
    return [p for p in wanted if p.exists()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2012, 2026)))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    files = list_files()
    with ThreadPoolExecutor(args.workers) as ex:
        for year, got in zip(args.years, ex.map(lambda y: fetch_year(y, files), args.years)):
            log.info("year %s: %d monthly rasters", year, len(got))


if __name__ == "__main__":
    main()
