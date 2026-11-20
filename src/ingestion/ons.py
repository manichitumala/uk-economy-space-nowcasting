"""Download ONS economic statistics used as targets and conventional predictors.

* Quarterly regional GDP, England and Wales (indicative, July 2026, current prices, SA)
* Annual regional GDP by ITL region and by local authority (multiple vintages)
* UK monthly GDP index (ECY2) via the ONS time-series API
* Claimant count by ITL1 region via the Nomis API

Usage:  python -m src.ingestion.ons
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import path

log = logging.getLogger(__name__)
ONS = "https://www.ons.gov.uk/file?uri="
HEADERS = {"User-Agent": "Mozilla/5.0 (uk-economy-space-nowcasting research project)"}

QUARTERLY_GDP = (
    "/economy/grossdomesticproductgdp/datasets/quarterlyregionalgdpenglandandwales/"
    "july2026/quarterlyregionalgdppublicationtables2025q3.xlsx"
)
_ANNUAL_ITL = "/economy/grossdomesticproductgdp/datasets/regionalgrossdomesticproductallnutslevelregions/"
_ANNUAL_LA = "/economy/grossdomesticproductgdp/datasets/regionalgrossdomesticproductlocalauthorities/"
# vintage (last year covered) -> file; ONS changes file names between editions
ANNUAL_ITL_VINTAGES = {
    2024: _ANNUAL_ITL + "1998to2024/regionalgrossdomesticproductgdpbyallinternationalterritoriallevelitlregions.xlsx",
    2023: _ANNUAL_ITL + "1998to2023/regionalgrossdomesticproductgdpbyallinternationalterritoriallevelitlregions.xlsx",
    2022: _ANNUAL_ITL + "1998to2022/regionalgrossdomesticproductgdpbyallitlregions.xlsx",
    2021: _ANNUAL_ITL + "1998to2021/regionalgrossdomesticproductgdpallitlregions.xlsx",
    2020: _ANNUAL_ITL + "1998to2020/regionalgrossdomesticproductgdpallitlregions.xlsx",
    2019: _ANNUAL_ITL + "1998to2019/regionalgrossdomesticproductallitlregions.xlsx",
}
ANNUAL_LA = _ANNUAL_LA + "1998to2024/regionalgrossdomesticproductgdplocalauthorities.xlsx"

ONS_TS = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/grossdomesticproductgdp/timeseries/{cdid}/{dataset}"
NOMIS_CLAIMANT = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_162_1.data.csv?"
    "geography=2013265921...2013265930&date=2012-01-2025-12&gender=0&age=0&measure=1&measures=20100"
    "&select=date_name,date,geography_name,geography_code,obs_value"
)
# Local authorities (December 2024 boundaries = Nomis TYPE424); paged, Nomis caps rows at 25k
NOMIS_CLAIMANT_LA = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_162_1.data.csv?"
    "geography=TYPE424&date=2012-01-2025-12&gender=0&age=0&measure=1&measures=20100"
    "&select=date,geography_code,obs_value&recordlimit=25000&recordoffset={offset}"
)


def _get_nomis_paged(template: str, dst: Path, force: bool = False) -> Path:
    if dst.exists() and not force:
        return dst
    frames, offset = [], 0
    while True:
        r = requests.get(template.format(offset=offset), headers=HEADERS, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        frames.append(df)
        if len(df) < 25000:
            break
        offset += len(df)
    pd.concat(frames, ignore_index=True).to_csv(dst, index=False)
    log.info("downloaded %s (%d rows)", dst.name, sum(map(len, frames)))
    return dst


def ons_dir() -> Path:
    d = path("raw") / "ons"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, dst: Path, force: bool = False) -> Path:
    if dst.exists() and not force:
        return dst
    for attempt in range(6):  # ONS rate-limits bursts with HTTP 429
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
    r.raise_for_status()
    time.sleep(2)
    dst.write_bytes(r.content)
    log.info("downloaded %s (%.1f kB)", dst.name, len(r.content) / 1e3)
    return dst


def download_all(force: bool = False) -> None:
    d = ons_dir()
    _get(ONS + QUARTERLY_GDP, d / "quarterly_regional_gdp_2025q3.xlsx", force)
    for v, uri in ANNUAL_ITL_VINTAGES.items():
        _get(ONS + uri, d / f"annual_gdp_itl_vintage{v}.xlsx", force)
    _get(ONS + ANNUAL_LA, d / "annual_gdp_la_vintage2024.xlsx", force)
    _get(ONS_TS.format(cdid="ecy2", dataset="mgdp"), d / "uk_monthly_gdp_ecy2.csv", force)
    _get(NOMIS_CLAIMANT, d / "claimant_count_itl1.csv", force)
    _get_nomis_paged(NOMIS_CLAIMANT_LA, d / "claimant_count_la.csv", force)


def read_ons_timeseries_csv(p: Path) -> pd.DataFrame:
    """Parse the ONS 'generator' CSV (metadata rows followed by period,value rows)."""
    raw = pd.read_csv(p, header=None, names=["period", "value"], dtype=str)
    return raw[raw["period"].str.match(r"^\d{4}( [A-Z]{3}| Q\d)?$", na=False)].assign(
        value=lambda x: pd.to_numeric(x["value"], errors="coerce")
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    download_all()
