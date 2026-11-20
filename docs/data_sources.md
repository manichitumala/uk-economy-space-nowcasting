# Data sources

| Dataset | Provider / licence | Coverage used | Access | Module |
|---|---|---|---|---|
| Monthly VIIRS-like night-time lights v4 (15″ ≈ 500 m, nW cm⁻² sr⁻¹) | Cheng, Geng, Li, … Zhou (2026), *ESSD* 18, 3449-3479; [doi:10.25442/hku.31321315](https://doi.org/10.25442/hku.31321315), CC BY 4.0 | Apr 2012 - Dec 2025 (165 months), cropped to the UK | Open (figshare), no login | `src/ingestion/satellite.py` |
| Quarterly regional GDP, England and Wales (indicative, July 2026) | ONS, OGL v3 | Q1 2012 - Q3 2025, ITL1, current prices, SA, £m, with industry detail | Open xlsx | `src/ingestion/ons.py` |
| Regional GDP, all ITL regions (6 vintages: 1998-2019 … 1998-2024) | ONS, OGL v3 | Annual, CP and CVM | Open xlsx | `src/ingestion/ons.py` |
| Regional GDP, local authorities (1998-2024 edition) | ONS, OGL v3 | 361 LADs (Dec 2024 codes), annual CVM & CP | Open xlsx | `src/ingestion/ons.py` |
| UK monthly GDP index (ECY2, CVM SA) | ONS, OGL v3 | Jan 1997 - Sep 2026 | ONS time-series API | `src/ingestion/ons.py` |
| Claimant count by region and by LAD (NM_162_1) | ONS via Nomis, OGL v3 | Jan 2012 - Dec 2025 | Nomis API | `src/ingestion/ons.py` |
| ITL1 (Jan 2021) & LAD (Dec 2024) boundaries, BGC | ONS Open Geography Portal, OGL v3 | UK | ArcGIS REST | `src/ingestion/boundaries.py` |

## Why not NASA Black Marble or Earth Engine directly?

The original plan named NASA VNP46A2/A3 (via LAADS or Google Earth Engine) and NOAA/EOG monthly
composites. All three require authenticated accounts (NASA Earthdata token, EOG Keycloak login,
Earth Engine Cloud project). The HKU product is built *from* the EOG monthly composites
(2012 onward), is openly downloadable, and keeps the native EOG 15-arc-second grid, so the
pipeline is fully reproducible without credentials. The extraction module is source-agnostic:
any monthly GeoTIFF on the same grid can be dropped into `data/raw/viirs/` as `uk_YYYYMM.tif`.

The cost of this choice is documented in [data_availability_audit.md](data_availability_audit.md):
the HKU product gap-fills missing pixels with a scaled copy of the same year's December, and it
does not ship the EOG cloud-free-coverage band.

## Release dates used for point-in-time rules

| Statistic | Rule | Source |
|---|---|---|
| Annual regional GDP, editions 2018-2024 | 19 Dec 2019, 26 May 2021, 30 May 2022, 25 Apr 2023, 24 Apr 2024, 17 Apr 2025, 23 Sep 2026 | ONS previous-releases page |
| Quarterly regional GDP | quarter end + 130 days (cadence of the 2019-2023 experimental series) | assumption, `configs/data_config.yaml` |
| UK monthly GDP | month end + 42 days | assumption (first estimate ~6 weeks) |
| Claimant count | month end + 18 days | assumption (mid-month publication) |
| Night-time lights composite | month end + 30 days | assumption (EOG monthly latency) |
