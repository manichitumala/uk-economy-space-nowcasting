# 🛰️ Nowcasting the UK Economy from Space

**Can satellite night-time lights improve nowcasts of UK regional GDP before official statistics
are published?** This repository is an end-to-end, point-in-time research system that answers
the question with real data. The answer: **not for short-run growth.** It also shows where the
signal does exist and why it fails.

[📄 Full research report](reports/research_report.md) ·
[📊 Model evaluation tables](reports/model_evaluation.md) ·
[🔎 Data audit](docs/data_availability_audit.md) ·
[🧪 Forecasting protocol](docs/forecasting_protocol.md)

![Genuine observations](reports/figures/01_genuine_observation_share.png)

## Key findings

| | Result |
|---|---|
| **Data audit** | 77 of 165 monthly VIIRS composites (47%) are *not observations* for England & Wales. Summer months are filled with a scaled copy of the **same year's December**, which is future information. They are detected per pixel and masked. |
| **Levels** | Lights locate economic activity: the elasticity of LA GDP to radiance is **0.73**, R² **0.55** (361 local authorities). |
| **Long run** | Weak link with 2015→2024 growth divergence (t = 3.8, R² = 0.03). |
| **Annual growth** | No relationship after LA and year fixed effects (t = −0.07, within-R² ≈ 0). |
| **Quarterly nowcasts (Track B)** | Best economic + satellite model vs economic only: **+0.7% MAE** (95% CI −3.3% to +4.3%, DM p = 0.73). Not significant in any model family. |
| **Annual LA nowcasts (Track A)** | The conventional bridge equation is best. Adding satellite data makes it 1.3% worse. An apparent +3.3% Ridge gain comes entirely from one small-sample year. |
| **Shock detection** | Central London fell **−25 log points** relative to the national trend in the Jan–Mar 2021 lockdown (z ≈ −3.3). The April–June 2020 lockdown cannot be seen because it falls in the imputed summer window. |
| **Live nowcast** | 2025 Q4 regional growth (unverified: ONS has not yet published Q4) and 2025 LA growth (due ~Sept 2027). |

## What's inside

```
src/ingestion/      satellite.py (stream 2 GB/yr zips, crop UK) · ons.py (ONS xlsx, ONS API, Nomis) · boundaries.py
src/processing/     quality_checks.py (December-imputation detector) · raster_processing.py (spherical
                    pixel area, zone rasters) · regional_aggregation.py (ITL1 + 361 LADs × 165 months)
                    economic.py (tidy targets with release dates and vintages)
src/features/       build_features.py: features computed only from data public at each forecast origin
src/models/         baselines.py · train.py (walk-forward, conformal 80% PIs) · evaluate.py (DM-HLN,
                    block bootstrap) · analysis.py (econometrics, event study, SHAP, figures) · predict.py
dashboard/app.py    Streamlit + Plotly: live estimates map, satellite explorer, performance,
                    alternative-data research, explainability
tests/              20 tests incl. leakage tests that corrupt all post-origin data
docs/ · reports/    proposal, sources, audit, protocol · research report, tables, figures, results.json
```

**Stack:** Python 3.11 · rasterio · GeoPandas · pandas · DuckDB/Parquet · scikit-learn · XGBoost ·
LightGBM · statsmodels · SHAP · Plotly · Streamlit · pytest · GitHub Actions · Docker.

## Methodology in one paragraph

Monthly 15-arc-second VIIRS rasters (HKU v4, CC BY 4.0, built from EOG composites) are cropped to
the UK. They are aggregated with exact spherical pixel areas to the nine English ITL1 regions plus
Wales and to 361 local authorities. Each pixel is then tested for being a December-scaled copy.
Every source carries an availability date. Features for a target period are computed only from
data public at its origin (quarter or year end + 35 days), and training uses only targets
published by then (≤ t-2). Three feature sets are compared under identical expanding-window
backtests: **A** economic (lagged regional GDP, UK monthly GDP, claimant count), **B** satellite
and **C** both. The models are naive baselines, a bridge equation, Ridge, Elastic Net, XGBoost and
LightGBM. They are compared with Diebold–Mariano tests on period-average losses and a
period-block bootstrap, because all regions share each period's shock.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt

streamlit run dashboard/app.py      # works immediately from the committed processed data
pytest                              # unit, leakage and data-quality tests

python run_pipeline.py              # full rebuild: ONS + Nomis + boundaries + ~28 GB raster stream
python run_pipeline.py --from features   # rebuild from cached rasters / aggregates
```

or with Docker: `docker build -t uk-space-nowcast . && docker run -p 8501:8501 uk-space-nowcast`.

No credentials are needed. All sources are open: ONS and Nomis under OGL v3, and figshare for
the HKU night-time lights.

## Honest limitations

* Quarterly regional GDP exists in **one vintage** (ONS indicative, July 2026, current prices), so
  lagged-GDP features are pseudo-real-time. Annual vintages show mean revisions of 1.26 pp.
* The open satellite product has no cloud-free-coverage band. A NASA Black Marble (VNP46A3)
  robustness run needs Earthdata credentials and is the main outstanding extension.
* Release lags other than the annual GDP dates are documented assumptions.
* With 33–39 quarterly blocks, gains below about 5% cannot be detected. The confidence intervals
  rule out larger ones.

## Data credits

Cheng, H., Geng, M., Li, X. et al. (2026), *ESSD* 18, 3449–3479,
[doi:10.25442/hku.31321315](https://doi.org/10.25442/hku.31321315) (CC BY 4.0). Office for
National Statistics and Nomis: Crown copyright, Open Government Licence v3.0. Boundaries: ONS
Open Geography Portal (OGL v3). Code: MIT.
