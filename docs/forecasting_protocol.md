# Forecasting protocol

## Point-in-time integrity

Every source carries an **availability date** (see `docs/data_sources.md`). For a target period
the pipeline fixes a **forecast origin** and filters each source to `available_date <= origin`
*before* computing features (`src/features/build_features.py`). Four dates are kept distinct:
observation period, processing (imputation) status, public availability, and forecast origin.

| | Track B (quarterly, ITL1) | Track A (annual, LAD) |
|---|---|---|
| Target | q/q log growth ×100 of GVA (CP, SA); and growth minus E&W | real (CVM) log growth ×100; and growth minus UK mean |
| Origin | quarter end + 35 days | year end + 35 days |
| Latest known target at origin | t-2 | Y-2 |
| Economic features | regional growth t-2..t-4, E&W growth t-2, relative growth t-2, historical mean, production share, UK monthly GDP (available months of t vs t-1), regional claimant-count q/q change (and relative) | LA growth Y-2, Y-3, UK monthly GDP (available months of Y vs Y-1), LA claimant-count y/y change (and relative) |
| Satellite features | mean over genuine months of: y/y log change in total radiance, lit area, radiance s.d.; y/y change in bright-pixel share and top-10% concentration; change in y/y light growth vs previous quarter; relative-to-E&W light growth; number of genuine months | same, over genuine months of the year |
| Test periods | 2016Q1-2025Q3 (39 quarters × 10 regions) | 2016-2024 (9 years × 361 LAs) |
| Live period | 2025Q4 | 2025 |

## Walk-forward evaluation

Expanding window. At test period P, models are trained only on rows with targets published by
P's origin, i.e. periods ≤ P-2 (`src/models/train.py::walk_forward`). COVID quarters
(2020Q1-2021Q2) are excluded from Track B *training*, and evaluation is reported both with and
without them. Track A training targets are winsorised at the 1st/99th percentiles (tiny LAs such as
the City of London and Isles of Scilly have extreme growth rates); evaluation uses raw targets.

## Prediction intervals

Split-conformal: 80% interval = point + [10th, 90th] percentile of the model's own **past
out-of-sample errors** whose targets were already published (falls back to in-sample residuals
until 30 exist). Coverage is evaluated out of sample.

## Comparison tests

Panel errors are cross-sectionally dependent (all regions share the period's macro shock), so:

* **Diebold-Mariano** (Harvey-Leybourne-Newbold correction, Student-t) on the time series of
  period-average absolute and squared errors;
* **Block bootstrap** (2,000 draws) of the relative MAE improvement, resampling whole periods;
* for Track A, an LA-resampling bootstrap is also reported, with the caveat that it ignores the
  common year shock and therefore overstates precision.

## Leakage tests

`tests/test_no_leakage.py` corrupts every observation published after an origin (GDP, lights,
UK GDP, claimants) and asserts the features at that origin are unchanged; it also asserts the
walk-forward engine never trains on a period later than P-2.
