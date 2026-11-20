# Data availability and comparability audit

This audit was the first milestone. Its findings shaped every later modelling decision.

## 1. Economic targets

**Quarterly regional GDP.** The only consistent quarterly series is ONS's *indicative* England &
Wales dataset of 31 July 2026: Q1 2012-Q3 2025, ITL1, **current prices**, seasonally adjusted,
"official statistics in development". Consequences:

* **One vintage only.** No real-time vintages exist for the new methods; lagged regional GDP
  features are therefore *pseudo*-real-time (latest values, but only for quarters that would have
  been published by the origin).
* **Nominal, not real.** No regional deflators exist quarterly. The primary target is nominal q/q
  growth; the *relative* target (region minus England & Wales) removes the common price component.
* **Partly interpolated.** ONS states that regional proportions come from the Annual Regional
  Accounts and are constrained to national quarterly totals; only the tail uses quarterly inputs.
  Within-year regional dynamics are therefore smooth by construction.
* **Sample size.** 10 regions × 54 growth quarters = 540 observations; after lags, 510 with
  features, 450 with complete economic features.

**Annual regional GDP.** Real (CVM) growth for ITL1-ITL3 and for 361 local authorities, with six
published vintages for ITL1. Revisions are large (`reports/tables/annual_itl1_revisions.csv`):

| | value |
|---|---|
| Region-years compared (first release vs Sept 2026) | 50 |
| Mean absolute revision | **1.26 pp** |
| Largest absolute revision | 3.8 pp |
| Sign changes | 5 |

London's 2019 growth, for example, was first published as 2.2% and now stands at 0.7%.

## 2. Satellite data: the summer-imputation problem

For each month the pipeline computes, per pixel, the ratio to the **same year's December**. If a
large share of lit pixels share one exact ratio, those pixels are scalar copies, not observations
(`src/processing/quality_checks.py`). Share of genuine England & Wales lit pixels:

```
        J     F     M     A     M     J     J     A     S     O     N     D
2012                     0%    0%    0%    0%    0%   99%  100%  100%  100%
2013   26%   26%   26%   26%    0%    0%    0%    0%   25%   26%   26%  100%
2014  100%  100%  100%    3%    0%    0%    0%    0%  100%  100%  100%  100%
...   (2015-2025: Jan-Mar and Oct-Dec ≈100%; Apr 0-40%; May-Aug 0%; Sep 44-100%)
2025  100%  100%  100%   35%    0%    0%    0%    0%  100%  100%   93%  100%
```

Full matrix: `reports/tables/genuine_share_by_month.csv`, figure `01_genuine_observation_share.png`.

* **May-August are always 100% imputed** (twilight during the short UK summer nights; EOG excludes
  stray-light-affected data). April is 60-100% imputed, September sometimes partly.
* **Jan-Nov 2013 north of ~52°N is imputed**, leaving only southern England genuine that year.
* **These pixels leak the future.** A June value equal to k × December embeds information from
  six months later. Using it in a Q2 nowcast would be look-ahead leakage. All months with
  `valid_share < 0.5` are therefore excluded from features.
* Result: **88 of 165 months** are usable for England & Wales. Genuine months per quarter:
  Q1 2.6, **Q2 0.0**, Q3 0.7, Q4 2.8. The satellite signal cannot inform Q2 at all, and Q3 rests
  on September alone.

## 3. Satellite data: residual cloud contamination

The HKU product does not include EOG's cloud-free-observation count. Genuine months still show
non-economic swings, e.g. **London radiance fell ~45% in Nov-Dec 2024** (persistent
"anticyclonic gloom") and recovered in January 2025. The imputation then copied the depressed
December 2024 into summer 2024. Mitigation: monthly year-on-year log changes are winsorised at
±30 log points; relative-to-national features remove common atmospheric shocks.

## 4. Geography

* LAD boundaries (Dec 2024) match the 361 LA codes of the 2024 GDP edition **exactly**; Nomis
  claimant-count geography TYPE424 matches the same 361 codes.
* ITL1 (Jan 2021) boundaries are used for regions. Zonal statistics use the pixel-centre rule and
  true spherical pixel area (pixels shrink ~22% between 50°N and 60°N).

## 5. Timeliness: is the satellite signal early enough?

At the Track B origin (quarter end + 35 days) the latest published regional GDP is for t-2, so the
nowcast is effectively 2 steps ahead; satellite data for all genuine months of t are available.
For Track A (origin: year end + 35 days) LA GDP is known only to Y-2. Satellite timing is not the
constraint; **satellite information content** is.
