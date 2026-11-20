# Research proposal

## Question

**Can satellite observations of artificial night-time light improve nowcasts of UK regional
economic activity before official statistics are published?**

Night-time radiance is not a measurement of GDP. The relevant question is whether it carries
information *beyond* what conventional, already-published indicators provide, at the moment a
nowcast would actually be made.

## Why it is hard in the UK

* **Publication gaps.** ONS paused experimental quarterly regional GDP in July 2023 and published
  an *indicative*, current-price series (Q1 2012-Q3 2025) only on 31 July 2026. Annual regional
  and local-authority GDP arrive about 21 months after the reference year.
* **Weak physics.** The UK is fully electrified; services output need not change lighting; LED
  street-light retrofits reduce VIIRS-measured radiance (the VIIRS Day/Night Band is insensitive
  below ~500 nm); energy-saving dimming policies change radiance independently of output.
* **Latitude.** At 50-56°N, summer nights are too short for usable VIIRS composites.

## Hypotheses

| | Hypothesis | Test |
|---|---|---|
| H1 | Lights locate economic activity (levels) | Cross-sectional elasticity of LA GDP on radiance |
| H2 | Lights track long-run divergence | Long-difference regressions, 2014/15 → 2019/24 |
| H3 | Lights track short-run growth | Two-way fixed-effects annual growth regressions |
| H4 | Lights improve *point-in-time* nowcasts | Walk-forward backtests, A vs C, DM tests |
| H5 | Lights detect large local shocks | COVID-19 event study, central London vs national |

## Two research tracks

* **Track B (primary): quarterly regional nowcasting.** 9 English ITL1 regions + Wales; target:
  q/q growth in GVA (current prices, SA); also growth *relative* to England & Wales.
* **Track A (supporting): annual local-authority nowcasting.** 361 local authorities; target:
  real (CVM) GDP growth, and growth relative to the UK mean.

## Experimental design

Three feature sets under identical protocols: **A** economic only, **B** satellite only,
**C** economic + satellite. Models: naive baselines, OLS bridge equation, Ridge, Elastic Net,
XGBoost, LightGBM. If C does not beat A out of sample, that is reported as the result.

See [forecasting_protocol.md](forecasting_protocol.md) for the point-in-time rules.
