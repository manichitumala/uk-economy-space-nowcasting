"""Plain-English labels shared by the report figures and the dashboard."""

FEATURE = {
    "g_lag2": "Region's last published growth", "g_lag3": "Region's growth one quarter earlier",
    "g_lag4": "Region's growth two quarters earlier", "ew_lag2": "England & Wales last published growth",
    "rel_lag2": "Region vs. national, last published", "g_hist_mean": "Region's long-run average growth",
    "uk_gdp_nowcast": "UK monthly GDP (latest months)", "claim_qoq": "Change in benefit claimants",
    "claim_rel": "Claimants vs. national trend", "production_share": "Share of output from industry",
    "claim_yoy": "Change in benefit claimants (yearly)",
    "light_yoy": "Night-light brightness change (yearly)", "light_yoy_rel": "Brightness change vs. national",
    "lit_area_yoy": "Change in lit-up area", "std_yoy": "Change in brightness spread",
    "bright_share_yoy_d": "Change in share of very bright areas",
    "top10_share_yoy_d": "Change in concentration of light", "light_yoy_d": "Acceleration in brightness change",
    "light_obs_months": "Months of usable satellite data",
}
SAT_FEATURES = {k for k in FEATURE if k.startswith(("light", "lit_", "std_", "bright", "top10"))}
