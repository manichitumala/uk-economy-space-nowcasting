"""Research analyses: forecast comparisons, significance tests, econometric diagnostics,
revision analysis, shock event study, explainability, and report figures.

Outputs
  reports/results.json      headline numbers consumed by the report and dashboard
  reports/tables/*.csv      full result tables
  reports/figures/*.png     static figures for the report

Usage:  python -m src.models.analysis
"""
from __future__ import annotations

import json
import logging
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models import evaluate as E
from src.models.train import Q_ECON, Q_SAT, specs, fit_final
from src.processing import economic as econ
from src.utils.config import ROOT, path, regions

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

REPORTS = ROOT / "reports"
TABLES, FIGS = REPORTS / "tables", REPORTS / "figures"
for d in (TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)

# Reference palette (validated): blue, orange, aqua; recessive ink and grid.
C_ECON, C_SAT, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COVID = ("2020Q1", "2021Q2")
FAMILIES = ["ridge", "enet", "xgb", "lgbm"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _save(fig, name):
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


# ============================================================================ forecasts
def metric_table(bt: pd.DataFrame, period: str, rel: bool) -> pd.DataFrame:
    rows = []
    for m, d in bt.groupby("model"):
        rows.append({"model": m, "group": d["group"].iloc[0], **E.summarise(d),
                     "cov80": E.coverage(d["y"], d["lo80"], d["hi80"]),
                     "width80": float((d["hi80"] - d["lo80"]).mean())})
    t = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    if not rel:
        t = t.drop(columns="dir_acc")
    return t


def comparison_tests(bt: pd.DataFrame, period: str) -> pd.DataFrame:
    """C (econ+sat) vs A (econ) within each model family, plus bridge vs bridge+sat."""
    pairs = [(f"{f}_A_econ", f"{f}_C_econ_sat") for f in FAMILIES]
    pairs += [("bridge_conventional", "ridge_B_sat_plus_bridge"),
              ("historical_mean", "ridge_A_econ"), ("historical_mean", "ridge_C_econ_sat")]
    rows = []
    for a, b in pairs:
        if not {a, b} <= set(bt["model"]):
            continue
        L = E.period_losses(bt, period, [a, b])
        L2 = E.period_losses(bt, period, [a, b], loss="sq")
        dm = E.diebold_mariano(L[a].to_numpy(), L[b].to_numpy())
        dm2 = E.diebold_mariano(L2[a].to_numpy(), L2[b].to_numpy())
        bs = E.block_bootstrap_mae_diff(bt, period, a, b)
        rows.append({"benchmark": a, "challenger": b,
                     "mae_benchmark": E.mae(bt[bt.model == a]["y"], bt[bt.model == a]["pred"]),
                     "mae_challenger": E.mae(bt[bt.model == b]["y"], bt[bt.model == b]["pred"]),
                     **bs, "dm_stat_abs": dm["dm_stat"], "dm_p_abs": dm["p_value"],
                     "dm_stat_sq": dm2["dm_stat"], "dm_p_sq": dm2["p_value"], "T": dm["T"]})
    return pd.DataFrame(rows)


def track_b(results: dict) -> pd.DataFrame:
    bt = pd.read_parquet(path("processed") / "backtest_quarterly.parquet")
    bt = bt[bt["y"].notna()]
    cov = bt["quarter"].between(*COVID)
    out = {}
    for tgt in ["target", "target_rel"]:
        for lab, mask in [("all", bt["target_name"] == tgt),
                          ("ex_covid", (bt["target_name"] == tgt) & ~cov)]:
            sub = bt[mask]
            mt = metric_table(sub, "qidx", rel=tgt.endswith("rel"))
            ct = comparison_tests(sub, "qidx")
            mt.to_csv(TABLES / f"trackB_{tgt}_{lab}_metrics.csv", index=False)
            ct.to_csv(TABLES / f"trackB_{tgt}_{lab}_tests.csv", index=False)
            out[f"{tgt}_{lab}"] = {"metrics": mt.round(4).to_dict("records"),
                                   "tests": ct.round(4).to_dict("records")}
    # by region (ex-COVID, nominal growth), A vs C ridge + best baseline
    sub = bt[(bt["target_name"] == "target") & ~cov]
    reg = (sub[sub["model"].isin(["historical_mean", "ridge_A_econ", "ridge_C_econ_sat",
                                  "lgbm_A_econ", "lgbm_C_econ_sat"])]
           .groupby(["code", "model"]).apply(lambda d: E.mae(d["y"], d["pred"]))
           .unstack().rename(index=regions()))
    reg.to_csv(TABLES / "trackB_mae_by_region.csv")
    out["mae_by_region"] = reg.round(4).reset_index().to_dict("records")
    # by quarter-of-year: do satellite months (Q1/Q4 have 3 genuine months) help there?
    sub = sub.assign(qtr=sub["quarter"].str[-1])
    byq = (sub[sub["model"].isin(["ridge_A_econ", "ridge_C_econ_sat", "lgbm_A_econ", "lgbm_C_econ_sat"])]
           .groupby(["qtr", "model"]).apply(lambda d: E.mae(d["y"], d["pred"])).unstack())
    byq.to_csv(TABLES / "trackB_mae_by_quarter_of_year.csv")
    out["mae_by_quarter_of_year"] = byq.round(4).reset_index().to_dict("records")
    results["track_b"] = out
    return bt


def track_a(results: dict) -> pd.DataFrame:
    bt = pd.read_parquet(path("processed") / "backtest_annual_la.parquet")
    bt = bt[bt["y"].notna()]
    out = {}
    for tgt in ["target", "target_rel"]:
        for lab, mask in [("all", bt["target_name"] == tgt),
                          ("ex_covid", (bt["target_name"] == tgt) & ~bt["year"].isin([2020, 2021]))]:
            sub = bt[mask]
            mt = metric_table(sub, "year", rel=tgt.endswith("rel"))
            ct = comparison_tests(sub, "year")
            mt.to_csv(TABLES / f"trackA_{tgt}_{lab}_metrics.csv", index=False)
            ct.to_csv(TABLES / f"trackA_{tgt}_{lab}_tests.csv", index=False)
            out[f"{tgt}_{lab}"] = {"metrics": mt.round(4).to_dict("records"),
                                   "tests": ct.round(4).to_dict("records")}
    # LA-level resampling bootstrap (years are few): resample LAs with replacement
    sub = bt[bt["target_name"] == "target"]
    rng = np.random.default_rng(0)
    for a, b in [("ridge_A_econ", "ridge_C_econ_sat"), ("bridge_conventional", "ridge_B_sat_plus_bridge")]:
        la_l = {m: sub[sub.model == m].assign(l=lambda d: (d.y - d.pred).abs()).groupby("code")["l"].mean()
                for m in (a, b)}
        codes = la_l[a].index.to_numpy()
        imp = []
        for _ in range(2000):
            s = rng.choice(codes, len(codes))
            imp.append(100 * (la_l[a][s].mean() - la_l[b][s].mean()) / la_l[a][s].mean())
        out[f"la_bootstrap_{a}_vs_{b}"] = {"ci_low": float(np.percentile(imp, 2.5)),
                                          "ci_high": float(np.percentile(imp, 97.5))}
    results["track_a"] = out
    return bt


# ============================================================================ econometrics
def _twfe(df, y, xs):
    def tw(c):
        return df[c] - df.groupby("code")[c].transform("mean") - df.groupby("year")[c].transform("mean") + df[c].mean()
    X = pd.concat([tw(c) for c in xs], axis=1)
    r = sm.OLS(tw(y), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["code"])[0]})
    return {"coef": r.params.round(5).to_dict(), "t": r.tvalues.round(3).to_dict(),
            "p": r.pvalues.round(4).to_dict(), "within_r2": float(r.rsquared), "n": int(r.nobs)}


def econometrics(results: dict) -> dict:
    a = pd.read_parquet(path("processed") / "features_annual_la.parquet")
    h = a[~a["is_live"]]
    lm = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    lad = lm[(lm["level"] == "lad") & (lm["valid_share"] >= 0.5)]
    out = {}

    # levels: cross-section by year
    lev = []
    for y in range(2014, 2025):
        L = lad[lad["month"].dt.year == y].groupby("code")["total_radiance"].mean()
        x = h[h["year"] == y].set_index("code").join(L.rename("L")).dropna(subset=["L", "gdp_cp_level"])
        x = x[x["L"] > 0]
        r = sm.OLS(np.log(x["gdp_cp_level"]), sm.add_constant(np.log(x["L"]))).fit(cov_type="HC1")
        lev.append({"year": y, "elasticity": r.params.iloc[1], "se": r.bse.iloc[1], "r2": r.rsquared, "n": len(x)})
    lev = pd.DataFrame(lev)
    lev.to_csv(TABLES / "levels_cross_section.csv", index=False)
    out["levels"] = lev.round(4).to_dict("records")

    # long differences
    g = h.pivot(index="code", columns="year", values="target")
    ld_rows, ld_frames = [], {}
    for y0, y1 in [(2014, 2019), (2015, 2024), (2014, 2024)]:
        m0 = lad[lad["month"].dt.year == y0].assign(mm=lambda d: d["month"].dt.month)
        m1 = lad[lad["month"].dt.year == y1].assign(mm=lambda d: d["month"].dt.month)
        j = m0.merge(m1, on=["code", "mm"])
        dl = 100 * np.log(j["total_radiance_y"] / j["total_radiance_x"]).groupby(j["code"]).mean()
        dy = g.loc[:, y0 + 1:y1].sum(axis=1)
        d = pd.concat([dl.rename("dlight"), dy.rename("dgdp")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        r = sm.OLS(d["dgdp"], sm.add_constant(d["dlight"])).fit(cov_type="HC1")
        rk = d.rank().corr().iloc[0, 1]
        ld_rows.append({"from": y0, "to": y1, "beta": r.params.iloc[1], "t": r.tvalues.iloc[1],
                        "p": r.pvalues.iloc[1], "r2": r.rsquared, "spearman": rk, "n": len(d)})
        ld_frames[(y0, y1)] = d
    ld = pd.DataFrame(ld_rows)
    ld.to_csv(TABLES / "long_differences.csv", index=False)
    out["long_differences"] = ld.round(4).to_dict("records")

    # two-way fixed effects annual growth regressions
    p = h.dropna(subset=["light_yoy", "target", "claim_yoy"])
    p = p[p["year"] >= 2015]
    out["twfe"] = {
        "light_only": _twfe(p, "target", ["light_yoy"]),
        "light_and_area": _twfe(p, "target", ["light_yoy", "lit_area_yoy"]),
        "light_and_claimant": _twfe(p, "target", ["light_yoy", "claim_yoy"]),
        "light_ex_covid": _twfe(p[~p["year"].isin([2020, 2021])], "target", ["light_yoy"]),
    }
    # heterogeneity: urban (top-quartile radiance density) vs rest
    dens = h.groupby("code")["light_level"].mean()
    urban = dens >= dens.quantile(0.75)
    out["twfe"]["urban_top_quartile"] = _twfe(p[p["code"].map(urban)], "target", ["light_yoy"])
    out["twfe"]["non_urban"] = _twfe(p[~p["code"].map(urban)], "target", ["light_yoy"])

    # national-level (mean across LAs) correlation by year
    nat = h.groupby("year")[["light_yoy", "target"]].mean()
    out["national_by_year"] = nat.round(3).reset_index().to_dict("records")
    out["national_corr"] = float(nat.corr().iloc[0, 1])
    results["econometrics"] = out
    return {"levels_year": 2019, "long_diff": ld_frames}


def quarterly_correlations(results: dict):
    q = pd.read_parquet(path("processed") / "features_quarterly.parquet")
    q = q[~q["is_live"] & ~q["covid"]]
    rows = []
    for col in Q_SAT + Q_ECON:
        for tgt in ["target", "target_rel"]:
            d = q[[col, tgt]].dropna()
            rows.append({"feature": col, "target": tgt, "pearson": d.corr().iloc[0, 1],
                         "spearman": d.rank().corr().iloc[0, 1], "n": len(d)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "trackB_feature_correlations.csv", index=False)
    results["trackB_feature_correlations"] = t.round(4).to_dict("records")


# ============================================================================ data audit
def data_audit(results: dict):
    lm = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    itl = lm[(lm["level"] == "itl1") & lm["code"].isin(regions())]
    w = itl.assign(w=itl["zone_area_km2"])
    t = w.groupby("month").apply(lambda x: (x["valid_share"] * x["w"]).sum() / x["w"].sum())
    mat = t.to_frame("v").assign(y=t.index.year, m=t.index.month).pivot(index="y", columns="m", values="v")
    mat.to_csv(TABLES / "genuine_share_by_month.csv")
    genuine_months = int((t >= 0.5).sum())
    results["audit"] = {"months_total": int(len(t)), "months_genuine_ew": genuine_months,
                        "share_genuine": genuine_months / len(t),
                        "genuine_share_matrix": mat.round(3).reset_index().to_dict("records")}
    q = pd.read_parquet(path("processed") / "features_quarterly.parquet")
    results["audit"]["mean_genuine_months_by_quarter"] = (
        q.groupby("qtr")["light_obs_months"].mean().round(3).to_dict())

    # revisions of annual ITL1 real growth across vintages
    v = econ.annual_growth_vintages()
    v = v[v["code"].isin(regions())]
    first = v[v["year"] == v["vintage"]][["code", "year", "growth_cvm"]].rename(columns={"growth_cvm": "first"})
    latest = v[v["vintage"] == v["vintage"].max()][["code", "year", "growth_cvm"]].rename(columns={"growth_cvm": "latest"})
    rv = first.merge(latest, on=["code", "year"])
    rv = rv[rv["year"] < v["vintage"].max()]
    rv["revision"] = rv["latest"] - rv["first"]
    rv.to_csv(TABLES / "annual_itl1_revisions.csv", index=False)
    results["audit"]["annual_revisions"] = {
        "mean_abs_revision_pp": float(rv["revision"].abs().mean()),
        "max_abs_revision_pp": float(rv["revision"].abs().max()),
        "sign_changes": int((np.sign(rv["first"]) != np.sign(rv["latest"])).sum()),
        "n": int(len(rv)),
        "mean_abs_first_release_growth": float(rv["first"].abs().mean()),
    }
    return mat, rv


# ============================================================================ shock study
def shock_event_study(results: dict) -> pd.DataFrame:
    """Central-London CBD lights vs the national LA average, relative to the same month of 2019."""
    lm = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    lad = lm[lm["level"] == "lad"].copy()
    lad.loc[lad["valid_share"] < 0.5, "total_radiance"] = np.nan
    p = lad.pivot(index="month", columns="code", values="total_radiance")
    cbd = ["E09000001", "E09000033"]   # City of London, Westminster
    ew = [c for c in p.columns if c[0] in "EW"]
    base = p[p.index.year == 2019].groupby(p[p.index.year == 2019].index.month).mean()
    dev = pd.DataFrame(index=p.index)
    for name, cols in [("cbd", cbd), ("national", ew)]:
        cur = p[cols].sum(axis=1, min_count=len(cols))
        b = base[cols].sum(axis=1).reindex(p.index.month).to_numpy()
        dev[name] = 100 * np.log(cur.to_numpy() / b)
    dev["cbd_minus_national"] = dev["cbd"] - dev["national"]
    # placebo noise band: same statistic vs. 2018 baseline over 2015-2019 (pre-COVID)
    base18 = p[p.index.year == 2017].groupby(p[p.index.year == 2017].index.month).mean()
    pre = p[(p.index.year >= 2015) & (p.index.year <= 2019) & (p.index.year != 2017)]
    pre_dev = []
    for name, cols in [("cbd", cbd), ("national", ew)]:
        cur = pre[cols].sum(axis=1, min_count=len(cols))
        b = base18[cols].sum(axis=1).reindex(pre.index.month).to_numpy()
        pre_dev.append(100 * np.log(cur.to_numpy() / b))
    placebo = pd.Series(pre_dev[0] - pre_dev[1]).dropna()
    sd = float(placebo.std())
    lock = dev.loc["2021-01-01":"2021-03-01", "cbd_minus_national"]
    post = dev.loc["2021-09-01":"2021-12-01", "cbd_minus_national"]
    pre_c = dev.loc["2019-09-01":"2020-02-01", "cbd_minus_national"]
    dev.to_csv(TABLES / "shock_event_study.csv")
    results["shock"] = {
        "placebo_sd_pp": sd,
        "lockdown_jan_mar_2021_mean_pp": float(lock.mean()),
        "lockdown_z": float(lock.mean() / (sd / np.sqrt(lock.notna().sum()))),
        "reopening_sep_dec_2021_mean_pp": float(post.mean()),
        "pre_covid_sep19_feb20_mean_pp": float(pre_c.mean()),
    }
    return dev.assign(placebo_sd=sd)


# ============================================================================ explainability
def explainability(results: dict):
    import shap
    q = pd.read_parquet(path("processed") / "features_quarterly.parquet")
    hist = q[~q["is_live"]].copy()
    out = {}
    sp = {s.name: s for s in specs("B", "target")}
    m, train = fit_final(hist, "target", sp["xgb_C_econ_sat"], exclude=hist["covid"])
    X = train[sp["xgb_C_econ_sat"].features]
    sv = shap.TreeExplainer(m).shap_values(X.to_numpy(float))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    imp_share = (imp / imp.sum())
    imp_share.to_csv(TABLES / "shap_xgb_C_trackB.csv", header=["share"])
    out["shap_xgb_C_trackB"] = imp_share.round(4).to_dict()
    out["shap_satellite_share"] = float(imp_share[[c for c in imp_share.index if c in Q_SAT]].sum())
    # ridge standardised coefficients
    r, train = fit_final(hist, "target", sp["ridge_C_econ_sat"], exclude=hist["covid"])
    coefs = pd.Series(r[-1].coef_, index=sp["ridge_C_econ_sat"].features).sort_values(key=np.abs, ascending=False)
    coefs.to_csv(TABLES / "ridge_C_trackB_std_coefs.csv", header=["coef"])
    out["ridge_C_trackB_std_coefs"] = coefs.round(4).to_dict()
    out["ridge_C_trackB_alpha"] = float(r[-1].alpha_)
    results["explainability"] = out
    return imp_share


# ============================================================================ figures
def figures(bt_b, bt_a, mat, rv, ld_info, dev, shap_share, results):
    _style()
    # 1. genuine-observation heatmap
    fig, ax = plt.subplots(figsize=(8, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(mat)), mat.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(mat.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, "" if v >= 0.995 else f"{v:.0%}",
                    ha="center", va="center", fontsize=7, color=INK if v < 0.6 else "white")
    ax.set_title("Share of England & Wales lit pixels that are genuine observations")
    fig.colorbar(im, ax=ax, fraction=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
    ax.text(0, -0.13, "Summer months (and Jan-Nov 2013 north of 52N) are December-scaled copies in the source data",
            transform=ax.transAxes, fontsize=8, color=INK2)
    _save(fig, "01_genuine_observation_share")

    # 2. levels scatter
    a = pd.read_parquet(path("processed") / "features_annual_la.parquet")
    lm = pd.read_parquet(path("interim") / "lights_monthly.parquet")
    lad = lm[(lm["level"] == "lad") & (lm["valid_share"] >= 0.5) & (lm["month"].dt.year == 2019)]
    L = lad.groupby("code")["total_radiance"].mean()
    x = a[a["year"] == 2019].set_index("code").join(L.rename("L")).dropna(subset=["L"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(np.log(x["L"]), np.log(x["gdp_cp_level"]), s=14, color=C_ECON, alpha=0.6,
               edgecolor=SURFACE, linewidth=0.5)
    b = np.polyfit(np.log(x["L"]), np.log(x["gdp_cp_level"]), 1)
    xs = np.linspace(np.log(x["L"]).min(), np.log(x["L"]).max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=INK2, linewidth=1.5)
    lev19 = next(r for r in results["econometrics"]["levels"] if r["year"] == 2019)
    ax.set_title(f"Levels: lights locate economic activity (elasticity {lev19['elasticity']:.2f}, R² {lev19['r2']:.2f})")
    ax.set_xlabel("log total radiance, 2019 (genuine months)")
    ax.set_ylabel("log GDP, 2019 (£m, current prices)")
    for c, lab in [("E09000001", "City of London"), ("E09000033", "Westminster"), ("E06000053", "Isles of Scilly")]:
        if c in x.index:
            ax.annotate(lab, (np.log(x.at[c, "L"]), np.log(x.at[c, "gdp_cp_level"])), fontsize=8,
                        color=INK2, xytext=(5, -3), textcoords="offset points")
    _save(fig, "02_levels_scatter")

    # 3. long differences
    d = ld_info[(2015, 2024)]
    r = next(r for r in results["econometrics"]["long_differences"] if r["from"] == 2015)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    dd = d[(d["dlight"].abs() < 150)]
    ax.scatter(dd["dlight"], dd["dgdp"], s=14, color=C_SAT, alpha=0.6, edgecolor=SURFACE, linewidth=0.5)
    b = np.polyfit(dd["dlight"], dd["dgdp"], 1)
    xs = np.linspace(dd["dlight"].min(), dd["dlight"].max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=INK2, linewidth=1.5)
    ax.set_title(f"Long differences 2015-2024: weak link (beta {r['beta']:.2f}, R² {r['r2']:.3f})")
    ax.set_xlabel("change in log radiance x100")
    ax.set_ylabel("cumulative real GDP growth, % (log pts)")
    _save(fig, "03_long_differences")

    # 4. Track B: MAE by model (ex-COVID nominal growth)
    mt = pd.DataFrame(results["track_b"]["target_ex_covid"]["metrics"]).sort_values("mae", ascending=False)
    col = {"baseline": GRID, "A_econ": C_ECON, "B_sat": C_THIRD, "C_econ_sat": C_SAT}
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(mt["model"], mt["mae"], color=[col[g] if g != "baseline" else "#b9b8b2" for g in mt["group"]],
            height=0.7, edgecolor=SURFACE, linewidth=2)
    for yv, v in zip(mt["model"], mt["mae"]):
        ax.text(v + 0.01, yv, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("MAE, percentage points (2016Q1-2025Q3 excl. 2020Q1-2021Q2)")
    ax.set_title("Quarterly regional GDP growth nowcasts: out-of-sample MAE")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b9b8b2", label="Naive baseline"), Patch(color=C_ECON, label="A: economic only"),
                       Patch(color=C_THIRD, label="B: satellite only"), Patch(color=C_SAT, label="C: economic + satellite")],
              loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=4, fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, "04_trackB_mae_by_model")

    # 5. Track B London actual vs nowcasts
    sub = bt_b[(bt_b["target_name"] == "target") & (bt_b["code"] == "TLI")]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    act = sub[sub.model == "ridge_A_econ"].set_index("quarter")
    xq = range(len(act))
    ax.plot(xq, act["y"], color=INK, linewidth=2, label="Actual (ONS, Jul 2026)")
    for m, c, lab in [("ridge_A_econ", C_ECON, "Ridge A: economic"), ("ridge_C_econ_sat", C_SAT, "Ridge C: + satellite")]:
        s = sub[sub.model == m].set_index("quarter").reindex(act.index)
        ax.plot(xq, s["pred"], color=c, linewidth=2, label=lab)
    s = sub[sub.model == "ridge_C_econ_sat"].set_index("quarter").reindex(act.index)
    ax.fill_between(xq, s["lo80"], s["hi80"], color=C_SAT, alpha=0.15, linewidth=0, label="80% interval (C)")
    ax.set_xticks(list(xq)[::4], act.index[::4])
    ax.set_ylim(-8, 8)
    ax.set_ylabel("q/q growth, % (nominal)")
    ax.set_title("London: walk-forward nowcasts vs. outturn\n"
                 "(y-axis clipped at ±8%; COVID quarters exceed it)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "05_trackB_london_nowcasts")

    # 6. Track A MAE by model
    mt = pd.DataFrame(results["track_a"]["target_all"]["metrics"]).sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(mt["model"], mt["mae"], color=[col[g] if g != "baseline" else "#b9b8b2" for g in mt["group"]],
            height=0.7, edgecolor=SURFACE, linewidth=2)
    for yv, v in zip(mt["model"], mt["mae"]):
        ax.text(v + 0.03, yv, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("MAE, percentage points (361 local authorities, 2016-2024)")
    ax.set_title("Annual local-authority real GDP growth nowcasts: out-of-sample MAE")
    ax.legend(handles=[Patch(color="#b9b8b2", label="Naive baseline"), Patch(color=C_ECON, label="A: economic only"),
                       Patch(color=C_THIRD, label="B: satellite only"), Patch(color=C_SAT, label="C: economic + satellite")],
              loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=4, fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, "06_trackA_mae_by_model")

    # 7. Shock event study
    fig, ax = plt.subplots(figsize=(8.5, 4))
    dv = dev.loc["2020-01-01":"2022-12-01", "cbd_minus_national"]
    sd = dev["placebo_sd"].iloc[0]
    ax.axhspan(-1.96 * sd, 1.96 * sd, color=GRID, alpha=0.6, linewidth=0, label="±1.96 sd pre-COVID placebo noise")
    ax.plot(dv.index, dv, color=C_SAT, marker="o", markersize=4, linewidth=1.5,
            label="City of London + Westminster minus national (vs same month 2019)")
    for start, end in [("2020-03-23", "2020-06-15"), ("2020-11-05", "2020-12-02"), ("2021-01-06", "2021-04-12")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=C_ECON, alpha=0.08, linewidth=0)
    ax.text(pd.Timestamp("2021-01-10"), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 5, "lockdowns", fontsize=8, color=C_ECON)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("log-point deviation x100")
    ax.set_title("Shock detection: central-London lights during COVID-19\n"
                 "(baseline = same month of 2019; gaps = imputed summer months)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _save(fig, "07_covid_cbd_event_study")

    # 8. SHAP
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    s = shap_share.sort_values()
    ax.barh(s.index, s.values, color=[C_SAT if c in Q_SAT else C_ECON for c in s.index], height=0.7,
            edgecolor=SURFACE, linewidth=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("XGBoost (C) mean |SHAP| share, quarterly nominal growth")
    ax.legend(handles=[Patch(color=C_ECON, label="Economic"), Patch(color=C_SAT, label="Satellite")],
              loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, "08_shap_trackB")

    # 9. Revisions
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(rv["first"], rv["latest"], s=22, color=C_ECON, alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
    lim = [min(rv["first"].min(), rv["latest"].min()) - 1, max(rv["first"].max(), rv["latest"].max()) + 1]
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("first-published real growth, %")
    ax.set_ylabel("latest (Sept 2026) estimate, %")
    ra = results["audit"]["annual_revisions"]
    ax.set_title(f"ITL1 annual growth revisions: mean |revision| {ra['mean_abs_revision_pp']:.2f} pp")
    _save(fig, "09_annual_revisions")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    results: dict = {}
    bt_b = track_b(results)
    bt_a = track_a(results)
    ld_info = econometrics(results)["long_diff"]
    quarterly_correlations(results)
    mat, rv = data_audit(results)
    dev = shock_event_study(results)
    shap_share = explainability(results)
    figures(bt_b, bt_a, mat, rv, ld_info, dev, shap_share, results)
    with open(REPORTS / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    write_model_evaluation(results)
    log.info("analysis complete")


# ============================================================================ markdown export
def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                        for v in r) + " |")
    return "\n".join(lines)


def write_model_evaluation(results: dict) -> None:
    """Generate reports/model_evaluation.md from results.json (always in sync with the run)."""
    out = ["# Model evaluation (auto-generated by `python -m src.models.analysis`)", ""]
    labels = {"target": "growth", "target_rel": "relative growth"}
    for key, title, period in [("track_b", "Track B - quarterly regional GDP (10 regions, 2016Q1-2025Q3)", "quarters"),
                               ("track_a", "Track A - annual local-authority real GDP (361 LAs, 2016-2024)", "years")]:
        out += [f"## {title}", ""]
        for tgt in ["target", "target_rel"]:
            for lab in ["ex_covid", "all"]:
                res = results[key][f"{tgt}_{lab}"]
                mt = pd.DataFrame(res["metrics"])
                keep = [c for c in ["model", "group", "n", "mae", "rmse", "dir_acc", "cov80", "width80"] if c in mt]
                out += [f"### {labels[tgt]}, {'excluding COVID' if lab == 'ex_covid' else 'all periods'}", "",
                        _md(mt[keep]), ""]
                tt = pd.DataFrame(res["tests"])[["benchmark", "challenger", "mae_benchmark", "mae_challenger",
                                                 "improvement_pct", "ci_low", "ci_high", "dm_p_abs", "dm_p_sq", "T"]]
                out += [f"Paired tests ({period} as blocks; improvement in % of benchmark MAE, 95% bootstrap CI, "
                        "DM p-values on absolute and squared errors):", "", _md(tt), ""]
    (REPORTS / "model_evaluation.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
