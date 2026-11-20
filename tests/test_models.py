import numpy as np
import pandas as pd

from src.models import baselines as B
from src.models import evaluate as E
from src.models.train import Spec, walk_forward


def test_dm_detects_clear_improvement():
    rng = np.random.default_rng(0)
    e_a = np.abs(rng.normal(0, 2, 200))
    e_b = np.abs(rng.normal(0, 1, 200))
    r = E.diebold_mariano(e_a, e_b)
    assert r["mean_diff"] > 0 and r["p_value"] < 0.01


def test_dm_no_difference_for_identical_losses():
    rng = np.random.default_rng(1)
    e = np.abs(rng.normal(size=100))
    r = E.diebold_mariano(e, e + rng.normal(0, 1e-3, 100))
    assert r["p_value"] > 0.05


def test_baselines():
    X = np.array([[1.0], [np.nan], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    assert np.allclose(B.ConstantModel("mean").fit(X, y).predict(X), 4.0)
    assert np.allclose(B.ConstantModel("zero").fit(X, y).predict(X), 0.0)
    assert np.allclose(B.ColumnModel(0).fit(X, y).predict(X), [1.0, 4.0, 3.0])


def test_conformal_intervals_cover_roughly_80pct():
    rng = np.random.default_rng(2)
    n_p, n_c = 60, 10
    df = pd.DataFrame({"p": np.repeat(np.arange(n_p), n_c), "code": list(range(n_c)) * n_p})
    df["x"] = rng.normal(size=len(df))
    df["y"] = 2 * df["x"] + rng.normal(size=len(df))
    res = walk_forward(df, "p", "y", [Spec("ridge", ["x"], B.ridge, "A")], first_test=10)
    cov = E.coverage(res["y"], res["lo80"], res["hi80"])
    assert 0.72 < cov < 0.88


def test_block_bootstrap_sign():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(40):
        y = rng.normal(size=5)
        rows += [{"p": p, "model": "bad", "y": v, "pred": v + rng.normal(0, 2)} for v in y]
        rows += [{"p": p, "model": "good", "y": v, "pred": v + rng.normal(0, .5)} for v in y]
    r = E.block_bootstrap_mae_diff(pd.DataFrame(rows), "p", "bad", "good", n_boot=300)
    assert r["improvement_pct"] > 0 and r["ci_low"] > 0
