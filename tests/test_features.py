import numpy as np
import pandas as pd
from affine import Affine

from src.features import build_features as bf
from src.processing import raster_processing as rp
from src.processing.quality_checks import imputed_mask, modal_ratio
from src.processing.regional_aggregation import zone_stats


def test_pixel_area_matches_spherical_band():
    # 1 deg x 1 deg cell at the equator is ~12,364 km2
    t = Affine(1.0, 0, 0, 0, -1.0, 1.0)
    a = rp.pixel_area_km2(t, (1, 1))
    assert abs(a[0, 0] - 12364) < 15
    # area shrinks with latitude
    t2 = Affine(1.0, 0, 0, 0, -1.0, 60.0)
    assert rp.pixel_area_km2(t2, (1, 1))[0, 0] < 0.51 * a[0, 0]


def test_imputation_detected_for_scaled_december():
    rng = np.random.default_rng(0)
    dec = rng.gamma(2, 5, size=(200, 200))
    fake = 0.87 * dec
    real = dec * rng.uniform(0.7, 1.3, size=dec.shape)
    assert abs(modal_ratio(fake, dec) - 0.87) < 1e-3
    assert imputed_mask(fake, dec)[dec > 0.5].mean() > 0.99
    assert imputed_mask(real, dec).mean() == 0


def test_partial_imputation_is_localised():
    rng = np.random.default_rng(1)
    dec = rng.gamma(2, 5, size=(100, 100))
    month = dec * rng.uniform(0.7, 1.3, size=dec.shape)
    month[:40] = 0.9 * dec[:40]                       # northern rows gap-filled
    m = imputed_mask(month, dec)
    assert m[:40][dec[:40] > 0.5].mean() > 0.99
    assert m[40:].mean() < 0.01


def test_zone_stats_area_weighting():
    rad = np.array([0.0, 10.0, 30.0, 100.0])
    area = np.array([1.0, 1.0, 2.0, 1.0])
    s = zone_stats(rad, area, np.zeros(4, bool), np.ones(4, bool), 0.5, 20.0)
    assert s["total_radiance"] == 0 + 10 + 60 + 100
    assert abs(s["mean_radiance"] - 170 / 5) < 1e-9
    assert s["lit_area_km2"] == 4.0
    assert abs(s["bright_share"] - 3 / 4) < 1e-9
    assert s["valid_share"] == 1.0


def test_yoy_only_defined_between_genuine_months():
    months = pd.date_range("2014-01-01", "2015-12-01", freq="MS")
    df = pd.DataFrame({"code": "X", "month": months, "total_radiance": 100.0,
                       "lit_area_km2": 10.0, "std_radiance": 1.0, "bright_share": .1,
                       "top10_share": .4, "valid_share": 1.0})
    df.loc[df["month"] == "2015-03-01", "total_radiance"] = 110.0
    df.loc[df["month"] == "2014-06-01", "valid_share"] = 0.0     # imputed month
    out = bf.monthly_light_yoy(bf.lights_with_availability(df)).set_index("month")
    assert abs(out.at[pd.Timestamp("2015-03-01"), "light_yoy"] - 100 * np.log(1.1)) < 1e-9
    assert np.isnan(out.at[pd.Timestamp("2015-06-01"), "light_yoy"])
    assert np.isnan(out.at[pd.Timestamp("2014-03-01"), "light_yoy"])  # no prior year
