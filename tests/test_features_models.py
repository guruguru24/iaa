import numpy as np
import pandas as pd
import pytest

from src.dataset import onlevel_factor
from src.features import BandEncoder, add_features, feature_columns, weather_features
from src.models import ConstantModel, TweedieGLM
from src.validation import evaluate, temporal_split

D = pd.Timestamp


def test_band_encoder_fits_on_train_only_and_maps_unseen_to_base():
    tr = pd.DataFrame({"x": np.arange(100.0), "c": ["a"] * 60 + ["b"] * 40})
    enc = BandEncoder(n_bins=4).fit(tr, ["x", "c"])
    assert enc.base_["c"] == "a" and "c=b" in enc.columns_
    te = pd.DataFrame({"x": [-50.0, 1000.0, np.nan], "c": ["b", "zzz", None]})
    X = enc.transform(te).toarray()
    assert X.shape == (3, len(enc.columns_))
    cols = enc.columns_
    assert X[0, cols.index("c=b")] == 1
    assert X[1].sum() == 1          # x in top band; unseen 'zzz' -> base (no column)
    assert X[2].sum() == 0          # missing x not seen in train -> base; None c -> 'missing' not seen -> base


def test_encoder_deterministic():
    tr = pd.DataFrame({"x": np.random.default_rng(0).normal(size=500)})
    a = BandEncoder(5).fit(tr, ["x"]).transform(tr)
    b = BandEncoder(5).fit(tr, ["x"]).transform(tr)
    assert (a != b).nnz == 0


def test_onlevel_factor():
    assert onlevel_factor(pd.Series([2021, 2024]), 2025, 0.05).round(6).tolist() == [round(1.05 ** 4, 6), 1.05]


def test_add_features_joins_and_derived():
    rows = pd.DataFrame({"row_id": ["B1", "B2"], "customer_id": ["c1", "cX"], "address_id": ["a1", "a1"],
                         "period_start": [D("2023-01-01"), D("2024-01-01")], "construction_type": ["timber", "brick"],
                         "voluntary_excess": [250, 0], "compulsory_excess": [350, 350], "sum_insured": [200_000.0, 100_000.0],
                         "renewal_decision": ["renewed", "lapsed_by_customer"], "written_premium": [100.0, 120.0]})
    addr = pd.DataFrame({"address_id": ["a1"], "year_built": [1990], "bedrooms": [4], "region": ["R01"]})
    cust = pd.DataFrame({"customer_id": ["c1"], "customer_since_year": [2015], "surname": ["X"]})
    w = pd.DataFrame({"address_id": ["a1"] * 3, "year": [2023, 2024, 2025], **{c: [1.0, 2.0, 3.0] for c in
                      ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm", "mean_temperature_c"]}})
    out = add_features(rows, "buildings", addr, cust, weather_features(w, "same_year"))
    assert out["property_age"].tolist() == [33, 34]
    assert out["customer_missing"].tolist() == [False, True]
    assert out["excess_total"].tolist() == [600, 350]
    assert out["dry_summer_index"].tolist() == [1.0, 2.0]
    assert out["timber_post2023"].tolist() == [True, False]
    lr = add_features(rows, "buildings", addr, cust, weather_features(w, "long_run", [2023, 2024]))
    assert lr["dry_summer_index_lr"].tolist() == [1.5, 1.5]      # 2025 not used
    cols = feature_columns(out)
    for banned in ("renewal_decision", "written_premium", "surname", "row_id", "customer_since_year"):
        assert banned not in cols


def _toy(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"policy_year": rng.choice([2021, 2022, 2023, 2024], n), "seg": rng.choice(["a", "b"], n),
                       "exposure": rng.uniform(0.1, 1.0, n)})
    mu = np.where(df.seg == "b", 300.0, 100.0) * df.exposure
    claim = rng.random(n) < 0.1 * df.exposure * 2
    df["target"] = np.where(claim, rng.gamma(2.0, mu / 0.2 / 2.0 / df.exposure * df.exposure), 0.0)
    return df


def test_models_scale_with_exposure_and_beat_constant():
    tr, va = temporal_split(_toy(), [2021, 2022, 2023], [2024])
    const = ConstantModel().fit(tr, "target")
    glm = TweedieGLM(["seg"]).fit(tr, "target")
    p_glm = glm.predict(va)
    assert np.all(np.isfinite(p_glm)) and np.all(p_glm > 0)
    double = va.assign(exposure=va.exposure * 2)
    np.testing.assert_allclose(glm.predict(double), 2 * p_glm)
    assert evaluate(va.target, p_glm)["d2"] > evaluate(va.target, const.predict(va))["d2"]
    assert glm.relativities()["seg=b"] == pytest.approx(3.0, rel=0.25)


def test_capped_glm_loading_restores_level():
    tr = _toy()
    tr["target_cap"] = tr["target"].clip(upper=tr["target"].quantile(0.995))
    glm = TweedieGLM(["seg"]).fit(tr, "target_cap", loading_target="target")
    assert glm.predict(tr).sum() == pytest.approx(tr["target"].sum(), rel=0.02)


def test_apply_claim_overrides():
    from src.data import apply_claim_overrides
    c = pd.DataFrame({"claim_id": ["a", "b"], "incurred_to_date": [10.0, 20.0]})
    assert apply_claim_overrides(c, {"b": 2.0})["incurred_to_date"].tolist() == [10.0, 2.0]
    assert c["incurred_to_date"].tolist() == [10.0, 20.0]          # original untouched
    with pytest.raises(KeyError):
        apply_claim_overrides(c, {"zzz": 1.0})


def test_lgbm_tweedie_basics():
    from src.models import LGBMTweedie
    df = _toy(8000)
    df["policy_id"] = np.arange(len(df)) % 2000
    df["flag"] = df["seg"].eq("b")
    tr, va = temporal_split(df, [2021, 2022, 2023], [2024])
    m = LGBMTweedie(["seg", "flag"], params={"min_child_samples": 50}, n_estimators=300).fit(tr, "target")
    p = m.predict(va)
    assert np.all(np.isfinite(p)) and np.all(p > 0)
    np.testing.assert_allclose(m.predict(va.assign(exposure=va.exposure * 2)), 2 * p)
    np.testing.assert_allclose(LGBMTweedie(["seg", "flag"], params={"min_child_samples": 50}, n_estimators=300).fit(tr, "target").predict(va), p)
    unseen = va.head(3).assign(seg="zzz")
    assert np.all(np.isfinite(m.predict(unseen)))


def test_lgbm_balance_reproduces_training_total():
    from src.models import LGBMTweedie
    df = _toy(8000)
    df["policy_id"] = np.arange(len(df)) % 2000
    m = LGBMTweedie(["seg"], params={"min_child_samples": 50, "learning_rate": 0.01}, n_estimators=20).fit(df, "target")
    assert m.predict(df).sum() == pytest.approx(df["target"].sum(), rel=1e-9)
