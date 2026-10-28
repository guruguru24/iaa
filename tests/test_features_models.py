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


def test_level_sensitivity_finds_calibration_factor():
    from src.validation import level_sensitivity
    rng = np.random.default_rng(0)
    mu = rng.gamma(2.0, 50.0, 20_000)
    y = np.where(rng.random(20_000) < 0.1, rng.gamma(2.0, mu * 5), 0.0)
    out = level_sensitivity(y, mu * 0.9)                      # predictions 10% too low
    best = out[out.get("best", False) == True]["scale"].iloc[0]
    assert 1.05 <= best <= 1.17
    assert out.loc[out.scale == 1.1, "d2"].iloc[0] > out.loc[out.scale == 1.0, "d2"].iloc[0]


def test_segment_ae():
    from src.validation import segment_ae
    df = pd.DataFrame({"f": ["a", "a", "b"], "y": [10.0, 0.0, 30.0], "p": [5.0, 5.0, 20.0], "exposure": 1.0})
    out = segment_ae(df, "y", "p", "f").set_index("level")
    assert out.loc["a", "ae"] == 1.0 and out.loc["b", "ae"] == 1.5
    assert out["exposure_share"].sum() == pytest.approx(1.0)


def test_recency_weights_shift_fit_towards_recent_years():
    from src.models import fit_weights
    df = pd.DataFrame({"policy_year": [2021, 2023], "exposure": [1.0, 2.0]})
    assert fit_weights(df).tolist() == [1.0, 2.0]
    assert fit_weights(df, recency=0.5, ref_year=2024).tolist() == [0.125, 1.0]
    # segment b relativity drifts from 1x (2021) to 3x (2023): decayed fit sits closer to the recent value
    rng = np.random.default_rng(1)
    n = 30_000
    d = pd.DataFrame({"policy_year": rng.choice([2021, 2022, 2023], n), "seg": rng.choice(["a", "b"], n), "exposure": 1.0})
    rel = np.where(d.seg == "b", 1.0 + (d.policy_year - 2021), 1.0)
    d["target"] = rng.poisson(0.2 * rel) * 100.0
    flat = TweedieGLM(["seg"]).fit(d, "target").relativities()["seg=b"]
    recent = TweedieGLM(["seg"]).fit(d, "target", recency=0.3, ref_year=2024).relativities()["seg=b"]
    assert flat < recent and recent > 2.4


def test_lgbm_time_stop_refits_on_all_rows():
    from src.models import LGBMTweedie
    df = _toy(8000)
    df["policy_id"] = np.arange(len(df)) % 2000
    m = LGBMTweedie(["seg"], params={"min_child_samples": 50}, n_estimators=200, stop="time").fit(df, "target")
    assert m.n_trees_ >= 1 and m.model_.n_estimators == m.n_trees_      # refit without early stopping
    assert m.predict(df).sum() == pytest.approx(df["target"].sum(), rel=1e-9)   # balance on all rows
    p = m.predict(df.head(5))
    assert np.all(np.isfinite(p)) and np.all(p > 0)


def test_experience_modifier():
    from src.models import ExperienceModifier
    rng = np.random.default_rng(0)
    n = 40_000
    hist = rng.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]).astype(float)
    hist[:2000] = np.nan                                       # history not observable
    risk = np.where(np.isnan(hist), 1.0, 1.0 + 0.5 * np.nan_to_num(hist))
    y = rng.poisson(0.1 * risk) * 1000.0
    df = pd.DataFrame({"hist_claims_1y": hist})
    pred = np.full(n, 100.0)                                   # base ignores history
    mod = ExperienceModifier(k=50).fit(df, y, pred)
    assert mod.factors_[0] < 1 < mod.factors_[1] < mod.factors_[2]
    adj = mod.apply(df, pred)
    obs = ~np.isnan(hist)
    assert adj[obs].sum() == pytest.approx(pred[obs].sum())   # level unchanged on fitted rows
    assert (adj[~obs] == 100.0).all()                         # unknown history -> 1.0
    assert evaluate(y, adj)["d2"] > evaluate(y, pred)["d2"]


def test_freqsev_gbm_recovers_peril_structure_and_level():
    from src.models import FreqSevGBM
    rng = np.random.default_rng(3)
    n = 40_000
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(n)], "policy_year": rng.choice([2022, 2023, 2024], n),
                         "seg": rng.choice(["a", "b"], n), "exposure": rng.uniform(0.3, 1.0, n)})
    lam = {"fire": np.where(rows.seg == "b", 0.03, 0.01), "theft": np.full(n, 0.05)}
    claims = []
    for p, l in lam.items():
        k = rng.poisson(l * rows.exposure)
        idx = np.repeat(np.arange(n), k)
        claims.append(pd.DataFrame({"row_id": rows.row_id.values[idx], "peril": p, "policy_year": rows.policy_year.values[idx],
                                    "net": rng.gamma(2.0, 5000 if p == "fire" else 500, len(idx))}))
    c = pd.concat(claims, ignore_index=True)
    rows["target_cap"] = rows.row_id.map(c.assign(v=c.net.clip(upper=50_000)).groupby("row_id").v.sum()).fillna(0)
    rows["target"] = rows.row_id.map(c.groupby("row_id").net.sum()).fillna(0)
    cf = pd.DataFrame({"peril": ["fire", "theft"], "policy_year": 2024, "count_dev_factor": 1.0})
    m = FreqSevGBM(["seg"], cap=50_000, trend=0.0, to_year=2024, params_freq={"min_child_samples": 50},
                   params_sev={"min_child_samples": 20}, min_sev_claims=50)
    m.fit(rows, c, "target_cap", cf, loading_target="target")
    p = m.predict(rows)
    assert p.sum() == pytest.approx(rows["target"].sum(), rel=1e-6)            # balance x loading restores level
    pp = m.peril_predictions(rows).div(rows.exposure, axis=0)
    ratio = pp.loc[rows.seg == "b", "fire"].mean() / pp.loc[rows.seg == "a", "fire"].mean()
    assert 2.0 < ratio < 4.0                                                    # fire 3x as frequent in seg b
    assert np.all(np.isfinite(m.predict(rows.head(3).assign(seg="zzz"))))      # unseen level


def test_simplex_weights_prefers_the_informative_model():
    from src.stacking import apply_weights, simplex_weights
    rng = np.random.default_rng(0)
    mu = rng.gamma(2.0, 50.0, 30_000)
    y = np.where(rng.random(30_000) < 0.2, rng.gamma(2.0, mu * 2.5), 0.0)
    preds = {"good": mu, "flat": np.full_like(mu, mu.mean()), "noisy": mu * rng.lognormal(0, 1.0, mu.size)}
    w = simplex_weights(y, preds)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3) and min(w.values()) >= 0
    assert w["good"] > 0.6
    blended = apply_weights(preds, w)
    assert evaluate(y, blended)["d2"] >= evaluate(y, preds["flat"])["d2"]


def test_harmonise_ages_matches_emerging_bands():
    from src.features import harmonise_ages
    out = harmonise_ages(pd.DataFrame({"driver_age": [17, 24, 25, 49, 50, 64, 65, 90]}))
    assert out["driver_age_band"].tolist() == ["17-24", "17-24", "25-34", "35-49", "50-64", "50-64", "65+", "65+"]
    already = pd.DataFrame({"driver_age_band": ["65+"]})
    assert harmonise_ages(already).equals(already)


def test_glm_reuses_encoder_for_comparable_coefficients():
    a, b = _toy(seed=1), _toy(seed=2)
    ga = TweedieGLM(["seg"]).fit(a, "target")
    gb = TweedieGLM(["seg"]).fit(b, "target", encoder=ga.enc_)
    assert gb.enc_ is ga.enc_ and list(gb.relativities().index) == list(ga.relativities().index)


def test_add_features_motor_without_telematics():
    rows = pd.DataFrame({"row_id": ["M1"], "customer_id": ["c1"], "address_id": ["a1"], "period_start": [pd.Timestamp("2024-01-01")],
                         "voluntary_excess": [0], "compulsory_excess": [250]})
    out = add_features(rows, "motor", pd.DataFrame({"address_id": ["a1"], "year_built": [2000]}),
                       pd.DataFrame({"customer_id": ["c1"], "customer_since_year": [2020]}))
    assert "telematics_missing" not in out and out["excess_total"].iloc[0] == 250


def test_weight_col_multiplies_weights():
    from src.models import fit_weights
    df = pd.DataFrame({"policy_year": [2023, 2024], "exposure": [1.0, 0.5], "m": [5.0, 1.0]})
    assert fit_weights(df, weight_col="m").tolist() == [5.0, 0.5]
    assert fit_weights(df, weight_col="m", recency=0.5, ref_year=2024).tolist() == [2.5, 0.5]
