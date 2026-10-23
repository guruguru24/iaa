import numpy as np
import pandas as pd
import pytest

from src.models import ExperienceModifier, LGBMTweedie, TweedieGLM
from src.tool import glm_breakdown, price, typical_values


def _toy(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"seg": rng.choice(["a", "b", "c"], n), "age": rng.integers(18, 80, n).astype(float),
                       "flag": rng.random(n) < 0.3, "exposure": rng.uniform(0.2, 1.0, n),
                       "policy_year": rng.choice([2022, 2023, 2024], n), "policy_id": np.arange(n) % 1500,
                       "hist_claims_1y": rng.choice([0, 1, 2], n, p=[0.85, 0.1, 0.05]).astype(float)})
    mu = 100 * np.where(df.seg == "b", 2.0, 1.0) * (1 + (df.age < 25)) * (1 + 0.4 * df.hist_claims_1y)
    df["target"] = np.where(rng.random(n) < 0.15 * df.exposure, rng.gamma(2.0, mu / 0.15 / 2), 0.0)
    return df


def _line_bundle(df):
    factors = ["seg", "age", "flag"]
    glm = TweedieGLM(factors, n_bins=5).fit(df, "target")
    gbm = LGBMTweedie(factors, params={"min_child_samples": 50}, n_estimators=50).fit(df, "target")
    mod = ExperienceModifier(k=50).fit(df, df["target"], 0.5 * glm.predict(df) + 0.5 * gbm.predict(df))
    return {"factors": factors, "defaults": typical_values(df, factors), "models": {"glm": glm, "gbm": gbm, "modifier": mod}}


def test_typical_values_types():
    d = typical_values(_toy(), ["seg", "age", "flag"])
    assert isinstance(d["seg"], str) and isinstance(d["age"], float) and isinstance(d["flag"], bool)


def test_glm_breakdown_reproduces_glm_prediction():
    df = _toy()
    b = _line_bundle(df)
    row = df.iloc[[7]]
    base, tab = glm_breakdown(b["models"]["glm"], row)
    assert base * tab["relativity"].prod() * row["exposure"].iloc[0] == pytest.approx(b["models"]["glm"].predict(row)[0], rel=1e-9)
    assert set(tab["factor"]) == {"seg", "age", "flag"}


def test_price_single_dict_uses_defaults_and_scales_with_exposure():
    b = _line_bundle(_toy())
    one = price(b, {"seg": "b", "exposure": 1.0})
    half = price(b, {"seg": "b", "exposure": 0.5})
    assert one["expected_cost"].iloc[0] == pytest.approx(2 * half["expected_cost"].iloc[0])
    worse = price(b, {"seg": "b", "exposure": 1.0, "hist_claims_1y": 2})
    assert worse["experience_factor"].iloc[0] > one["experience_factor"].iloc[0]
    batch = price(b, pd.DataFrame({"seg": ["a", "zzz"], "age": [30.0, np.nan]}))     # unseen level, missing value
    assert np.all(np.isfinite(batch["expected_cost"])) and len(batch) == 2
