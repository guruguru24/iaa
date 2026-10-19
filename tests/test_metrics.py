import numpy as np
import pandas as pd

from src.metrics import actual_to_expected, gini, lift_table, tweedie_d2
from src.submission import check_submission


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(2.0, 100.0, n)
    claim = rng.random(n) < 0.1
    y = np.where(claim, rng.gamma(2.0, mu * 5), 0.0)
    return y, mu


def test_constant_prediction_scores_zero():
    y, _ = _toy()
    assert abs(tweedie_d2(y, np.full_like(y, y.mean()))) < 1e-9


def test_informative_prediction_beats_constant():
    y, mu = _toy()
    assert tweedie_d2(y, mu) > tweedie_d2(y, np.full_like(y, y.mean() * 1.3))


def test_ae_and_gini():
    y, mu = _toy()
    assert abs(actual_to_expected(mu, mu) - 1) < 1e-12
    assert abs(gini(y, y) - 1) < 1e-9
    assert len(lift_table(y, mu)) == 10


def test_submission_checks():
    sample = pd.DataFrame({"row_id": ["M1", "B1"], "prediction": [0, 0]})
    ok = pd.DataFrame({"row_id": ["M1", "B1"], "prediction": [1.0, 0.0]})
    bad = pd.DataFrame({"row_id": ["M1", "M1"], "prediction": [-1.0, np.nan]})
    assert check_submission(ok, sample) == []
    assert len(check_submission(bad, sample)) >= 3
