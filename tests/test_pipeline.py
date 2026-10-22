import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_submission, level_check
from src.submission import check_submission


def test_build_submission_follows_sample_order_and_passes_checks():
    sample = pd.DataFrame({"row_id": ["B1", "M1", "C1", "M2"], "prediction": 0.0})
    preds = {"motor": pd.DataFrame({"row_id": ["M2", "M1"], "prediction": [2.0, 1.0]}),
             "buildings": pd.DataFrame({"row_id": ["B1"], "prediction": [3.0]}),
             "contents": pd.DataFrame({"row_id": ["C1"], "prediction": [4.0]})}
    sub = build_submission(preds, sample)
    assert sub["row_id"].tolist() == ["B1", "M1", "C1", "M2"]
    assert sub["prediction"].tolist() == [3.0, 1.0, 4.0, 2.0]
    assert check_submission(sub, sample) == []
    missing = build_submission({"motor": preds["motor"]}, sample)
    assert check_submission(missing, sample) != []          # NaN for missing lines is caught


def test_level_check():
    tr = pd.DataFrame({"policy_year": [2023, 2024], "target_dev": [100.0, 110.0], "exposure": [1.0, 1.0]})
    te = pd.DataFrame({"exposure": [2.0]})
    out = level_check(tr, te, np.array([231.0]), trend=0.05, to_year=2025)
    assert out["pred_bc"] == pytest.approx(115.5)
    assert out["trended_bc_2024"] == pytest.approx(115.5)
    assert out["pred_vs_2024_trended"] == pytest.approx(1.0)
