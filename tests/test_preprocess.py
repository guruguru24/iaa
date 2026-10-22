import pandas as pd
import pytest

from src.preprocess import fix_construction_type, fix_exposure, fix_sum_insured, preprocess

D = pd.Timestamp


def _df():
    return pd.DataFrame({
        "period_start": [D("2021-01-01")] * 4,
        "period_end": [D("2021-12-31"), D("2021-12-31"), D("2021-06-30"), D("2021-12-31")],
        "exposure": [1.0, 365.0, 181 / 365, 1.0139],
        "sum_insured": [200_000.0, 250_000.0, 300_000_000.0, 180_000.0],
        "construction_type": ["Brick", "BRICK", "W", None],
        "renewal_decision": ["renewed"] * 4,
    })


def test_fix_exposure():
    out = fix_exposure(_df())
    assert out["exposure"].round(6).tolist() == [1.0, 1.0, round(181 / 365, 6), 1.0]
    assert out["exposure_fixed"].tolist() == [False, True, False, True]
    assert out["exposure_raw"].iloc[1] == 365.0


def test_fix_sum_insured_uses_given_median():
    out = fix_sum_insured(_df(), median=220_000)
    assert out["sum_insured"].tolist() == [200_000.0, 250_000.0, 300_000.0, 180_000.0]
    assert out["sum_insured_fixed"].sum() == 1
    # a test set is corrected with the TRAIN median, not its own
    assert fix_sum_insured(_df().iloc[[2]], median=220_000)["sum_insured"].iloc[0] == 300_000.0


def test_fix_construction_type():
    out = fix_construction_type(_df())
    assert out["construction_type"].iloc[:3].tolist() == ["brick", "brick", "timber"]
    assert pd.isna(out["construction_type"].iloc[3])
    assert fix_construction_type(_df().assign(construction_type="Mud"))["construction_type"].iloc[0] == "other"


def test_preprocess_drops_leakage_and_keeps_rows():
    out = preprocess(_df(), "buildings", {"sum_insured_median": 220_000})
    assert "renewal_decision" not in out
    assert len(out) == 4
    with pytest.raises(KeyError):
        preprocess(_df(), "contents", {})
