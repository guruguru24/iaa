import numpy as np
import pandas as pd
import pytest

from src.data import exposure_from_dates, flag_exposure
from src.target import build_row_target, claim_net_amount, match_claims_to_rows

D = pd.Timestamp


def _rows():
    return pd.DataFrame({
        "row_id": ["M1", "M2", "M3"],
        "policy_id": ["P1", "P1", "P2"],
        "period_start": [D("2023-01-01"), D("2024-01-01"), D("2024-03-01")],
        "period_end": [D("2023-12-31"), D("2024-06-30"), D("2024-12-31")],
        "voluntary_excess": [250, 250, 0],
        "compulsory_excess": [250, 250, 350],
    })


def _claims():
    return pd.DataFrame({
        "claim_id": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        "policy_id": ["P1", "P1", "P1", "P1", "P2", "P2", "P2"],
        "loss_date": [D("2023-12-31"),  # last day of M1 -> in period (inclusive end)
                      D("2024-01-01"),  # first day of M2
                      D("2024-07-15"),  # after M2 ends -> out of period
                      D("2024-03-03"),  # declined
                      D("2024-05-01"),  # below excess -> net 0
                      D("2024-06-01"),  # recovery
                      D("2024-02-01")],  # before M3 start -> out of period
        "status": ["closed", "open", "closed", "repudiated", "closed", "closed", "closed"],
        "incurred_to_date": [1500.0, 800.0, 999.0, 0.0, 300.0, 2000.0, 777.0],
        "recoveries": [0.0, 0.0, 0.0, 0.0, 0.0, 1600.0, 0.0],
    })


def test_claim_net_amount_floor_and_order():
    assert claim_net_amount([1000], [500], [0]).tolist() == [500]
    assert claim_net_amount([300], [500], [0]).tolist() == [0]
    # recovery bigger than post-excess amount -> floored at 0, unfloored negative
    assert claim_net_amount([1000], [500], [800]).tolist() == [0]
    assert claim_net_amount([1000], [500], [800], floor_at_zero=False).tolist() == [-300]
    # with floor: identical to max(g - e - r, 0)
    g, e, r = np.random.default_rng(0).uniform(0, 3000, (3, 500))
    np.testing.assert_allclose(claim_net_amount(g, e, r), np.maximum(g - e - r, 0))


def test_match_inclusive_period_and_out_of_period():
    m = match_claims_to_rows(_claims(), _rows()).set_index("claim_id")
    assert m.loc["c1", "row_id"] == "M1"
    assert m.loc["c2", "row_id"] == "M2"
    assert pd.isna(m.loc["c3", "row_id"]) and pd.isna(m.loc["c7", "row_id"])


def test_build_row_target_toy_case():
    t = build_row_target(_rows(), _claims()).set_index("row_id")
    assert t.loc["M1", "target_net"] == pytest.approx(1000.0)   # 1500 - 500
    assert t.loc["M2", "target_net"] == pytest.approx(300.0)    # open 800 - 500; c4 declined; c3 out of period
    assert t.loc["M2", "open_net"] == pytest.approx(300.0)
    assert t.loc["M3", "target_net"] == pytest.approx(50.0)     # c5 below excess=0, c6 2000-350-1600=50
    assert t.loc["M3", "n_claims_net"] == 1
    assert t.loc["M3", "n_claims_reported"] == 2
    assert (t["target_net"] >= 0).all()


def test_overlapping_rows_raise():
    rows = _rows()
    rows.loc[1, "period_start"] = D("2023-12-01")  # overlaps M1
    with pytest.raises(ValueError):
        match_claims_to_rows(_claims(), rows)


def test_exposure_from_dates_and_flags():
    df = pd.DataFrame({"period_start": [D("2021-01-01")] * 3, "period_end": [D("2021-12-31")] * 3,
                       "exposure": [1.0, 1.0139, 365.0]})
    assert exposure_from_dates(df.period_start, df.period_end).tolist() == [1.0] * 3
    assert flag_exposure(df).tolist() == ["ok", "over_small", "days_not_years"]
