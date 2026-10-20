import numpy as np
import pandas as pd
import pytest

from src.eda import band, burning_cost_table, ibnr_count_estimate, lag_cdf, one_way

D = pd.Timestamp


def _rows():
    return pd.DataFrame({
        "policy_year": [2023, 2023, 2024, 2024],
        "exposure_used": [1.0, 0.5, 1.0, 1.0],
        "target_net": [1000.0, 0.0, 300.0, 0.0],
        "n_claims_net": [2, 0, 1, 0],
        "written_premium": [500.0, 250.0, 500.0, 500.0],
        "age": [20, 30, np.nan, 40],
    })


def test_burning_cost_table():
    t = burning_cost_table(_rows(), "policy_year").set_index("policy_year")
    assert t.loc[2023, "exposure"] == 1.5
    assert t.loc[2023, "frequency"] == pytest.approx(2 / 1.5)
    assert t.loc[2023, "severity"] == pytest.approx(500.0)
    assert t.loc[2023, "burning_cost"] == pytest.approx(1000 / 1.5)
    assert t.loc[2024, "share_rows_with_claim"] == 0.5
    assert t.loc[2024, "loss_ratio"] == pytest.approx(0.3)


def test_band_and_one_way():
    s = pd.Series(np.r_[np.arange(100.0), np.nan])
    b = band(s, 4)
    assert b.nunique() == 5 and (b == "missing").sum() == 1
    assert band(pd.Series(["a", "b", None]), 4).tolist() == ["a", "b", "missing"]
    ow = one_way(_rows(), "age", n_bins=10)
    # exposure-weighted relativities average to 1
    assert (ow["relativity"] * ow["exposure"]).sum() / ow["exposure"].sum() == pytest.approx(1.0)
    assert set(ow["level"]) == {"20.0", "30.0", "40.0", "missing"}


def test_lag_cdf():
    F = lag_cdf([0, 10, 10, 30])
    assert F([-1, 0, 10, 29, 30, 100]).tolist() == [0, 0.25, 0.75, 0.75, 1.0, 1.0]


def test_ibnr_count_estimate_grosses_up_recent_year():
    # mature year: half the claims reported at lag 0, half at lag 100
    mature = pd.DataFrame({"claim_id": ["a", "b"], "loss_date": [D("2021-01-01")] * 2,
                           "report_date": [D("2021-01-01"), D("2021-04-11")]})
    # recent claim with only 10 days to cutoff -> P(reported) = 0.5 -> counts as 2
    recent = pd.DataFrame({"claim_id": ["c"], "loss_date": [D("2024-12-20")], "report_date": [D("2024-12-20")]})
    g = ibnr_count_estimate(pd.concat([mature, recent]), D("2024-12-30"), [2021]).set_index("loss_year")
    assert g.loc[2021, "count_dev_factor"] == pytest.approx(1.0)
    assert g.loc[2024, "est_ultimate"] == pytest.approx(2.0)
