import numpy as np
import pandas as pd

from src.history import history_features

D = pd.Timestamp


def _rows():
    return pd.DataFrame({
        "row_id": ["M1", "M2", "M3", "X1"],
        "policy_id": ["P1", "P1", "P1", "P9"],
        "customer_id": ["K1", "K1", "K1", "K2"],
        "period_start": [D("2023-01-01"), D("2024-01-01"), D("2025-01-01"), D("2025-01-01")],
        "period_end": [D("2023-12-31"), D("2024-12-31"), D("2025-12-31"), D("2025-12-31")],
        "exposure": [1.0, 1.0, 1.0, 1.0],
    })


def _claims():
    return pd.DataFrame({
        "claim_id": ["a", "b", "c", "d", "e", "f"],
        "policy_id": ["P1", "P1", "P1", "P1", "B1", "P1"],
        "loss_date": [D("2023-03-01"),   # reported 2023 -> history for M2, M3
                      D("2023-12-20"),   # reported 2024-02 -> NOT known at M2 start, known at M3
                      D("2024-05-01"),   # M2's own period -> must not count for M2; counts for M3 (1y window)
                      D("2025-02-01"),   # M3's own period -> never counts
                      D("2023-06-01"),   # other line (buildings) of the same customer
                      D("2022-06-01")],  # declined, reported 2022
        "report_date": [D("2023-03-05"), D("2024-02-10"), D("2024-05-03"), D("2025-02-02"), D("2023-06-05"), D("2022-06-10")],
        "status": ["closed", "closed", "closed", "open", "closed", "repudiated"],
    })


def _map():
    return pd.DataFrame({"policy_id": ["P1", "B1", "P9"], "customer_id": ["K1", "K1", "K2"], "line": ["motor", "buildings", "motor"]})


def test_history_uses_only_claims_reported_before_start():
    h = history_features(_rows(), _claims(), _map()).set_index("row_id")
    assert h.loc["M1", "hist_claims"] == 0 and h.loc["M1", "hist_exposure"] == 0.0
    assert h.loc["M2", "hist_claims"] == 1            # a only (b reported after M2 start; c is M2's own period)
    assert h.loc["M3", "hist_claims"] == 3            # a, b, c
    assert h.loc["M3", "hist_claims_1y"] == 1         # c (loss within 365 days before 2025-01-01)
    assert h.loc["M2", "hist_exposure"] == 1.0 and h.loc["M3", "hist_exposure"] == 2.0
    assert h.loc["M3", "hist_rate"] == 1.5
    assert np.isnan(h.loc["M1", "hist_rate"])
    assert h.loc["M2", "hist_declined"] == 1          # f
    assert h.loc["M2", "cust_claims_other"] == 1      # e (buildings) counts for the motor row
    assert h.loc["X1", ["hist_claims", "cust_claims_other"]].tolist() == [0, 0]


def test_no_same_period_leakage_on_random_data():
    # property check: no counted claim may have loss_date or report_date on/after its row's start
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(400)], "policy_id": [f"P{i % 100}" for i in range(400)],
                         "customer_id": [f"K{i % 100}" for i in range(400)],
                         "period_start": pd.to_datetime([f"{2021 + i // 100}-01-01" for i in range(400)])})
    rows["period_end"] = rows["period_start"] + pd.offsets.YearEnd(0)
    rows["exposure"] = 1.0
    loss = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1460, 500), unit="D")
    claims = pd.DataFrame({"claim_id": range(500), "policy_id": [f"P{i}" for i in rng.integers(0, 100, 500)], "loss_date": loss,
                           "report_date": loss + pd.to_timedelta(rng.integers(0, 400, 500), unit="D"), "status": "closed"})
    pmap = rows[["policy_id", "customer_id"]].drop_duplicates().assign(line="motor")
    h = history_features(rows, claims, pmap).merge(rows, on="row_id")
    expected = rows.merge(claims, on="policy_id").query("report_date < period_start").groupby("row_id").size()
    assert (h.set_index("row_id")["hist_claims"] == expected.reindex(h.row_id).fillna(0).astype(int).values).all()
