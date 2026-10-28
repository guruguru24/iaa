import numpy as np
import pandas as pd

from src.history import history_features

D = pd.Timestamp


def _rows():
    return pd.DataFrame({
        "row_id": ["M1", "M2", "M3", "X1"],
        "policy_id": ["P1", "P1", "P1", "P9"],
        "customer_id": ["K1", "K1", "K1", "K2"],
        "period_start": [D("2021-01-01"), D("2022-01-01"), D("2023-01-01"), D("2023-01-01")],
        "period_end": [D("2021-12-31"), D("2022-12-31"), D("2023-12-31"), D("2023-12-31")],
        "exposure": [1.0, 1.0, 1.0, 1.0],
    })


def _claims():
    return pd.DataFrame({
        "claim_id": ["a", "b", "c", "d", "e", "f"],
        "policy_id": ["P1", "P1", "P1", "P1", "B1", "P1"],
        "loss_date": [D("2021-03-01"),   # reported 2021 -> in M2's 1y window, M3's 2y window
                      D("2021-12-20"),   # reported 2022-02 -> NOT known at M2 start; in M3's 2y window
                      D("2022-05-01"),   # M2's own period -> not for M2; M3's 1y window
                      D("2023-02-01"),   # M3's own period -> never counts
                      D("2022-06-01"),   # other line (buildings), same customer
                      D("2021-06-01")],  # declined
        "report_date": [D("2021-03-05"), D("2022-02-10"), D("2022-05-03"), D("2023-02-02"), D("2022-06-05"), D("2021-06-10")],
        "status": ["closed", "closed", "closed", "open", "closed", "repudiated"],
    })


def _map():
    return pd.DataFrame({"policy_id": ["P1", "B1", "P9"], "customer_id": ["K1", "K1", "K2"], "line": ["motor", "buildings", "motor"]})


def test_windows_and_reporting_rule():
    h = history_features(_rows(), _claims(), _map()).set_index("row_id")
    # M1 (2021): no observable window -> NaN, not 0
    assert h.loc["M1", ["hist_claims_1y", "hist_exposure_1y", "hist_claims_2y"]].isna().all()
    # M2 (2022): 1y window observable, 2y not
    assert h.loc["M2", "hist_claims_1y"] == 1            # a (b reported after start, c own period)
    assert h.loc["M2", "hist_declined_1y"] == 1          # f
    assert h.loc["M2", "hist_exposure_1y"] == 1.0
    assert np.isnan(h.loc["M2", "hist_claims_2y"])
    # M3 (2023): both windows observable
    assert h.loc["M3", "hist_claims_1y"] == 1            # c
    assert h.loc["M3", "hist_claims_2y"] == 3            # a, b, c
    assert h.loc["M3", "hist_exposure_2y"] == 2.0
    assert h.loc["M3", "cust_other_1y"] == 1             # e (buildings)
    assert h.loc["X1", ["hist_claims_1y", "hist_exposure_1y", "cust_other_1y"]].tolist() == [0.0, 0.0, 0.0]
    assert h.loc["M3", "hist_all_claims"] == 3 and h.loc["M3", "hist_all_exposure"] == 2.0


def test_no_same_period_leakage_on_random_data():
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
    h = history_features(rows, claims, pmap).merge(rows, on="row_id").set_index("row_id")
    pairs = rows.merge(claims, on="policy_id")
    exp = pairs[(pairs.report_date < pairs.period_start) & (pairs.loss_date >= pairs.period_start - pd.Timedelta(days=365))].groupby("row_id").size()
    obs = h[h.period_start >= D("2022-01-01")]
    assert (obs["hist_claims_1y"] == exp.reindex(obs.index).fillna(0).values).all()
    assert h.loc[h.period_start < D("2022-01-01"), "hist_claims_1y"].isna().all()


def test_emerging_data_start_makes_first_year_unobservable():
    from src.history import DATA_START_BY_MARKET
    h = history_features(_rows(), _claims(), _map(), DATA_START_BY_MARKET["w2"]).set_index("row_id")
    # with data starting 2023, the 2023 rows have no observable 1y window -> NaN, not claim-free
    assert h.loc[["M3", "X1"], "hist_claims_1y"].isna().all()
