import numpy as np
import pandas as pd
import pytest

from src.development import apply_to_rows, develop, peril_year_table, year_dev_factors
from src.trend import annual_trend, design_matrix, row_capped_target, year_effects


def _claim_frame():
    return pd.DataFrame({
        "claim_id": ["a", "b", "c", "d", "e"],
        "row_id": ["R1", "R1", "R2", "R3", None],
        "line": ["motor"] * 5,
        "peril": ["own_damage", "bodily_injury", "bodily_injury", "own_damage", "own_damage"],
        "policy_year": [2024, 2024, 2024, 2023, np.nan],
        "status": ["closed", "open", "closed", "closed", "closed"],
        "net": [100.0, 1000.0, 200_000.0, 50.0, 999.0],
        "in_period": [True, True, True, True, False],
        "declined": [False] * 5,
    })


def test_develop_and_year_factors():
    t = peril_year_table(_claim_frame())
    assert t["net"].sum() == pytest.approx(201_150.0)  # out-of-period claim excluded
    f = pd.DataFrame({"line": ["motor"], "peril": ["bodily_injury"], "policy_year": [2024], "count_dev_factor": [1.5]})
    d = develop(t, f, {"_default": 1.0, "bodily_injury": 1.2}).set_index(["policy_year", "peril"])
    # BI 2024: (closed 200000 + open 1000*1.2) * 1.5
    assert d.loc[(2024, "bodily_injury"), "ultimate"] == pytest.approx((200_000 + 1_200) * 1.5)
    assert d.loc[(2024, "own_damage"), "ultimate"] == pytest.approx(100.0)  # no factor -> 1.0
    yf = year_dev_factors(develop(t, f)).set_index("policy_year")
    assert yf.loc[2023, "dev_factor"] == pytest.approx(1.0)
    rows = pd.DataFrame({"policy_year": [2023, 2024], "target_net": [50.0, 100.0]})
    out = apply_to_rows(rows, "motor", yf.reset_index())
    assert out.iloc[0] == 50.0 and out.iloc[1] == pytest.approx(100.0 * yf.loc[2024, "dev_factor"])


def test_row_capped_target():
    rows = pd.DataFrame({"row_id": ["R1", "R2", "R3", "R4"]})
    s = row_capped_target(rows, _claim_frame(), cap=500.0)
    assert s.tolist() == [600.0, 500.0, 50.0, 0.0]


def test_design_matrix_shapes():
    df = pd.DataFrame({"policy_year": [2021, 2022, 2023, 2021], "f": ["a", "b", "a", "a"]})
    X, names = design_matrix(df, ["f"])
    assert names == ["year=2022", "year=2023", "f_b"]
    assert X.shape == (4, 3) and X.sum() == 3
    # a factor with a single level in the subset contributes no columns instead of failing
    X1, names1 = design_matrix(df.assign(g="only"), ["f", "g"])
    assert names1 == names and X1.shape == (4, 3)


def test_year_effects_recover_trend_under_mix_shift():
    # true +10%/yr frequency trend; risky segment grows over time -> crude overstates, adjusted recovers
    rng = np.random.default_rng(0)
    n = 60_000
    year = rng.choice([2021, 2022, 2023], n)
    risky = rng.random(n) < np.select([year == 2021, year == 2022], [0.2, 0.4], 0.6)
    lam = 0.1 * 1.1 ** (year - 2021) * np.where(risky, 2.0, 1.0)
    df = pd.DataFrame({"policy_year": year, "seg": np.where(risky, "r", "s"), "exposure": 1.0, "n": rng.poisson(lam)})
    e = year_effects(df, ["seg"], y="n", weight="exposure", kind="poisson").set_index("policy_year")
    assert e.loc[2023, "mix_adjusted"] == pytest.approx(1.21, rel=0.05)
    assert e.loc[2023, "crude"] > 1.35
    assert annual_trend(e["mix_adjusted"], [2021, 2022, 2023]) == pytest.approx(0.10, abs=0.03)
    # claim-level frames carry float years (NaN-capable column): same result, integer index
    ef = year_effects(df.assign(policy_year=df.policy_year.astype(float)), ["seg"], y="n", weight="exposure",
                      kind="poisson").set_index("policy_year")
    assert ef.index.tolist() == [2021, 2022, 2023]
    assert ef.loc[2023, "mix_adjusted"] == pytest.approx(e.loc[2023, "mix_adjusted"])


def test_weighted_lag_cdf():
    from src.eda import lag_cdf
    F = lag_cdf([10, 400], weights=[100.0, 300.0])
    assert F([5, 10, 399, 400]).tolist() == [0.0, 0.25, 0.25, 1.0]


def test_cost_factor_exceeds_count_factor_when_late_claims_are_dearer():
    from src.development import development_factors
    D = pd.Timestamp
    mature = pd.DataFrame({"claim_id": [f"m{i}" for i in range(4)], "line": "motor", "peril": "bodily_injury",
                           "status": "closed", "loss_date": D("2021-01-01"),
                           "report_date": [D("2021-01-11")] * 2 + [D("2022-02-05")] * 2,   # lags 10 and 400 days
                           "incurred_to_date": [100.0, 100.0, 300.0, 300.0], "recoveries": 0.0})
    recent = pd.DataFrame({"claim_id": ["r1"], "line": "motor", "peril": "bodily_injury", "status": "open",
                           "loss_date": [D("2024-09-21")], "report_date": [D("2024-10-01")],
                           "incurred_to_date": [100.0], "recoveries": [0.0]})
    f = development_factors(pd.concat([mature, recent]), mature_years=[2021], cutoff=D("2024-12-30")).set_index("policy_year")
    assert f.loc[2024, "count_dev_factor"] == pytest.approx(2.0)    # half the claims reported within 100 days
    assert f.loc[2024, "cost_dev_factor"] == pytest.approx(4.0)     # only a quarter of the cost
    assert f.loc[2021, "cost_dev_factor"] == pytest.approx(1.0)


def test_expected_severity_override():
    from src.development import expected_severity_override
    d = pd.DataFrame({"line": "motor", "peril": "bodily_injury", "policy_year": [2021, 2022, 2023],
                      "n": [10, 10, 5], "net": [200.0, 200.0, 2_100.0], "net_capped": [200.0, 200.0, 100.0],
                      "count_dev_factor": [1.0, 1.0, 2.0], "ultimate": [200.0, 200.0, 2_100.0]})
    out = expected_severity_override(d, "motor", "bodily_injury", [2023], [2021, 2022], trend=0.10).set_index("policy_year")
    # base severity 400/20 = 20; 5 x 2.0 claims x 20 x 1.1^1.5 + excess over cap 2000
    assert out.loc[2023, "ultimate"] == pytest.approx(10 * 20 * 1.1 ** 1.5 + 2_000)
    assert out.loc[2021, "ultimate"] == 200.0
