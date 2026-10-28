import numpy as np
import pandas as pd
import pytest

from src.office_premium import build_table, build_up, expense_rates

EXP = pd.DataFrame({
    "item": ["fixed_expense_per_policy", "variable_expense", "commission_direct", "commission_comparison_site",
             "commission_broker", "premium_tax", "cat_reinsurance", "target_profit_margin"],
    "developed": [45, 0.12, 0.0, 0.10, 0.15, 0.12, 0.06, 0.05],
    "emerging": [85, 0.15, 0.0, np.nan, 0.20, 0.08, 0.12, 0.08],
    "unit": ["CU"] + ["share of premium"] * 5 + ["share of property risk premium", "share of premium"],
})
R = expense_rates(EXP, "emerging")


def test_rates_read_from_table_not_hard_coded():
    assert R["fixed_expense_per_policy"] == 85 and expense_rates(EXP, "developed")["fixed_expense_per_policy"] == 45


def test_components_add_up_and_op_at_least_rp():
    for line in ("motor", "buildings"):
        for ch in ("direct", "broker"):
            for tax in ("within", "on_top"):
                b = build_up(500.0, line, ch, 0.5, R, tax_mode=tax)
                assert b["check_sum"] == pytest.approx(b["office_premium"]) and b["office_premium"] >= 500.0


def test_worked_numbers_broker_buildings():
    b = build_up(100.0, "buildings", "broker", 1.0, R)
    assert b["cat_reinsurance"] == pytest.approx(12.0)
    assert b["office_premium"] == pytest.approx((100 + 12 + 85) / (1 - 0.15 - 0.20 - 0.08 - 0.08))


def test_cat_ri_only_property_and_commission_by_channel():
    assert build_up(100.0, "motor", "broker", 1.0, R)["cat_reinsurance"] == 0.0
    assert build_up(100.0, "contents", "broker", 1.0, R)["cat_reinsurance"] > 0.0
    assert build_up(100.0, "motor", "direct", 1.0, R)["commission"] == 0.0
    assert build_up(100.0, "motor", "broker", 1.0, R)["office_premium"] > build_up(100.0, "motor", "direct", 1.0, R)["office_premium"]
    with pytest.raises(ValueError):
        build_up(100.0, "motor", "comparison_site", 1.0, R)


def test_monotone_in_each_loading():
    base = build_up(100.0, "buildings", "broker", 1.0, R)["office_premium"]
    for item in ("fixed_expense_per_policy", "variable_expense", "commission_broker", "premium_tax", "cat_reinsurance",
                 "target_profit_margin"):
        bumped = {**R, item: R[item] * 1.1}
        assert build_up(100.0, "buildings", "broker", 1.0, bumped)["office_premium"] > base, item


def test_fixed_pro_rata_and_table():
    assert build_up(100.0, "motor", "direct", 0.25, R, fixed_mode="pro_rata")["fixed_expense"] == pytest.approx(85 * 0.25)
    ref = pd.DataFrame({"row_id": ["M1", "B1"], "line": ["motor", "buildings"], "channel": ["direct", "broker"],
                        "exposure": [1.0, 0.5], "new_business": [True, False], "written_premium": [300.0, 200.0]})
    t = build_table(ref, pd.DataFrame({"row_id": ["B1", "M1"], "prediction": [80.0, 120.0]}), R)
    assert t.loc[t.row_id == "M1", "risk_premium"].iloc[0] == 120.0
    assert t["office_vs_written"].notna().all()
    with pytest.raises(ValueError):
        build_table(ref, pd.DataFrame({"row_id": ["M1"], "prediction": [1.0]}), R)
