"""Modelling frames: train rows with developed / capped / on-levelled targets, and test rows, with features.

Target per row (decision #7): reported net x line-year development factor (central), each claim optionally capped,
then trended ("on-levelled") to the year being predicted: x (1 + g) ** (to_year - policy_year).
"""
from __future__ import annotations

import pandas as pd

from src.data import exposure_from_dates, load_claims, load_config, load_policies, load_table
from src.development import apply_to_rows, developed_table, development_factors, year_dev_factors
from src.eda import claim_frame, line_frame
from src.features import add_features, weather_features
from src.history import history_features, policy_customer_map
from src.preprocess import clean_customers, preprocess, sum_insured_reference
from src.trend import row_capped_target


def onlevel_factor(policy_year: pd.Series, to_year: int, annual_trend: float) -> pd.Series:
    return (1.0 + annual_trend) ** (to_year - policy_year)


def reference_tables(market: str = "w1") -> dict:
    return {"addresses": load_table("addresses", market), "customers": clean_customers(load_table("customers", market)),
            "weather": load_table("weather", market), "policy_customer": policy_customer_map(market)}


def _add_history(target: pd.DataFrame, all_rows: pd.DataFrame, claims: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Join prior-claims features computed over `all_rows` (every row that can precede the target rows)."""
    h = history_features(all_rows, claims, refs["policy_customer"])
    return target.merge(h, on="row_id", how="left")


def train_frame(line: str, market: str = "w1", caps=(None,), scenario: str = "central", claims=None,
                refs=None, weather_mode: str = "same_year", weather_years=None, history: bool = False) -> pd.DataFrame:
    """Train rows with features and targets: target_dev (uncapped) and target_dev_cap{cap} per requested cap."""
    cfg = load_config()["assumptions"]["development"]
    claims = load_claims(market) if claims is None else claims
    refs = reference_tables(market) if refs is None else refs
    rows = line_frame(line, market, claims)          # applies approved G2 feature fixes
    rows["exposure"] = rows["exposure_dates"]         # approved G2 exposure rule
    cf = claim_frame(line, rows, claims)
    yf = year_dev_factors(developed_table(cf, development_factors(claims, cfg["mature_years"]), cfg, scenario))
    rows["target_dev"] = apply_to_rows(rows, line, yf)
    for cap in caps:
        if cap is not None:
            rows[f"target_cap"] = row_capped_target(rows, cf, cap)
            rows[f"target_dev_cap{int(cap)}"] = apply_to_rows(rows, line, yf, col="target_cap")
    rows = rows.drop(columns=["target_cap"], errors="ignore")
    if history:
        rows = _add_history(rows, rows, claims, refs)
    w = weather_features(refs["weather"], weather_mode, weather_years)
    return add_features(rows, line, refs["addresses"], refs["customers"], w)


def test_frame(line: str, market: str = "w1", refs=None, weather_mode: str = "same_year", weather_years=None,
               history: bool = False, claims=None) -> pd.DataFrame:
    """Test rows cleaned with the TRAIN sum_insured median, with the same features."""
    refs = reference_tables(market) if refs is None else refs
    train = load_policies(line, market)
    ref = {"sum_insured_median": sum_insured_reference(train)} if "sum_insured" in train else {}
    te = preprocess(load_policies(line, market, "test"), line, ref)
    te["policy_year"] = te["period_start"].dt.year
    if history:
        claims = load_claims(market) if claims is None else claims
        prev = train.assign(exposure=exposure_from_dates(train["period_start"], train["period_end"]))
        cols = ["row_id", "policy_id", "customer_id", "period_start", "period_end", "exposure"]
        te = _add_history(te, pd.concat([prev[cols], te[cols]], ignore_index=True), claims, refs)
    w = weather_features(refs["weather"], weather_mode, weather_years)
    return add_features(te, line, refs["addresses"], refs["customers"], w)
