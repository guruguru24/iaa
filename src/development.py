"""Development of reported net cost (valued 31 Dec 2024) to ultimate, by line x peril x policy year (gate G3).

Rows cover calendar-year periods, so policy_year == loss year for every in-period claim.

    ultimate = (closed_net + open_net * ibner_open) * count_dev_factor

- count_dev_factor: IBNR claim counts from the reporting-delay CDF (src.eda.ibnr_count_estimate), per line x peril.
  Assumes late-reported claims cost the same on average as reported ones of the same peril and year
  (may understate bodily injury, where late reports tend to be larger).
- ibner_open: multiplier on open-claim case estimates. With a single valuation date there is no payment history,
  so it cannot be estimated from the data: 1.0 is the central assumption, other values are sensitivities.
"""
from __future__ import annotations

import pandas as pd

from src.eda import ibnr_count_estimate

IBNER_SCENARIOS = {  # placeholders for discussion at G3, not estimates
    "low": {"_default": 0.95},
    "central": {"_default": 1.0},
    "high": {"_default": 1.10, "bodily_injury": 1.30, "subsidence": 1.30},
}


def count_factors(claims: pd.DataFrame, mature_years=(2021, 2022), cutoff=None) -> pd.DataFrame:
    """Count development factors per line x peril x loss year."""
    cutoff = claims["report_date"].max() if cutoff is None else cutoff
    out = [ibnr_count_estimate(g, cutoff, mature_years).assign(line=line, peril=peril)
           for (line, peril), g in claims.groupby(["line", "peril"])]
    return pd.concat(out, ignore_index=True).rename(columns={"loss_year": "policy_year"})


def peril_year_table(claim_frame: pd.DataFrame) -> pd.DataFrame:
    """Reported net, open net and claim count per line x policy year x peril (in-period, non-declined claims)."""
    c = claim_frame[claim_frame["in_period"] & ~claim_frame["declined"]]
    c = c.assign(open_net=c["net"].where(c["status"] == "open", 0.0))
    return (c.groupby(["line", "policy_year", "peril"])
             .agg(n=("claim_id", "size"), net=("net", "sum"), open_net=("open_net", "sum")).reset_index())


def develop(table: pd.DataFrame, factors: pd.DataFrame, ibner_open: dict | float = 1.0) -> pd.DataFrame:
    """Add count_dev_factor, ibner and ultimate columns to a peril_year_table."""
    ibner = ibner_open if isinstance(ibner_open, dict) else {"_default": ibner_open}
    t = table.merge(factors[["line", "peril", "policy_year", "count_dev_factor"]],
                    on=["line", "peril", "policy_year"], how="left")
    t["count_dev_factor"] = t["count_dev_factor"].fillna(1.0)
    t["ibner"] = t["peril"].map(lambda p: ibner.get(p, ibner["_default"]))
    t["ultimate"] = (t["net"] - t["open_net"] + t["open_net"] * t["ibner"]) * t["count_dev_factor"]
    return t


def year_dev_factors(developed: pd.DataFrame) -> pd.DataFrame:
    """Ultimate / reported per line x policy year."""
    g = developed.groupby(["line", "policy_year"])[["net", "ultimate"]].sum()
    g["dev_factor"] = g["ultimate"] / g["net"]
    return g.reset_index()


def apply_to_rows(rows: pd.DataFrame, line: str, yf: pd.DataFrame, col: str = "target_net") -> pd.Series:
    """Row target grossed up by its line x policy-year development factor (level adjustment, not row-specific IBNR)."""
    f = yf[yf["line"] == line].set_index("policy_year")["dev_factor"]
    return rows[col] * rows["policy_year"].map(f).fillna(1.0)
