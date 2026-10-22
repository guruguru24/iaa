"""Development of reported net cost (valued 31 Dec 2024) to ultimate, by line x peril x policy year (gate G3).

Rows cover calendar-year periods, so policy_year == loss year for every in-period claim.

Approved G3 method (decision #7, logs/decision_log.md):
- IBNR: cost-based development factor per line x peril x loss year, from the reporting-delay CDF of mature
  loss years weighted by net cost (late-reported claims may cost more than early ones).
      ultimate = (closed_net + open_net * ibner_open) * cost_dev_factor
  ibner_open stays 1.0 centrally: with a single valuation date it cannot be estimated from the data.
- Motor bodily injury, immature years: expected-severity (Bornhuetter-Ferguson style) override
      ultimate = reported_n * count_dev_factor * base_severity * (1 + trend) ** (year - mean(base_years))
                 + reported net above the cap (kept as reported; large-loss loading handles it, decision #6)
  because reported BI severity falls as the open share rises (case estimates below settlement values).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda import ibnr_count_estimate, lag_cdf

IBNER_SCENARIOS = {  # sensitivities for non-overridden perils (not estimates)
    "low": {"_default": 0.95},
    "central": {"_default": 1.0},
    "high": {"_default": 1.10, "subsidence": 1.30},
}


def cost_factor_estimate(claims: pd.DataFrame, cutoff: pd.Timestamp, mature_years) -> pd.DataFrame:
    """Cost-based development factor by loss year: per loss date, reported cost / share of cost reported by then."""
    c = claims[claims["status"] != "repudiated"]
    c = c.assign(loss_year=c["loss_date"].dt.year, lag=(c["report_date"] - c["loss_date"]).dt.days,
                 window=(cutoff - c["loss_date"]).dt.days,
                 net=(c["incurred_to_date"] - c["recoveries"]).clip(lower=0))
    m = c[c["loss_year"].isin(mature_years)]
    F = lag_cdf(m["lag"], m["net"])
    c = c.assign(p=np.clip(F(c["window"]), 1e-3, 1.0))
    g = c.groupby("loss_year").apply(
        lambda d: (d["net"] / d["p"]).sum() / d["net"].sum() if d["net"].sum() > 0 else 1.0, include_groups=False)
    return g.rename("cost_dev_factor").reset_index()


def development_factors(claims: pd.DataFrame, mature_years=(2021, 2022), cutoff=None) -> pd.DataFrame:
    """Count- and cost-based development factors per line x peril x policy (= loss) year."""
    cutoff = claims["report_date"].max() if cutoff is None else cutoff
    out = []
    for (line, peril), g in claims.groupby(["line", "peril"]):
        cnt = ibnr_count_estimate(g, cutoff, mature_years)[["loss_year", "count_dev_factor"]]
        out.append(cnt.merge(cost_factor_estimate(g, cutoff, mature_years), on="loss_year", how="left")
                   .assign(line=line, peril=peril))
    f = pd.concat(out, ignore_index=True).rename(columns={"loss_year": "policy_year"})
    f["cost_dev_factor"] = f["cost_dev_factor"].fillna(1.0)
    return f


def count_factors(claims: pd.DataFrame, mature_years=(2021, 2022), cutoff=None) -> pd.DataFrame:
    """Count development factors per line x peril x loss year (kept for the G3 evidence scripts)."""
    cutoff = claims["report_date"].max() if cutoff is None else cutoff
    out = [ibnr_count_estimate(g, cutoff, mature_years).assign(line=line, peril=peril)
           for (line, peril), g in claims.groupby(["line", "peril"])]
    return pd.concat(out, ignore_index=True).rename(columns={"loss_year": "policy_year"})


def peril_year_table(claim_frame: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
    """Reported net, open net, capped net and claim count per line x policy year x peril (in-period, non-declined)."""
    c = claim_frame[claim_frame["in_period"] & ~claim_frame["declined"]]
    c = c.assign(policy_year=c["policy_year"].astype(int),  # float in claim frames (NaN for out-of-period)
                 open_net=c["net"].where(c["status"] == "open", 0.0),
                 net_capped=c["net"] if cap is None else c["net"].clip(upper=cap))
    return (c.groupby(["line", "policy_year", "peril"])
             .agg(n=("claim_id", "size"), net=("net", "sum"), open_net=("open_net", "sum"),
                  net_capped=("net_capped", "sum")).reset_index())


def develop(table: pd.DataFrame, factors: pd.DataFrame, ibner_open: dict | float = 1.0,
            factor_col: str = "count_dev_factor") -> pd.DataFrame:
    """Add dev_factor, ibner and ultimate columns to a peril_year_table."""
    ibner = ibner_open if isinstance(ibner_open, dict) else {"_default": ibner_open}
    cols = ["line", "peril", "policy_year"] + [c for c in ("count_dev_factor", "cost_dev_factor") if c in factors]
    t = table.merge(factors[cols], on=["line", "peril", "policy_year"], how="left")
    t["dev_factor"] = t[factor_col].fillna(1.0)
    t["ibner"] = t["peril"].map(lambda p: ibner.get(p, ibner["_default"]))
    t["ultimate"] = (t["net"] - t["open_net"] + t["open_net"] * t["ibner"]) * t["dev_factor"]
    return t


def expected_severity_override(developed: pd.DataFrame, line: str, peril: str, years, base_years,
                               trend: float) -> pd.DataFrame:
    """Replace ultimate for (line, peril, years) by developed count x trended base-year capped severity + excess over cap.

    Needs peril_year_table(..., cap=...) columns (n, net_capped) and count_dev_factor from development_factors.
    """
    d = developed.copy()
    sel = (d["line"] == line) & (d["peril"] == peril)
    base = d[sel & d["policy_year"].isin(base_years)]
    base_sev = base["net_capped"].sum() / base["n"].sum()
    mid = float(np.mean(base_years))
    tgt = sel & d["policy_year"].isin(years)
    n_ult = d.loc[tgt, "n"] * d.loc[tgt, "count_dev_factor"].fillna(1.0)
    sev = base_sev * (1 + trend) ** (d.loc[tgt, "policy_year"] - mid)
    d.loc[tgt, "ultimate"] = n_ult * sev + (d.loc[tgt, "net"] - d.loc[tgt, "net_capped"])
    d.loc[tgt, "override"] = f"expected severity {base_sev:,.0f} x (1+{trend:.0%})^(year-{mid})"
    return d


def developed_table(claim_frame: pd.DataFrame, factors: pd.DataFrame, cfg: dict, scenario: str = "central") -> pd.DataFrame:
    """Apply the approved G3 method for one line's claim frame. cfg = config['assumptions']['development']."""
    t = peril_year_table(claim_frame, cap=cfg["severity_cap"])
    d = develop(t, factors, IBNER_SCENARIOS[scenario], factor_col=f"{cfg['basis']}_dev_factor")
    for o in cfg.get("expected_severity", []):
        if (d["line"] == o["line"]).any():
            d = expected_severity_override(d, o["line"], o["peril"], o["years"], o["base_years"], o["trend"][scenario])
    return d


def year_dev_factors(developed: pd.DataFrame) -> pd.DataFrame:
    """Ultimate / reported per line x policy year."""
    g = developed.groupby(["line", "policy_year"])[["net", "ultimate"]].sum()
    g["dev_factor"] = g["ultimate"] / g["net"]
    return g.reset_index()


def apply_to_rows(rows: pd.DataFrame, line: str, yf: pd.DataFrame, col: str = "target_net") -> pd.Series:
    """Row target grossed up by its line x policy-year development factor (level adjustment, not row-specific IBNR)."""
    f = yf[yf["line"] == line].set_index("policy_year")["dev_factor"]
    return rows[col] * rows["policy_year"].map(f).fillna(1.0)
