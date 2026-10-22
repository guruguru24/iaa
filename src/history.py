"""Prior-claims experience features (known at inception).

For each row starting on day s, only claims REPORTED before s count (a claim that happened earlier but was reported
after s was unknown when the row was priced). Counts only: claim amounts are valued at 31 Dec 2024 and would carry
post-inception development. CLAUDE.md rule 3 forbids same-period claim information; these use earlier periods only.

Features
  hist_exposure      years on cover of the same policy before s (history length; 0 for the first observed row)
  hist_claims        same policy, non-declined claims reported before s
  hist_claims_1y     ... of which loss date within 365 days before s
  hist_declined      same policy, declined claims reported before s
  hist_rate          hist_claims / hist_exposure (NaN without history)
  cust_claims_other  same customer, claims on OTHER lines' policies reported before s (cross-line signal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import LINES, _read

HIST_COLS = ["hist_exposure", "hist_claims", "hist_claims_1y", "hist_declined", "hist_rate", "cust_claims_other"]


def policy_customer_map(market: str = "w1") -> pd.DataFrame:
    """policy_id -> customer_id, line over all train and test policy files."""
    parts = []
    for line in LINES:
        for name in (f"policies_{line}", f"test_{line}"):
            parts.append(_read(name, market, columns=["policy_id", "customer_id"]).assign(line=line))
    return pd.concat(parts, ignore_index=True).drop_duplicates("policy_id")


def history_features(rows: pd.DataFrame, claims: pd.DataFrame, policy_customer: pd.DataFrame) -> pd.DataFrame:
    """Features for `rows` (row_id, policy_id, customer_id, period_start, period_end, exposure).

    `rows` must contain every observed row of the line that can precede the target rows (e.g. train + test),
    because history exposure is built from the same policy's earlier rows.
    """
    r = rows[["row_id", "policy_id", "customer_id", "period_start", "period_end", "exposure"]].copy()

    # history exposure: same-policy rows that END before this row starts (rows never overlap within a policy)
    p = r[["row_id", "policy_id", "period_start", "period_end", "exposure"]]
    m = p.merge(p, on="policy_id", suffixes=("", "_prev"))
    m = m[m["period_end_prev"] < m["period_start"]]
    hist_exp = m.groupby("row_id")["exposure_prev"].sum()

    # same-policy claims reported before the row starts
    c = claims[["claim_id", "policy_id", "loss_date", "report_date", "status"]]
    mc = r[["row_id", "policy_id", "period_start"]].merge(c, on="policy_id")
    mc = mc[mc["report_date"] < mc["period_start"]]
    declined = mc["status"].eq("repudiated")
    g_ok = mc[~declined].groupby("row_id")
    hist_claims = g_ok.size()
    recent = mc[~declined & (mc["loss_date"] >= mc["period_start"] - pd.Timedelta(days=365))].groupby("row_id").size()
    hist_declined = mc[declined].groupby("row_id").size()

    # same customer, other lines' claims reported before the row starts
    line_of_policy = policy_customer.set_index("policy_id")["line"]
    pc = policy_customer.rename(columns={"line": "claim_line"})[["policy_id", "customer_id", "claim_line"]]
    cc = claims.loc[claims["status"] != "repudiated", ["policy_id", "report_date"]].merge(pc, on="policy_id")
    mx = r[["row_id", "customer_id", "policy_id", "period_start"]].merge(
        cc[["customer_id", "claim_line", "report_date"]], on="customer_id")
    mx = mx[(mx["report_date"] < mx["period_start"]) & (mx["claim_line"] != mx["policy_id"].map(line_of_policy))]
    cust_other = mx.groupby("row_id").size()

    out = pd.DataFrame(index=r["row_id"])
    out["hist_exposure"] = hist_exp.reindex(out.index).fillna(0.0)
    out["hist_claims"] = hist_claims.reindex(out.index).fillna(0).astype(int)
    out["hist_claims_1y"] = recent.reindex(out.index).fillna(0).astype(int)
    out["hist_declined"] = hist_declined.reindex(out.index).fillna(0).astype(int)
    out["hist_rate"] = np.where(out["hist_exposure"] > 0, out["hist_claims"] / out["hist_exposure"].where(out["hist_exposure"] > 0), np.nan)
    out["cust_claims_other"] = cust_other.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()
