"""Target reconstruction: claim-level net amounts aggregated to policy rows.

Definition (docs/00): ultimate cost of claims with loss date in the row's period, net of excess,
net of recoveries, excluding declined claims. Claims data gives paid/incurred *to date* only, so the
row target built here is net *incurred to date*; development to ultimate is a separate step (gate G3).

Verified W1 facts this module relies on (see logs/data_quality_log.md):
- Row period is inclusive [period_start, period_end]; each claim matches at most one row.
- Claim amounts are gross of excess (13–27% of non-declined claims are below the policy excess).
- 'repudiated' = declined; those rows carry zero amounts. Closed claims have paid == incurred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECLINED = {"repudiated"}


def claim_net_amount(gross, excess, recoveries, floor_at_zero: bool = True):
    """Per-claim net cost = (gross - excess)+ - recoveries, optionally floored at 0.

    With the floor, this equals max(gross - excess - recoveries, 0), i.e. the order of applying
    excess and recoveries does not matter.
    """
    gross = np.asarray(gross, float)
    net = np.maximum(gross - np.asarray(excess, float), 0.0) - np.asarray(recoveries, float)
    return np.maximum(net, 0.0) if floor_at_zero else net


def match_claims_to_rows(claims: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach each claim to the policy row whose inclusive period contains its loss date.

    Returns claims with row_id (NaN when no row contains the loss date) and the row's excesses.
    """
    keep = ["row_id", "policy_id", "period_start", "period_end", "voluntary_excess", "compulsory_excess"]
    m = claims.merge(rows[keep], on="policy_id", how="inner")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])]
    if m["claim_id"].duplicated().any():
        raise ValueError("a claim matched more than one row (overlapping periods)")
    out = claims.merge(m[["claim_id", "row_id", "voluntary_excess", "compulsory_excess"]], on="claim_id", how="left")
    return out


def claim_level_net(claims_matched: pd.DataFrame, amount: str = "incurred_to_date",
                    floor_at_zero: bool = True) -> pd.DataFrame:
    """Add excess, gross and net columns; declined claims get net 0 and is_counted False."""
    c = claims_matched.copy()
    c["excess"] = c["voluntary_excess"].fillna(0) + c["compulsory_excess"].fillna(0)
    c["declined"] = c["status"].isin(DECLINED)
    c["gross"] = np.where(c["declined"], 0.0, c[amount])
    c["net"] = np.where(c["declined"], 0.0, claim_net_amount(c[amount], c["excess"], c["recoveries"], floor_at_zero))
    c["in_period"] = c["row_id"].notna()
    c["is_counted"] = c["in_period"] & ~c["declined"] & (c["net"] > 0)
    return c


def build_row_target(rows: pd.DataFrame, claims: pd.DataFrame, amount: str = "incurred_to_date",
                     floor_at_zero: bool = True) -> pd.DataFrame:
    """Row-level target columns: target_net, n_claims_net (>0 net), gross, open-claim share."""
    c = claim_level_net(match_claims_to_rows(claims, rows), amount, floor_at_zero)
    c = c[c["in_period"]]
    c["open_net"] = np.where(c["status"] == "open", c["net"], 0.0)
    agg = c.groupby("row_id").agg(
        target_net=("net", "sum"),
        gross_incurred=("gross", "sum"),
        open_net=("open_net", "sum"),
        n_claims_net=("is_counted", "sum"),
        n_claims_reported=("claim_id", "size"),
    )
    out = rows[["row_id"]].merge(agg, left_on="row_id", right_index=True, how="left")
    out = out.fillna({k: 0 for k in agg.columns})
    out[["n_claims_net", "n_claims_reported"]] = out[["n_claims_net", "n_claims_reported"]].astype(int)
    return out.set_index(rows.index)
