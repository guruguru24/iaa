"""Target reconstruction: claim-level net amounts aggregated to policy rows.

Definition (docs/00): ultimate cost of claims with loss date in the row's period, net of excess,
net of recoveries, excluding declined claims. Claims data gives paid/incurred *to date* only, so the
row target built here is net *incurred to date*; development to ultimate is a separate step (gate G3).

Facts this module relies on (see logs/data_quality_log.md):
- Official data description: paid_to_date is already NET of excess, incurred = paid + case estimate.
  So excess is NOT subtracted by default (subtract_excess=False). DQ2's earlier "gross" reading was wrong.
- Row period is inclusive [period_start, period_end]; each claim matches at most one row (verified W1).
- 'repudiated' = declined; those rows carry zero amounts. Closed claims have paid == incurred (verified W1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECLINED = {"repudiated"}


def claim_net_amount(amount, recoveries, excess=0.0, subtract_excess: bool = False, floor_at_zero: bool = True):
    """Per-claim net cost = amount - recoveries (- excess only if amounts were gross), optionally floored at 0."""
    amount = np.asarray(amount, float)
    if subtract_excess:
        amount = np.maximum(amount - np.asarray(excess, float), 0.0)
    net = amount - np.asarray(recoveries, float)
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
                    subtract_excess: bool = False, floor_at_zero: bool = True) -> pd.DataFrame:
    """Add excess, amount (before recoveries) and net columns; declined claims get net 0 and is_counted False."""
    c = claims_matched.copy()
    c["excess"] = c["voluntary_excess"].fillna(0) + c["compulsory_excess"].fillna(0)
    c["declined"] = c["status"].isin(DECLINED)
    c["amount"] = np.where(c["declined"], 0.0, c[amount])
    c["net"] = np.where(c["declined"], 0.0,
                        claim_net_amount(c[amount], c["recoveries"], c["excess"], subtract_excess, floor_at_zero))
    c["in_period"] = c["row_id"].notna()
    c["is_counted"] = c["in_period"] & ~c["declined"] & (c["net"] > 0)
    return c


def build_row_target(rows: pd.DataFrame, claims: pd.DataFrame, amount: str = "incurred_to_date",
                     subtract_excess: bool = False, floor_at_zero: bool = True) -> pd.DataFrame:
    """Row-level target columns: target_net, amount before recoveries, open-claim net, claim counts."""
    c = claim_level_net(match_claims_to_rows(claims, rows), amount, subtract_excess, floor_at_zero)
    c = c[c["in_period"]]
    c["open_net"] = np.where(c["status"] == "open", c["net"], 0.0)
    agg = c.groupby("row_id").agg(
        target_net=("net", "sum"),
        amount_before_recoveries=("amount", "sum"),
        open_net=("open_net", "sum"),
        n_claims_net=("is_counted", "sum"),
        n_claims_reported=("claim_id", "size"),
    )
    out = rows[["row_id"]].merge(agg, left_on="row_id", right_index=True, how="left")
    out = out.fillna({k: 0 for k in agg.columns})
    out[["n_claims_net", "n_claims_reported"]] = out[["n_claims_net", "n_claims_reported"]].astype(int)
    return out.set_index(rows.index)
