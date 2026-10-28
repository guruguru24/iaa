"""Prior-claims experience features (known at inception).

For each row starting on day s, only claims REPORTED before s count (a claim that happened earlier but was reported
after s was unknown when the row was priced). Counts only: claim amounts are valued at 31 Dec 2024 and would carry
post-inception development. CLAUDE.md rule 3 forbids same-period claim information; these use earlier periods only.

v2 (after w1_feature_test_20261023_0844): model features use FIXED look-back windows and are NaN when the window is
not inside the observed data (data starts 2021-01-01). v1 set them to 0 for 2021 rows, which confused "history not
observable" with "claim-free" and hurt the 2023 fold.

Model features (HIST_COLS), window w in {1y, 2y} = [s - 365*k days, s):
  hist_claims_{w}     same policy, non-declined claims with loss date in the window, reported before s
  hist_exposure_{w}   same policy, years on cover inside the window
  hist_declined_{w}   same policy, declined claims in the window
  cust_other_{w}      same customer, claims on OTHER lines' policies in the window
Diagnostics only (prefix hist_all_, excluded from models): all observed history before s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import LINES, _read

WINDOWS = {"1y": 365, "2y": 730}
HIST_COLS = [f"{f}_{w}" for w in WINDOWS for f in ("hist_claims", "hist_exposure", "hist_declined", "cust_other")]
HIST_DIAG_COLS = ["hist_all_claims", "hist_all_exposure"]
DATA_START = pd.Timestamp("2021-01-01")
DATA_START_BY_MARKET = {"w1": DATA_START, "w2": pd.Timestamp("2023-01-01")}   # emerging history starts 2023


def policy_customer_map(market: str = "w1") -> pd.DataFrame:
    """policy_id -> customer_id, line over all train and test policy files."""
    parts = []
    for line in LINES:
        for name in (f"policies_{line}", f"test_{line}"):
            parts.append(_read(name, market, columns=["policy_id", "customer_id"]).assign(line=line))
    return pd.concat(parts, ignore_index=True).drop_duplicates("policy_id")


def _overlap_years(start_prev, end_prev, lo, hi) -> pd.Series:
    """Years of [start_prev, end_prev] (inclusive) falling in [lo, hi)."""
    a = np.maximum(start_prev, lo)
    b = np.minimum(end_prev + pd.Timedelta(days=1), hi)
    return ((b - a).dt.days.clip(lower=0)) / 365.0


def history_features(rows: pd.DataFrame, claims: pd.DataFrame, policy_customer: pd.DataFrame,
                     data_start: pd.Timestamp = DATA_START) -> pd.DataFrame:
    """Features for `rows` (row_id, policy_id, customer_id, period_start, period_end, exposure).

    `rows` must contain every observed row of the line that can precede the target rows (e.g. train + test).
    """
    r = rows[["row_id", "policy_id", "customer_id", "period_start", "period_end", "exposure"]].copy()
    out = pd.DataFrame(index=pd.Index(r["row_id"], name="row_id"))
    s = r.set_index("row_id")["period_start"]

    # same-policy earlier rows (for exposure) and claims reported before s
    p = r[["row_id", "policy_id", "period_start", "period_end"]]
    m = p.merge(p, on="policy_id", suffixes=("", "_prev"))
    m = m[m["period_end_prev"] < m["period_start"]]
    c = claims[["policy_id", "loss_date", "report_date", "status"]]
    mc = r[["row_id", "policy_id", "period_start"]].merge(c, on="policy_id")
    mc = mc[mc["report_date"] < mc["period_start"]]
    declined = mc["status"].eq("repudiated")

    line_of_policy = policy_customer.set_index("policy_id")["line"]
    pc = policy_customer.rename(columns={"line": "claim_line"})[["policy_id", "customer_id", "claim_line"]]
    cc = claims.loc[claims["status"] != "repudiated", ["policy_id", "loss_date", "report_date"]].merge(pc, on="policy_id")
    mx = r[["row_id", "customer_id", "policy_id", "period_start"]].merge(
        cc[["customer_id", "claim_line", "loss_date", "report_date"]], on="customer_id")
    mx = mx[(mx["report_date"] < mx["period_start"]) & (mx["claim_line"] != mx["policy_id"].map(line_of_policy))]

    for w, days in WINDOWS.items():
        lo_m = m["period_start"] - pd.Timedelta(days=days)
        expo = _overlap_years(m["period_start_prev"], m["period_end_prev"], lo_m, m["period_start"]).groupby(m["row_id"]).sum()
        in_w = mc["loss_date"] >= mc["period_start"] - pd.Timedelta(days=days)
        in_x = mx["loss_date"] >= mx["period_start"] - pd.Timedelta(days=days)
        cols = {f"hist_exposure_{w}": expo,
                f"hist_claims_{w}": mc[in_w & ~declined].groupby("row_id").size(),
                f"hist_declined_{w}": mc[in_w & declined].groupby("row_id").size(),
                f"cust_other_{w}": mx[in_x].groupby("row_id").size()}
        observable = (s - pd.Timedelta(days=days)) >= data_start     # window fully inside the observed data
        for name, v in cols.items():
            out[name] = v.reindex(out.index).fillna(0).astype(float).where(observable.reindex(out.index))

    out["hist_all_claims"] = mc[~declined].groupby("row_id").size().reindex(out.index).fillna(0)
    out["hist_all_exposure"] = (m.assign(e=_overlap_years(m["period_start_prev"], m["period_end_prev"],
                                                          pd.Series(data_start, index=m.index), m["period_start"]))
                                .groupby("row_id")["e"].sum().reindex(out.index).fillna(0.0))
    return out.reset_index()
