"""Office premium build-up from our risk premium (docs/07, Week 2, panel-judged). All rates come from `expenses`.

Central assumptions (decision #37):
  - premium tax is a share of the gross premium -> inside the denominator (alternative: added on top)
  - fixed expense is charged per policy in full (alternative: pro-rata to exposure)
  - commission for a channel blank in the market (`n/a`) raises an error (no silent fallback)

    loaded_cost    = RP + cat_RI x RP (buildings/contents only) + fixed
    office_premium = loaded_cost / (1 - variable - commission[channel] - premium_tax - profit)      # tax "within"
    office_premium = loaded_cost / (1 - variable - commission[channel] - profit) x (1 + premium_tax)  # tax "on_top"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROPERTY_LINES = {"buildings", "contents"}


def expense_rates(expenses: pd.DataFrame, market: str = "emerging") -> dict:
    """item -> value for the market column (NaN kept: means not applicable in that market)."""
    return dict(zip(expenses["item"], expenses[market].astype(float)))


def build_up(rp: float, line: str, channel: str, exposure: float, rates: dict,
             tax_mode: str = "within", fixed_mode: str = "per_policy") -> dict:
    """Components (CU) of the office premium for one policy row."""
    comm = rates.get(f"commission_{channel}", np.nan)
    if pd.isna(comm):
        raise ValueError(f"commission for channel '{channel}' is not applicable/blank in this market")
    cat = rates["cat_reinsurance"] * rp if line in PROPERTY_LINES else 0.0
    fixed = rates["fixed_expense_per_policy"] * (exposure if fixed_mode == "pro_rata" else 1.0)
    loaded = rp + cat + fixed
    var, tax, profit = rates["variable_expense"], rates["premium_tax"], rates["target_profit_margin"]
    if tax_mode == "within":
        op = loaded / (1.0 - var - comm - tax - profit)
        base = op
    elif tax_mode == "on_top":
        base = loaded / (1.0 - var - comm - profit)
        op = base * (1.0 + tax)
    else:
        raise ValueError(tax_mode)
    out = {"risk_premium": rp, "cat_reinsurance": cat, "fixed_expense": fixed, "variable_expense": var * base,
           "commission": comm * base, "profit": profit * base, "premium_tax": tax * base, "office_premium": op}
    out["check_sum"] = sum(v for k, v in out.items() if k not in ("office_premium",))   # must equal office_premium
    return out


def build_table(reference: pd.DataFrame, predictions: pd.DataFrame, rates: dict, **kw) -> pd.DataFrame:
    """One row per reference policy: inputs, components, office premium, current written premium and ratio."""
    df = reference[["row_id", "line", "channel", "exposure", "new_business", "written_premium"]].merge(
        predictions[["row_id", "prediction"]], on="row_id", how="left", validate="one_to_one")
    if df["prediction"].isna().any():
        raise ValueError("reference policies missing from predictions: " + ", ".join(df.loc[df.prediction.isna(), "row_id"]))
    comp = pd.DataFrame([build_up(r.prediction, r.line, r.channel, r.exposure, rates, **kw) for r in df.itertuples()])
    out = pd.concat([df.drop(columns="prediction").reset_index(drop=True), comp], axis=1)
    out["office_vs_written"] = out["office_premium"] / out["written_premium"]
    return out
