"""Temporal validation helpers (docs/03 §1-2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import actual_to_expected, gini, tweedie_d2


def temporal_split(df: pd.DataFrame, train_years, valid_years, year_col: str = "policy_year"):
    return df[df[year_col].isin(train_years)].copy(), df[df[year_col].isin(valid_years)].copy()


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"d2": tweedie_d2(y_true, y_pred), "ae": actual_to_expected(y_true, y_pred),
            "gini": gini(y_true, y_pred), "pred_total": y_pred.sum(), "actual_total": y_true.sum()}


def level_sensitivity(y_true, y_pred, scales=(0.9, 0.95, 1.0, 1.05, 1.1)) -> pd.DataFrame:
    """D2 when all predictions are multiplied by a common factor; also the best factor on a fine grid.

    Shows how much of the gap is level (fixable by calibration) versus ranking.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    grid = np.round(np.arange(0.80, 1.2001, 0.01), 2)
    d2 = {s: tweedie_d2(y_true, y_pred * s) for s in sorted(set(grid) | set(scales))}
    best = max(d2, key=d2.get)
    rows = [{"scale": s, "d2": d2[s]} for s in scales] + [{"scale": best, "d2": d2[best], "best": True}]
    return pd.DataFrame(rows)


def segment_ae(df: pd.DataFrame, actual: str, pred: str, factor: str, exposure: str = "exposure",
               n_bins: int = 5) -> pd.DataFrame:
    """A/E by level of one factor (numeric factors in quantile bands), with exposure share and claim count."""
    from src.eda import band
    d = df.assign(level=band(df[factor], n_bins))
    g = d.groupby("level", observed=True).agg(exposure=(exposure, "sum"), actual=(actual, "sum"), expected=(pred, "sum"),
                                              rows_with_claim=(actual, lambda s: (s > 0).sum()))
    g["ae"] = g["actual"] / g["expected"]
    g["exposure_share"] = g["exposure"] / g["exposure"].sum()
    g.insert(0, "factor", factor)
    return g.reset_index()
