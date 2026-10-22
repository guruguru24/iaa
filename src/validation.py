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
