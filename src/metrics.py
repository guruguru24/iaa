"""Scoring metric and actuarial diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score, mean_tweedie_deviance

POWER = 1.5


def tweedie_d2(y_true, y_pred, power: float = POWER, sample_weight=None) -> float:
    """Competition metric: Tweedie D² (power 1.5) vs constant baseline."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    return float(d2_tweedie_score(y_true, y_pred, power=power, sample_weight=sample_weight))


def tweedie_deviance(y_true, y_pred, power: float = POWER) -> float:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    return float(mean_tweedie_deviance(y_true, y_pred, power=power))


def actual_to_expected(y_true, y_pred) -> float:
    """A/E ratio: total actual / total expected."""
    return float(np.sum(y_true) / np.sum(y_pred))


def ae_by_group(df: pd.DataFrame, actual: str, expected: str, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(n=(actual, "size"), actual=(actual, "sum"), expected=(expected, "sum"))
    g["ae"] = g["actual"] / g["expected"]
    return g


def lift_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """A/E and mean actual by prediction decile."""
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    out = df.groupby("bin").agg(n=("y", "size"), actual=("y", "mean"), expected=("p", "mean"))
    out["ae"] = out["actual"] / out["expected"]
    return out


def gini(y_true, y_pred) -> float:
    """Normalised Gini (ordered Lorenz) of predictions vs actual."""
    y_true = np.asarray(y_true, float)
    order = np.argsort(-np.asarray(y_pred, float), kind="mergesort")
    cum = np.cumsum(y_true[order]) / y_true.sum()
    n = len(y_true)
    g = cum.sum() / n - (n + 1) / (2 * n)
    order_best = np.argsort(-y_true, kind="mergesort")
    cum_best = np.cumsum(y_true[order_best]) / y_true.sum()
    g_best = cum_best.sum() / n - (n + 1) / (2 * n)
    return float(g / g_best)
