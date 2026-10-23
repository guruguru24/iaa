"""Stage 3 (decision #14): simplex stacking of base-model predictions, fitted on one temporal fold.

Weights are non-negative and sum to 1 (each base model is already level-balanced), chosen to minimise the mean
Tweedie deviance (p = 1.5, the scoring loss) on the fitting fold. Cross-fit: weights fitted on fold A are scored on fold B.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_tweedie_deviance


def simplex_weights(y, preds: dict, power: float = 1.5) -> dict:
    names = list(preds)
    P = np.column_stack([np.clip(np.asarray(preds[n], float), 1e-9, None) for n in names])
    y = np.asarray(y, float)

    def loss(z):
        w = np.exp(z - z.max()); w /= w.sum()
        return mean_tweedie_deviance(y, P @ w, power=power)

    best = min((minimize(loss, z0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-12})
                for z0 in (np.zeros(len(names)), np.eye(len(names))[0] * 2)), key=lambda r: r.fun)
    w = np.exp(best.x - best.x.max()); w /= w.sum()
    return dict(zip(names, np.round(w, 4)))


def apply_weights(preds: dict, weights: dict) -> np.ndarray:
    return sum(weights[n] * np.asarray(preds[n], float) for n in weights)
