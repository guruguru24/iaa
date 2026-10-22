"""Model ladder (docs/02 §1). Targets are per unit exposure; predictions are multiplied back by exposure.

Rung 0: constant burning cost per line.  Rung 2: Tweedie GLM (p=1.5, log link) on train-fitted bands.
Capped variants fit on capped targets and multiply by a large-loss loading = uncapped / capped total on train
(decision #6: excess over the cap spread across the line).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor

from src.features import BandEncoder


class ConstantModel:
    def fit(self, df: pd.DataFrame, target: str, exposure: str = "exposure"):
        self.rate_ = df[target].sum() / df[exposure].sum()
        return self

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self.rate_ * df[exposure].to_numpy()


class TweedieGLM:
    def __init__(self, factors, power: float = 1.5, alpha: float = 1e-4, n_bins: int = 10):
        self.factors, self.power, self.alpha, self.n_bins = list(factors), power, alpha, n_bins

    def fit(self, df: pd.DataFrame, target: str, exposure: str = "exposure", loading_target: str | None = None):
        self.enc_ = BandEncoder(self.n_bins).fit(df, self.factors)
        X = self.enc_.transform(df)
        w = df[exposure].to_numpy()
        self.glm_ = TweedieRegressor(power=self.power, link="log", alpha=self.alpha, solver="newton-cholesky", max_iter=200)
        self.glm_.fit(X, df[target].to_numpy() / w, sample_weight=w)
        self.loading_ = 1.0 if loading_target is None else df[loading_target].sum() / df[target].sum()
        return self

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self.glm_.predict(self.enc_.transform(df)) * df[exposure].to_numpy() * self.loading_

    def relativities(self) -> pd.Series:
        return pd.Series(np.exp(self.glm_.coef_), index=self.enc_.columns_)
