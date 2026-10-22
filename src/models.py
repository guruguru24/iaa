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


def fit_weights(df: pd.DataFrame, exposure: str = "exposure", recency: float | None = None,
                ref_year: int | None = None) -> np.ndarray:
    """Exposure weights, optionally decayed by age: exposure * recency ** (ref_year - policy_year)."""
    w = df[exposure].to_numpy(dtype=float)
    if recency is None or recency == 1.0:
        return w
    return w * recency ** (ref_year - df["policy_year"].to_numpy(dtype=float))


class ConstantModel:
    def fit(self, df: pd.DataFrame, target: str, exposure: str = "exposure"):
        self.rate_ = df[target].sum() / df[exposure].sum()
        return self

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self.rate_ * df[exposure].to_numpy()


class TweedieGLM:
    def __init__(self, factors, power: float = 1.5, alpha: float = 1e-4, n_bins: int = 10):
        self.factors, self.power, self.alpha, self.n_bins = list(factors), power, alpha, n_bins

    def fit(self, df: pd.DataFrame, target: str, exposure: str = "exposure", loading_target: str | None = None,
            recency: float | None = None, ref_year: int | None = None):
        self.enc_ = BandEncoder(self.n_bins).fit(df, self.factors)
        X = self.enc_.transform(df)
        w = df[exposure].to_numpy()
        self.glm_ = TweedieRegressor(power=self.power, link="log", alpha=self.alpha, solver="newton-cholesky", max_iter=200)
        self.glm_.fit(X, df[target].to_numpy() / w, sample_weight=fit_weights(df, exposure, recency, ref_year))
        self.train_balance_ = df[target].sum() / (self.glm_.predict(X) * w).sum()  # diagnostic only, not applied
        self.loading_ = 1.0 if loading_target is None else df[loading_target].sum() / df[target].sum()
        return self

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self.glm_.predict(self.enc_.transform(df)) * df[exposure].to_numpy() * self.loading_

    def relativities(self) -> pd.Series:
        return pd.Series(np.exp(self.glm_.coef_), index=self.enc_.columns_)


class LGBMTweedie:
    """Rung 4: LightGBM, objective tweedie (p=1.5), target per exposure with exposure weight (same basis as the GLM).

    Categories are fixed on the training rows (unseen test levels -> NaN). Early stopping uses a holdout of
    policies inside the training years only, so validation/test rows never steer training.
    """

    def __init__(self, factors, power: float = 1.5, seed: int = 2025, params: dict | None = None,
                 n_estimators: int = 3000, holdout: float = 0.1):
        self.factors, self.power, self.seed, self.n_estimators, self.holdout = list(factors), power, seed, n_estimators, holdout
        self.params = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 300, "subsample": 0.8,
                       "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1, **(params or {})}
        self.patience = 300

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.factors].copy()
        for c, cats in self.cats_.items():
            known = set(cats)
            s = X[c].astype("object").map(lambda v: str(v) if pd.notna(v) and str(v) in known else None)
            X[c] = pd.Categorical(s, categories=cats)  # unseen / missing -> NaN
        return X

    def fit(self, df: pd.DataFrame, target: str, exposure: str = "exposure", loading_target: str | None = None,
            group: str = "policy_id", recency: float | None = None, ref_year: int | None = None):
        import lightgbm as lgb
        self.cats_ = {}
        for c in self.factors:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                self.cats_[c] = sorted(s.dropna().astype(str).unique())
        X = self._X(df)
        w = df[exposure].to_numpy()
        y = df[target].to_numpy() / w
        w = fit_weights(df, exposure, recency, ref_year)   # level is restored by the balance step below
        rng = np.random.default_rng(self.seed)
        groups = df[group].unique()
        hold = df[group].isin(rng.choice(groups, int(len(groups) * self.holdout), replace=False)).to_numpy()
        self.model_ = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=self.power,
                                        n_estimators=self.n_estimators, random_state=self.seed, **self.params)
        self.model_.fit(X[~hold], y[~hold], sample_weight=w[~hold], eval_X=(X[hold],), eval_y=(y[hold],),
                        eval_sample_weight=[w[hold]], callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        # balance correction: early-stopped boosting does not reproduce the training total; rescale so that
        # predicted = actual on all training rows (level matters as much as ranking for D2)
        self.balance_ = 1.0
        self.balance_ = df[target].sum() / self._raw(df, exposure).sum()
        self.loading_ = 1.0 if loading_target is None else df[loading_target].sum() / df[target].sum()
        return self

    def _raw(self, df: pd.DataFrame, exposure: str) -> np.ndarray:
        return self.model_.predict(self._X(df), num_iteration=self.model_.best_iteration_) * df[exposure].to_numpy() * self.balance_

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self._raw(df, exposure) * self.loading_
