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
                 n_estimators: int = 3000, holdout: float = 0.1, stop: str = "policy", refit_scale: float = 1.1):
        """stop='policy': early stopping on a random 10% of policies; stop='time': on the last training year, then
        refit on all training rows with best_iteration x refit_scale trees (no early stopping)."""
        self.factors, self.power, self.seed, self.n_estimators, self.holdout = list(factors), power, seed, n_estimators, holdout
        self.stop, self.refit_scale = stop, refit_scale
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
        if self.stop == "time":
            hold = (df["policy_year"] == df["policy_year"].max()).to_numpy()
        else:
            rng = np.random.default_rng(self.seed)
            groups = df[group].unique()
            hold = df[group].isin(rng.choice(groups, int(len(groups) * self.holdout), replace=False)).to_numpy()
        self.model_ = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=self.power,
                                        n_estimators=self.n_estimators, random_state=self.seed, **self.params)
        self.model_.fit(X[~hold], y[~hold], sample_weight=w[~hold], eval_X=(X[hold],), eval_y=(y[hold],),
                        eval_sample_weight=[w[hold]], callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        self.n_trees_ = max(int(self.model_.best_iteration_ or self.n_estimators), 1)
        if self.stop == "time":   # the last year steered the tree count; now learn from it too
            self.n_trees_ = max(int(round(self.n_trees_ * self.refit_scale)), 1)
            self.model_ = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=self.power,
                                            n_estimators=self.n_trees_, random_state=self.seed, **self.params)
            self.model_.fit(X, y, sample_weight=w)
        # balance correction: early-stopped boosting does not reproduce the training total; rescale so that
        # predicted = actual on all training rows (level matters as much as ranking for D2)
        self.balance_ = 1.0
        self.balance_ = df[target].sum() / self._raw(df, exposure).sum()
        self.loading_ = 1.0 if loading_target is None else df[loading_target].sum() / df[target].sum()
        return self

    def _raw(self, df: pd.DataFrame, exposure: str) -> np.ndarray:
        return self.model_.predict(self._X(df), num_iteration=self.n_trees_) * df[exposure].to_numpy() * self.balance_

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self._raw(df, exposure) * self.loading_


class ExperienceModifier:
    """Bonus-malus style multiplier on a base prediction by claims in the last year (experience rating).

    Fitted on rows whose look-back window is observable: factor_k = Z_k * (A/E_k / A/E_all) + (1 - Z_k), with
    credibility Z_k = n_k / (n_k + k) on the bucket's rows-with-claim count; then rescaled so the base total on those
    rows is unchanged. Rows with unobservable history (NaN) get 1.0.
    """

    def __init__(self, col: str = "hist_claims_1y", max_bucket: int = 3, k: float = 200.0):
        self.col, self.max_bucket, self.k = col, max_bucket, k

    def _bucket(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].clip(upper=self.max_bucket).fillna(-1).astype(int)

    def fit(self, df: pd.DataFrame, actual, pred):
        b = self._bucket(df).to_numpy()
        actual, pred = np.asarray(actual, float), np.asarray(pred, float)
        obs = b >= 0
        overall = actual[obs].sum() / pred[obs].sum()
        self.factors_ = {}
        for k in sorted(set(b[obs])):
            m = b == k
            n = (actual[m] > 0).sum()
            z = n / (n + self.k)
            self.factors_[k] = z * (actual[m].sum() / pred[m].sum() / overall) + (1 - z)
        f = pd.Series(b[obs]).map(self.factors_).to_numpy()
        scale = pred[obs].sum() / (pred[obs] * f).sum()
        self.factors_ = {k: v * scale for k, v in self.factors_.items()}
        return self

    def apply(self, df: pd.DataFrame, pred) -> np.ndarray:
        f = self._bucket(df).map(self.factors_).fillna(1.0).to_numpy()
        return np.asarray(pred, float) * f
