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
                ref_year: int | None = None, weight_col: str | None = None) -> np.ndarray:
    """Exposure weights, optionally decayed by age (exposure * recency ** (ref_year - policy_year)) and multiplied
    by a per-row column (e.g. up-weighting emerging rows in a joint fit)."""
    w = df[exposure].to_numpy(dtype=float)
    if weight_col is not None:
        w = w * df[weight_col].to_numpy(dtype=float)
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
            recency: float | None = None, ref_year: int | None = None, encoder: BandEncoder | None = None,
            weight_col: str | None = None):
        """encoder: reuse another fit's bands and base levels (coefficients then compare level by level)."""
        self.enc_ = encoder if encoder is not None else BandEncoder(self.n_bins).fit(df, self.factors)
        X = self.enc_.transform(df)
        w = df[exposure].to_numpy()
        self.glm_ = TweedieRegressor(power=self.power, link="log", alpha=self.alpha, solver="newton-cholesky", max_iter=200)
        self.glm_.fit(X, df[target].to_numpy() / w, sample_weight=fit_weights(df, exposure, recency, ref_year, weight_col))
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
            group: str = "policy_id", recency: float | None = None, ref_year: int | None = None,
            weight_col: str | None = None):
        import lightgbm as lgb
        self.cats_ = {}
        for c in self.factors:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                self.cats_[c] = sorted(s.dropna().astype(str).unique())
        X = self._X(df)
        w = df[exposure].to_numpy()
        y = df[target].to_numpy() / w
        w = fit_weights(df, exposure, recency, ref_year, weight_col)   # level is restored by the balance step below
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


def _fit_cats(df: pd.DataFrame, factors) -> dict:
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in factors
            if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])}


def _cat_X(df: pd.DataFrame, factors, cats: dict) -> pd.DataFrame:
    X = df[list(factors)].copy()
    for c, levels in cats.items():
        known = set(levels)
        X[c] = pd.Categorical(X[c].astype("object").map(lambda v: str(v) if pd.notna(v) and str(v) in known else None),
                              categories=levels)
    return X


def _lgbm_time_stopped(X, y, w, years, objective: str, params: dict, seed: int, patience: int = 300,
                       n_estimators: int = 6000, refit_scale: float = 1.1):
    """Early stopping on the last training year, then refit on all rows with best_iteration x refit_scale trees."""
    import lightgbm as lgb
    hold = years == years.max()
    base = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 300, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1, **params}
    if hold.all() or (~hold).sum() == 0:
        n = 100
    else:
        m = lgb.LGBMRegressor(objective=objective, n_estimators=n_estimators, random_state=seed, **base)
        m.fit(X[~hold], y[~hold], sample_weight=None if w is None else w[~hold], eval_X=(X[hold],), eval_y=(y[hold],),
              eval_sample_weight=None if w is None else [w[hold]], callbacks=[lgb.early_stopping(patience, verbose=False)])
        n = max(int(round((m.best_iteration_ or n_estimators) * refit_scale)), 1)
    m = lgb.LGBMRegressor(objective=objective, n_estimators=n, random_state=seed, **base)
    m.fit(X, y, sample_weight=w)
    return m, n


class FreqSevGBM:
    """Stage 2 (decision #14): per-peril frequency (Poisson GBM) x severity (Gamma GBM on capped claims).

    Frequency target: claims of the peril per row x count development factor (peril, year), per unit exposure.
    Severity target: claim net capped at `cap`, on-levelled to `to_year` at `trend`. Perils with fewer than
    `min_sev_claims` claims use the mean on-levelled capped severity. Prediction = sum over perils of
    freq x exposure x sev, rescaled to the training capped total (balance) and multiplied by the large-loss loading.
    """

    def __init__(self, factors, cap: float, trend: float, to_year: int, seed: int = 2025,
                 params_freq: dict | None = None, params_sev: dict | None = None, min_sev_claims: int = 300):
        self.factors, self.cap, self.trend, self.to_year, self.seed = list(factors), cap, trend, to_year, seed
        self.params_freq, self.params_sev, self.min_sev_claims = params_freq or {}, params_sev or {}, min_sev_claims

    def fit(self, rows: pd.DataFrame, claims_c: pd.DataFrame, capped_target: str, count_factors: pd.DataFrame,
            exposure: str = "exposure", loading_target: str | None = None):
        """rows: training rows (features, policy_year, exposure, on-levelled capped/uncapped targets).
        claims_c: claim-level (row_id, peril, net, policy_year) for in-period, non-declined, net > 0 claims of these rows."""
        self.cats_ = _fit_cats(rows, self.factors)
        X = _cat_X(rows, self.factors, self.cats_)
        e = rows[exposure].to_numpy(dtype=float)
        years = rows["policy_year"].to_numpy()
        c = claims_c[claims_c["row_id"].isin(rows["row_id"])]
        self.perils_ = sorted(c["peril"].unique())
        cf = count_factors.set_index(["peril", "policy_year"])["count_dev_factor"]
        self.freq_, self.sev_, self.trees_ = {}, {}, {}
        for p in self.perils_:
            cp = c[c["peril"] == p]
            n = rows["row_id"].map(cp.groupby("row_id").size()).fillna(0).to_numpy(dtype=float)
            dev = pd.Series(list(zip([p] * len(rows), years))).map(cf).fillna(1.0).to_numpy()
            self.freq_[p], nf = _lgbm_time_stopped(X, n * dev / e, e, years, "poisson", self.params_freq, self.seed)
            sev = cp["net"].clip(upper=self.cap) * (1 + self.trend) ** (self.to_year - cp["policy_year"].astype(int))
            if len(cp) >= self.min_sev_claims:
                Xs = _cat_X(cp[["row_id"]].merge(rows[["row_id"] + self.factors], on="row_id", how="left"), self.factors, self.cats_)
                self.sev_[p], ns = _lgbm_time_stopped(Xs, sev.to_numpy(), None, cp["policy_year"].astype(int).to_numpy(),
                                                      "gamma", self.params_sev, self.seed)
            else:
                self.sev_[p], ns = float(sev.mean()), 0
            self.trees_[p] = (nf, ns)
        self.balance_ = 1.0
        self.balance_ = rows[capped_target].sum() / self._raw(rows, exposure).sum()
        self.loading_ = 1.0 if loading_target is None else rows[loading_target].sum() / rows[capped_target].sum()
        return self

    def peril_predictions(self, df: pd.DataFrame, exposure: str = "exposure") -> pd.DataFrame:
        X = _cat_X(df, self.factors, self.cats_)
        out = {}
        for p in self.perils_:
            f = self.freq_[p].predict(X)
            s = self.sev_[p].predict(X) if not isinstance(self.sev_[p], float) else self.sev_[p]
            out[p] = f * s * df[exposure].to_numpy()
        return pd.DataFrame(out, index=df.index)

    def _raw(self, df, exposure):
        return self.peril_predictions(df, exposure).sum(axis=1).to_numpy() * self.balance_

    def predict(self, df: pd.DataFrame, exposure: str = "exposure") -> np.ndarray:
        return self._raw(df, exposure) * self.loading_
