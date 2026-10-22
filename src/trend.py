"""Mix-adjusted year effects (gate G3): GLMs with a policy-year factor plus main rating factors.

The year coefficients estimate how cost moved from year to year for an unchanged risk mix; compare them with the
crude (unadjusted) year ratios to see how much of the movement is mix. Estimation only, not the pricing model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

from src.eda import band

TREND_FACTORS = {
    "motor": ["driver_age", "ncb_years", "vehicle_value_group", "vehicle_age", "cover_type", "annual_mileage",
              "vehicle_use", "overnight_parking", "channel", "voluntary_excess", "fuel_type", "body_type",
              "new_business", "payment_method", "region"],
    "buildings": ["sum_insured", "construction_type", "occupancy", "policyholder_age", "voluntary_excess", "channel",
                  "accidental_damage_cover", "new_business", "payment_method", "region", "flood_zone", "property_type"],
    "contents": ["sum_insured", "tenure", "security_level", "high_value_items", "accidental_damage_cover",
                 "policyholder_age", "voluntary_excess", "channel", "new_business", "payment_method", "region"],
}


def design_matrix(df: pd.DataFrame, factors, year_col: str = "policy_year", base_year: int = 2021, n_bins: int = 5):
    """Sparse one-hot matrix: year dummies (base year dropped) + banded factors (most common level dropped)."""
    blocks, names = [], []
    years = sorted(y for y in df[year_col].unique() if y != base_year)
    for y in years:
        blocks.append(sp.csr_matrix((df[year_col].to_numpy() == y).astype(float)).T)
        names.append(f"year={y}")
    for f in factors:
        lv = band(df[f], n_bins)
        dummies = pd.get_dummies(lv, prefix=f, sparse=True, dtype=float)
        dummies = dummies.drop(columns=f"{f}_{lv.value_counts().idxmax()}")
        blocks.append(sp.csr_matrix(dummies.sparse.to_coo()))
        names += list(dummies.columns)
    return sp.hstack(blocks).tocsr(), names


def _year_effects(model, names) -> pd.Series:
    coef = pd.Series(model.coef_, index=names)
    return np.exp(coef[[n for n in names if n.startswith("year=")]]).rename(lambda n: int(n.split("=")[1]))


def year_effects(df: pd.DataFrame, factors, y: str, weight: str | None, kind: str,
                 base_year: int = 2021, alpha: float = 1e-6) -> pd.DataFrame:
    """Crude and mix-adjusted year effects relative to base_year. kind in {'poisson','gamma','tweedie'}."""
    model = {"poisson": PoissonRegressor, "gamma": GammaRegressor}.get(kind)
    model = model(alpha=alpha, max_iter=1000) if model else TweedieRegressor(power=1.5, link="log", alpha=alpha, max_iter=1000)
    w = None if weight is None else df[weight].to_numpy()
    X, names = design_matrix(df, factors, base_year=base_year)
    model.fit(X, df[y].to_numpy(), sample_weight=w)
    adj = _year_effects(model, names)
    ww = np.ones(len(df)) if w is None else w
    crude = (df.assign(_yw=df[y] * ww, _w=ww).groupby("policy_year")[["_yw", "_w"]].sum()
             .pipe(lambda g: g["_yw"] / g["_w"]))
    crude = crude / crude[base_year]
    return pd.DataFrame({"crude": crude, "mix_adjusted": adj}).fillna({"mix_adjusted": 1.0}).rename_axis("policy_year").reset_index()


def annual_trend(effects: pd.Series, years) -> float:
    """Log-linear annual rate fitted through the given years' effects."""
    e = effects.loc[list(years)]
    return float(np.exp(np.polyfit(np.asarray(years, float), np.log(e.to_numpy()), 1)[0]) - 1)


def row_capped_target(rows: pd.DataFrame, claim_frame: pd.DataFrame, cap: float) -> pd.Series:
    """Row target with each claim's net capped at `cap` (large-loss treatment, decision #6)."""
    c = claim_frame[claim_frame["in_period"] & ~claim_frame["declined"]]
    s = c.assign(capped=c["net"].clip(upper=cap)).groupby("row_id")["capped"].sum()
    return rows["row_id"].map(s).fillna(0.0)
