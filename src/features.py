"""Model features: joins (customers, addresses, weather), derived fields, exclusions and a train-fitted encoder.

Rules (docs/02 §3, data_description Part B4/B5, decisions #4/#5):
- Only information known at inception. Excluded: renewal_decision (outcome), customer_service_contacts (may contain
  in-period claim contacts), written_premium / discount_applied (insurer's own price; benchmark only, G4 variant).
- Fairness / low-rationale fields excluded by default: surname, email_domain, marketing_segment, campaign_code,
  branch_id, vehicle_colour, make_model (300 levels; GBM stage), date_of_birth (age fields already exist).
- Weather: 'same_year' joins the row's own year; 'long_run' uses the address mean over `weather_years` only.
- Bands and category levels are fitted on training rows only (BandEncoder); unseen test levels map to the base level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

ID_COLS = ["row_id", "policy_id", "customer_id", "address_id"]
ALWAYS_EXCLUDE = ["renewal_decision", "customer_service_contacts", "written_premium", "discount_applied", "occupation_raw",
                  "period_start", "period_end", "line", "exposure", "exposure_raw", "exposure_fixed",
                  "sum_insured_raw", "sum_insured_fixed", "construction_type_raw", "compulsory_excess"]
LOW_RATIONALE = ["surname", "email_domain", "marketing_segment", "campaign_code", "branch_id", "vehicle_colour",
                 "make_model", "date_of_birth", "document_preference", "marketing_consent"]
WEATHER_COLS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index",
                "rainfall_mm", "mean_temperature_c"]


def weather_features(weather: pd.DataFrame, mode: str, weather_years=None) -> pd.DataFrame:
    """Weather keyed by (address_id, policy_year) for 'same_year', or by address_id for 'long_run'."""
    if mode == "same_year":
        return weather.rename(columns={"year": "policy_year"})[["address_id", "policy_year"] + WEATHER_COLS]
    if mode == "long_run":
        w = weather if weather_years is None else weather[weather["year"].isin(weather_years)]
        return w.groupby("address_id")[WEATHER_COLS].mean().add_suffix("_lr").reset_index()
    raise ValueError(f"unknown weather mode {mode!r}")


def add_features(df: pd.DataFrame, line: str, addresses: pd.DataFrame, customers: pd.DataFrame,
                 weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join reference tables and derive inception-time features. `weather` comes from weather_features()."""
    out = df.copy()
    if "policy_year" not in out:
        out["policy_year"] = out["period_start"].dt.year
    out = out.merge(addresses, on="address_id", how="left")
    cust = customers.drop(columns=[c for c in LOW_RATIONALE if c in customers.columns])
    out = out.merge(cust, on="customer_id", how="left", indicator="_cust")
    out["customer_missing"] = out.pop("_cust").eq("left_only")
    if weather is not None:
        keys = ["address_id", "policy_year"] if "policy_year" in weather else ["address_id"]
        out = out.merge(weather, on=keys, how="left")
    out["customer_tenure"] = out["policy_year"] - out["customer_since_year"]
    out["property_age"] = out["policy_year"] - out["year_built"]
    out["excess_total"] = out["voluntary_excess"] + out["compulsory_excess"]
    if "sum_insured" in out:
        out["si_per_bedroom"] = out["sum_insured"] / out["bedrooms"].clip(lower=1)
    if line == "motor" and "telematics_score" in out:          # emerging motor has no telematics (Part C2)
        out["telematics_missing"] = out["telematics_score"].isna()
    if line == "buildings":
        out["timber_post2023"] = out["construction_type"].eq("timber") & (out["policy_year"] >= 2023)  # decision #5
    return out


def feature_columns(df: pd.DataFrame, extra_exclude=()) -> list[str]:
    drop = set(ID_COLS + ALWAYS_EXCLUDE + LOW_RATIONALE + ["policy_year", "customer_since_year", "year_built"]) | set(extra_exclude)
    target_like = {c for c in df.columns if c.startswith(("target", "n_claims", "open_net", "amount_before", "exposure_", "hist_all_"))}
    return [c for c in df.columns if c not in drop and c not in target_like]


class BandEncoder:
    """One-hot encoder fitted on training rows: numeric -> quantile bands (+ missing), categorical -> levels.

    The most common level of each factor is the base (dropped). Unseen levels at transform time -> base.
    """

    def __init__(self, n_bins: int = 10, max_levels: int = 50):
        self.n_bins, self.max_levels = n_bins, max_levels

    def _levels(self, s: pd.Series, f: str) -> pd.Series:
        if f in self.edges_:
            b = pd.cut(s.astype(float), self.edges_[f], include_lowest=True).astype(str)
            return b.where(s.notna(), "missing")
        return s.astype(str).where(s.notna(), "missing")

    def fit(self, df: pd.DataFrame, factors) -> "BandEncoder":
        self.factors_, self.edges_, self.levels_, self.base_ = list(factors), {}, {}, {}
        for f in self.factors_:
            s = df[f]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > self.n_bins:
                q = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, self.n_bins + 1)))
                q[0], q[-1] = -np.inf, np.inf
                self.edges_[f] = q
            lv = self._levels(s, f)
            counts = lv.value_counts()
            keep = list(counts.index[: self.max_levels])
            self.base_[f] = counts.index[0]
            self.levels_[f] = [x for x in keep if x != self.base_[f]]
        self.columns_ = [f"{f}={lv}" for f in self.factors_ for lv in self.levels_[f]]
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        blocks = []
        for f in self.factors_:
            lv = self._levels(df[f], f)
            pos = {level: i for i, level in enumerate(self.levels_[f])}
            idx = lv.map(pos).fillna(-1).astype(int).to_numpy()  # unseen/base -> -1 -> all zeros
            rows = np.flatnonzero(idx >= 0)
            blocks.append(sp.csr_matrix((np.ones(len(rows)), (rows, idx[rows])), shape=(len(df), len(self.levels_[f]))))
        return sp.hstack(blocks).tocsr()


AGE_BANDS = ([-np.inf, 24, 34, 49, 64, np.inf], ["17-24", "25-34", "35-49", "50-64", "65+"])
AGE_BAND_SOURCES = {"driver_age": "driver_age_band", "policyholder_age": "policyholder_age_band"}


def harmonise_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Developed exact ages -> the emerging market's age bands (data_description Part C2). No-op if already banded."""
    out = df.copy()
    for src, dst in AGE_BAND_SOURCES.items():
        if src in out and dst not in out:
            out[dst] = pd.cut(out[src], AGE_BANDS[0], labels=AGE_BANDS[1]).astype("object")
    return out
