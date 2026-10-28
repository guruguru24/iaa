"""Row-level cleaning rules applied identically to train and test (gate G2 approved 2026-10-22, decision #4).

Each rule keeps the raw value in `<col>_raw` and adds a `<col>_fixed` flag so its impact stays auditable.
Reference statistics (e.g. sum_insured median) come from TRAIN only and are passed to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import exposure_from_dates, flag_exposure

LEAKAGE_COLS = ["renewal_decision"]  # outcome after the period, train only (DQ14)

# construction_type: official levels are brick, stone, timber, non-standard. 'T' appears only in 2021-22 and 'W'
# only in 2023-24 with matching volumes -> timber relabelled (Timber -> Wood). Approved G2, decision #4 (DQ11).
CONSTRUCTION_MAP = {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "timber": "timber",
                    "w": "timber", "t": "timber",
                    # emerging codes (data_description Part C2; decision #26): MIX = mixed or non-standard
                    "brk": "brick", "stn": "stone", "wod": "timber", "mix": "non_standard"}

# Occupation levels in the developed market; emerging free text maps onto these (decision #26)
DEV_OCCUPATIONS = ["Accountant", "Administrator", "Builder", "Care worker", "Chef", "Doctor", "Driver", "Electrician",
                   "Engineer", "Hairdresser", "Manager", "Mechanic", "Nurse", "Retail assistant", "Retired",
                   "Sales representative", "Software developer", "Student", "Teacher", "Unemployed"]


def fix_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """DQ7: replace supplied exposure by (days+1)/365 where they disagree."""
    out = df.copy()
    out["exposure_raw"] = out["exposure"]
    out["exposure_fixed"] = flag_exposure(out).ne("ok")
    out["exposure"] = exposure_from_dates(out["period_start"], out["period_end"])
    return out


def sum_insured_reference(train: pd.DataFrame) -> float:
    return float(train["sum_insured"].median())


def fix_sum_insured(df: pd.DataFrame, median: float, ratio: float = 50.0, factor: float = 1000.0) -> pd.DataFrame:
    """DQ10: values above `ratio` x train median are treated as entered `factor` times too large."""
    out = df.copy()
    out["sum_insured_raw"] = out["sum_insured"]
    out["sum_insured_fixed"] = out["sum_insured"] > ratio * median
    out.loc[out["sum_insured_fixed"], "sum_insured"] = out.loc[out["sum_insured_fixed"], "sum_insured"] / factor
    return out


def fix_construction_type(df: pd.DataFrame, mapping: dict = CONSTRUCTION_MAP) -> pd.DataFrame:
    """DQ11: normalise case/abbreviations; unmapped values become 'other'."""
    out = df.copy()
    out["construction_type_raw"] = out["construction_type"]
    key = out["construction_type"].str.strip().str.lower()
    out["construction_type"] = key.map(mapping).fillna("other").where(key.notna())
    return out


def clean_occupation(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Free-text occupation -> (developed occupation level, self_employed flag).

    Rule (emerging data): strip spaces, ignore case, a leading 'Self-employed ' becomes the flag; anything not in the
    developed list -> 'other'. Missing stays missing (flag missing too).
    """
    norm = s.astype("object").where(s.notna()).map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    se = norm.map(lambda v: v.startswith("self-employed ") if isinstance(v, str) else None)
    base = norm.map(lambda v: v[len("self-employed "):] if isinstance(v, str) and v.startswith("self-employed ") else v)
    lookup = {o.lower(): o for o in DEV_OCCUPATIONS}
    occ = base.map(lambda v: lookup.get(v, "other") if isinstance(v, str) else None)
    return occ, se


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Normalise occupation; add `self_employed` only when the data has self-employed entries (developed has none,
    so Week 1 inputs stay byte-identical)."""
    if "occupation" not in customers:
        return customers
    out = customers.copy()
    occ, se = clean_occupation(out["occupation"])
    out["occupation_raw"] = out["occupation"]
    out["occupation"] = occ
    if se.fillna(False).astype(bool).any():
        out["self_employed"] = se
    if out["occupation"].equals(customers["occupation"]):
        out = out.drop(columns="occupation_raw")
    return out


def preprocess(df: pd.DataFrame, line: str, ref: dict) -> pd.DataFrame:
    """Apply all rules for one line. `ref` holds train-derived statistics, e.g. {'sum_insured_median': ...}."""
    out = fix_exposure(df).drop(columns=LEAKAGE_COLS, errors="ignore")
    if line in ("buildings", "contents"):
        out = fix_sum_insured(out, ref["sum_insured_median"])
    if line == "buildings":
        out = fix_construction_type(out)
    return out
