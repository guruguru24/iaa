"""Row-level cleaning rules applied identically to train and test (gate G2: proposals until approved).

Each rule keeps the raw value in `<col>_raw` and adds a `<col>_fixed` flag so its impact stays auditable.
Reference statistics (e.g. sum_insured median) come from TRAIN only and are passed to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import exposure_from_dates, flag_exposure

LEAKAGE_COLS = ["renewal_decision"]  # outcome after the period, train only (DQ14)

# construction_type: official levels are brick, stone, timber, non-standard. 'w'/'t' are unresolved (DQ11);
# they are kept as their own levels until the check script shows what they are.
CONSTRUCTION_MAP = {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "timber": "timber",
                    "w": "w_unknown", "t": "t_unknown"}


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


def preprocess(df: pd.DataFrame, line: str, ref: dict) -> pd.DataFrame:
    """Apply all rules for one line. `ref` holds train-derived statistics, e.g. {'sum_insured_median': ...}."""
    out = fix_exposure(df).drop(columns=LEAKAGE_COLS, errors="ignore")
    if line in ("buildings", "contents"):
        out = fix_sum_insured(out, ref["sum_insured_median"])
    if line == "buildings":
        out = fix_construction_type(out)
    return out
