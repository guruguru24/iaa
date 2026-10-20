"""Load raw data (read-only) and basic exposure handling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LINES = ("motor", "buildings", "contents")
ROW_PREFIX = {"motor": "M", "buildings": "B", "contents": "C"}
REPO = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))


def raw_dir(market: str = "w1") -> Path:
    return REPO / load_config()["markets"][market]["raw_dir"]


def _read(name: str, market: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(raw_dir(market) / f"{name}.parquet", columns=columns)


def load_policies(line: str, market: str = "w1", kind: str = "train") -> pd.DataFrame:
    """Policy rows for one line. kind='train' -> policies_{line}, kind='test' -> test_{line}."""
    if line not in LINES:
        raise ValueError(f"unknown line {line!r}")
    name = f"policies_{line}" if kind == "train" else f"test_{line}"
    df = _read(name, market)
    for col in ("period_start", "period_end"):
        df[col] = pd.to_datetime(df[col])
    df["line"] = line
    return df


def load_claims(market: str = "w1", line: str | None = None) -> pd.DataFrame:
    df = _read("claims", market)
    for col in ("loss_date", "report_date"):
        df[col] = pd.to_datetime(df[col])
    return df if line is None else df[df["line"] == line].reset_index(drop=True)


def load_table(name: str, market: str = "w1") -> pd.DataFrame:
    """addresses / customers / weather / sample_submission."""
    return _read(name, market)


def exposure_from_dates(period_start: pd.Series, period_end: pd.Series) -> pd.Series:
    """Earned years for an inclusive period [start, end]: (days + 1) / 365.

    Verified on W1 data: matches the supplied exposure on 99.4% of train rows and 100% of test rows.
    """
    return ((period_end - period_start).dt.days + 1) / 365.0


def flag_exposure(df: pd.DataFrame, tol: float = 1e-4) -> pd.Series:
    """Classify supplied exposure vs dates: 'ok', 'over_small' (ratio <= 2, e.g. 1.0139), 'days_not_years' (ratio > 2)."""
    implied = exposure_from_dates(df["period_start"], df["period_end"])
    ratio = df["exposure"] / implied
    out = np.where(np.isclose(df["exposure"], implied, atol=tol), "ok",
                   np.where(ratio > 2, "days_not_years", "over_small"))
    return pd.Series(out, index=df.index, name="exposure_flag")
