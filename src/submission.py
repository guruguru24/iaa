"""Build and check the submission file (pre-submission gate, doc 03 §4)."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty list = pass."""
    problems = []
    if list(sub.columns) != ["row_id", "prediction"]:
        problems.append(f"header must be row_id,prediction; got {list(sub.columns)}")
    if sub["row_id"].duplicated().any():
        problems.append("duplicate row_id")
    missing = set(sample["row_id"]) - set(sub["row_id"])
    extra = set(sub["row_id"]) - set(sample["row_id"])
    if missing:
        problems.append(f"{len(missing)} row_ids missing")
    if extra:
        problems.append(f"{len(extra)} unexpected row_ids")
    pred = sub["prediction"]
    if pred.isna().any() or not np.isfinite(pred).all():
        problems.append("NaN/inf predictions")
    if (pred < 0).any():
        problems.append("negative predictions")
    return problems


def write_submission(sub: pd.DataFrame, sample: pd.DataFrame, market: str, out_dir="outputs/submissions") -> Path:
    problems = check_submission(sub, sample)
    if problems:
        raise ValueError("Submission failed checks: " + "; ".join(problems))
    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        githash = "nogit"
    path = Path(out_dir) / f"{market}_{datetime.now():%Y%m%d_%H%M}_{githash}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"Wrote {path} sha256={sha} - log it in logs/decision_log.md")
    return path
