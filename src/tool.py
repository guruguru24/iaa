"""Pricing-tool back end: build and load the model bundle, price one policy or a batch, explain with GLM relativities.

    python -m src.tool --market w1        # fits the final models (same code path as src.pipeline) and saves
                                          # outputs/models/{market}_bundle.joblib (git-ignored; rebuildable)

The bundle stores, per line: the fitted GLM, GBM and experience modifier, the feature list, typical values for each
feature (2024 rows: median / most common level) as tool defaults, and version metadata (git hash, config assumptions).
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import reference_tables
from src.pipeline import fit_predict, gbm_spec, prepare_line

BUNDLE = REPO / "outputs" / "models" / "{market}_bundle.joblib"


def typical_values(df: pd.DataFrame, factors) -> dict:
    """Median for numeric factors, most common level otherwise (bool kept as bool)."""
    out = {}
    for f in factors:
        s = df[f].dropna()
        if s.empty:
            out[f] = None
        elif pd.api.types.is_bool_dtype(s):
            out[f] = bool(s.mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(s):
            out[f] = float(s.median())
        else:
            out[f] = str(s.mode().iloc[0])
    return out


def build_bundle(market: str = "w1") -> dict:
    cfg = load_config()
    claims, refs = load_claims(market), reference_tables(market)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO).strip()
    except Exception:
        git = "nogit"
    bundle = {"meta": {"market": market, "built": f"{datetime.now():%Y-%m-%d %H:%M}", "git": git,
                       "assumptions": cfg["assumptions"], "model_by_line": cfg.get("model_by_line"),
                       "experience_modifier": cfg.get("experience_modifier")}, "lines": {}}
    for line in LINES:
        tr, tr_on, te, factors, g, cap = prepare_line(line, cfg, market, claims, refs)
        gname, gspec = gbm_spec(line, cfg)
        p, info, models = fit_predict("blend", tr_on, te, factors, cap, cfg["seed"], gspec, cfg.get("experience_modifier"),
                                      return_models=True)
        recent = tr[tr["policy_year"] == tr["policy_year"].max()]
        bundle["lines"][line] = {"factors": factors, "defaults": typical_values(recent, factors), "models": models,
                                 "gbm_config": gname, "cap": cap, "trend": g, "info": info,
                                 "test_check": {"rows": len(te), "pred_total": float(p.sum())}}
        print(f"[{line}] bundled ({len(factors)} factors, gbm {gname})", flush=True)
        del tr, tr_on, te
    path = BUNDLE.with_name(BUNDLE.name.format(market=market))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"saved {path.relative_to(REPO)}")
    return bundle


def load_bundle(market: str = "w1") -> dict:
    return joblib.load(BUNDLE.with_name(BUNDLE.name.format(market=market)))


def to_frame(line_bundle: dict, rows: pd.DataFrame | dict) -> pd.DataFrame:
    """Rows with all model columns: missing features take the bundle's typical values; exposure defaults to 1."""
    df = pd.DataFrame([rows]) if isinstance(rows, dict) else rows.copy()
    for f, v in line_bundle["defaults"].items():
        if f not in df:
            df[f] = v
        else:
            df[f] = df[f].where(df[f].notna(), v)
    if "exposure" not in df:
        df["exposure"] = 1.0
    if "hist_claims_1y" not in df:
        df["hist_claims_1y"] = 0
    return df


def price(line_bundle: dict, rows: pd.DataFrame | dict) -> pd.DataFrame:
    """GLM, GBM, blend, experience factor and final expected cost for each row (per the row's exposure)."""
    df = to_frame(line_bundle, rows)
    m = line_bundle["models"]
    out = pd.DataFrame({"exposure": df["exposure"].to_numpy()}, index=df.index)
    out["glm"] = m["glm"].predict(df)
    out["gbm"] = m["gbm"].predict(df)
    out["blend"] = 0.5 * out["glm"] + 0.5 * out["gbm"]
    mod = m.get("modifier")
    out["experience_factor"] = mod.apply(df, np.ones(len(df))) if mod is not None else 1.0
    out["expected_cost"] = out["blend"] * out["experience_factor"]
    return out


def glm_breakdown(glm, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Base rate (per unit exposure, incl. loading) and one relativity per factor for a single row.

    base x product(relativities) x exposure = GLM prediction.
    """
    enc = glm.enc_
    coef = pd.Series(glm.glm_.coef_, index=enc.columns_)
    rows = []
    for f in enc.factors_:
        level = enc._levels(row[f], f).iloc[0]
        col = f"{f}={level}"
        rel = float(np.exp(coef[col])) if col in coef.index else 1.0
        rows.append({"factor": f, "value": row[f].iloc[0], "band": level if col in coef.index else f"{level} (base)",
                     "relativity": rel})
    base = float(np.exp(glm.glm_.intercept_)) * glm.loading_
    tab = pd.DataFrame(rows)
    return base, tab.reindex(tab["relativity"].map(lambda r: abs(np.log(r))).sort_values(ascending=False).index)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="w1")
    build_bundle(ap.parse_args(argv).market)


if __name__ == "__main__":
    main()
