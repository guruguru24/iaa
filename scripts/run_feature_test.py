"""Prior-claims test v2 on two temporal folds (2021-22 -> 2023, 2021-23 -> 2024).

Variants (50/50 GLM+GBM blend unless noted):
  base          current features
  history       + fixed-window history features (NaN when the window is not observable; src/history.py v2)
  base+modifier base blend x experience modifier by claims in the last year (bonus-malus, src/models.ExperienceModifier)
  history+premium (--premium) as history + written_premium (rejected in run 0844; kept for reference only)

Pre-registered rule: adopt only if blend D2 gain >= +0.001 in BOTH folds and overall A/E stays within 0.95-1.05.

Usage (from the repo root; ~10 min per line):
    .venv\\Scripts\\python scripts\\run_feature_test.py
    .venv\\Scripts\\python scripts\\run_feature_test.py --lines contents
"""
import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import ExperienceModifier, LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--premium", action="store_true")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
rows, signal, factors_out = [], [], []


def fit_blend(tr, va, fs, capped):
    glm = TweedieGLM(fs).fit(tr, capped, loading_target="target_dev")
    gbm = LGBMTweedie(fs, seed=cfg["seed"]).fit(tr, capped, loading_target="target_dev")
    p_glm, p_gbm = glm.predict(va), gbm.predict(va)
    return p_glm, p_gbm, 0.5 * p_glm + 0.5 * p_gbm, 0.5 * glm.predict(tr) + 0.5 * gbm.predict(tr)


for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"], history=True)
    base = [f for f in feature_columns(df) if f not in HIST_COLS]
    capped = f"target_dev_cap{cap}"
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        y = va["target_dev"].to_numpy()
        variants = {"base": base, "history": base + HIST_COLS}
        if a.premium:
            variants["history+premium"] = base + HIST_COLS + ["written_premium"]
        preds = {}
        for vname, fs in variants.items():
            p_glm, p_gbm, p_bl, p_tr = fit_blend(tr, va, fs, capped)
            preds[vname] = p_bl
            for m, p in (("glm", p_glm), ("gbm", p_gbm), ("blend", p_bl)):
                rows.append({"line": line, "valid_year": vy, "variant": vname, "model": m, **evaluate(y, p)})
            if vname == "base":
                mod = ExperienceModifier().fit(tr, tr["target_dev"], p_tr)
                preds["base+modifier"] = mod.apply(va, p_bl)
                rows.append({"line": line, "valid_year": vy, "variant": "base+modifier", "model": "blend",
                             **evaluate(y, preds["base+modifier"])})
                factors_out.append({"line": line, "valid_year": vy, **{f"mod_{k}": v for k, v in mod.factors_.items()}})
        b = va["hist_claims_1y"].clip(upper=3).map(lambda v: "unknown" if pd.isna(v) else str(int(v)))
        s = pd.DataFrame({"bucket": b, "y": y, "e": va["exposure"], **{f"p_{k}": v for k, v in preds.items()}})
        gs = s.groupby("bucket").agg(exposure=("e", "sum"), actual=("y", "sum"), **{k: (f"p_{k}", "sum") for k in preds})
        gs["bc_actual"] = gs["actual"] / gs["exposure"]
        for k in preds:
            gs[f"ae_{k}"] = gs["actual"] / gs[k]
        signal.append(gs.reset_index().assign(line=line, valid_year=vy))
        print(f"[{line} -> {vy}] done ({time.time() - t0:.0f}s)", flush=True)
    del df

res = pd.DataFrame(rows)
out = REPO / "outputs" / "reports" / f"w1_feature_test_v2_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print("\n=== D2 by variant")
print(res.pivot_table(index=["line", "model", "valid_year"], columns="variant", values="d2").round(4).to_string())
bl = res[res.model == "blend"].pivot_table(index=["line", "valid_year"], columns="variant", values="d2")
gain = bl.drop(columns="base").sub(bl["base"], axis=0)
print("\n=== Blend D2 gain vs base (rule: >= +0.001 in BOTH folds)")
print(gain.round(4).to_string())
print("pass:"); print((gain >= 0.001).groupby(level="line").all().to_string())
print("\n=== Blend A/E by variant")
print(res[res.model == "blend"].pivot_table(index=["line", "valid_year"], columns="variant", values="ae").round(3).to_string())
print("\n=== Experience modifier factors (claims in last year: 0,1,2,3+)")
print(pd.DataFrame(factors_out).round(3).to_string(index=False))
S = pd.concat(signal)
print("\n=== A/E by claims in the last year (unknown = window not observable)")
print(S[["line", "valid_year", "bucket", "exposure", "bc_actual"] + [c for c in S.columns if c.startswith("ae_")]].round(3).to_string(index=False))
print(f"saved {out.relative_to(REPO)}")
