"""Feature-set test on two temporal folds (2021-22 -> 2023, 2021-23 -> 2024): base vs + prior-claims history
(src/history.py) vs + history + written_premium (G4 governance variant, Part B4).

Pre-registered rule (decision #12 style): adopt a variant only if the 50/50 blend D2 improves by >= 0.001 in BOTH folds
and overall A/E stays within 0.95-1.05. written_premium additionally needs an owner G4 decision (circularity).

Usage (from the repo root; ~10 min per line):
    .venv\\Scripts\\python scripts\\run_feature_test.py
    .venv\\Scripts\\python scripts\\run_feature_test.py --lines contents --no-premium
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
from src.models import LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--no-premium", action="store_true")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
rows, signal = [], []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"], history=True)
    base = [f for f in feature_columns(df) if f not in HIST_COLS]
    variants = {"base": base, "history": base + HIST_COLS}
    if not a.no_premium:
        variants["history+premium"] = base + HIST_COLS + ["written_premium"]
    capped = f"target_dev_cap{cap}"
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        y = va["target_dev"].to_numpy()
        preds = {}
        for vname, fs in variants.items():
            p_glm = TweedieGLM(fs).fit(tr, capped, loading_target="target_dev").predict(va)
            p_gbm = LGBMTweedie(fs, seed=cfg["seed"]).fit(tr, capped, loading_target="target_dev").predict(va)
            preds[vname] = 0.5 * p_glm + 0.5 * p_gbm
            for m, p in (("glm", p_glm), ("gbm", p_gbm), ("blend", preds[vname])):
                rows.append({"line": line, "valid_year": vy, "variant": vname, "model": m, **evaluate(y, p)})
        if vy == 2024:   # signal check: actual cost per exposure and base-model A/E by prior-claims bucket
            b = pd.cut(va["hist_claims"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"]).astype(str)
            b = np.where(va["hist_exposure"] > 0, b, "no_history")
            s = pd.DataFrame({"bucket": b, "y": y, "e": va["exposure"], "p_base": preds["base"], "p_hist": preds["history"]})
            gs = s.groupby("bucket").agg(exposure=("e", "sum"), actual=("y", "sum"), base=("p_base", "sum"), hist=("p_hist", "sum"))
            gs["bc_actual"] = gs["actual"] / gs["exposure"]
            gs["ae_base"] = gs["actual"] / gs["base"]
            gs["ae_history"] = gs["actual"] / gs["hist"]
            signal.append(gs.reset_index().assign(line=line))
        print(f"[{line} -> {vy}] done ({time.time() - t0:.0f}s)", flush=True)
    del df

res = pd.DataFrame(rows)
out = REPO / "outputs" / "reports" / f"w1_feature_test_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 220)
print("\n=== D2 by variant (blend / glm / gbm)")
print(res.pivot_table(index=["line", "model", "valid_year"], columns="variant", values="d2").round(4).to_string())
bl = res[res.model == "blend"].pivot_table(index=["line", "valid_year"], columns="variant", values="d2")
for v in bl.columns.drop("base"):
    bl[f"gain_{v}"] = bl[v] - bl["base"]
print("\n=== Blend D2 gain vs base (rule: >= +0.001 in BOTH folds)")
print(bl.round(4).to_string())
print("\n=== Blend A/E by variant")
print(res[res.model == "blend"].pivot_table(index=["line", "valid_year"], columns="variant", values="ae").round(3).to_string())
print("\n=== 2024 signal: cost per exposure and A/E by prior same-policy claims")
print(pd.concat(signal)[["line", "bucket", "exposure", "bc_actual", "ae_base", "ae_history"]].round(3).to_string(index=False))
print(f"saved {out.relative_to(REPO)}")
