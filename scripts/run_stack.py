"""Round-2 regularisation grid around c3 + stage-3 cross-fitted stacking (decision #17, pre-registered).

Base models per fold (2021-22 -> 2023, 2021-23 -> 2024): GLM; Tweedie GBM configs c0 (current), c3, c7-c10;
FreqSevGBM with c3-like frequency trees. Stack: simplex weights fitted on one fold, scored on the other (cross-fit).
Rules (pre-registered): a single config or the cross-fitted stack replaces the current blend (GLM + c0, 50/50) only if
its D2 gain is >= +0.001 in BOTH folds; stack weights for the final model are then refitted on the 2024 fold.

Usage (from the repo root; ~20-30 min per line):
    .venv\\Scripts\\python scripts\\run_stack.py
    .venv\\Scripts\\python scripts\\run_stack.py --lines contents --no-freqsev
"""
import argparse
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.development import development_factors
from src.eda import claim_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import FreqSevGBM, LGBMTweedie, TweedieGLM
from src.stacking import apply_weights, simplex_weights
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
GBM = {
    "c0": dict(stop="policy", params={}),
    "c3": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 1000}),
    "c7": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 2000}),
    "c8": dict(stop="time", params={"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 3000, "colsample_bytree": 0.6}),
    "c9": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 2000, "reg_lambda": 10.0}),
    "c10": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 1000}, power=1.3),
}
FREQ_PARAMS = {"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 1000}

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--no-freqsev", action="store_true")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
dev = development_factors(claims, asm["development"]["mature_years"])
rows, weights_out = [], []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"])
    factors = [f for f in feature_columns(df) if f not in HIST_COLS]
    capped = f"target_dev_cap{cap}"
    cf = claim_frame(line, df, claims)
    cc = cf.loc[cf["in_period"] & ~cf["declined"] & (cf["net"] > 0), ["row_id", "peril", "net", "policy_year"]]
    cnt = dev.loc[dev["line"] == line, ["peril", "policy_year", "count_dev_factor"]]
    fold_preds, fold_y = {}, {}
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        p = {"glm": TweedieGLM(factors).fit(tr, capped, loading_target="target_dev").predict(va)}
        for name, spec in GBM.items():
            m = LGBMTweedie(factors, seed=cfg["seed"], params=spec["params"], stop=spec["stop"], power=spec.get("power", 1.5),
                            n_estimators=6000)
            p[name] = m.fit(tr, capped, loading_target="target_dev").predict(va)
            print(f"[{line} -> {vy}] {name}: trees {m.n_trees_} ({time.time() - t0:.0f}s)", flush=True)
        if not a.no_freqsev:
            fs = FreqSevGBM(factors, cap=cap, trend=g, to_year=vy, seed=cfg["seed"], params_freq=FREQ_PARAMS)
            p["freqsev"] = fs.fit(tr, cc[cc["row_id"].isin(tr["row_id"])], capped, cnt, loading_target="target_dev").predict(va)
        p["current_blend"] = 0.5 * p["glm"] + 0.5 * p["c0"]
        for name in [n for n in GBM if n != "c0"]:
            p[f"blend_glm_{name}"] = 0.5 * p["glm"] + 0.5 * p[name]
        fold_preds[vy], fold_y[vy] = p, va["target_dev"].to_numpy()
        print(f"[{line} -> {vy}] base models done ({time.time() - t0:.0f}s)", flush=True)
    del df, cf

    bases = ["glm"] + list(GBM) + ([] if a.no_freqsev else ["freqsev"])
    for fit_y, score_y in ((2023, 2024), (2024, 2023)):
        w = simplex_weights(fold_y[fit_y], {b: fold_preds[fit_y][b] for b in bases})
        weights_out.append({"line": line, "fitted_on": fit_y, **w})
        fold_preds[score_y]["stack_crossfit"] = apply_weights({b: fold_preds[score_y][b] for b in bases}, w)
    for vy in (2023, 2024):
        for name, pred in fold_preds[vy].items():
            rows.append({"line": line, "valid_year": vy, "model": name, **evaluate(fold_y[vy], pred)})

res = pd.DataFrame(rows)
stamp = f"{datetime.now():%Y%m%d_%H%M}"
res.to_csv(REPO / "outputs" / "reports" / f"w1_stack_{stamp}.csv", index=False)
pd.DataFrame(weights_out).to_csv(REPO / "outputs" / "reports" / f"w1_stack_weights_{stamp}.csv", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
piv = res.pivot_table(index=["line", "valid_year"], columns="model", values="d2")
print("\n=== D2 by model")
print(piv.round(4).to_string())
gain = piv.drop(columns="current_blend").sub(piv["current_blend"], axis=0)
print("\n=== D2 gain vs current blend (GLM + c0), rule >= +0.001 in BOTH folds")
print(gain.round(4).to_string())
ok = (gain >= 0.001).groupby(level="line").all()
print("pass (models passing per line):")
for line, r in ok.iterrows():
    print(f"  {line}: {[m for m, v in r.items() if v] or 'none'}")
print("\n=== Cross-fit stack weights (fitted on one fold, scored on the other)")
print(pd.DataFrame(weights_out).round(3).to_string(index=False))
print("\n=== A/E (stack_crossfit, current_blend)")
print(res[res.model.isin(["stack_crossfit", "current_blend"])].pivot_table(index=["line", "valid_year"], columns="model", values="ae").round(3).to_string())
print(f"saved outputs/reports/w1_stack_{stamp}.csv")
