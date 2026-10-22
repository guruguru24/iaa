"""Stage 1 of decision #14: GBM early-stopping mode and a small hyper-parameter grid on two temporal folds.

Configs (all Tweedie, capped target + loading, training-total balance):
  c0  current: random-policy early stopping, lr .05, leaves 31, min_child 300
  c1  time stop (last training year steers tree count, then refit on all years), same params
  c2  time, lr .03, leaves 63, min_child 200, colsample .7
  c3  time, lr .03, leaves 15, min_child 1000
  c4  time, lr .03, leaves 31, min_child 300, reg_lambda 10
  c5  time, as c1 with tweedie power 1.3      c6  time, as c1 with power 1.7
Reports GBM alone and 50/50 blend with the GLM (same features). Pre-registered rule: a config replaces c0 only if its
D2 gain vs c0 is >= +0.001 in BOTH folds (for the metric used: GBM alone or blend).

Usage (from the repo root; ~20 min per line, motor longest):
    .venv\\Scripts\\python scripts\\run_gbm_tuning.py
    .venv\\Scripts\\python scripts\\run_gbm_tuning.py --lines contents --configs c0 c1 c2 --history
"""
import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
CONFIGS = {
    "c0": dict(stop="policy", params={}),
    "c1": dict(stop="time", params={}),
    "c2": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 200, "colsample_bytree": 0.7}),
    "c3": dict(stop="time", params={"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 1000}),
    "c4": dict(stop="time", params={"learning_rate": 0.03, "reg_lambda": 10.0}),
    "c5": dict(stop="time", params={}, power=1.3),
    "c6": dict(stop="time", params={}, power=1.7),
}

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
ap.add_argument("--history", action="store_true", help="include prior-claims features (only once adopted)")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
rows = []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"], history=a.history)
    factors = feature_columns(df) if a.history else [f for f in feature_columns(df) if f not in HIST_COLS]
    capped = f"target_dev_cap{cap}"
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        y = va["target_dev"].to_numpy()
        p_glm = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev").predict(va)
        rows.append({"line": line, "valid_year": vy, "config": "glm", "metric": "glm", "trees": 0, **evaluate(y, p_glm)})
        for c in a.configs:
            spec = CONFIGS[c]
            m = LGBMTweedie(factors, seed=cfg["seed"], params=spec["params"], stop=spec["stop"],
                            power=spec.get("power", 1.5), n_estimators=6000)
            p = m.fit(tr, capped, loading_target="target_dev").predict(va)
            for metric, pp in (("gbm", p), ("blend", 0.5 * p_glm + 0.5 * p)):
                rows.append({"line": line, "valid_year": vy, "config": c, "metric": metric, "trees": m.n_trees_,
                             "balance": m.balance_, **evaluate(y, pp)})
            print(f"[{line} -> {vy}] {c}: trees {m.n_trees_}, gbm D2 {rows[-2]['d2']:.4f}, blend D2 {rows[-1]['d2']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    del df

res = pd.DataFrame(rows)
out = REPO / "outputs" / "reports" / f"w1_gbm_tuning_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 220)
for metric in ("gbm", "blend"):
    piv = res[res.metric == metric].pivot_table(index=["line", "valid_year"], columns="config", values="d2")
    print(f"\n=== D2, {metric} (glm alone for reference: column glm)")
    piv = piv.join(res[res.metric == "glm"].set_index(["line", "valid_year"])["d2"].rename("glm"))
    print(piv.round(4).to_string())
    gain = piv[[c for c in a.configs if c != "c0"]].sub(piv["c0"], axis=0) if "c0" in piv else None
    if gain is not None:
        ok = (gain >= 0.001).groupby(level="line").all()
        print(f"--- gain vs c0 ({metric}); pass = >= +0.001 in both folds")
        print(gain.round(4).to_string())
        print(ok.to_string())
print("\n=== trees and A/E (gbm)")
print(res[res.metric == "gbm"].pivot_table(index=["line", "valid_year"], columns="config", values=["trees", "ae"]).round(3).to_string())
print(f"saved {out.relative_to(REPO)}")
