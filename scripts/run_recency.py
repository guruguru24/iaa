"""Recency weighting test: fit weight = exposure * decay ** (valid_year - policy_year), decay in {1.0, 0.8, 0.6, 0.4}.

Folds 2021-22 -> 2023 and 2021-23 -> 2024; GLM, GBM and 50/50 blend. Also A/E of the segments found drifting in
scripts/run_diagnostics.py (contents credit_band E, region R07/R08, dense_urban; buildings region R08).
Writes outputs/reports/w1_recency_<time>.csv. Never uses the leaderboard.

Usage (from the repo root; ~10 min per line):
    .venv\\Scripts\\python scripts\\run_recency.py
    .venv\\Scripts\\python scripts\\run_recency.py --lines contents buildings
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
from src.models import LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
DECAYS = [1.0, 0.8, 0.6, 0.4]
WATCH = {"contents": [("credit_band", "E"), ("region", "R07"), ("region", "R08"), ("urban_density", "dense_urban")],
         "buildings": [("region", "R08"), ("region", "R07")],
         "motor": [("cover_type", "third_party_fire_theft")]}

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
rows = []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"])
    factors = feature_columns(df)
    capped = f"target_dev_cap{cap}"
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        cols = ["target_dev", capped]
        tr[cols] = tr[cols].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        y = va["target_dev"].to_numpy()
        for decay in DECAYS:
            kw = {"loading_target": "target_dev", "recency": decay, "ref_year": vy}
            p_glm = TweedieGLM(factors).fit(tr, capped, **kw).predict(va)
            p_gbm = LGBMTweedie(factors, seed=cfg["seed"]).fit(tr, capped, **kw).predict(va)
            for name, p in (("glm", p_glm), ("gbm", p_gbm), ("blend", 0.5 * p_glm + 0.5 * p_gbm)):
                r = {"line": line, "valid_year": vy, "decay": decay, "model": name, **evaluate(y, p)}
                for f, lv in WATCH[line]:
                    m = (va[f].astype(str) == lv).to_numpy()
                    r[f"ae_{f}={lv}"] = y[m].sum() / p[m].sum() if m.any() else float("nan")
                rows.append(r)
        print(f"[{line} -> {vy}] done ({time.time() - t0:.0f}s)", flush=True)
    del df

res = pd.DataFrame(rows)
out = REPO / "outputs" / "reports" / f"w1_recency_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print("\n=== D2 by decay (1.0 = current equal weights)")
print(res.pivot_table(index=["line", "model", "valid_year"], columns="decay", values="d2").round(4).to_string())
print("\n=== A/E by decay (blend)")
print(res[res.model == "blend"].pivot_table(index=["line", "valid_year"], columns="decay", values="ae").round(3).to_string())
watch_cols = [c for c in res.columns if c.startswith("ae_")]
print("\n=== Watched segments A/E (blend)")
print(res[res.model == "blend"].set_index(["line", "valid_year", "decay"])[watch_cols].dropna(how="all", axis=1).round(3).to_string())
print(f"saved {out.relative_to(REPO)}")
