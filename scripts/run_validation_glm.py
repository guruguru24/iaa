"""Temporal validation, rungs 0 and 2: train 2021-23 (on-levelled to 2024 at the central trend) -> validate 2024.

Actual for 2024 = developed (central) uncapped target. Compares constant vs Tweedie GLM (uncapped and two caps with
large-loss loading), for both weather variants (A8). Writes outputs/reports/w1_validation_glm.csv.

Usage (from the repo root; several minutes):
    .venv\\Scripts\\python scripts\\run_validation_glm.py
    .venv\\Scripts\\python scripts\\run_validation_glm.py --weather same_year --lines contents
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.features import feature_columns
from src.metrics import tweedie_d2
from src.models import ConstantModel, TweedieGLM
from src.validation import evaluate, temporal_split

CAPS = {"motor": (100_000, 250_000), "buildings": (100_000, 250_000), "contents": (25_000, 50_000)}
TRAIN, VALID = [2021, 2022, 2023], [2024]

ap = argparse.ArgumentParser()
ap.add_argument("--weather", default="both", choices=["both", "same_year", "long_run"])
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()
weathers = ["same_year", "long_run"] if a.weather == "both" else [a.weather]

cfg = load_config()
trend = cfg["assumptions"]["trend_2025"]
claims = load_claims("w1")
refs = reference_tables("w1")
results, preds = [], {}

for line in a.lines:
    g = trend[line]["central"]
    for wmode in weathers:
        t0 = time.time()
        df = train_frame(line, "w1", caps=CAPS[line], claims=claims, refs=refs, weather_mode=wmode,
                         weather_years=TRAIN if wmode == "long_run" else None)
        factors = feature_columns(df)
        tr, va = temporal_split(df, TRAIN, VALID)
        del df
        targets = ["target_dev"] + [f"target_dev_cap{c}" for c in CAPS[line]]
        tr[targets] = tr[targets].mul(onlevel_factor(tr["policy_year"], 2024, g), axis=0)
        y = va["target_dev"].to_numpy()
        models = {"const": ConstantModel().fit(tr, "target_dev"),
                  "glm_uncapped": TweedieGLM(factors).fit(tr, "target_dev")}
        for i, c in enumerate(CAPS[line]):
            models[f"glm_cap_{'low' if i == 0 else 'high'}"] = TweedieGLM(factors).fit(tr, f"target_dev_cap{c}", loading_target="target_dev")
        for name, m in models.items():
            p = m.predict(va)
            preds[(wmode, name, line)] = (y, p)
            results.append({"line": line, "weather": wmode, "model": name, "n_valid": len(va), "n_factors": len(factors),
                            "loading": getattr(m, "loading_", np.nan), **evaluate(y, p)})
        print(f"[{line} / {wmode}] {len(factors)} factors, {time.time() - t0:.0f}s")
        del tr, va

res = pd.DataFrame(results)
pooled = []
for (wmode, name) in {(k[0], k[1]) for k in preds}:
    ks = [k for k in preds if k[0] == wmode and k[1] == name]
    if len(ks) == len(a.lines):
        y = np.concatenate([preds[k][0] for k in ks]); p = np.concatenate([preds[k][1] for k in ks])
        pooled.append({"line": "POOLED", "weather": wmode, "model": name, "n_valid": len(y), **evaluate(y, p)})
res = pd.concat([res, pd.DataFrame(pooled)], ignore_index=True)
res.to_csv(REPO / "outputs" / "reports" / "w1_validation_glm.csv", index=False)
pd.set_option("display.width", 220)
print(res[["line", "weather", "model", "d2", "ae", "gini", "loading"]].sort_values(["line", "weather", "model"]).round(4).to_string(index=False))
print(f"caps: {CAPS}")
