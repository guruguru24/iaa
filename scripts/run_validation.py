"""Temporal validation: train 2021-23 (on-levelled to 2024 at the central trend) -> validate 2024 (developed, uncapped).

Models: const, GLM uncapped, GLM per cap (+ large-loss loading); with --gbm also LightGBM at the interim cap and a
50/50 GLM+GBM blend. --cap-grid tries finer caps (A3). --sensitivity adds motor with CL30973963 at 1.19M (decision #6).
POOLED rows combine all lines run in the same call (the submission metric). Results: outputs/reports/w1_validation_<time>.csv

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\run_validation.py --cap-grid --sensitivity          # GLM caps + loading check
    .venv\\Scripts\\python scripts\\run_validation.py --gbm                             # GLM vs GBM vs blend, pooled
    .venv\\Scripts\\python scripts\\run_validation.py --gbm --lines contents            # quick single line
"""
import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, apply_claim_overrides, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.features import feature_columns
from src.models import ConstantModel, LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

GRID = {"motor": (50_000, 100_000, 150_000, 250_000), "buildings": (50_000, 100_000, 150_000, 250_000),
        "contents": (10_000, 15_000, 25_000, 50_000)}
TRAIN, VALID = [2021, 2022, 2023], [2024]

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--weather", default=None, choices=["same_year", "long_run"])
ap.add_argument("--cap-grid", action="store_true")
ap.add_argument("--gbm", action="store_true")
ap.add_argument("--sensitivity", action="store_true")
ap.add_argument("--no-glm-grid", action="store_true", help="skip uncapped GLM (faster GBM runs)")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
weather = a.weather or asm["weather_mode"]
claims0 = load_claims("w1")
refs = reference_tables("w1")
results, preds = [], {}

for line in a.lines:
    g = asm["trend_2025"][line]["central"]
    cap0 = asm["large_loss_cap"][line]
    caps = sorted(set(GRID[line]) | {cap0}) if a.cap_grid else [cap0]
    variants = {"empirical": claims0}
    if a.sensitivity and line == "motor":
        variants["CL30973963_at_1.19M"] = apply_claim_overrides(claims0, asm["large_loss_sensitivity"])
    for vname, claims in variants.items():
        t0 = time.time()
        df = train_frame(line, "w1", caps=caps, claims=claims, refs=refs, weather_mode=weather,
                         weather_years=TRAIN if weather == "long_run" else None)
        factors = feature_columns(df)
        tr, va = temporal_split(df, TRAIN, VALID)
        del df
        targets = ["target_dev"] + [f"target_dev_cap{c}" for c in caps]
        tr[targets] = tr[targets].mul(onlevel_factor(tr["policy_year"], 2024, g), axis=0)
        y = va["target_dev"].to_numpy()
        models = {"const": ConstantModel().fit(tr, "target_dev")}
        if not a.no_glm_grid:
            models["glm_uncapped"] = TweedieGLM(factors).fit(tr, "target_dev")
        for c in caps:
            models[f"glm_cap{c}"] = TweedieGLM(factors).fit(tr, f"target_dev_cap{c}", loading_target="target_dev")
        if a.gbm:
            models["gbm"] = LGBMTweedie(factors, seed=cfg["seed"]).fit(tr, f"target_dev_cap{cap0}", loading_target="target_dev")
        p_all = {n: m.predict(va) for n, m in models.items()}
        p_all["glm_interim"] = p_all[f"glm_cap{cap0}"]  # same name across lines so POOLED uses each line's interim cap
        if a.gbm:
            p_all["blend_glm_gbm"] = 0.5 * p_all[f"glm_cap{cap0}"] + 0.5 * p_all["gbm"]
        for name, p in p_all.items():
            preds[(line, vname, name)] = (y, p)
            m = models.get(name)
            results.append({"line": line, "variant": vname, "weather": weather, "model": name,
                            "loading": getattr(m, "loading_", np.nan),
                            "gbm_trees": getattr(getattr(m, "model_", None), "best_iteration_", np.nan),
                            "balance": getattr(m, "balance_", getattr(m, "train_balance_", np.nan)),
                            **evaluate(y, p)})
        print(f"[{line} / {vname} / {weather}] {len(factors)} factors, {time.time() - t0:.0f}s", flush=True)
        del tr, va

# pooled: every line's empirical variant, plus the motor sensitivity variant combined with the others' empirical
if len(a.lines) > 1:
    models_common = set.intersection(*[{k[2] for k in preds if k[0] == l} for l in a.lines])
    motor_variants = sorted({k[1] for k in preds if k[0] == "motor"}) or ["empirical"]
    for mv in motor_variants:
        for name in models_common:
            keys = [(l, mv if l == "motor" else "empirical", name) for l in a.lines]
            if all(k in preds for k in keys):
                y = np.concatenate([preds[k][0] for k in keys]); p = np.concatenate([preds[k][1] for k in keys])
                results.append({"line": "POOLED", "variant": mv, "weather": weather, "model": name, **evaluate(y, p)})

res = pd.DataFrame(results)
out = REPO / "outputs" / "reports" / f"w1_validation_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 220)
print(res[["line", "variant", "model", "d2", "ae", "gini", "loading", "balance", "gbm_trees"]].round(4).to_string(index=False))
print(f"saved {out.relative_to(REPO)}")
