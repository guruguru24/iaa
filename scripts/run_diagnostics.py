"""Diagnostics promised in decision #10 + improvement candidates, all on temporal validation (never the leaderboard).

1. Rolling origin: 2021->2022, 2021-22->2023, 2021-23->2024 (train on-levelled to the valid year at the central trend);
   GLM, GBM and blends with GBM weight 0, .25, .5, .75, 1.
2. Main fold (->2024), 50/50 blend: level sensitivity (D2 if all predictions x factor), A/E by predicted decile.
   A/E by segment on every fold; segments off (outside 0.85-1.15, >= 2% exposure) in >= 2 folds are listed.
3. GBM variants on the main fold: current, slower/deeper, + make_model (motor).

Writes outputs/reports/w1_diagnostics_<time>/*.csv and prints a compact summary.

Usage (from the repo root; ~10 min per line):
    .venv\\Scripts\\python scripts\\run_diagnostics.py
    .venv\\Scripts\\python scripts\\run_diagnostics.py --lines contents
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
from src.metrics import lift_table
from src.models import LGBMTweedie, TweedieGLM
from src.validation import evaluate, level_sensitivity, segment_ae, temporal_split

FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEGMENTS = {
    "motor": ["driver_age", "ncb_years", "vehicle_value_group", "cover_type", "region", "channel", "new_business",
              "annual_mileage", "fuel_type", "telematics_missing", "vehicle_age", "overnight_parking", "vehicle_use",
              "body_type", "named_young_driver", "driver_assist", "voluntary_excess", "payment_method", "credit_band"],
    "buildings": ["region", "construction_type", "sum_insured", "flood_zone", "property_type", "occupancy",
                  "property_age", "soil_type", "dry_summer_index", "timber_post2023", "roof_type", "trees_within_5m",
                  "basement", "urban_density", "listed_building", "accidental_damage_cover", "voluntary_excess",
                  "days_unoccupied", "smoke_alarms", "channel", "credit_band"],
    "contents": ["region", "tenure", "sum_insured", "security_level", "high_value_items", "policyholder_age",
                 "urban_density", "accidental_damage_cover", "lodgers", "works_from_home", "bicycle_cover",
                 "voluntary_excess", "property_type", "channel", "credit_band"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--skip-variants", action="store_true", help="skip GBM variants (faster)")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
out_dir = REPO / "outputs" / "reports" / f"w1_diagnostics_{datetime.now():%Y%m%d_%H%M}"
out_dir.mkdir(parents=True, exist_ok=True)
folds, levels, lifts, segs, variants = [], [], [], [], []


def onlevel(tr, cap, to_year, g):
    cols = ["target_dev", f"target_dev_cap{cap}"]
    tr[cols] = tr[cols].mul(onlevel_factor(tr["policy_year"], to_year, g), axis=0)
    return tr


for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"])
    factors = feature_columns(df)
    capped = f"target_dev_cap{cap}"

    # 1. rolling origin
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr = onlevel(tr, cap, vy, g)
        p_glm = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev").predict(va)
        p_gbm = LGBMTweedie(factors, seed=cfg["seed"]).fit(tr, capped, loading_target="target_dev").predict(va)
        y = va["target_dev"].to_numpy()
        for w in WEIGHTS:
            folds.append({"line": line, "valid_year": vy, "w_gbm": w, **evaluate(y, (1 - w) * p_glm + w * p_gbm)})
        vb = va.assign(p_blend=0.5 * p_glm + 0.5 * p_gbm)
        for f in SEGMENTS[line]:
            if f in vb:
                segs.append(segment_ae(vb, "target_dev", "p_blend", f).assign(line=line, valid_year=vy))
        if vy == 2024:
            va = va.assign(p_blend=0.5 * p_glm + 0.5 * p_gbm, p_glm=p_glm, p_gbm=p_gbm)
            main_tr, main_va = tr, va
    print(f"[{line}] rolling folds done ({time.time() - t0:.0f}s)", flush=True)

    # 2. main-fold diagnostics on the 50/50 blend
    y = main_va["target_dev"].to_numpy()
    for name in ("p_glm", "p_gbm", "p_blend"):
        levels.append(level_sensitivity(y, main_va[name]).assign(line=line, model=name))
        lt = lift_table(y, main_va[name]).reset_index().assign(line=line, model=name)
        lifts.append(lt)

    # 3. GBM variants on the main fold
    if not a.skip_variants:
        cands = {"gbm_current": (factors, {}, 300),
                 "gbm_slow_deep": (factors, {"learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 200}, 500)}
        if line == "motor":
            cands["gbm_make_model"] = (factors + ["make_model"], {}, 300)
        for name, (fs, params, patience) in cands.items():
            m = LGBMTweedie(fs, seed=cfg["seed"], params=params, n_estimators=6000)
            m.patience = patience
            p = m.fit(main_tr, capped, loading_target="target_dev").predict(main_va)
            for w in (0.5, 1.0):
                pp = (1 - w) * main_va["p_glm"] + w * p
                variants.append({"line": line, "variant": name, "w_gbm": w, "trees": m.model_.best_iteration_,
                                 "balance": m.balance_, **evaluate(y, pp)})
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del df, main_tr, main_va

F, L, LI, S = (pd.DataFrame(folds), pd.concat(levels), pd.concat(lifts), pd.concat(segs))
V = pd.DataFrame(variants)
for name, t in (("folds", F), ("level", L), ("lift", LI), ("segments", S), ("gbm_variants", V)):
    t.to_csv(out_dir / f"{name}.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
print("\n=== 1. Rolling origin: D2 by GBM weight (0 = GLM only)")
print(F.pivot_table(index=["line", "valid_year"], columns="w_gbm", values="d2").round(4).to_string())
print("--- A/E by GBM weight")
print(F.pivot_table(index=["line", "valid_year"], columns="w_gbm", values="ae").round(3).to_string())
print("\n=== 2a. Level sensitivity on 2024 (D2 at x scale; 'best' = D2-maximising scale)")
is_best = L["best"].eq(True) if "best" in L else pd.Series(False, index=L.index)
lv = L[~is_best].pivot_table(index=["line", "model"], columns="scale", values="d2", aggfunc="first")
lv = lv.join(L[is_best].set_index(["line", "model"])[["scale", "d2"]].rename(columns={"scale": "best_scale", "d2": "best_d2"}))
print(lv.round(4).to_string())
print("\n=== 2b. A/E by predicted decile (2024, blend)")
print(LI[LI.model == "p_blend"].pivot_table(index="line", columns="bin", values="ae").round(2).to_string())
print("\n=== 2c. Segments outside A/E 0.85-1.15 (>= 2% exposure) in at least 2 of 3 folds (blend); A/E per valid year")
S["off"] = ((S.ae < 0.85) | (S.ae > 1.15)) & (S.exposure_share >= 0.02)
piv = S.pivot_table(index=["line", "factor", "level"], columns="valid_year", values="ae")
info = S.groupby(["line", "factor", "level"]).agg(n_off=("off", "sum"), exposure_share=("exposure_share", "mean"),
                                                 claims_per_year=("rows_with_claim", "mean"))
pers = piv.join(info)
side = np.sign(piv - 1)
pers["same_side"] = side.abs().sum(axis=1) == side.sum(axis=1).abs()
pers = pers[pers.n_off >= 2].sort_values(["line", "n_off"], ascending=[True, False])
print(pers.round(3).to_string() if len(pers) else "none")
print("(same_side = A/E on the same side of 1 in every fold -> systematic rather than noise)")
if len(V):
    print("\n=== 3. GBM variants on 2024 (w_gbm 1.0 = GBM alone, 0.5 = blend with GLM)")
    print(V[["line", "variant", "w_gbm", "trees", "balance", "d2", "ae", "gini"]].round(4).to_string(index=False))
print(f"\nsaved {out_dir.relative_to(REPO)}")
