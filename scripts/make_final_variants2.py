"""Two more W1 variants on top of final b3, built in one pass. Never submits (G5 per file). NOT temporally validated.

  b3        rebuild of final b3 (long-run weather; GLM + GBM: buildings/contents c7, motor c0; 1y experience modifier)
            - must equal candidates/w1_final_b3_20261024_0024.csv
  b3_seed5  b3 with each GBM averaged over 5 seeds (variance reduction only, no tuning)
  b3_mod2y  b3 with the experience modifier keyed on claims in the last 2 years (fitted on rows with a 2-year window)

Usage (from the repo root; ~40 min):
    .venv\\Scripts\\python scripts\\make_final_variants2.py
"""
import hashlib
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config, load_table
from src.dataset import onlevel_factor, reference_tables, test_frame, train_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import ExperienceModifier, LGBMTweedie, TweedieGLM
from src.pipeline import build_submission, gbm_spec
from src.submission import check_submission

TO_YEAR = 2025
B_FILE = REPO / "outputs" / "submissions" / "candidates" / "w1_final_b3_20261024_0024.csv"
SEEDS = 5

cfg = load_config()
asm = cfg["assumptions"]
weather, wyears = asm["weather_mode"], asm.get("weather_years")
claims = load_claims("w1")
refs = reference_tables("w1")
variants = {k: {} for k in ("b3", "b3_seed5", "b3_mod2y")}
notes = []


def gbm_fit(spec, tr, te, factors, capped, seed=None):
    m = LGBMTweedie(factors, seed=cfg["seed"] if seed is None else seed, params=spec.get("params") or {}, stop=spec.get("stop", "policy"),
                    n_estimators=6000 if spec.get("stop") == "time" else 3000)
    m.fit(tr, capped, loading_target="target_dev")
    return m.predict(te), m.predict(tr)


for line in LINES:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    capped = f"target_dev_cap{cap}"
    tr = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=weather, weather_years=wyears, history=True)
    te = test_frame(line, "w1", refs=refs, weather_mode=weather, weather_years=wyears, history=True, claims=claims)
    factors = [f for f in feature_columns(tr) if f not in HIST_COLS]
    on = onlevel_factor(tr["policy_year"], TO_YEAR, g)
    tr = tr.assign(target_dev=tr["target_dev"] * on, **{capped: tr[capped] * on})
    glm = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev")
    g_te, g_tr = glm.predict(te), glm.predict(tr)
    gname, gspec = gbm_spec(line, cfg)
    if line == "buildings":                     # b3 uses c7 for buildings (decision #22)
        gname, gspec = "c7", cfg["gbm_configs"]["c7"]
    b_te, b_tr = gbm_fit(gspec, tr, te, factors, capped)
    s_te, s_tr = [b_te], [b_tr]
    for k in range(1, SEEDS):
        x_te, x_tr = gbm_fit(gspec, tr, te, factors, capped, seed=cfg["seed"] + k)
        s_te.append(x_te); s_tr.append(x_tr)
    bl_te, bl_tr = 0.5 * g_te + 0.5 * b_te, 0.5 * g_tr + 0.5 * b_tr
    sd_te, sd_tr = 0.5 * g_te + 0.5 * np.mean(s_te, axis=0), 0.5 * g_tr + 0.5 * np.mean(s_tr, axis=0)
    mod = ExperienceModifier().fit(tr, tr["target_dev"], bl_tr)
    mod_sd = ExperienceModifier().fit(tr, tr["target_dev"], sd_tr)
    mod2 = ExperienceModifier(col="hist_claims_2y").fit(tr, tr["target_dev"], bl_tr)
    out = {"b3": mod.apply(te, bl_te), "b3_seed5": mod_sd.apply(te, sd_te), "b3_mod2y": mod2.apply(te, bl_te)}
    for k, v in out.items():
        variants[k][line] = pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": v})
    notes.append(f"{line}: gbm {gname}; 1y factors {np.round(list(mod.factors_.values()), 3).tolist()}; "
                 f"2y factors {np.round(list(mod2.factors_.values()), 3).tolist()}; "
                 f"test unknown 2y history {te['hist_claims_2y'].isna().mean():.1%}; "
                 f"seed spread of GBM test total {np.std([x.sum() for x in s_te]) / np.mean([x.sum() for x in s_te]):.4f}")
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del tr, te

sample = load_table("sample_submission")
out_dir = REPO / "outputs" / "submissions" / "candidates"
stamp = f"{datetime.now():%Y%m%d_%H%M}"
rows = []
for name, preds in variants.items():
    sub = build_submission(preds, sample)
    problems = check_submission(sub, sample)
    path = out_dir / f"w1_final2_{name}_{stamp}.csv"
    if not problems:
        sub.to_csv(path, index=False)
    rows.append({"variant": name, "checks": "PASS" if not problems else "; ".join(problems),
                 "file": path.name if not problems else "", "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if not problems else "",
                 **{f"total_{l}": float(preds[l]["prediction"].sum()) for l in LINES}})
man = pd.DataFrame(rows)
for l in LINES:
    man[f"vs_base_{l}"] = man[f"total_{l}"] / man.loc[man.variant == "b3", f"total_{l}"].iloc[0]
man.to_csv(out_dir / f"manifest_final2_{stamp}.csv", index=False)

if B_FILE.exists():
    b = pd.read_csv(B_FILE, dtype={"row_id": str})
    s = pd.read_csv(out_dir / f"w1_final2_b3_{stamp}.csv", dtype={"row_id": str})
    m = b.merge(s, on="row_id", suffixes=("_b", "_new"))
    notes.append(f"b3 rebuild vs submitted b3 file: rows {len(m)}/{len(b)}, max rel diff {(m.prediction_new / m.prediction_b - 1).abs().max():.2e}, "
                 f"sha equal {hashlib.sha256(B_FILE.read_bytes()).hexdigest() == man.loc[man.variant == 'b3', 'sha256'].iloc[0]}")
pd.set_option("display.width", 220)
print(man[["variant", "checks", "file"] + [f"vs_base_{l}" for l in LINES]].assign(sha=man.sha256.str[:12]).round(4).to_string(index=False))
print("\n".join(notes))
