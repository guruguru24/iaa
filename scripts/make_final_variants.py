"""Near-pass variants of the W1 final (b), built in one pass (shared fits). Never submits (G5 per file).

  base  = (b): long-run weather; GLM + GBM blend per configs/config.yaml (contents c7, others c0) - must equal the (b) file
  b1    = base, buildings GBM c7 instead of c0      (validation +0.0006 / +0.0014: same sign, 2023 below the +0.001 rule)
  b2    = base x experience modifier                (validation 6/6 cells positive, +0.0002..+0.0012; decision #16)
  b3    = b1 x experience modifier
Leaderboard use: tie-break check of near-pass, sign-consistent validation gains (decision #21), not model selection.

Usage (from the repo root; ~25 min):
    .venv\\Scripts\\python scripts\\make_final_variants.py
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
B_FILE = REPO / "outputs" / "submissions" / "w1_blend_20261024_0007_d4aa68f.csv"

cfg = load_config()
asm = cfg["assumptions"]
weather, wyears = asm["weather_mode"], asm.get("weather_years")
claims = load_claims("w1")
refs = reference_tables("w1")
variants = {k: {} for k in ("base", "b1", "b2", "b3")}
notes = []


def gbm_fit(spec, tr, te, factors, capped):
    m = LGBMTweedie(factors, seed=cfg["seed"], params=spec.get("params") or {}, stop=spec.get("stop", "policy"),
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
    b_te, b_tr = gbm_fit(gspec, tr, te, factors, capped)
    base_te, base_tr = 0.5 * g_te + 0.5 * b_te, 0.5 * g_tr + 0.5 * b_tr
    if line == "buildings":
        c7 = cfg["gbm_configs"]["c7"]
        c_te, c_tr = gbm_fit(c7, tr, te, factors, capped)
        b1_te, b1_tr = 0.5 * g_te + 0.5 * c_te, 0.5 * g_tr + 0.5 * c_tr
    else:
        b1_te, b1_tr = base_te, base_tr
    mod = ExperienceModifier().fit(tr, tr["target_dev"], base_tr)
    mod1 = ExperienceModifier().fit(tr, tr["target_dev"], b1_tr)
    out = {"base": base_te, "b1": b1_te, "b2": mod.apply(te, base_te), "b3": mod1.apply(te, b1_te)}
    for k, v in out.items():
        variants[k][line] = pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": v})
    notes.append(f"{line}: gbm {gname}; modifier factors (claims last yr 0/1/2/3+) {np.round(list(mod.factors_.values()), 3).tolist()}; "
                 f"test rows with unknown 1y history {te['hist_claims_1y'].isna().mean():.1%}; "
                 f"totals b2/base {out['b2'].sum() / out['base'].sum():.4f}")
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del tr, te

sample = load_table("sample_submission")
out_dir = REPO / "outputs" / "submissions" / "candidates"
stamp = f"{datetime.now():%Y%m%d_%H%M}"
rows = []
for name, preds in variants.items():
    sub = build_submission(preds, sample)
    problems = check_submission(sub, sample)
    path = out_dir / f"w1_final_{name}_{stamp}.csv"
    if not problems:
        sub.to_csv(path, index=False)
    rows.append({"variant": name, "checks": "PASS" if not problems else "; ".join(problems),
                 "file": path.name if not problems else "", "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if not problems else "",
                 **{f"total_{l}": float(preds[l]["prediction"].sum()) for l in LINES}})
man = pd.DataFrame(rows)
for l in LINES:
    man[f"vs_base_{l}"] = man[f"total_{l}"] / man.loc[man.variant == "base", f"total_{l}"].iloc[0]
man.to_csv(out_dir / f"manifest_final_{stamp}.csv", index=False)

if B_FILE.exists():
    b = pd.read_csv(B_FILE, dtype={"row_id": str})
    s = pd.read_csv(out_dir / f"w1_final_base_{stamp}.csv", dtype={"row_id": str})
    m = b.merge(s, on="row_id", suffixes=("_b", "_new"))
    notes.append(f"base vs (b) file: rows {len(m)}/{len(b)}, max rel diff {(m.prediction_new / m.prediction_b - 1).abs().max():.2e}, "
                 f"sha equal {hashlib.sha256(B_FILE.read_bytes()).hexdigest() == man.loc[man.variant == 'base', 'sha256'].iloc[0]}")
pd.set_option("display.width", 220)
print(man[["variant", "checks", "file"] + [f"vs_base_{l}" for l in LINES]].assign(sha=man.sha256.str[:12]).round(4).to_string(index=False))
print("\n".join(notes))
