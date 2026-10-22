"""Build W1 submission candidates, each with a pre-registered purpose. Never submits (gate G5 per file).

Base fit per line (same as src.pipeline: 2021-24 on-levelled to 2025, capped target + loading, same-year weather):
  C1 blend50          reproduces the submitted file (consistency check; not to be resubmitted)
  C2 glm_only         tests "blend > GLM" (validation pooled D2 0.1388 vs 0.1370)
  C3 blend25          GBM weight 0.25 (motor rolling folds: 0.25 ~ 0.5; others prefer 0.5)
  C4 trend_low        level only: low trend + low development scenario   (approximate: predictions rescaled by the
  C5 trend_high       level only: high trend + high development scenario  ratio of on-levelled training totals)
  C6 loading_1.19M    motor large-loss loading with CL30973963 at 1.19M (decision #9 sensitivity; others unchanged)
Optional refits:
  C7 weather_longrun  --longrun: address long-run weather (2021-24 means) instead of same-year (A8 / G4 question)
  C8 recency{d}       --recency d: fit weights decayed by d per year (only if scripts/run_recency.py supports it)

Writes outputs/submissions/candidates/*.csv and outputs/submissions/candidates/manifest_<time>.csv.

Usage (from the repo root; base ~20 min, each optional refit ~15 min):
    .venv\\Scripts\\python scripts\\make_candidates.py
    .venv\\Scripts\\python scripts\\make_candidates.py --longrun --recency 0.8
"""
import argparse
import hashlib
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, apply_claim_overrides, load_claims, load_config, load_table
from src.dataset import onlevel_factor, reference_tables, test_frame, train_frame
from src.development import developed_table, development_factors, year_dev_factors
from src.eda import claim_frame, line_frame
from src.features import feature_columns
from src.models import LGBMTweedie, TweedieGLM
from src.pipeline import build_submission
from src.submission import check_submission

TO_YEAR = 2025
PURPOSE = {
    "blend50": "Reproduce submitted file (must match w1_blend_20261022_2257 sha 72be5f20); do not resubmit",
    "glm_only": "Test blend > GLM seen in validation (pooled D2 +0.0018); expect LB lower than blend50",
    "blend25": "Test GBM weight 0.25 vs 0.5 (validation near-tie); informational",
    "trend_low": "2025 level is untestable offline: low trend/development; validation level curve flat (+-5% ~ 0.001 D2); no switch unless large",
    "trend_high": "As trend_low, high side",
    "loading_1.19M": "Decision #9 sensitivity: motor loading with CL30973963 at 1.19M (motor total -2.5%)",
    "weather_longrun": "A8/G4: long-run weather vs same-year 2025 weather (validation: bldg A/E 1.065 vs 1.000)",
}

ap = argparse.ArgumentParser()
ap.add_argument("--longrun", action="store_true")
ap.add_argument("--recency", type=float, default=None)
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
dev_cfg = asm["development"]
claims = load_claims("w1")
refs = reference_tables("w1")
sample = load_table("sample_submission")
dev_factors = development_factors(claims, dev_cfg["mature_years"])
cands = {k: {} for k in ("blend50", "glm_only", "blend25", "trend_low", "trend_high", "loading_1.19M")}
if a.longrun:
    cands["weather_longrun"] = {}
if a.recency:
    cands[f"recency{a.recency}"] = {}
    PURPOSE[f"recency{a.recency}"] = f"Recency weights decay {a.recency}: submit only if scripts/run_recency.py met the pre-set rule"
notes = []


def onlevel_total(target_by_year: pd.Series, g: float) -> float:
    return float((target_by_year * (1 + g) ** (TO_YEAR - target_by_year.index)).sum())


def fit_both(tr, te, factors, cap, **kw):
    capped = f"target_dev_cap{cap}"
    p_glm = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev", **kw).predict(te)
    gbm = LGBMTweedie(factors, seed=cfg["seed"]).fit(tr, capped, loading_target="target_dev", **kw)
    return p_glm, gbm.predict(te)


for line in LINES:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    tr = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode="same_year")
    te = test_frame(line, "w1", refs=refs, weather_mode="same_year")
    factors = feature_columns(tr)
    on = onlevel_factor(tr["policy_year"], TO_YEAR, g)
    tr_on = tr.assign(target_dev=tr["target_dev"] * on, **{f"target_dev_cap{cap}": tr[f"target_dev_cap{cap}"] * on})
    p_glm, p_gbm = fit_both(tr_on, te, factors, cap)
    ids = te["row_id"].to_numpy()
    blend = 0.5 * p_glm + 0.5 * p_gbm
    cands["blend50"][line] = blend
    cands["glm_only"][line] = p_glm
    cands["blend25"][line] = 0.75 * p_glm + 0.25 * p_gbm

    # level-only scenarios: ratio of on-levelled uncapped training totals (scenario dev factors + trend) to central
    rows = line_frame(line, "w1", claims)
    cf = claim_frame(line, rows, claims)
    net_by_year = rows.groupby("policy_year")["target_net"].sum()
    totals = {}
    for scen in ("low", "central", "high"):
        yf = year_dev_factors(developed_table(cf, dev_factors, dev_cfg, scen)).set_index("policy_year")["dev_factor"]
        totals[scen] = onlevel_total(net_by_year * yf.reindex(net_by_year.index).fillna(1.0), asm["trend_2025"][line][scen])
    cands["trend_low"][line] = blend * totals["low"] / totals["central"]
    cands["trend_high"][line] = blend * totals["high"] / totals["central"]

    # loading sensitivity (motor only): uncapped / capped on-levelled totals with the claim at 1.19M
    if line == "motor":
        alt = apply_claim_overrides(claims, asm["large_loss_sensitivity"])
        tr_alt = train_frame(line, "w1", caps=(cap,), claims=alt, refs=refs, weather_mode="same_year")
        on_alt = onlevel_factor(tr_alt["policy_year"], TO_YEAR, g)
        load_alt = (tr_alt["target_dev"] * on_alt).sum() / (tr_alt[f"target_dev_cap{cap}"] * on_alt).sum()
        load_c = tr_on["target_dev"].sum() / tr_on[f"target_dev_cap{cap}"].sum()
        cands["loading_1.19M"][line] = blend * load_alt / load_c
        notes.append(f"motor loading central {load_c:.4f} -> 1.19M {load_alt:.4f}")
        del tr_alt
    else:
        cands["loading_1.19M"][line] = blend
    notes.append(f"{line}: level ratios low {totals['low'] / totals['central']:.4f}, high {totals['high'] / totals['central']:.4f}")

    if a.recency:
        pg, pb = fit_both(tr_on, te, factors, cap, recency=a.recency, ref_year=TO_YEAR)
        cands[f"recency{a.recency}"][line] = 0.5 * pg + 0.5 * pb
    if a.longrun:
        yrs = [2021, 2022, 2023, 2024]
        tr_lr = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode="long_run", weather_years=yrs)
        te_lr = test_frame(line, "w1", refs=refs, weather_mode="long_run", weather_years=yrs)
        on_lr = onlevel_factor(tr_lr["policy_year"], TO_YEAR, g)
        tr_lr = tr_lr.assign(target_dev=tr_lr["target_dev"] * on_lr, **{f"target_dev_cap{cap}": tr_lr[f"target_dev_cap{cap}"] * on_lr})
        pg, pb = fit_both(tr_lr, te_lr, feature_columns(tr_lr), cap)
        if not (te_lr["row_id"].to_numpy() == ids).all():
            raise ValueError("test row order differs between weather variants")
        cands["weather_longrun"][line] = 0.5 * pg + 0.5 * pb
        del tr_lr, te_lr
    for c in cands.values():
        c[line] = pd.DataFrame({"row_id": ids, "prediction": c[line]})
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del tr, te, tr_on, rows, cf

out_dir = REPO / "outputs" / "submissions" / "candidates"
out_dir.mkdir(parents=True, exist_ok=True)
stamp = f"{datetime.now():%Y%m%d_%H%M}"
manifest = []
for name, preds in cands.items():
    sub = build_submission(preds, sample)
    problems = check_submission(sub, sample)
    path = out_dir / f"w1_{name}_{stamp}.csv"
    if not problems:
        sub.to_csv(path, index=False)
    line_tot = {f"total_{l}": float(preds[l]["prediction"].sum()) for l in LINES}
    manifest.append({"candidate": name, "purpose": PURPOSE.get(name, ""), "file": path.name if not problems else "",
                     "checks": "PASS" if not problems else "; ".join(problems),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if not problems else "", **line_tot})
man = pd.DataFrame(manifest)
base = man.set_index("candidate").loc["blend50", [f"total_{l}" for l in LINES]]
for l in LINES:
    man[f"vs_blend50_{l}"] = man[f"total_{l}"] / base[f"total_{l}"]
man.to_csv(out_dir / f"manifest_{stamp}.csv", index=False)

submitted = REPO / "outputs" / "submissions" / "w1_blend_20261022_2257_85bf3c8.csv"
if submitted.exists():
    s0 = pd.read_csv(submitted, dtype={"row_id": str})
    s1 = pd.read_csv(out_dir / f"w1_blend50_{stamp}.csv", dtype={"row_id": str})
    m = s0.merge(s1, on="row_id", suffixes=("_sub", "_new"))
    rel = (m["prediction_new"] / m["prediction_sub"] - 1).abs()
    notes.append(f"blend50 vs submitted file: rows matched {len(m)}/{len(s0)}, max rel diff {rel.max():.2e}, "
                 f"total ratio {m.prediction_new.sum() / m.prediction_sub.sum():.6f}")

pd.set_option("display.width", 240)
print(man[["candidate", "checks", "sha256"] + [f"vs_blend50_{l}" for l in LINES]].assign(sha256=man.sha256.str[:12]).round(4).to_string(index=False))
print("\n".join(notes))
print(f"saved {out_dir.relative_to(REPO)}  (manifest_{stamp}.csv)")
