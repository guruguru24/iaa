"""Week 2 adaptation candidates (docs/05 §3) on emerging validation: train 2023 -> validate 2024.

The developed model uses developed 2021-23 only (no same-calendar-year look-ahead), harmonised features,
on-levelled to 2024. Emerging targets use developed development factors (DQ22); emerging trend = developed central
(provisional, gate G3). Never joins across markets; joint fits prefix IDs with the market.

Candidates
  C0_const         emerging 2023 burning cost
  C0_glm           emerging-only GLM (all emerging features incl. region, gender, self_employed)
  C0_glm_nogender  same without gender / self_employed (G4 question)
  C1               harmonised developed blend (GLM+GBM) x level factor fitted on emerging 2023
  C3_z{25,50,75}   Z x C0_glm + (1 - Z) x C1
  C4               joint GLM+GBM on developed 2021-23 + emerging 2023, `market` feature, emerging rows weighted x5
  C5               emerging GBM with the developed prediction (log, per exposure) as a feature; and 50/50 with C1
Metrics on emerging 2024: D2 all rows and new-business rows only, A/E; bootstrap (200, by row) 90% band of the D2
difference vs C1. Selection (pre-registered): simplest candidate within the noise band of the best (docs/05 §3).

Usage (from the repo root; ~25 min):
    .venv\\Scripts\\python scripts\\run_w2_candidates.py
    .venv\\Scripts\\python scripts\\run_w2_candidates.py --lines contents
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.development import development_factors
from src.features import feature_columns, harmonise_ages
from src.history import HIST_COLS
from src.metrics import tweedie_d2
from src.models import LGBMTweedie, TweedieGLM
from src.pipeline import gbm_spec
from src.validation import evaluate

NOT_COMMON = {"region", "credit_band", "driver_age", "policyholder_age", "annual_mileage", "telematics_score",
              "telematics_opt_in", "telematics_missing", "gender", "self_employed"}
OUT = REPO / "outputs" / "reports" / "w2_candidates"
OUT.mkdir(parents=True, exist_ok=True)
EMG_WEIGHT = 5.0
B = 200

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
dev_claims, emg_claims = load_claims("w1"), load_claims("w2")
dev_refs, emg_refs = reference_tables("w1"), reference_tables("w2")
borrowed = development_factors(dev_claims, asm["development"]["mature_years"])
emg_cfg = {**asm["development"], "expected_severity": []}
rows, boot = [], []


def onlevel(df, cap, to_year, g):
    on = onlevel_factor(df["policy_year"], to_year, g)
    return df.assign(target_dev=df["target_dev"] * on, **{f"target_dev_cap{cap}": df[f"target_dev_cap{cap}"] * on})


def gbm(factors, spec, stop=None):
    return LGBMTweedie(factors, seed=cfg["seed"], params=spec.get("params") or {}, stop=stop or spec.get("stop", "policy"),
                       n_estimators=6000 if (stop or spec.get("stop")) == "time" else 3000)


for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    capped = f"target_dev_cap{cap}"
    _, spec = gbm_spec(line, cfg)
    dev = harmonise_ages(train_frame(line, "w1", caps=(cap,), claims=dev_claims, refs=dev_refs, weather_mode="long_run",
                                     weather_years=[2021, 2022, 2023]))
    dev = onlevel(dev[dev["policy_year"] <= 2023], cap, 2024, g)
    emg = train_frame(line, "w2", caps=(cap,), claims=emg_claims, refs=emg_refs, weather_mode="long_run",
                      weather_years=[2023], dev_factors=borrowed, dev_cfg=emg_cfg)
    tr, va = onlevel(emg[emg["policy_year"] == 2023], cap, 2024, g), emg[emg["policy_year"] == 2024].copy()
    common = [f for f in feature_columns(dev) if f in set(feature_columns(emg)) and f not in NOT_COMMON | set(HIST_COLS)]
    emg_all = [f for f in feature_columns(emg) if f not in set(HIST_COLS)]
    emg_nog = [f for f in emg_all if f not in {"gender", "self_employed"}]
    y = va["target_dev"].to_numpy()
    nb = va["new_business"].astype(bool).to_numpy()
    p = {}

    # C0: emerging only
    p["C0_const"] = tr["target_dev"].sum() / tr["exposure"].sum() * va["exposure"].to_numpy()
    p["C0_glm"] = TweedieGLM(emg_all, alpha=1e-3).fit(tr, capped, loading_target="target_dev").predict(va)
    p["C0_glm_nogender"] = TweedieGLM(emg_nog, alpha=1e-3).fit(tr, capped, loading_target="target_dev").predict(va)

    # C1: harmonised developed blend x emerging level
    dglm = TweedieGLM(common).fit(dev, capped, loading_target="target_dev")
    dgbm = gbm(common, spec).fit(dev, capped, loading_target="target_dev")
    dev_pred = lambda d: 0.5 * dglm.predict(d) + 0.5 * dgbm.predict(d)
    level = tr["target_dev"].sum() / dev_pred(tr).sum()
    p["C1"] = dev_pred(va) * level
    print(f"[{line}] developed model done, emerging level factor {level:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # C3: credibility blends
    for z in (0.25, 0.5, 0.75):
        p[f"C3_z{int(z * 100)}"] = z * p["C0_glm"] + (1 - z) * p["C1"]

    # C4: joint fit, market feature, emerging up-weighted, IDs prefixed
    j = pd.concat([dev.assign(market="developed", w_mult=1.0, policy_id="dev_" + dev["policy_id"].astype(str)),
                   tr.assign(market="emerging", w_mult=EMG_WEIGHT, policy_id="emg_" + tr["policy_id"].astype(str))],
                  ignore_index=True)
    jf = common + ["market"]
    vj = va.assign(market="emerging")
    jglm = TweedieGLM(jf).fit(j, capped, loading_target="target_dev", weight_col="w_mult")
    jgbm = gbm(jf, spec, stop="policy").fit(j, capped, loading_target="target_dev", weight_col="w_mult")
    p["C4"] = 0.5 * jglm.predict(vj) + 0.5 * jgbm.predict(vj)
    del j

    # C5: emerging GBM with the developed prediction as a feature
    tr5 = tr.assign(dev_pred_log=np.log(dev_pred(tr) / tr["exposure"]))
    va5 = va.assign(dev_pred_log=np.log(dev_pred(va) / va["exposure"]))
    m5 = LGBMTweedie(emg_all + ["dev_pred_log"], seed=cfg["seed"], params={"min_child_samples": 200}).fit(tr5, capped, loading_target="target_dev")
    p["C5"] = m5.predict(va5)
    p["C5_C1_5050"] = 0.5 * p["C5"] + 0.5 * p["C1"]

    for name, pred in p.items():
        rows.append({"line": line, "candidate": name, "d2_all": evaluate(y, pred)["d2"], "ae_all": y.sum() / pred.sum(),
                     "d2_new_business": tweedie_d2(y[nb], pred[nb]) if nb.any() else np.nan,
                     "ae_new_business": y[nb].sum() / pred[nb].sum() if nb.any() else np.nan,
                     "n_valid": len(y), "n_valid_nb": int(nb.sum())})
    rng = np.random.default_rng(cfg["seed"])
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    base = np.array([tweedie_d2(y[i], p["C1"][i]) for i in idx])
    for name, pred in p.items():
        diff = np.array([tweedie_d2(y[i], pred[i]) for i in idx]) - base
        boot.append({"line": line, "candidate": name, "d2_minus_C1_p05": np.quantile(diff, 0.05),
                     "d2_minus_C1_median": np.median(diff), "d2_minus_C1_p95": np.quantile(diff, 0.95)})
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del dev, emg

R = pd.DataFrame(rows).merge(pd.DataFrame(boot), on=["line", "candidate"])
R.to_csv(OUT / "candidates.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)
print("\n=== Emerging validation 2024 (train 2023). Bootstrap = 90% band of D2 minus C1")
for line, d in R.groupby("line", sort=False):
    print(f"\n--- {line} (valid rows {d.n_valid.iloc[0]:,}, new business {d.n_valid_nb.iloc[0]:,})")
    print(d.drop(columns=["line", "n_valid", "n_valid_nb"]).sort_values("d2_all", ascending=False).round(4).to_string(index=False))
print(f"\nsaved {OUT.relative_to(REPO)}")
