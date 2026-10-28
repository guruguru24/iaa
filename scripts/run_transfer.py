"""Week 2 Day 2 (docs/05 §2): what carries over from the developed market to the emerging market.

Per line:
  (i)   lift lost by harmonising: developed temporal validation (2021-23 -> 2024), full Week 1 features vs the
        features common to both markets (age bands; no region, credit, mileage, telematics; gender not available)
  (ii)  transfer: harmonised developed model (2021-24, on-levelled to 2024) applied to emerging 2023-24:
        LEVEL = overall A/E by year and new vs renewal; SHAPE = A/E by factor level relative to the overall A/E
  (iii) carry-over table: emerging-only GLM fitted with the developed GLM's bands/base levels -> relativity per level,
        emerging vs developed, with emerging claim counts; factor verdict carry / adjust / refit / weak

Emerging targets use the developed market's development factors (same reporting delays, DQ22); the bodily-injury
expected-severity override is not applied (no mature emerging years). Never joins across markets.

Usage (from the repo root; ~30 min):
    .venv\\Scripts\\python scripts\\run_transfer.py
    .venv\\Scripts\\python scripts\\run_transfer.py --lines contents --skip-lift
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
from src.eda import band
from src.features import feature_columns, harmonise_ages
from src.history import HIST_COLS
from src.models import LGBMTweedie, TweedieGLM
from src.pipeline import gbm_spec
from src.validation import evaluate, temporal_split

NOT_COMMON = {"region", "credit_band", "driver_age", "policyholder_age", "annual_mileage", "telematics_score",
              "telematics_opt_in", "telematics_missing", "gender", "self_employed"}
OUT = REPO / "outputs" / "reports" / "w2_transfer"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
ap.add_argument("--skip-lift", action="store_true")
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
dev_claims, emg_claims = load_claims("w1"), load_claims("w2")
dev_refs, emg_refs = reference_tables("w1"), reference_tables("w2")
borrowed = development_factors(dev_claims, asm["development"]["mature_years"])       # developed patterns
emg_dev_cfg = {**asm["development"], "expected_severity": []}
lift, level, shape, carry = [], [], [], []
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


def blend_fit(tr, factors, capped, line):
    _, spec = gbm_spec(line, cfg)
    glm = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev")
    gbm = LGBMTweedie(factors, seed=cfg["seed"], params=spec.get("params") or {}, stop=spec.get("stop", "policy"),
                      n_estimators=6000 if spec.get("stop") == "time" else 3000).fit(tr, capped, loading_target="target_dev")
    return glm, (lambda d: 0.5 * glm.predict(d) + 0.5 * gbm.predict(d))


def onlevel(df, cap, to_year, g):
    on = onlevel_factor(df["policy_year"], to_year, g)
    return df.assign(target_dev=df["target_dev"] * on, **{f"target_dev_cap{cap}": df[f"target_dev_cap{cap}"] * on})


for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    capped = f"target_dev_cap{cap}"
    dev = harmonise_ages(train_frame(line, "w1", caps=(cap,), claims=dev_claims, refs=dev_refs, weather_mode="long_run",
                                     weather_years=[2021, 2022, 2023, 2024]))
    emg = train_frame(line, "w2", caps=(cap,), claims=emg_claims, refs=emg_refs, weather_mode="long_run",
                      weather_years=[2023, 2024], dev_factors=borrowed, dev_cfg=emg_dev_cfg)
    full = [f for f in feature_columns(dev) if f not in HIST_COLS and f not in {"driver_age_band", "policyholder_age_band"}]
    common = [f for f in feature_columns(dev) if f in set(feature_columns(emg)) and f not in NOT_COMMON | set(HIST_COLS)]
    print(f"[{line}] features full {len(full)}, common {len(common)}; emerging rows {len(emg):,}", flush=True)

    # (i) lift lost by harmonising, on developed temporal validation
    if not a.skip_lift:
        tr, va = temporal_split(dev, [2021, 2022, 2023], [2024])
        tr = onlevel(tr, cap, 2024, g)
        for name, fs in (("full_W1_features", full), ("common_features", common)):
            _, pred = blend_fit(tr, fs, capped, line)
            lift.append({"line": line, "features": name, "n": len(fs), **evaluate(va["target_dev"], pred(va))})
        print(f"[{line}] lift check done ({time.time() - t0:.0f}s)", flush=True)

    # (ii) harmonised developed model -> emerging 2023-24
    glm_dev, pred_dev = blend_fit(onlevel(dev, cap, 2024, g), common, capped, line)
    e = emg.assign(pred=pred_dev(emg) / (1 + g) ** (2024 - emg["policy_year"]))      # 2023 rows back to 2023 level
    for (yr, nb), d in e.groupby(["policy_year", "new_business"]):
        level.append({"line": line, "year": yr, "new_business": nb, "exposure": d["exposure"].sum(),
                      "claims_rows": int((d["target_dev"] > 0).sum()), "ae": d["target_dev"].sum() / d["pred"].sum()})
    overall = e["target_dev"].sum() / e["pred"].sum()
    level.append({"line": line, "year": "all", "new_business": "all", "exposure": e["exposure"].sum(),
                  "claims_rows": int((e["target_dev"] > 0).sum()), "ae": overall})
    for f in common:
        x = e.assign(lv=band(e[f], 5)).groupby("lv", observed=True).agg(exp=("exposure", "sum"), y=("target_dev", "sum"),
                                                                       p=("pred", "sum"), k=("target_dev", lambda s: (s > 0).sum()))
        for lv, r in x.iterrows():
            shape.append({"line": line, "factor": f, "level": lv, "exposure_share": r.exp / x["exp"].sum(),
                          "claims_rows": int(r.k), "rel_ae": (r.y / r.p) / overall})

    # (iii) carry-over table: emerging GLM on the developed GLM's encoder
    glm_emg = TweedieGLM(common, alpha=1e-3).fit(onlevel(emg, cap, 2024, g), capped, loading_target="target_dev",
                                                 encoder=glm_dev.enc_)
    rd, re_ = glm_dev.relativities(), glm_emg.relativities()
    X = glm_dev.enc_.transform(emg)
    k = pd.Series(np.asarray(X[(emg["target_dev"] > 0).to_numpy()].sum(axis=0)).ravel(), index=glm_dev.enc_.columns_)
    for col in rd.index:
        f, lv = col.split("=", 1)
        carry.append({"line": line, "factor": f, "level": lv, "rel_dev": rd[col], "rel_emg": re_[col],
                      "ratio": re_[col] / rd[col], "emg_claims_rows": int(k[col])})
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del dev, emg, e

L, S, C = pd.DataFrame(level), pd.DataFrame(shape), pd.DataFrame(carry)
for name, df in (("lift", pd.DataFrame(lift)), ("level", L), ("shape", S), ("carry_levels", C)):
    df.to_csv(OUT / f"{name}.csv", index=False)

# factor verdicts: credible levels (>= 100 emerging claim rows) decide; exposure-free summary by median |log ratio|
C["credible"] = C["emg_claims_rows"] >= 100
C["abs_log_ratio"] = np.log(C["ratio"]).abs()
verdict = []
for (line, f), d in C.groupby(["line", "factor"]):
    cr = d[d["credible"]]
    spread_dev = np.log(d["rel_dev"]).abs().max()
    m = cr["abs_log_ratio"].median() if len(cr) else np.nan
    v = ("weak (little effect in both)" if spread_dev < 0.05 and (cr.empty or np.log(cr["rel_emg"]).abs().max() < 0.05)
         else "carry over" if len(cr) == 0 or m < np.log(1.10)
         else "adjust" if m < np.log(1.30) else "refit on emerging")
    verdict.append({"line": line, "factor": f, "levels": len(d), "credible_levels": len(cr),
                    "median_ratio_dev_to_emg": float(np.exp(m)) if len(cr) else np.nan, "verdict": v})
V = pd.DataFrame(verdict)
V.to_csv(OUT / "carry_over_table.csv", index=False)

if lift:
    print("\n=== (i) Lift lost by harmonising (developed, validate 2024)")
    print(pd.DataFrame(lift)[["line", "features", "n", "d2", "ae"]].round(4).to_string(index=False))
print("\n=== (ii) LEVEL: harmonised developed model on emerging (A/E; developed level carried as is)")
print(L.round(3).to_string(index=False))
print("\n=== (ii) SHAPE: factor levels where developed relativities break (rel A/E outside 0.8-1.25, >= 5% exposure, >= 100 claim rows)")
br = S[(S.exposure_share >= 0.05) & (S.claims_rows >= 100) & ((S.rel_ae < 0.8) | (S.rel_ae > 1.25))]
print(br.round(3).to_string(index=False) if len(br) else "none")
print("\n=== (iii) Carry-over table (verdict per factor)")
print(V.sort_values(["line", "verdict"]).round(3).to_string(index=False))
print(f"\nsaved {OUT.relative_to(REPO)}")
