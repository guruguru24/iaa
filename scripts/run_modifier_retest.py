"""Pre-registered re-test of the experience modifier (decision #16, deferred in #18) on the final models.

Part A (Week 1, the pre-registered test): final W1 model per line (50/50 GLM + GBM with the line's GBM config, caps +
loading) on two temporal folds (2021-22 -> 2023, 2021-23 -> 2024), with vs without the experience modifier
(config `experience_modifier`). Rule (#16, unchanged): adopt only if blend D2 gain >= +0.001 in BOTH folds and
A/E stays within 0.95-1.05.

Part B (Week 2, information only - the Week 2 finals are already selected, decision #36): G6 model per line
(C1/C4, single seed) trained as in run_w2_improve (train emerging 2023 -> validate 2024), with vs without the
modifier fitted on the model's own training rows. Emerging history starts 2023, so only 2024 rows are rated.
Variant `nb_unknown` (owner request): new-business rows are treated as history unknown (factor 1.0) in both the fit
and the application, because a new customer's zero claims with this insurer is not evidence of being claim-free.
Rule written before the run (as for the region factor, #34): useful only if the bootstrap 5th percentile of the
D2 gain over the base is > 0.

Usage (from the repo root; ~30-40 min):
    .venv\\Scripts\\python scripts\\run_modifier_retest.py
    .venv\\Scripts\\python scripts\\run_modifier_retest.py --part A --lines contents
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
from src.models import ExperienceModifier, LGBMTweedie, TweedieGLM
from src.pipeline import fit_predict, gbm_spec
from src.pipeline_w2 import common_features, credibility_level, onlevel
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
B = 200
OUT = REPO / "outputs" / "reports" / "modifier_retest"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["A", "B", "both"], default="both")
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()

cfg = load_config()
asm, w2, mcfg = cfg["assumptions"], cfg["w2"], cfg["experience_modifier"]
make_mod = lambda: ExperienceModifier(col=mcfg["col"], max_bucket=mcfg["max_bucket"], k=mcfg["k"])
nb_unknown = lambda d: d.assign(**{mcfg["col"]: d[mcfg["col"]].mask(d["new_business"].astype(bool))})
dev_claims, dev_refs = load_claims("w1"), reference_tables("w1")

if a.part in ("A", "both"):
    rows = []
    for line in a.lines:
        t0 = time.time()
        g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
        _, spec = gbm_spec(line, cfg)
        df = train_frame(line, "w1", caps=(cap,), claims=dev_claims, refs=dev_refs, weather_mode=asm["weather_mode"],
                         history=True)
        factors = [f for f in feature_columns(df) if f not in HIST_COLS]
        capped = f"target_dev_cap{cap}"
        for train_years, vy in FOLDS:
            tr, va = temporal_split(df, train_years, [vy])
            tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
            p, _, models = fit_predict("blend", tr, va, factors, cap, cfg["seed"], spec, None, return_models=True)
            p_tr = 0.5 * models["glm"].predict(tr) + 0.5 * models["gbm"].predict(tr)
            mod = make_mod().fit(tr, tr["target_dev"], p_tr)
            y = va["target_dev"].to_numpy()
            for name, pred in (("final", p), ("final+modifier", mod.apply(va, p))):
                rows.append({"line": line, "valid_year": vy, "variant": name, **evaluate(y, pred),
                             **({f"mod_{k}": v for k, v in mod.factors_.items()} if name != "final" else {})})
            print(f"[A {line} -> {vy}] done ({time.time() - t0:.0f}s)", flush=True)
        del df
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "part_a_w1.csv", index=False)
    d2 = A.pivot_table(index=["line", "valid_year"], columns="variant", values="d2")
    ae = A.pivot_table(index=["line", "valid_year"], columns="variant", values="ae")
    s = pd.DataFrame({"d2_final": d2["final"], "d2_modifier": d2["final+modifier"],
                      "gain": d2["final+modifier"] - d2["final"], "ae_modifier": ae["final+modifier"]})
    verdict = s.groupby(level="line").apply(lambda x: bool((x["gain"] >= 0.001).all() and x["ae_modifier"].between(0.95, 1.05).all()))
    print("\n=== Part A (W1, pre-registered #16): gain >= +0.001 in BOTH folds and A/E 0.95-1.05")
    print(s.round(4).to_string())
    print("pass:", verdict.to_dict())
    print(A.filter(regex="^(line|valid_year|mod_)").dropna().round(3).to_string(index=False))

if a.part in ("B", "both"):
    emg_claims, emg_refs = load_claims("w2"), reference_tables("w2")
    borrowed = development_factors(dev_claims, asm["development"]["mature_years"])
    emg_cfg = {**asm["development"], "expected_severity": []}
    rows = []
    for line in a.lines:
        t0 = time.time()
        g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
        capped, method = f"target_dev_cap{cap}", w2["model_by_line"][line]
        _, spec = gbm_spec(line, cfg)
        dev = harmonise_ages(train_frame(line, "w1", caps=(cap,), claims=dev_claims, refs=dev_refs, weather_mode="long_run",
                                         weather_years=[2021, 2022, 2023], history=True))
        dev = onlevel(dev[dev["policy_year"] <= 2023], cap, 2024, g)
        emg = train_frame(line, "w2", caps=(cap,), claims=emg_claims, refs=emg_refs, weather_mode="long_run",
                          weather_years=[2023], dev_factors=borrowed, dev_cfg=emg_cfg, history=True)
        tr, va = onlevel(emg[emg["policy_year"] == 2023], cap, 2024, g), emg[emg["policy_year"] == 2024].copy()
        feats = [f for f in common_features(dev, emg) if f not in HIST_COLS]
        stop = spec.get("stop", "policy")
        if method == "C1":
            glm = TweedieGLM(feats).fit(dev, capped, loading_target="target_dev")
            gbm = LGBMTweedie(feats, seed=cfg["seed"], params=spec.get("params") or {}, stop=stop,
                              n_estimators=6000 if stop == "time" else 3000).fit(dev, capped, loading_target="target_dev")
            raw = lambda d: 0.5 * glm.predict(d) + 0.5 * gbm.predict(d)
            level = tr["target_dev"].sum() / raw(tr).sum()
            fit_rows = dev                                  # the model's training rows
        else:  # C4
            j = pd.concat([dev.assign(market="developed", w_mult=1.0, policy_id="dev_" + dev["policy_id"].astype(str)),
                           tr.assign(market="emerging", w_mult=w2["emerging_weight"], policy_id="emg_" + tr["policy_id"].astype(str))],
                          ignore_index=True)
            jf = feats + ["market"]
            glm = TweedieGLM(jf).fit(j, capped, loading_target="target_dev", weight_col="w_mult")
            gbm = LGBMTweedie(jf, seed=cfg["seed"], params=spec.get("params") or {}, stop="policy",
                              n_estimators=3000).fit(j, capped, loading_target="target_dev", weight_col="w_mult")
            raw = lambda d: 0.5 * glm.predict(d) + 0.5 * gbm.predict(d)
            em = lambda d: d.assign(market="emerging")
            level = credibility_level(tr["target_dev"].sum() / raw(em(tr)).sum(), w2["c4_level_credibility"])
            fit_rows = j
            raw = (lambda f: (lambda d: f(em(d))))(raw)
        p_fit = 0.5 * glm.predict(fit_rows) + 0.5 * gbm.predict(fit_rows)
        p = raw(va) * level
        y, nb = va["target_dev"].to_numpy(), va["new_business"].astype(bool).to_numpy()
        rng = np.random.default_rng(cfg["seed"])
        idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
        for variant, prep in (("nb_claim_free", lambda d: d), ("nb_unknown", nb_unknown)):
            mod = make_mod().fit(prep(fit_rows), fit_rows["target_dev"], p_fit)
            pm = mod.apply(prep(va), p)
            rated = prep(va)[mcfg["col"]].notna().to_numpy()
            gains = np.array([tweedie_d2(y[i], pm[i]) - tweedie_d2(y[i], p[i]) for i in idx])
            rows.append({"line": line, "method": method, "variant": variant, "d2_base": tweedie_d2(y, p),
                         "d2_modifier": tweedie_d2(y, pm), "gain_p05": np.quantile(gains, 0.05),
                         "gain_median": np.median(gains), "gain_p95": np.quantile(gains, 0.95),
                         "ae_base": y.sum() / p.sum(), "ae_modifier": y.sum() / pm.sum(), "rated_share": rated.mean(),
                         "d2_nb_base": tweedie_d2(y[nb], p[nb]), "d2_nb_modifier": tweedie_d2(y[nb], pm[nb]),
                         **{f"mod_{k}": v for k, v in mod.factors_.items()}})
        print(f"[B {line}] done ({time.time() - t0:.0f}s)", flush=True)
        del dev, emg
    Bdf = pd.DataFrame(rows)
    Bdf.to_csv(OUT / "part_b_w2_v2.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== Part B (W2, information only): useful only if gain_p05 > 0")
    print(Bdf.round(4).to_string(index=False))
print(f"saved {OUT.relative_to(REPO)}")
