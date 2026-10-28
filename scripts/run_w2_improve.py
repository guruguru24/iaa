"""Week 2 improvements on emerging validation (train 2023 -> validate 2024), on top of the G6 models (decision #27/#28).

Variants per line (method from config `w2.model_by_line`; C4 level with credibility as in decision #28):
  base           G6 model, single seed
  seed5          GBM averaged over 5 seeds (Week 1 lesson, decision #23)
  region_k100    base x emerging-region credibility factor (k = 100 claim rows)   - C1/C4 use no region at all
  region_k300    same with k = 300 (more shrinkage)
  seed5_region   seed5 x region factor (k = 300)
Pre-registered rules: seed averaging is kept if its median bootstrap gain is >= 0 (prior evidence from Week 1);
the region factor is adopted only if the bootstrap 5th percentile of its D2 gain over the matching base is > 0.

Usage (from the repo root; ~20 min):
    .venv\\Scripts\\python scripts\\run_w2_improve.py
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import reference_tables, train_frame
from src.development import development_factors
from src.features import harmonise_ages
from src.metrics import tweedie_d2
from src.models import CredibilityFactor, LGBMTweedie, TweedieGLM, seed_average
from src.pipeline import gbm_spec
from src.pipeline_w2 import common_features, credibility_level, onlevel

B = 200
OUT = REPO / "outputs" / "reports" / "w2_improve"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()

cfg = load_config()
asm, w2 = cfg["assumptions"], cfg["w2"]
dev_claims, emg_claims = load_claims("w1"), load_claims("w2")
dev_refs, emg_refs = reference_tables("w1"), reference_tables("w2")
borrowed = development_factors(dev_claims, asm["development"]["mature_years"])
emg_cfg = {**asm["development"], "expected_severity": []}
rows = []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    capped, method = f"target_dev_cap{cap}", w2["model_by_line"][line]
    _, spec = gbm_spec(line, cfg)
    dev = harmonise_ages(train_frame(line, "w1", caps=(cap,), claims=dev_claims, refs=dev_refs, weather_mode="long_run",
                                     weather_years=[2021, 2022, 2023]))
    dev = onlevel(dev[dev["policy_year"] <= 2023], cap, 2024, g)
    emg = train_frame(line, "w2", caps=(cap,), claims=emg_claims, refs=emg_refs, weather_mode="long_run",
                      weather_years=[2023], dev_factors=borrowed, dev_cfg=emg_cfg)
    tr, va = onlevel(emg[emg["policy_year"] == 2023], cap, 2024, g), emg[emg["policy_year"] == 2024].copy()
    feats = common_features(dev, emg)
    stop = spec.get("stop", "policy")
    make_gbm = lambda s, fs, st=stop: LGBMTweedie(fs, seed=s, params=spec.get("params") or {}, stop=st,
                                                 n_estimators=6000 if st == "time" else 3000)
    seeds = [cfg["seed"] + i for i in range(5)]
    if method == "C1":
        glm = TweedieGLM(feats).fit(dev, capped, loading_target="target_dev")
        g1 = seed_average(lambda s: make_gbm(s, feats), dev, capped, seeds[:1], loading_target="target_dev")
        g5 = seed_average(lambda s: make_gbm(s, feats), dev, capped, seeds, loading_target="target_dev")
        raw = {"base": lambda d: 0.5 * glm.predict(d) + 0.5 * g1(d), "seed5": lambda d: 0.5 * glm.predict(d) + 0.5 * g5(d)}
        level = {k: tr["target_dev"].sum() / f(tr).sum() for k, f in raw.items()}
    else:  # C4
        j = pd.concat([dev.assign(market="developed", w_mult=1.0, policy_id="dev_" + dev["policy_id"].astype(str)),
                       tr.assign(market="emerging", w_mult=w2["emerging_weight"], policy_id="emg_" + tr["policy_id"].astype(str))],
                      ignore_index=True)
        jf = feats + ["market"]
        glm = TweedieGLM(jf).fit(j, capped, loading_target="target_dev", weight_col="w_mult")
        g1 = seed_average(lambda s: make_gbm(s, jf, "policy"), j, capped, seeds[:1], loading_target="target_dev", weight_col="w_mult")
        g5 = seed_average(lambda s: make_gbm(s, jf, "policy"), j, capped, seeds, loading_target="target_dev", weight_col="w_mult")
        del j
        em = lambda d: d.assign(market="emerging")
        raw = {"base": lambda d: 0.5 * glm.predict(em(d)) + 0.5 * g1(em(d)), "seed5": lambda d: 0.5 * glm.predict(em(d)) + 0.5 * g5(em(d))}
        level = {k: credibility_level(tr["target_dev"].sum() / f(tr).sum(), w2["c4_level_credibility"]) for k, f in raw.items()}
    p, ptr = {}, {}
    for k, f in raw.items():
        p[k], ptr[k] = f(va) * level[k], f(tr) * level[k]
    for k_cred, src in ((100, "base"), (300, "base"), (300, "seed5")):
        cf = CredibilityFactor("region", k=k_cred).fit(tr, tr["target_dev"], ptr[src])
        name = f"region_k{k_cred}" if src == "base" else "seed5_region"
        p[name] = cf.apply(va, p[src])
        print(f"[{line}] {name} factors: " + ", ".join(f"{r} {v:.2f}" for r, v in sorted(cf.factors_.items())), flush=True)
    y, nb = va["target_dev"].to_numpy(), va["new_business"].astype(bool).to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    ref = {"base": "base", "seed5": "base", "region_k100": "base", "region_k300": "base", "seed5_region": "seed5"}
    for name, pred in p.items():
        r = ref[name]
        gains = np.array([tweedie_d2(y[i], pred[i]) - tweedie_d2(y[i], p[r][i]) for i in idx]) if name != r else np.zeros(B)
        rows.append({"line": line, "method": method, "variant": name, "vs": r, "d2": tweedie_d2(y, pred),
                     "d2_nb": tweedie_d2(y[nb], pred[nb]), "ae": y.sum() / pred.sum(),
                     "gain_p05": np.quantile(gains, 0.05), "gain_median": np.median(gains), "gain_p95": np.quantile(gains, 0.95)})
    print(f"[{line}] done ({time.time() - t0:.0f}s)", flush=True)
    del dev, emg

R = pd.DataFrame(rows)
R.to_csv(OUT / "improve.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== Emerging validation 2024 (train 2023); gain = bootstrap D2 difference vs `vs`")
print(R.round(4).to_string(index=False))
print("\nPre-registered: seed5 kept if gain_median >= 0; region adopted only if gain_p05 > 0")
print(f"saved {OUT.relative_to(REPO)}")
