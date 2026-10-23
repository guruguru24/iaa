"""Stage 2 of decision #14: per-peril frequency x severity GBM vs the current blend, on two temporal folds.

Models: GLM (capped + loading), Tweedie GBM (current c0), FreqSevGBM (src/models.py). Blends: glm+tweedie (current,
the reference), glm+freqsev, tweedie+freqsev, equal thirds. Pre-registered rule: a candidate replaces the current blend
only if its D2 gain is >= +0.001 in BOTH folds and A/E stays within 0.95-1.05.

Usage (from the repo root; ~15-25 min per line):
    .venv\\Scripts\\python scripts\\run_freqsev.py
    .venv\\Scripts\\python scripts\\run_freqsev.py --lines contents
"""
import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config
from src.dataset import onlevel_factor, reference_tables, train_frame
from src.development import development_factors
from src.eda import claim_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import FreqSevGBM, LGBMTweedie, TweedieGLM
from src.validation import evaluate, temporal_split

FOLDS = [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]

ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
a = ap.parse_args()

cfg = load_config()
asm = cfg["assumptions"]
claims = load_claims("w1")
refs = reference_tables("w1")
dev = development_factors(claims, asm["development"]["mature_years"])
rows, peril_rows = [], []

for line in a.lines:
    t0 = time.time()
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    df = train_frame(line, "w1", caps=(cap,), claims=claims, refs=refs, weather_mode=asm["weather_mode"])
    factors = [f for f in feature_columns(df) if f not in HIST_COLS]
    capped = f"target_dev_cap{cap}"
    cf = claim_frame(line, df, claims)
    cc = cf.loc[cf["in_period"] & ~cf["declined"] & (cf["net"] > 0), ["row_id", "peril", "net", "policy_year"]]
    cnt = dev.loc[dev["line"] == line, ["peril", "policy_year", "count_dev_factor"]]
    for train_years, vy in FOLDS:
        tr, va = temporal_split(df, train_years, [vy])
        tr[["target_dev", capped]] = tr[["target_dev", capped]].mul(onlevel_factor(tr["policy_year"], vy, g), axis=0)
        y = va["target_dev"].to_numpy()
        p = {"glm": TweedieGLM(factors).fit(tr, capped, loading_target="target_dev").predict(va),
             "tweedie": LGBMTweedie(factors, seed=cfg["seed"]).fit(tr, capped, loading_target="target_dev").predict(va)}
        fs = FreqSevGBM(factors, cap=cap, trend=g, to_year=vy, seed=cfg["seed"]).fit(
            tr, cc[cc["row_id"].isin(tr["row_id"])], capped, cnt, loading_target="target_dev")
        p["freqsev"] = fs.predict(va)
        p["blend_glm_tweedie (current)"] = 0.5 * p["glm"] + 0.5 * p["tweedie"]
        p["blend_glm_freqsev"] = 0.5 * p["glm"] + 0.5 * p["freqsev"]
        p["blend_tweedie_freqsev"] = 0.5 * p["tweedie"] + 0.5 * p["freqsev"]
        p["blend_equal3"] = (p["glm"] + p["tweedie"] + p["freqsev"]) / 3
        for name, pred in p.items():
            rows.append({"line": line, "valid_year": vy, "model": name, **evaluate(y, pred)})
        # per-peril check: predicted vs developed actual capped cost by peril on the valid year
        pp = fs.peril_predictions(va).sum() * fs.balance_ * fs.loading_
        act = cc[cc["row_id"].isin(va["row_id"])].assign(v=lambda d: d["net"].clip(upper=cap)).groupby("peril")["v"].sum()
        for per in pp.index:
            peril_rows.append({"line": line, "valid_year": vy, "peril": per, "pred": pp[per], "actual_capped_reported": act.get(per, 0.0),
                               "freq_trees": fs.trees_[per][0], "sev_trees": fs.trees_[per][1]})
        print(f"[{line} -> {vy}] freqsev D2 {rows[-5]['d2']:.4f} | current blend {rows[-4]['d2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    del df, cf

res = pd.DataFrame(rows)
out = REPO / "outputs" / "reports" / f"w1_freqsev_{datetime.now():%Y%m%d_%H%M}.csv"
res.to_csv(out, index=False)
pd.set_option("display.width", 240)
piv = res.pivot_table(index=["line", "valid_year"], columns="model", values="d2")
print("\n=== D2 by model")
print(piv.round(4).to_string())
gain = piv.drop(columns="blend_glm_tweedie (current)").sub(piv["blend_glm_tweedie (current)"], axis=0)
print("\n=== D2 gain vs current blend (rule: >= +0.001 in BOTH folds)")
print(gain.round(4).to_string())
print("pass:"); print((gain >= 0.001).groupby(level="line").all().to_string())
print("\n=== A/E by model")
print(res.pivot_table(index=["line", "valid_year"], columns="model", values="ae").round(3).to_string())
print("\n=== Per-peril (valid year): predicted capped cost incl. loading vs reported capped actual (immature in 2024), trees")
print(pd.DataFrame(peril_rows).round(0).to_string(index=False))
print(f"saved {out.relative_to(REPO)}")
