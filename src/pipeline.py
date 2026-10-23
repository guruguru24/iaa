"""End-to-end W1 build: final fit on 2021-24 (on-levelled to 2025), predict test rows, checks, submission file.

    python -m src.pipeline --market w1 --model glm        # or gbm / blend (G4 decides)

Steps per line: train frame (approved G1/G2/G3 rules) -> on-level to 2025 at the central trend -> fit on the capped
target with the large-loss loading (decision #9) -> predict test rows x their exposure -> level checks.
Writes outputs/submissions/{market}_{time}_{git}.csv (via src.submission) and outputs/reports/{market}_pipeline_{time}.csv.
Never submits to Kaggle: that is gate G5 (owner approval, logs/decision_log.md).
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.data import LINES, REPO, apply_claim_overrides, load_claims, load_config, load_table
from src.dataset import onlevel_factor, reference_tables, test_frame, train_frame
from src.features import feature_columns
from src.history import HIST_COLS
from src.models import ExperienceModifier, LGBMTweedie, TweedieGLM
from src.submission import check_submission, write_submission

TARGET_YEAR = {"w1": 2025}


def gbm_spec(line: str, cfg: dict) -> tuple[str, dict]:
    """GBM config name and settings for a line (config: model_by_line, gbm_configs); default c0."""
    name = cfg.get("model_by_line", {}).get(line, {}).get("gbm", "c0")
    return name, cfg.get("gbm_configs", {}).get(name, {"stop": "policy", "params": {}})


def fit_predict(model: str, tr: pd.DataFrame, te: pd.DataFrame, factors, cap: int, seed: int,
                gbm: dict | None = None, modifier: dict | None = None, return_models: bool = False):
    """Fit the chosen model on the capped target (+ loading), optionally x experience modifier; predict test rows."""
    capped = f"target_dev_cap{cap}"
    gbm = gbm or {"stop": "policy", "params": {}}
    info, preds, train_preds, models = {}, {}, {}, {}
    if model in ("glm", "blend"):
        m = TweedieGLM(factors).fit(tr, capped, loading_target="target_dev")
        preds["glm"], train_preds["glm"], info["glm_loading"], models["glm"] = m.predict(te), m.predict(tr), m.loading_, m
    if model in ("gbm", "blend"):
        m = LGBMTweedie(factors, seed=seed, params=gbm.get("params") or {}, stop=gbm.get("stop", "policy"),
                        n_estimators=6000 if gbm.get("stop") == "time" else 3000)
        m = m.fit(tr, capped, loading_target="target_dev")
        preds["gbm"], train_preds["gbm"], models["gbm"] = m.predict(te), m.predict(tr), m
        info["gbm_loading"], info["gbm_trees"] = m.loading_, m.n_trees_
    blend = (lambda d: 0.5 * d["glm"] + 0.5 * d["gbm"]) if model == "blend" else (lambda d: d[model])
    p = blend(preds)
    if modifier:
        mod = ExperienceModifier(col=modifier.get("col", "hist_claims_1y"), max_bucket=modifier.get("max_bucket", 3),
                                 k=modifier.get("k", 200.0)).fit(tr, tr["target_dev"], blend(train_preds))
        p, models["modifier"] = mod.apply(te, p), mod
        info["modifier_factors"] = {int(k): round(float(v), 4) for k, v in mod.factors_.items()}
    return (p, info, models) if return_models else (p, info)


def prepare_line(line: str, cfg: dict, market: str, claims, refs, scenario: str = "central"):
    """Train frame (on-levelled to the target year) and test frame for one line, with the configured weather/history."""
    asm = cfg["assumptions"]
    to_year = TARGET_YEAR[market]
    weather = asm["weather_mode"]
    wyears = asm.get("weather_years") if weather == "long_run" else None
    history = bool(cfg.get("experience_modifier"))
    g, cap = asm["trend_2025"][line][scenario], asm["large_loss_cap"][line]
    tr = train_frame(line, market, caps=(cap,), claims=claims, refs=refs, weather_mode=weather, scenario=scenario,
                     weather_years=wyears, history=history)
    te = test_frame(line, market, refs=refs, weather_mode=weather, weather_years=wyears, history=history, claims=claims)
    factors = [f for f in feature_columns(tr) if f not in HIST_COLS]
    missing = [f for f in factors if f not in te]
    if missing:
        raise ValueError(f"{line}: features missing in test: {missing}")
    on = onlevel_factor(tr["policy_year"], to_year, g)
    tr_on = tr.assign(target_dev=tr["target_dev"] * on, **{f"target_dev_cap{cap}": tr[f"target_dev_cap{cap}"] * on})
    return tr, tr_on, te, factors, g, cap


def level_check(tr: pd.DataFrame, te: pd.DataFrame, pred: np.ndarray, trend: float, to_year: int) -> dict:
    """Predicted 2025 burning cost vs each training year's developed burning cost trended to 2025."""
    by_year = tr.groupby("policy_year").apply(lambda d: d["target_dev"].sum() / d["exposure"].sum(), include_groups=False)
    trended = by_year * (1 + trend) ** (to_year - by_year.index)
    pred_bc = pred.sum() / te["exposure"].sum()
    return {"pred_bc": pred_bc, **{f"trended_bc_{y}": v for y, v in trended.items()},
            "pred_vs_2024_trended": pred_bc / trended.get(to_year - 1, np.nan),
            "pred_vs_mean_trended": pred_bc / trended.mean()}


def build_submission(preds: dict[str, pd.DataFrame], sample: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-line predictions and order rows as in sample_submission."""
    sub = pd.concat(preds.values(), ignore_index=True)
    return sample[["row_id"]].merge(sub, on="row_id", how="left")


def run(market: str = "w1", model: str = "glm", scenario: str = "central", loading_sensitivity: bool = True) -> dict:
    cfg = load_config()
    asm = cfg["assumptions"]
    to_year = TARGET_YEAR[market]
    weather = asm["weather_mode"]
    wyears = asm.get("weather_years") if weather == "long_run" else None
    claims = load_claims(market)
    refs = reference_tables(market)
    preds, summary = {}, []
    for line in LINES:
        tr, tr_on, te, factors, g, cap = prepare_line(line, cfg, market, claims, refs, scenario)
        gname, gspec = gbm_spec(line, cfg)
        p, info = fit_predict(model, tr_on, te, factors, cap, cfg["seed"], gspec, cfg.get("experience_modifier"))
        row = {"line": line, "model": model, "gbm_config": gname, "scenario": scenario, "trend": g, "cap": cap, "n_train": len(tr), "n_test": len(te),
               "test_exposure": te["exposure"].sum(), "pred_total": p.sum(), **info, **level_check(tr, te, p, g, to_year)}
        if loading_sensitivity and line == "motor":
            alt = apply_claim_overrides(claims, asm["large_loss_sensitivity"])
            tr_alt = train_frame(line, market, caps=(cap,), claims=alt, refs=refs, weather_mode=weather, scenario=scenario,
                                 weather_years=wyears)
            loading_alt = (tr_alt["target_dev"] * onlevel_factor(tr_alt["policy_year"], to_year, g)).sum() / \
                          (tr_alt[f"target_dev_cap{cap}"] * onlevel_factor(tr_alt["policy_year"], to_year, g)).sum()
            key = "glm_loading" if "glm_loading" in info else "gbm_loading"
            row["loading_sensitivity_1.19M"] = loading_alt
            row["pred_total_sensitivity_1.19M"] = p.sum() * loading_alt / info[key]
            del tr_alt
        summary.append(row)
        preds[line] = pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": p})
        print(f"[{line}] pred_bc {row['pred_bc']:.1f} | vs 2024 trended {row['pred_vs_2024_trended']:.3f} "
              f"| vs mean trended {row['pred_vs_mean_trended']:.3f}", flush=True)
        del tr, te, tr_on
    sample = load_table("sample_submission", market)
    sub = build_submission(preds, sample)
    problems = check_submission(sub, sample)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    rep = pd.DataFrame(summary)
    rep.to_csv(REPO / "outputs" / "reports" / f"{market}_pipeline_{model}_{stamp}.csv", index=False)
    if problems:
        raise ValueError("submission checks failed: " + "; ".join(problems))
    path = write_submission(sub, sample, f"{market}_{model}", out_dir=REPO / "outputs" / "submissions")
    return {"summary": rep, "submission": path}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="w1", choices=list(TARGET_YEAR))
    ap.add_argument("--model", default="glm", choices=["glm", "gbm", "blend"])
    ap.add_argument("--scenario", default="central", choices=["low", "central", "high"])
    a = ap.parse_args(argv)
    out = run(a.market, a.model, a.scenario)
    pd.set_option("display.width", 220)
    print(out["summary"].round(4).T.to_string())


if __name__ == "__main__":
    main()
