"""Week 2 (emerging market) build per gate G6 (decision #27, configs/config.yaml `w2`).

    python -m src.pipeline_w2                # final fit, emerging test predictions, checks, submission + NB sensitivities

Per line:
  C1 (motor, contents): harmonised developed GLM+GBM blend (developed 2021-24, on-levelled to 2025) x emerging level
                        (actual / predicted on emerging 2023-24, both on-levelled to 2025)
  C4 (buildings):       joint GLM+GBM on developed + emerging with a `market` feature, emerging rows up-weighted
New-business loading multiplies new-business test rows (central 1.0; low/high written as sensitivity files).
Never joins across markets; joint fits prefix IDs with the market. Never submits (G5).
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.data import LINES, REPO, load_claims, load_config, load_table
from src.dataset import onlevel_factor, reference_tables, test_frame, train_frame
from src.development import development_factors
from src.features import feature_columns, harmonise_ages
from src.history import HIST_COLS
from src.models import LGBMTweedie, TweedieGLM, seed_average
from src.pipeline import build_submission, gbm_spec
from src.submission import check_submission, write_submission

NOT_COMMON = {"region", "credit_band", "driver_age", "policyholder_age", "annual_mileage", "telematics_score",
              "telematics_opt_in", "telematics_missing", "gender", "self_employed"}


def common_features(dev: pd.DataFrame, emg: pd.DataFrame) -> list[str]:
    return [f for f in feature_columns(dev) if f in set(feature_columns(emg)) and f not in NOT_COMMON | set(HIST_COLS)]


def onlevel(df: pd.DataFrame, cap: int, to_year: int, g: float) -> pd.DataFrame:
    on = onlevel_factor(df["policy_year"], to_year, g)
    return df.assign(target_dev=df["target_dev"] * on, **{f"target_dev_cap{cap}": df[f"target_dev_cap{cap}"] * on})


def credibility_level(ae: float, z: float) -> float:
    """Level adjustment Z x A/E + (1 - Z): Z = 0 keeps the model level, Z = 1 matches the emerging actuals."""
    return z * ae + (1.0 - z)


def nb_loading(pred: np.ndarray, new_business: pd.Series, factor: float) -> np.ndarray:
    """Multiply new-business rows by `factor` (G6 sensitivity)."""
    return np.where(new_business.fillna(False).astype(bool).to_numpy(), pred * factor, pred)


def _gbm(factors, spec, seed, stop=None):
    stop = stop or spec.get("stop", "policy")
    return LGBMTweedie(factors, seed=seed, params=spec.get("params") or {}, stop=stop,
                       n_estimators=6000 if stop == "time" else 3000)


def fit_line(line: str, cfg: dict, data: dict) -> dict:
    asm, w2 = cfg["assumptions"], cfg["w2"]
    to_year, method = w2["target_year"], w2["model_by_line"][line]
    g, cap = asm["trend_2025"][line]["central"], asm["large_loss_cap"][line]
    capped = f"target_dev_cap{cap}"
    _, spec = gbm_spec(line, cfg)
    dev = harmonise_ages(train_frame(line, "w1", caps=(cap,), claims=data["dev_claims"], refs=data["dev_refs"],
                                     weather_mode="long_run", weather_years=w2["developed_years"]))
    emg = train_frame(line, "w2", caps=(cap,), claims=data["emg_claims"], refs=data["emg_refs"], weather_mode="long_run",
                      weather_years=w2["train_years"], dev_factors=data["borrowed"], dev_cfg=data["emg_dev_cfg"])
    te = test_frame(line, "w2", refs=data["emg_refs"], weather_mode="long_run", weather_years=w2["train_years"])
    dev, emg = onlevel(dev, cap, to_year, g), onlevel(emg, cap, to_year, g)
    feats = common_features(dev, emg)
    info = {"line": line, "method": method, "trend": g, "cap": cap, "n_features": len(feats), "gbm_seeds": len(seeds)}
    seeds = [cfg["seed"] + i for i in range(int(w2.get("gbm_seeds", 1)))]
    if method == "C1":
        glm = TweedieGLM(feats).fit(dev, capped, loading_target="target_dev")
        gbm = seed_average(lambda s: _gbm(feats, spec, s), dev, capped, seeds, loading_target="target_dev")
        pred = lambda d: 0.5 * glm.predict(d) + 0.5 * gbm(d)
        level = emg["target_dev"].sum() / pred(emg).sum()
        p = pred(te) * level
        info["emerging_level"] = level
        info["level_by_year"] = {int(y): float(d["target_dev"].sum() / pred(d).sum()) for y, d in emg.groupby("policy_year")}
    elif method == "C4":
        k = w2["emerging_weight"]
        j = pd.concat([dev.assign(market="developed", w_mult=1.0, policy_id="dev_" + dev["policy_id"].astype(str)),
                       emg.assign(market="emerging", w_mult=k, policy_id="emg_" + emg["policy_id"].astype(str))], ignore_index=True)
        jf = feats + ["market"]
        glm = TweedieGLM(jf).fit(j, capped, loading_target="target_dev", weight_col="w_mult")
        gbm = seed_average(lambda s: _gbm(jf, spec, s, stop="policy"), j, capped, seeds, loading_target="target_dev", weight_col="w_mult")
        pred = lambda d: 0.5 * glm.predict(d.assign(market="emerging")) + 0.5 * gbm(d.assign(market="emerging"))
        # the joint fit is balanced on both markets together, so its emerging level leans to developed: move it part of
        # the way to emerging's own actuals with credibility Z (decision #28)
        ae = float(emg["target_dev"].sum() / pred(emg).sum())
        z = float(w2.get("c4_level_credibility", 0.0))
        level = credibility_level(ae, z)
        p = pred(te) * level
        info.update({"emerging_ae_in_sample": ae, "c4_level_credibility": z, "emerging_level": level})
        info["level_by_year"] = {int(y): float(d["target_dev"].sum() / pred(d).sum()) for y, d in emg.groupby("policy_year")}   # before Z
        del j
    else:
        raise ValueError(f"unknown method {method}")
    by_year = emg.groupby("policy_year").apply(lambda d: d["target_dev"].sum() / d["exposure"].sum(), include_groups=False)
    info.update({"pred_bc": float(p.sum() / te["exposure"].sum()),
                 **{f"emg_bc_{y}_onlevelled": float(v) for y, v in by_year.items()},
                 "test_rows": len(te), "test_nb_share": float(te["new_business"].astype(bool).mean())})
    return {"pred": p, "row_id": te["row_id"].to_numpy(), "new_business": te["new_business"], "info": info}


def run() -> dict:
    cfg = load_config()
    asm = cfg["assumptions"]
    dev_claims = load_claims("w1")
    data = {"dev_claims": dev_claims, "emg_claims": load_claims("w2"), "dev_refs": reference_tables("w1"),
            "emg_refs": reference_tables("w2"), "borrowed": development_factors(dev_claims, asm["development"]["mature_years"]),
            "emg_dev_cfg": {**asm["development"], "expected_severity": []}}
    fits = {}
    for line in LINES:
        fits[line] = fit_line(line, cfg, data)
        i = fits[line]["info"]
        print(f"[{line}] {i['method']}: pred_bc {i['pred_bc']:.1f} | emerging bc (on-levelled) "
              + ", ".join(f"{k[7:11]} {v:.1f}" for k, v in i.items() if k.startswith("emg_bc_")), flush=True)
    sample = load_table("sample_submission", "w2")
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    out = {}
    for name, factor in cfg["w2"]["new_business_loading"].items():
        preds = {l: pd.DataFrame({"row_id": f["row_id"], "prediction": nb_loading(f["pred"], f["new_business"], factor)})
                 for l, f in fits.items()}
        sub = build_submission(preds, sample)
        problems = check_submission(sub, sample)
        if problems:
            raise ValueError(f"{name}: " + "; ".join(problems))
        if name == "central":
            out[name] = write_submission(sub, sample, "w2", out_dir=REPO / "outputs" / "submissions")
        else:
            path = REPO / "outputs" / "submissions" / "candidates" / f"w2_nb_{name}_{stamp}.csv"
            sub.to_csv(path, index=False)
            out[name] = path
    rep = pd.DataFrame([f["info"] for f in fits.values()])
    rep.to_csv(REPO / "outputs" / "reports" / f"w2_pipeline_{stamp}.csv", index=False)
    return {"summary": rep, "files": out}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    res = run()
    pd.set_option("display.width", 220)
    print(res["summary"].round(4).T.to_string())
    for k, v in res["files"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
