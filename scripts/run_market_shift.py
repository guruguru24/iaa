"""Week 2 Day 1 (docs/05 §1, data_description Part C): what changed in the emerging market — before any modelling.

Sections run independently (a failure is reported, the rest continues). Never joins across markets.
  1 schema        file/column/dtype/level differences developed vs emerging (Part C check)
  2 definitions   claim status codes, perils, valuation date, years, paid==incurred on closed, recoveries,
                  exposure vs dates, excess levels, expenses table
  3 credibility   emerging rows / exposure / claims by line x peril x year
  4 mix           emerging train (2023-24) vs test (2025): new business, channel, NCB, renewals, age bands, ...; PSI
  5 adversarial   classifier train vs test per line: AUC and the features that separate them
  6 dev vs emg    PSI developed vs emerging on common policy / address / customer columns
  7 outcomes      frequency, severity, burning cost, peril mix, report delay, open share: emerging vs developed 2023-24
  8 climate       weather ranges emerging vs developed; share of emerging addresses outside the developed range
  9 data quality  emerging: exposure flags, sum insured outliers, case/spelling variants, nulls, duplicates, negatives

Usage (from the repo root; developed sections read the large Week 1 files, ~5-10 min):
    .venv\\Scripts\\python scripts\\run_market_shift.py
Outputs: outputs/reports/w2_shift/*.csv and a compact printout.
"""
import sys
import traceback

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import LINES, REPO, flag_exposure, load_claims, load_policies, load_table, raw_dir

DEV, EMG = "w1", "w2"
OUT = REPO / "outputs" / "reports" / "w2_shift"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)
ID_DATE = {"row_id", "policy_id", "customer_id", "address_id", "claim_id", "period_start", "period_end", "line"}
status = []


def section(name):
    def wrap(fn):
        print(f"\n{'=' * 12} {name}", flush=True)
        try:
            fn()
            status.append(f"{name}: ok")
        except Exception as e:
            status.append(f"{name}: FAILED ({type(e).__name__}: {e})")
            traceback.print_exc(limit=3)
        return fn
    return wrap


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    a, b = a.dropna(), b.dropna()
    if a.empty or b.empty:
        return np.nan
    if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a) and a.nunique() > bins:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pa, pb = (pd.cut(x, edges).value_counts(normalize=True, sort=False) for x in (a, b))
    else:
        pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        pa, pb = pa.reindex(idx, fill_value=0), pb.reindex(idx, fill_value=0)
    pa, pb = pa.clip(lower=1e-4), pb.clip(lower=1e-4)
    return float(((pb - pa) * np.log(pb / pa)).sum())


@section("1 schema (developed -> emerging)")
def _():
    files = sorted({f.stem for f in raw_dir(DEV).glob("*.parquet")} | {f.stem for f in raw_dir(EMG).glob("*.parquet")})
    rows, levels = [], []
    for f in files:
        pd_, pe = raw_dir(DEV) / f"{f}.parquet", raw_dir(EMG) / f"{f}.parquet"
        if not (pd_.exists() and pe.exists()):
            rows.append({"file": f, "status": "emerging only" if pe.exists() else "developed only"})
            continue
        d, e = pd.read_parquet(pd_), pd.read_parquet(pe)
        common = set(d.columns) & set(e.columns)
        rows.append({"file": f, "status": "both", "rows_dev": len(d), "rows_emg": len(e),
                     "added": sorted(set(e.columns) - set(d.columns)), "removed": sorted(set(d.columns) - set(e.columns)),
                     "dtype_changed": sorted(f"{c}:{d[c].dtype}->{e[c].dtype}" for c in common if str(d[c].dtype) != str(e[c].dtype))})
        for c in sorted(common - ID_DATE):
            if not pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]):
                ld, le = set(d[c].dropna().astype(str)), set(e[c].dropna().astype(str))
                if ld != le:
                    levels.append({"file": f, "column": c, "n_dev": len(ld), "n_emg": len(le),
                                   "emg_only": sorted(le - ld)[:12], "dev_only": sorted(ld - le)[:12]})
    s, lv = pd.DataFrame(rows), pd.DataFrame(levels)
    save(s, "1_schema"); save(lv, "1_levels")
    print(s.to_string(index=False))
    print("\nlevel differences (first 12 shown per side):")
    print(lv.to_string(index=False) if len(lv) else "none")


@section("2 definitions")
def _():
    rows = []
    for m in (DEV, EMG):
        c = load_claims(m)
        closed = c[c["status"] == "closed"]
        rows.append({"market": m, "claims": len(c), "status_codes": sorted(c["status"].astype(str).unique()),
                     "valuation (max report)": str(c["report_date"].max().date()),
                     "loss_dates": f"{c['loss_date'].min().date()}..{c['loss_date'].max().date()}",
                     "closed_paid_eq_incurred": round(float(np.isclose(closed["paid_to_date"], closed["incurred_to_date"]).mean()), 4),
                     "share_with_recoveries": round(float((c["recoveries"] > 0).mean()), 4),
                     "n_negative_amounts": int((c[["paid_to_date", "incurred_to_date", "recoveries"]] < 0).any(axis=1).sum()),
                     "report_before_loss": int((c["report_date"] < c["loss_date"]).sum())})
    print(pd.DataFrame(rows).T.to_string(header=False))
    per = []
    for m in (DEV, EMG):
        c = load_claims(m)
        per.append(c.groupby(["line", "peril"]).size().rename(m))
    print("\nclaims by line x peril:"); print(pd.concat(per, axis=1).fillna(0).astype(int).to_string())
    pol = []
    for m in (DEV, EMG):
        for line in LINES:
            for kind in ("train", "test"):
                p = load_policies(line, m, kind)
                pol.append({"market": m, "line": line, "kind": kind, "rows": len(p),
                            "years": f"{p['period_start'].dt.year.min()}-{p['period_start'].dt.year.max()}",
                            "exposure_eq_dates": round(float((flag_exposure(p) == "ok").mean()), 4),
                            "vol_excess": sorted(p["voluntary_excess"].dropna().unique().tolist()),
                            "comp_excess": sorted(p["compulsory_excess"].dropna().unique().tolist())})
    d = pd.DataFrame(pol); save(d, "2_policies"); print(d.to_string(index=False))
    ex = load_table("expenses", EMG); save(ex, "2_expenses"); print("\nexpenses:"); print(ex.to_string(index=False))


@section("3 credibility (emerging)")
def _():
    c = load_claims(EMG)
    c = c[c["status"] != "repudiated"]
    rows = []
    for line in LINES:
        p = load_policies(line, EMG)
        e = (((p["period_end"] - p["period_start"]).dt.days + 1) / 365).groupby(p["period_start"].dt.year).sum()
        n = c[c["line"] == line].groupby([c["loss_date"].dt.year, "peril"]).size()
        for (y, peril), k in n.items():
            rows.append({"line": line, "year": y, "peril": peril, "claims": int(k), "exposure": round(float(e.get(y, np.nan)), 1)})
    d = pd.DataFrame(rows); save(d, "3_credibility")
    print(d.pivot_table(index=["line", "peril"], columns="year", values="claims", aggfunc="sum").fillna(0).astype(int).to_string())


def _mix_cols(df):
    return [c for c in df.columns if c not in ID_DATE | {"exposure", "renewal_decision"}]


@section("4 mix: emerging train (2023-24) vs test (2025)")
def _():
    rows, key = [], []
    for line in LINES:
        tr, te = load_policies(line, EMG), load_policies(line, EMG, "test")
        for c in sorted(set(_mix_cols(tr)) & set(te.columns)):
            rows.append({"line": line, "feature": c, "psi_train_vs_test": psi(tr[c], te[c])})
        for c in ("new_business", "channel", "renewal_count", "ncb_years", "days_quote_to_inception", "discount_applied", "written_premium"):
            if c in tr:
                if pd.api.types.is_numeric_dtype(tr[c]) and not pd.api.types.is_bool_dtype(tr[c]):
                    key.append({"line": line, "feature": c, "train": round(float(tr[c].mean()), 3), "test": round(float(te[c].mean()), 3)})
                else:
                    for lv in sorted(set(tr[c].dropna().astype(str)) | set(te[c].dropna().astype(str))):
                        key.append({"line": line, "feature": f"{c}={lv}", "train": round(float((tr[c].astype(str) == lv).mean()), 3),
                                    "test": round(float((te[c].astype(str) == lv).mean()), 3)})
    d, k = pd.DataFrame(rows).sort_values("psi_train_vs_test", ascending=False), pd.DataFrame(key)
    save(d, "4_psi_train_test"); save(k, "4_mix_key")
    print(k.to_string(index=False))
    print("\nPSI train vs test > 0.2:"); print(d[d.psi_train_vs_test > 0.2].round(3).to_string(index=False) or "none")


@section("5 adversarial validation (train vs test, per line)")
def _():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from src.models import _cat_X, _fit_cats
    rows = []
    for line in LINES:
        tr, te = load_policies(line, EMG), load_policies(line, EMG, "test")
        cols = [c for c in sorted(set(_mix_cols(tr)) & set(te.columns)) if c != "written_premium"]
        df = pd.concat([tr[cols].assign(_y=0), te[cols].assign(_y=1)], ignore_index=True)
        cats = _fit_cats(df, cols)
        X, y = _cat_X(df, cols, cats), df["_y"].to_numpy()
        rng = np.random.default_rng(2025)
        hold = rng.random(len(df)) < 0.25
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1, random_state=2025)
        m.fit(X[~hold], y[~hold])
        auc = roc_auc_score(y[hold], m.predict_proba(X[hold])[:, 1])
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        imp = (imp / imp.sum()).head(8)
        rows += [{"line": line, "auc": round(auc, 3), "feature": f, "gain_share": round(v, 3)} for f, v in imp.items()]
        print(f"{line}: AUC {auc:.3f} (0.5 = test looks like train) | top: " + ", ".join(f"{f} {v:.0%}" for f, v in imp.head(5).items()))
    save(pd.DataFrame(rows), "5_adversarial")


@section("6 developed vs emerging (PSI on common columns)")
def _():
    rows = []
    for line in LINES:
        d, e = load_policies(line, DEV), load_policies(line, EMG)
        d = d[d["period_start"].dt.year >= 2023]
        for c in sorted(set(_mix_cols(d)) & set(e.columns)):
            rows.append({"table": line, "feature": c, "psi": psi(d[c], e[c])})
    for t in ("addresses", "customers"):
        d, e = load_table(t, DEV), load_table(t, EMG)
        for c in sorted((set(d.columns) & set(e.columns)) - ID_DATE):
            rows.append({"table": t, "feature": c, "psi": psi(d[c], e[c])})
    x = pd.DataFrame(rows).sort_values("psi", ascending=False); save(x, "6_psi_dev_emg")
    print("PSI > 0.25:"); print(x[x.psi > 0.25].round(3).to_string(index=False) or "none")


@section("7 outcomes: emerging vs developed (2023-24)")
def _():
    from src.eda import burning_cost_table, line_frame
    rows, perils, lags = [], [], []
    for m in (DEV, EMG):
        claims = load_claims(m)
        for line in LINES:
            lf = line_frame(line, m, claims, clean=False)
            lf = lf[lf["policy_year"] >= 2023]
            rows.append(burning_cost_table(lf, "policy_year").assign(market=m, line=line))
        c = claims[(claims["status"] != "repudiated") & (claims["loss_date"].dt.year >= 2023)]
        c = c.assign(lag=(c["report_date"] - c["loss_date"]).dt.days, net=(c["incurred_to_date"] - c["recoveries"]).clip(lower=0),
                     open=c["status"].eq("open"), year=c["loss_date"].dt.year)
        perils.append(c.groupby(["line", "peril"]).agg(n=("claim_id", "size"), mean_net=("net", "mean")).assign(market=m).reset_index())
        lags.append(c.groupby(["line", "year"]).agg(lag_median=("lag", "median"), lag_p90=("lag", lambda s: s.quantile(.9)),
                                                    open_share=("open", "mean")).assign(market=m).reset_index())
    o, pr, lg = pd.concat(rows, ignore_index=True), pd.concat(perils, ignore_index=True), pd.concat(lags, ignore_index=True)
    save(o, "7_outcomes"); save(pr, "7_perils"); save(lg, "7_lags")
    print(o.pivot_table(index=["line", "policy_year"], columns="market", values=["frequency", "severity", "burning_cost"]).round(3).to_string())
    print("\nmean net per claim by peril (2023-24):")
    print(pr.pivot_table(index=["line", "peril"], columns="market", values="mean_net").round(0).to_string())
    print("\nreport delay and open share:")
    print(lg.pivot_table(index=["line", "year"], columns="market", values=["lag_median", "open_share"]).round(3).to_string())


@section("8 climate: weather ranges")
def _():
    d, e = load_table("weather", DEV), load_table("weather", EMG)
    rows = []
    for c in [x for x in d.columns if x not in ("address_id", "year")]:
        lo, hi = d[c].quantile([0.005, 0.995])
        rows.append({"variable": c, "dev_p50": d[c].median(), "emg_p50": e[c].median(), "dev_p99.5": hi, "emg_max": e[c].max(),
                     "share_emg_outside_dev_range": float(((e[c] < lo) | (e[c] > hi)).mean()), "years_emg": sorted(e["year"].unique().tolist())})
    w = pd.DataFrame(rows); save(w, "8_climate"); print(w.round(3).to_string(index=False))


@section("9 data quality (emerging)")
def _():
    rows, text = [], []
    for line in LINES:
        for kind in ("train", "test"):
            p = load_policies(line, EMG, kind)
            r = {"line": line, "kind": kind, "rows": len(p), "dup_rows_excl_id": int(p.drop(columns="row_id").duplicated().sum()),
                 "row_id_unique": bool(p["row_id"].is_unique), **{f"exposure_{k}": int(v) for k, v in flag_exposure(p).value_counts().items()},
                 "end_before_start": int((p["period_end"] < p["period_start"]).sum()),
                 "negative_numeric_cols": [c for c in p.select_dtypes("number").columns if (p[c] < 0).any()],
                 "cols_with_nulls": {c: round(float(v), 3) for c, v in p.isna().mean().items() if v > 0}}
            if "sum_insured" in p:
                r["si_gt_50x_median"] = int((p["sum_insured"] > 50 * p["sum_insured"].median()).sum())
            rows.append(r)
            for c in p.select_dtypes(["object", "string"]).columns.difference(list(ID_DATE)):
                s = p[c].dropna().astype(str)
                norm = s.str.strip().str.lower()
                if s.nunique() != norm.nunique():
                    text.append({"line": line, "kind": kind, "column": c, "raw_levels": s.nunique(), "normalised_levels": norm.nunique(),
                                 "examples": sorted(s.unique())[:10]})
    for t in ("customers", "addresses"):
        x = load_table(t, EMG)
        for c in x.select_dtypes(["object", "string"]).columns.difference(list(ID_DATE)):
            s = x[c].dropna().astype(str)
            norm = s.str.strip().str.lower()
            if s.nunique() != norm.nunique() or s.nunique() > 60:
                text.append({"line": t, "kind": "-", "column": c, "raw_levels": s.nunique(), "normalised_levels": norm.nunique(),
                             "examples": s.value_counts().head(10).index.tolist()})
    q, tx = pd.DataFrame(rows), pd.DataFrame(text)
    save(q, "9_quality"); save(tx, "9_text_variants")
    print(q.T.to_string(header=False)); print("\ncase / spelling variants and free text:"); print(tx.to_string(index=False) if len(tx) else "none")


print("\n" + "\n".join(status))
print(f"saved {OUT.relative_to(REPO)}")
