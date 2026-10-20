"""Actuarial EDA and integrity checks (docs/01_eda_checklist.md sections 1-8).

Run (writes CSVs to outputs/reports/{market}_eda/ and a short summary to stdout):
    python -m src.eda --market w1                       # all sections
    python -m src.eda --market w1 --sections integrity target
    python -m src.eda --market w1 --exposure raw        # use supplied exposure instead of date-implied

Sections (checklist number): integrity (1), target (2), exposure (3), freqsev (4, 7), large (5), dev (6), oneway (8).
Every table is computed from the raw files; nothing is written back to data/raw.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LINES, REPO, ROW_PREFIX, exposure_from_dates, flag_exposure, load_claims, load_policies, load_table
from src.target import build_row_target, claim_level_net, match_claims_to_rows

ID_COLS = ["row_id", "policy_id", "customer_id", "address_id", "line"]
LEAKAGE_COLS = ["renewal_decision"]  # train-only, known after the period (DQ14)
NON_FACTOR_COLS = ID_COLS + LEAKAGE_COLS + ["period_start", "period_end", "exposure", "exposure_used",
                                            "exposure_dates", "exposure_flag", "policy_year"]
SECTIONS = ("integrity", "target", "exposure", "freqsev", "large", "dev", "oneway")


# ---------------------------------------------------------------- building blocks

def line_frame(line: str, market: str = "w1", claims: pd.DataFrame | None = None,
               exposure: str = "dates") -> pd.DataFrame:
    """Train policy rows + row target + exposure columns. exposure='dates' -> (days+1)/365, 'raw' -> as supplied."""
    p = load_policies(line, market)
    claims = load_claims(market) if claims is None else claims
    p["policy_year"] = p["period_start"].dt.year
    p["exposure_dates"] = exposure_from_dates(p["period_start"], p["period_end"])
    p["exposure_flag"] = flag_exposure(p)
    p["exposure_used"] = p["exposure_dates"] if exposure == "dates" else p["exposure"]
    t = build_row_target(p, claims[claims["line"] == line])
    return p.join(t.drop(columns="row_id"))


def claim_frame(line: str, rows: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Claim-level net amounts for one line, with the matched row's policy_year."""
    c = claim_level_net(match_claims_to_rows(claims[claims["line"] == line], rows))
    return c.merge(rows[["row_id", "policy_year"]], on="row_id", how="left")


def burning_cost_table(df: pd.DataFrame, by, exposure_col: str = "exposure_used",
                       target_col: str = "target_net", count_col: str = "n_claims_net") -> pd.DataFrame:
    """Exposure, claim count, net cost, frequency, severity, burning cost and loss ratio by group."""
    by = [by] if isinstance(by, str) else list(by)
    agg = {"rows": (target_col, "size"), "exposure": (exposure_col, "sum"), "n_claims": (count_col, "sum"),
           "net": (target_col, "sum"), "rows_with_claim": (count_col, lambda s: (s > 0).sum())}
    if "written_premium" in df.columns:
        agg["premium"] = ("written_premium", "sum")
    g = df.groupby(by, observed=True, dropna=False).agg(**agg)
    g["frequency"] = g["n_claims"] / g["exposure"]
    g["severity"] = g["net"] / g["n_claims"].where(g["n_claims"] > 0)
    g["burning_cost"] = g["net"] / g["exposure"]
    g["share_rows_with_claim"] = g["rows_with_claim"] / g["rows"]
    if "premium" in g:
        g["loss_ratio"] = g["net"] / g["premium"]
    return g.reset_index()


def band(s: pd.Series, n_bins: int = 10) -> pd.Series:
    """Quantile bands for numeric columns with many values; otherwise the value itself. NaN -> 'missing'."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > n_bins:
        out = pd.qcut(s, n_bins, duplicates="drop").astype(str)
    else:
        out = s.astype(str)
    return out.where(s.notna(), "missing")


def one_way(df: pd.DataFrame, factor: str, n_bins: int = 10, **kw) -> pd.DataFrame:
    """Observed burning cost by factor level, with relativity to the overall burning cost."""
    d = df.assign(level=band(df[factor], n_bins))
    g = burning_cost_table(d, "level", **kw)
    overall = g["net"].sum() / g["exposure"].sum()
    g["relativity"] = g["burning_cost"] / overall
    g.insert(0, "factor", factor)
    return g


def lag_cdf(lags_days) -> callable:
    """Empirical CDF of reporting delay in days."""
    lags = np.sort(np.asarray(lags_days, float))
    return lambda x: np.searchsorted(lags, np.asarray(x, float), side="right") / len(lags)


def ibnr_count_estimate(claims: pd.DataFrame, cutoff: pd.Timestamp, mature_years) -> pd.DataFrame:
    """Reported vs estimated ultimate claim counts by loss year.

    Each reported claim is weighted by 1 / F(cutoff - loss_date), with F the reporting-delay CDF from
    loss years in `mature_years`. Assumes the delay distribution is stable over time (check this).
    """
    c = claims.assign(loss_year=claims["loss_date"].dt.year,
                      lag=(claims["report_date"] - claims["loss_date"]).dt.days,
                      window=(cutoff - claims["loss_date"]).dt.days)
    F = lag_cdf(c.loc[c["loss_year"].isin(mature_years), "lag"])
    c["p_reported"] = np.clip(F(c["window"]), 1e-3, 1.0)
    g = c.groupby("loss_year").agg(reported=("claim_id", "size"), est_ultimate=("p_reported", lambda p: (1 / p).sum()))
    g["count_dev_factor"] = g["est_ultimate"] / g["reported"]
    return g.reset_index()


# ---------------------------------------------------------------- sections

def sec_integrity(market: str) -> dict[str, pd.DataFrame]:
    out, nulls, keys, ranges, unseen, si = {}, [], [], [], [], []
    files = sorted((REPO / "data" / "raw" / market).glob("*.parquet"))
    out["files"] = pd.DataFrame([{"file": f.name, **dict(zip(("rows", "cols"), pd.read_parquet(f).shape))} for f in files])
    cust = load_table("customers", market)
    addr = load_table("addresses", market)
    for name, df in (("customers", cust), ("addresses", addr), ("weather", load_table("weather", market))):
        nulls += [{"table": name, "column": c, "n_null": n, "share": n / len(df)} for c, n in df.isna().sum().items() if n]
    keys.append({"check": "customers.customer_id unique", "value": cust["customer_id"].is_unique})
    keys.append({"check": "addresses.address_id unique", "value": addr["address_id"].is_unique})
    for line in LINES:
        tr, te = load_policies(line, market), load_policies(line, market, "test")
        for kind, df in (("train", tr), ("test", te)):
            tag = f"{line}/{kind}"
            nulls += [{"table": tag, "column": c, "n_null": n, "share": n / len(df)} for c, n in df.isna().sum().items() if n]
            s = df.sort_values(["policy_id", "period_start"])
            prev_end = s.groupby("policy_id")["period_end"].shift()
            keys += [
                {"check": f"{tag} rows", "value": len(df)},
                {"check": f"{tag} row_id unique", "value": df["row_id"].is_unique},
                {"check": f"{tag} row_id prefix == {ROW_PREFIX[line]}", "value": df["row_id"].str[0].eq(ROW_PREFIX[line]).all()},
                {"check": f"{tag} exact duplicates excl. row_id", "value": int(df.drop(columns="row_id").duplicated().sum())},
                {"check": f"{tag} period_end < period_start", "value": int((df["period_end"] < df["period_start"]).sum())},
                {"check": f"{tag} overlapping periods within policy", "value": int((s["period_start"] <= prev_end).sum())},
                {"check": f"{tag} gaps > 1 day within policy", "value": int(((s["period_start"] - prev_end).dt.days > 1).sum())},
                {"check": f"{tag} orphan customer_id", "value": int((~df["customer_id"].isin(cust["customer_id"])).sum())},
                {"check": f"{tag} orphan address_id", "value": int((~df["address_id"].isin(addr["address_id"])).sum())},
            ]
            for flag, n in flag_exposure(df).value_counts().items():
                keys.append({"check": f"{tag} exposure_flag={flag}", "value": int(n)})
            d = df.select_dtypes("number").describe(percentiles=[.01, .5, .99]).T
            ranges.append(d.assign(table=tag, n_negative=(df[d.index] < 0).sum()).reset_index(names="column"))
            if "sum_insured" in df:
                med = df["sum_insured"].median()
                si.append({"table": tag, "median": med, "n_gt_50x_median": int((df["sum_insured"] > 50 * med).sum()),
                           "max": df["sum_insured"].max()})
        keys.append({"check": f"{line} train/test row_id overlap", "value": len(set(tr["row_id"]) & set(te["row_id"]))})
        keys.append({"check": f"{line} test policies seen in train", "value": te["policy_id"].isin(tr["policy_id"]).sum()})
        keys.append({"check": f"{line} test-only columns", "value": sorted(set(te.columns) - set(tr.columns))})
        keys.append({"check": f"{line} train-only columns", "value": sorted(set(tr.columns) - set(te.columns))})
        for c in tr.columns.drop(ID_COLS + LEAKAGE_COLS, errors="ignore"):
            if tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c]) or pd.api.types.is_bool_dtype(tr[c]):
                lv_tr, lv_te = set(tr[c].dropna().unique()), set(te[c].dropna().unique())
                unseen.append({"line": line, "column": c, "n_levels_train": len(lv_tr), "n_unseen_in_test": len(lv_te - lv_tr),
                               "unseen_examples": sorted(map(str, lv_te - lv_tr))[:5],
                               "levels_if_few": sorted(map(str, lv_tr)) if len(lv_tr) <= 15 else None})
    out |= {"nulls": pd.DataFrame(nulls), "keys": pd.DataFrame(keys), "ranges": pd.concat(ranges, ignore_index=True),
            "levels": pd.DataFrame(unseen), "sum_insured_outliers": pd.DataFrame(si)}
    return out


def sec_target(market: str, frames: dict) -> dict[str, pd.DataFrame]:
    claims = frames["claims"]
    out = {"status_by_peril": pd.crosstab([claims["line"], claims["peril"]], claims["status"]).reset_index()}
    amounts = []
    for status, g in claims.groupby("status"):
        amounts.append({"status": status, "n": len(g), "paid": g["paid_to_date"].sum(), "incurred": g["incurred_to_date"].sum(),
                        "recoveries": g["recoveries"].sum(), "share_paid_eq_incurred": np.isclose(g["paid_to_date"], g["incurred_to_date"]).mean(),
                        "n_paid_gt_incurred": int((g["paid_to_date"] > g["incurred_to_date"] + 1e-6).sum()),
                        "n_recov_gt_paid": int((g["recoveries"] > g["paid_to_date"] + 1e-6).sum())})
    out["amounts_by_status"] = pd.DataFrame(amounts)
    match, recon = [], []
    for line in LINES:
        rows, c = frames[line], frames[f"claims_{line}"]
        first_last = rows.groupby("policy_id").agg(first=("period_start", "min"), last=("period_end", "max"))
        un = c[~c["in_period"]].join(first_last, on="policy_id")
        nd = c[c["in_period"] & ~c["declined"]]
        unfloored = np.maximum(nd["gross"] - nd["excess"], 0) - nd["recoveries"]
        match.append({"line": line, "claims": len(c), "policy_not_found": int(un["first"].isna().sum()),
                      "out_of_period": int((~c["in_period"]).sum()),
                      "before_first_row": int((un["loss_date"] < un["first"]).sum()),
                      "after_last_row": int((un["loss_date"] > un["last"]).sum()),
                      "out_of_period_gross": un["incurred_to_date"].sum(),
                      "share_below_excess": (nd["gross"] < nd["excess"]).mean(),
                      "n_negative_before_floor": int((unfloored < 0).sum()),
                      "floor_effect_on_net": nd["net"].sum() - unfloored.sum()})
        a = c[c["in_period"]].groupby("policy_year")["net"].sum()
        b = rows.groupby("policy_year")["target_net"].sum()
        for y in b.index:
            ok = nd[nd["policy_year"] == y]
            recon.append({"line": line, "policy_year": y, "claim_level_net": a.get(y, 0.0), "row_level_net": b[y],
                          "diff": a.get(y, 0.0) - b[y], "gross_incurred": ok["gross"].sum(),
                          "excess_removed": (ok["gross"] - np.maximum(ok["gross"] - ok["excess"], 0)).sum(),
                          "recoveries": ok["recoveries"].sum(),
                          "open_share_of_net": ok.loc[ok["status"] == "open", "net"].sum() / ok["net"].sum()})
    out["claim_matching"] = pd.DataFrame(match)
    out["reconciliation"] = pd.DataFrame(recon)
    return out


def sec_exposure(market: str, frames: dict) -> dict[str, pd.DataFrame]:
    rows = []
    for line in LINES:
        for kind, df in (("train", frames[line]), ("test", load_policies(line, market, "test"))):
            df = df.assign(policy_year=df["period_start"].dt.year,
                           exposure_dates=exposure_from_dates(df["period_start"], df["period_end"]))
            g = df.groupby("policy_year").agg(rows=("row_id", "size"), exposure_raw=("exposure", "sum"),
                                              exposure_dates=("exposure_dates", "sum"),
                                              share_partial=("exposure_dates", lambda s: (s < 0.999).mean()),
                                              median_exposure=("exposure_dates", "median"))
            rows.append(g.reset_index().assign(line=line, kind=kind))
    out = {"exposure_by_year": pd.concat(rows, ignore_index=True)}
    tr = pd.concat([frames[l].assign(line=l) for l in LINES])
    if "renewal_decision" in tr:
        out["partial_by_renewal_decision"] = (tr.assign(partial=tr["exposure_dates"] < 0.999)
                                              .groupby(["line", "renewal_decision"]).agg(rows=("row_id", "size"), share_partial=("partial", "mean"))
                                              .reset_index())
    return out


def sec_freqsev(market: str, frames: dict) -> dict[str, pd.DataFrame]:
    by_year, by_peril, mix = [], [], []
    for line in LINES:
        df, c = frames[line], frames[f"claims_{line}"]
        by_year.append(burning_cost_table(df, "policy_year").assign(line=line))
        expo = df.groupby("policy_year")["exposure_used"].sum()
        cp = c[c["in_period"] & ~c["declined"]]
        g = cp.groupby(["policy_year", "peril"]).agg(n_reported=("claim_id", "size"), n_net_pos=("is_counted", "sum"),
                                                     net=("net", "sum"),
                                                     q50=("net", lambda s: s[s > 0].quantile(.5)),
                                                     q90=("net", lambda s: s[s > 0].quantile(.9)),
                                                     q99=("net", lambda s: s[s > 0].quantile(.99)),
                                                     q999=("net", lambda s: s[s > 0].quantile(.999))).reset_index()
        g["frequency"] = g["n_net_pos"] / g["policy_year"].map(expo)
        g["severity"] = g["net"] / g["n_net_pos"]
        by_peril.append(g.assign(line=line))
        num = [c_ for c_ in ("sum_insured", "vehicle_value_group", "vehicle_age", "driver_age", "policyholder_age",
                             "annual_mileage", "written_premium") if c_ in df]
        m = df.groupby("policy_year")[num].mean()
        m["premium_per_exposure"] = df.groupby("policy_year")["written_premium"].sum() / expo
        for col in ("channel", "new_business"):
            sh = pd.crosstab(df["policy_year"], df[col], normalize="index").add_prefix(f"{col}=")
            m = m.join(sh)
        mix.append(m.reset_index().assign(line=line))
    t = pd.concat(by_year, ignore_index=True)
    for col in ("frequency", "severity", "burning_cost"):
        t[f"{col}_yoy"] = t.groupby("line")[col].pct_change()
    return {"freqsev_by_year": t, "freqsev_by_peril_year": pd.concat(by_peril, ignore_index=True),
            "mix_by_year": pd.concat(mix, ignore_index=True)}


def sec_large(market: str, frames: dict, top_n: int = 20) -> dict[str, pd.DataFrame]:
    top, share, events = [], [], []
    addr = load_table("addresses", market)[["address_id", "region"]]
    for line in LINES:
        c = frames[f"claims_{line}"]
        c = c[c["in_period"] & ~c["declined"]]
        top.append(c.nlargest(top_n, "net")[["claim_id", "policy_id", "row_id", "peril", "loss_date", "report_date", "status",
                                              "paid_to_date", "incurred_to_date", "recoveries", "net"]].assign(line=line))
        s = c["net"].sort_values(ascending=False)
        for q in (0.001, 0.01, 0.05):
            k = max(1, int(len(s) * q))
            share.append({"line": line, "top_share_of_claims": q, "n_claims": k, "threshold": s.iloc[k - 1],
                          "share_of_net": s.iloc[:k].sum() / s.sum()})
        for thr in (25_000, 50_000, 100_000, 250_000):
            share.append({"line": line, "threshold": thr, "n_claims": int((s > thr).sum()),
                          "excess_over_threshold_share": (s - thr).clip(lower=0).sum() / s.sum()})
        e = c.merge(frames[line][["row_id", "address_id"]], on="row_id").merge(addr, on="address_id", how="left")
        ev = e.groupby(["loss_date", "peril", "region"]).agg(n_claims=("claim_id", "size"), net=("net", "sum")).reset_index()
        events.append(ev.nlargest(top_n, "n_claims").assign(line=line))
    return {"large_top_claims": pd.concat(top, ignore_index=True), "large_tail_share": pd.DataFrame(share),
            "large_event_clusters": pd.concat(events, ignore_index=True)}


def sec_dev(market: str, frames: dict, mature_years=(2021, 2022)) -> dict[str, pd.DataFrame]:
    claims = frames["claims"]
    cutoff = claims["report_date"].max()
    lag = claims.assign(lag=(claims["report_date"] - claims["loss_date"]).dt.days, loss_year=claims["loss_date"].dt.year)
    q = lag.groupby(["line", "peril", "loss_year"])["lag"].describe(percentiles=[.5, .9, .95, .99]).reset_index()
    ibnr, maturity = [], []
    for line in LINES:
        c = claims[claims["line"] == line]
        ibnr.append(ibnr_count_estimate(c, cutoff, mature_years).assign(line=line))
        cf = frames[f"claims_{line}"]
        cf = cf[cf["in_period"] & ~cf["declined"]]
        g = cf.groupby(["policy_year", "peril"]).agg(n=("claim_id", "size"), n_open=("status", lambda s: (s == "open").sum()),
                                                     paid=("paid_to_date", "sum"), incurred=("gross", "sum"), net=("net", "sum"))
        g["paid_to_incurred"] = g["paid"] / g["incurred"]
        g["open_share_count"] = g["n_open"] / g["n"]
        maturity.append(g.reset_index().assign(line=line))
    return {"dev_report_lag": q, "dev_ibnr_count": pd.concat(ibnr, ignore_index=True),
            "dev_maturity": pd.concat(maturity, ignore_index=True),
            "dev_cutoff": pd.DataFrame({"cutoff_report_date": [cutoff], "mature_years_for_lag_cdf": [list(mature_years)]})}


def sec_oneway(market: str, frames: dict, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    addr = load_table("addresses", market)
    cust = load_table("customers", market).drop(columns=["surname"], errors="ignore")
    out = {}
    for line in LINES:
        df = frames[line].merge(addr, on="address_id", how="left", suffixes=("", "_addr"))
        df = df.merge(cust, on="customer_id", how="left", suffixes=("", "_cust"))
        factors = [c for c in df.columns if c not in NON_FACTOR_COLS and c not in
                   ("target_net", "gross_incurred", "open_net", "n_claims_net", "n_claims_reported", "date_of_birth")]
        out[f"oneway_{line}"] = pd.concat([one_way(df, f, n_bins) for f in factors], ignore_index=True)
    return out


# ---------------------------------------------------------------- runner

def run(market: str = "w1", sections=SECTIONS, exposure: str = "dates") -> Path:
    out_dir = REPO / "outputs" / "reports" / f"{market}_eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    if set(sections) - {"integrity"}:
        frames["claims"] = load_claims(market)
        for line in LINES:
            frames[line] = line_frame(line, market, frames["claims"], exposure)
            frames[f"claims_{line}"] = claim_frame(line, frames[line], frames["claims"])
    funcs = {"integrity": lambda: sec_integrity(market), "target": lambda: sec_target(market, frames),
             "exposure": lambda: sec_exposure(market, frames), "freqsev": lambda: sec_freqsev(market, frames),
             "large": lambda: sec_large(market, frames), "dev": lambda: sec_dev(market, frames),
             "oneway": lambda: sec_oneway(market, frames)}
    for s in sections:
        for name, table in funcs[s]().items():
            path = out_dir / f"{s}__{name}.csv"
            table.to_csv(path, index=False)
            print(f"[{s}] {path.relative_to(REPO)}  ({len(table)} rows)")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="w1")
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=SECTIONS)
    ap.add_argument("--exposure", default="dates", choices=["dates", "raw"])
    a = ap.parse_args(argv)
    run(a.market, a.sections, a.exposure)


if __name__ == "__main__":
    main()
