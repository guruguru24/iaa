"""Compact view for G3 (trend, development) and the large-loss threshold (A3).

Usage (from the repo root):
    .venv\\Scripts\\python -m src.eda --market w1 --sections freqsev dev large
    .venv\\Scripts\\python scripts\\view_level_dev_large.py
"""
import pandas as pd

d = "outputs/reports/w1_eda/"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def show(title, df):
    print(f"\n=== {title}")
    print(df.to_string())


fy = pd.read_csv(d + "freqsev__freqsev_by_year.csv")
show("1. Level by policy year (exposure = from dates; 2024 immature)",
     fy[["line", "policy_year", "exposure", "n_claims", "frequency", "severity", "burning_cost",
         "frequency_yoy", "severity_yoy", "burning_cost_yoy", "loss_ratio"]].round(3))

fp = pd.read_csv(d + "freqsev__freqsev_by_peril_year.csv")
for col in ("frequency", "severity"):
    piv = fp.pivot_table(index=["line", "peril"], columns="policy_year", values=col)
    show(f"2. {col} by peril x policy year", piv.round(4 if col == "frequency" else 0))

mx = pd.read_csv(d + "freqsev__mix_by_year.csv")
keep = [c for c in ("line", "policy_year", "sum_insured", "vehicle_value_group", "driver_age", "policyholder_age",
                    "premium_per_exposure", "new_business=True") if c in mx]
show("3. Mix by year (means)", mx[keep].round(2))

ib = pd.read_csv(d + "dev__dev_ibnr_count.csv")
show("4. IBNR count: reported vs estimated ultimate by loss year (lag CDF from 2021-22)",
     ib.pivot_table(index=["line", "peril"], columns="loss_year", values="count_dev_factor").round(4))

mt = pd.read_csv(d + "dev__dev_maturity.csv")
agg = mt.groupby(["line", "policy_year"])[["n", "n_open", "paid", "incurred"]].sum()
agg["open_share_count"] = agg["n_open"] / agg["n"]
agg["paid_to_incurred"] = agg["paid"] / agg["incurred"]
show("5a. Maturity by line x policy year", agg[["open_share_count", "paid_to_incurred"]].unstack().round(3))
show("5b. paid/incurred by peril x policy year",
     mt.pivot_table(index=["line", "peril"], columns="policy_year", values="paid_to_incurred").round(3))

lg = pd.read_csv(d + "dev__dev_report_lag.csv")
show("6. Report lag (days) by peril, loss years 2021-22",
     lg[lg.loss_year.isin([2021, 2022])].groupby(["line", "peril"])[["50%", "90%", "99%", "max"]].mean().round(0))

ts = pd.read_csv(d + "large__large_tail_share.csv")
show("7a. Tail share: top x% of claims", ts[ts.top_share_of_claims.notna()][["line", "top_share_of_claims", "n_claims", "threshold", "share_of_net"]].round(3))
show("7b. Share of net above a cap", ts[ts.top_share_of_claims.isna()][["line", "threshold", "n_claims", "excess_over_threshold_share"]].round(4))

tc = pd.read_csv(d + "large__large_top_claims.csv")
show("8. Top 5 claims per line", tc.groupby("line").head(5)[["line", "claim_id", "peril", "loss_date", "status", "paid_to_date", "net"]].round(0))

ev = pd.read_csv(d + "large__large_event_clusters.csv")
show("9. Event clusters (same date, peril, region): top 5 per line", ev.groupby("line").head(5).round(0))
