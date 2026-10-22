"""Follow-up checks: (a) is T->W a pure rename of timber, and is timber's risk shift real? (b) the 11.9M BI claim (DQ8).

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\check_timber_bi.py
"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data import load_claims, load_policies
from src.eda import burning_cost_table, line_frame

pd.set_option("display.width", 220)
claims = load_claims("w1")

print("=== a1. Test file: which raw construction_type spellings appear in 2025?")
te = load_policies("buildings", kind="test")
print(te["construction_type"].value_counts(dropna=False).to_string())

print("\n=== a2. Timber (T 2021-22, W 2023-24) relative to brick within each year (same-year trend & maturity)")
df = line_frame("buildings", "w1", claims, clean=False)
u = df["construction_type"].str.upper()
df["ct"] = u.replace({"T": "TIMBER", "W": "TIMBER"})
t = burning_cost_table(df, ["policy_year", "ct"])
for col in ("frequency", "severity", "burning_cost"):
    piv = t.pivot(index="policy_year", columns="ct", values=col)
    print(f"--- {col} relative to BRICK"); print(piv.div(piv["BRICK"], axis=0).round(3).to_string())

print("\n=== a3. Timber largest claims by year (does one big fire drive W's severity?)")
c = claims[claims.line == "buildings"].merge(df[["row_id", "policy_id", "period_start", "period_end", "ct", "policy_year"]], on="policy_id")
c = c[(c.loss_date >= c.period_start) & (c.loss_date <= c.period_end) & (c.ct == "TIMBER")]
c["net"] = (c.incurred_to_date - c.recoveries).clip(lower=0)
print(c.groupby("policy_year").net.agg(["size", "sum", "max", lambda s: s.nlargest(5).sum() / s.sum()]).round(3).to_string())

print("\n=== b. Motor bodily_injury: open claims, incurred/paid ratio, and the 11.9M claim")
bi = claims[(claims.line == "motor") & (claims.peril == "bodily_injury") & (claims.status != "repudiated")].copy()
bi["ratio"] = bi.incurred_to_date / bi.paid_to_date.where(bi.paid_to_date > 0)
bi["loss_year"] = bi.loss_date.dt.year
print("incurred quantiles by status:")
print(bi.groupby("status").incurred_to_date.quantile([.5, .9, .99, .999, 1]).unstack().round(0).to_string())
print("open BI incurred/paid ratio quantiles by loss year:")
print(bi[bi.status == "open"].groupby("loss_year").ratio.quantile([.5, .9, .99, 1]).unstack().round(2).to_string())
print("top 10 BI by incurred:")
print(bi.nlargest(10, "incurred_to_date")[["claim_id", "loss_date", "report_date", "status", "paid_to_date", "incurred_to_date", "ratio"]].round({"paid_to_date": 2, "incurred_to_date": 2, "ratio": 2}).to_string())
