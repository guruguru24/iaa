"""Evidence for the G2 cleaning rules (exposure, sum_insured, construction_type). Reads raw data; prints a compact view.

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\check_preprocess.py
"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data import load_claims, load_policies
from src.eda import burning_cost_table, line_frame
from src.preprocess import fix_construction_type, fix_exposure, fix_sum_insured, sum_insured_reference

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
claims = load_claims("w1")

print("=== 1. Exposure (DQ7): flagged rows and effect on burning cost")
for line in ("motor", "buildings", "contents"):
    df = line_frame(line, "w1", claims, exposure="raw")
    df["exposure_dates"] = fix_exposure(df)["exposure"]
    fl = df.groupby("exposure_flag").agg(rows=("row_id", "size"), exposure_raw=("exposure", "sum"),
                                         exposure_dates=("exposure_dates", "sum"), net=("target_net", "sum"),
                                         n_claims=("n_claims_net", "sum"))
    print(f"--- {line}"); print(fl.round(1).to_string())
    print("burning cost raw exposure:", round(df.target_net.sum() / df.exposure.sum(), 1),
          "| date exposure:", round(df.target_net.sum() / df.exposure_dates.sum(), 1))

print("\n=== 2. Sum insured (DQ10): do /1000 values look like normal rows?")
for line in ("buildings", "contents"):
    tr, te = load_policies(line), load_policies(line, kind="test")
    med = sum_insured_reference(tr)
    for kind, df in (("train", tr), ("test", te)):
        f = fix_sum_insured(df, med)
        flagged, normal = f[f.sum_insured_fixed], f[~f.sum_insured_fixed]
        q = [.01, .25, .5, .75, .99]
        print(f"--- {line}/{kind}: train median {med:,.0f}, flagged {len(flagged)} rows")
        print("  normal SI quantiles       :", normal.sum_insured.quantile(q).round(0).tolist())
        print("  flagged SI/1000 quantiles :", flagged.sum_insured.quantile(q).round(0).tolist())
        print("  premium normal vs flagged :", round(normal.written_premium.median(), 1), "vs", round(flagged.written_premium.median(), 1))
        between = df[(df.sum_insured > normal.sum_insured.max()) & (df.sum_insured <= 50 * med)]
        print("  rows between normal max and threshold (grey zone):", len(between))

print("\n=== 3. Construction type (DQ11): what are 'W' and 'T'?")
df = fix_construction_type(line_frame("buildings", "w1", claims))
df["raw_upper"] = df["construction_type_raw"].str.upper()
t = burning_cost_table(df, "raw_upper")
t["premium_per_exposure"] = t["premium"] / t["exposure"]
print(t[["raw_upper", "rows", "exposure", "n_claims", "frequency", "severity", "burning_cost", "premium_per_exposure"]].round(3).to_string())
c = claims[claims.line == "buildings"].merge(df[["policy_id", "raw_upper"]].drop_duplicates("policy_id"), on="policy_id")
print("peril mix by raw construction type (share of claims)")
print(pd.crosstab(c.raw_upper, c.peril, normalize="index").round(3).to_string())
print("rows by raw level x year (does the spelling change over time?)")
print(pd.crosstab(df.construction_type_raw, df.policy_year).to_string())
