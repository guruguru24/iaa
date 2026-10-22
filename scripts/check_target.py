"""Print a compact view of the target checks (run `python -m src.eda --sections target` first).

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\check_target.py
"""
import pandas as pd

d = "outputs/reports/w1_eda/"
pd.set_option("display.width", 200)

s = pd.read_csv(d + "target__amounts_by_status.csv")
m = pd.read_csv(d + "target__claim_matching.csv")
p = pd.read_csv(d + "target__status_by_peril.csv")

print("=== 1. Coverage: all claims included? (expect 163150 each)")
print("sum over status :", s["n"].sum())
print("sum over line   :", m["claims"].sum())
print("sum over peril  :", p.drop(columns=["line", "peril"]).to_numpy().sum())

print("\n=== 2. Claim matching by line")
print(m.T.to_string())

print("\n=== 3. Amounts by status")
print(s.round(3).to_string())

print("\n=== 4. Excess check (net of excess -> frequency and amounts fall as excess rises)")
ex = pd.read_csv(d + "target__excess_check.csv")
print(ex[["line", "excess", "count", "1%", "5%", "50%", "frequency_all_reported"]].round(3).to_string())

print("\n=== 5. Reconciliation claim-level vs row-level (diff should be 0)")
r = pd.read_csv(d + "target__reconciliation.csv")
print(r[["line", "policy_year", "row_level_net", "diff", "recoveries", "open_share_of_net"]].round(3).to_string())
