"""Office premium build-up for the 20 emerging reference policies (docs/07). Uses the Week 2 final predictions.

Writes outputs/reports/w2_office_premium.csv, w2_office_premium_sensitivity.csv and w2_office_premium_example.md.

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\make_office_premium.py
"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data import REPO, load_table
from src.office_premium import build_table, build_up, expense_rates

PRED = REPO / "outputs" / "submissions" / "w2_20261028_2009_82f4d73.csv"     # Week 2 final (decision #36)
EXAMPLE = "B00229821"                                                         # broker, buildings: shows every component

rates = expense_rates(load_table("expenses", "w2"), "emerging")
ref = load_table("reference_policies", "w2")
pred = pd.read_csv(PRED, dtype={"row_id": str})
t = build_table(ref, pred, rates)
t.to_csv(REPO / "outputs" / "reports" / "w2_office_premium.csv", index=False)

sens = []
for tax in ("within", "on_top"):
    for fixed in ("per_policy", "pro_rata"):
        x = build_table(ref, pred, rates, tax_mode=tax, fixed_mode=fixed)
        sens.append({"tax": tax, "fixed": fixed, "total_office_premium": x["office_premium"].sum(),
                     "vs_central": x["office_premium"].sum() / t["office_premium"].sum(),
                     "median_office_vs_written": x["office_vs_written"].median()})
S = pd.DataFrame(sens)
S.to_csv(REPO / "outputs" / "reports" / "w2_office_premium_sensitivity.csv", index=False)

r = t[t.row_id == EXAMPLE].iloc[0]
comm = rates[f"commission_{r.channel}"]
den = 1 - rates["variable_expense"] - comm - rates["premium_tax"] - rates["target_profit_margin"]
lines = [f"# Worked example — {EXAMPLE} ({r.line}, {r.channel}, exposure {r.exposure:.4f})", "",
         f"1. Risk premium (model, row total): **{r.risk_premium:,.2f} CU**",
         f"2. Catastrophe reinsurance: {rates['cat_reinsurance']:.0%} × RP = {r.cat_reinsurance:,.2f}",
         f"3. Fixed expense per policy: {r.fixed_expense:,.2f}",
         f"4. Loaded cost = {r.risk_premium:,.2f} + {r.cat_reinsurance:,.2f} + {r.fixed_expense:,.2f} = {r.risk_premium + r.cat_reinsurance + r.fixed_expense:,.2f}",
         f"5. Premium-based loadings: variable {rates['variable_expense']:.0%} + commission ({r.channel}) {comm:.0%} + premium tax {rates['premium_tax']:.0%} "
         f"+ profit {rates['target_profit_margin']:.0%} = {1 - den:.0%}",
         f"6. Office premium = loaded cost / (1 − {1 - den:.2f}) = **{r.office_premium:,.2f} CU**",
         f"7. Of which: variable {r.variable_expense:,.2f}, commission {r.commission:,.2f}, tax {r.premium_tax:,.2f}, profit {r.profit:,.2f} (components sum to the premium)",
         f"8. Current written premium {r.written_premium:,.2f} → office / written = {r.office_vs_written:.2f}"]
(REPO / "outputs" / "reports" / "w2_office_premium_example.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

pd.set_option("display.width", 220)
cols = ["row_id", "line", "channel", "exposure", "risk_premium", "cat_reinsurance", "fixed_expense", "variable_expense",
        "commission", "profit", "premium_tax", "office_premium", "written_premium", "office_vs_written"]
print(t[cols].round(2).to_string(index=False))
print("\nsensitivity:"); print(S.round(3).to_string(index=False))
print("\n" + "\n".join(lines))
