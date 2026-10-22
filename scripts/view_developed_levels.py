"""Developed burning cost by line x policy year under the approved G3 method (decision #7), with low/high scenarios,
and each year's developed level trended to 2025 at the central trend (consistency check before modelling).

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\view_developed_levels.py
"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data import LINES, load_claims, load_config
from src.development import developed_table, development_factors, year_dev_factors
from src.eda import claim_frame, line_frame

pd.set_option("display.width", 220)
cfg = load_config()["assumptions"]
dev_cfg, trend = cfg["development"], cfg["trend_2025"]
claims = load_claims("w1")
factors = development_factors(claims, mature_years=dev_cfg["mature_years"])

print("=== 0. Development factors 2023-2024 by peril: count vs cost basis")
f = factors[factors.policy_year.isin([2023, 2024])]
print(f.pivot_table(index=["line", "peril"], columns="policy_year", values=["count_dev_factor", "cost_dev_factor"]).round(4).to_string())

for line in LINES:
    rows = line_frame(line, "w1", claims)
    cf = claim_frame(line, rows, claims)
    expo = rows.groupby("policy_year")["exposure_used"].sum()
    out = pd.DataFrame({"exposure": expo, "reported_bc": rows.groupby("policy_year")["target_net"].sum() / expo})
    for scen in ("low", "central", "high"):
        d = developed_table(cf, factors, dev_cfg, scen)
        yf = year_dev_factors(d).set_index("policy_year")
        out[f"dev_{scen}"] = yf["dev_factor"]
        out[f"bc_{scen}"] = yf["ultimate"] / expo
        if scen == "central" and "override" in d:
            ov = d[d["override"].notna()]
            if len(ov):
                print(f"\n--- {line}: expected-severity override (central)")
                print(ov[["policy_year", "peril", "n", "count_dev_factor", "net", "net_capped", "ultimate", "override"]].round(1).to_string(index=False))
    g = trend[line]["central"]
    out["bc_central_trended_2025"] = out["bc_central"] * (1 + g) ** (2025 - out.index)
    print(f"\n=== {line} (central trend {g:.1%}/yr)")
    print(out.round(3).to_string())
    del rows, cf
