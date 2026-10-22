"""G3 evidence: developed level by policy year, and crude vs mix-adjusted year effects (frequency, severity, burning cost).

Usage (from the repo root; takes a few minutes, one line at a time):
    .venv\\Scripts\\python scripts\\view_development_trend.py
"""
import sys
import time

sys.path.insert(0, ".")
import pandas as pd

from src.data import LINES, load_claims, load_table
from src.development import IBNER_SCENARIOS, apply_to_rows, count_factors, develop, peril_year_table, year_dev_factors
from src.eda import claim_frame, line_frame
from src.trend import TREND_FACTORS, annual_trend, row_capped_target, year_effects

pd.set_option("display.width", 220)
CAPS = {"motor": 100_000, "buildings": 100_000, "contents": 50_000}  # for trend estimation only (A3 still open)

claims = load_claims("w1")
addr = load_table("addresses")[["address_id", "region", "flood_zone", "property_type"]]
cfac = count_factors(claims)

for line in LINES:
    t0 = time.time()
    rows = line_frame(line, "w1", claims).merge(addr, on="address_id", how="left")
    cf = claim_frame(line, rows, claims)
    table = peril_year_table(cf)
    expo = rows.groupby("policy_year")["exposure_used"].sum()

    print(f"\n{'=' * 25} {line}")
    print("--- 1. Developed level by policy year (IBNER scenarios are placeholders)")
    lvl = pd.DataFrame({"exposure": expo, "reported_bc": table.groupby("policy_year")["net"].sum() / expo})
    for scen, ib in IBNER_SCENARIOS.items():
        yf = year_dev_factors(develop(table, cfac, ib)).set_index("policy_year")
        lvl[f"dev_{scen}"] = yf["dev_factor"]
    lvl["developed_bc_central"] = lvl["reported_bc"] * lvl["dev_central"]
    print(lvl.round(3).to_string())

    yf_c = year_dev_factors(develop(table, cfac, 1.0))
    rows["target_cap"] = row_capped_target(rows, cf, CAPS[line])
    rows["target_cap_dev"] = apply_to_rows(rows, line, yf_c, col="target_cap")
    rows["pp"] = rows["target_cap_dev"] / rows["exposure_used"]
    rows["freq"] = rows["n_claims_net"] / rows["exposure_used"]
    factors = TREND_FACTORS[line]

    cnt = cfac[cfac["line"] == line].groupby("policy_year")[["reported", "est_ultimate"]].sum()
    cnt = cnt["est_ultimate"] / cnt["reported"]
    fe = year_effects(rows, factors, "freq", "exposure_used", "poisson").set_index("policy_year")
    fe = fe.mul(cnt / cnt.loc[2021], axis=0)  # count-developed

    # keep only claim keys/amounts: claim_frame also carries the row's excess columns, which would clash on merge
    c = cf.loc[cf["in_period"] & ~cf["declined"] & (cf["net"] > 0), ["claim_id", "row_id", "policy_year", "net"]].merge(
        rows[["row_id"] + factors], on="row_id")
    c["sev_cap"] = c["net"].clip(upper=CAPS[line])
    se = year_effects(c, factors, "sev_cap", None, "gamma").set_index("policy_year")
    be = year_effects(rows, factors, "pp", "exposure_used", "tweedie").set_index("policy_year")

    eff = pd.concat({"freq(dev)": fe, "sev(cap)": se, "bc(cap,dev)": be}, axis=1)
    print(f"--- 2. Year effects vs 2021 (caps: {CAPS[line]:,}; severity not IBNER-adjusted)")
    print(eff.round(3).to_string())
    tr = {k: {"2021-23": annual_trend(eff[(k, "mix_adjusted")], [2021, 2022, 2023]),
              "2021-24": annual_trend(eff[(k, "mix_adjusted")], [2021, 2022, 2023, 2024])}
          for k in ("freq(dev)", "sev(cap)", "bc(cap,dev)")}
    print("--- 3. Annual mix-adjusted trend"); print(pd.DataFrame(tr).round(4).to_string())
    print(f"({time.time() - t0:.0f}s)")
    del rows, cf, c
