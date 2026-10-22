"""G3 follow-ups:
 1. motor: mix-adjusted frequency and severity trend BY PERIL (line-level severity looked flat while peril severities rose)
 2. buildings: is the 2024 subsidence jump (count factor 1.64) supported by dry summers? (weather by year)
 3. buildings/contents: trend with vs without sum_insured in the GLM (SI is indexed, so it can absorb inflation)

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\check_trend_followups.py
"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data import load_claims, load_table
from src.development import count_factors, develop, peril_year_table, year_dev_factors, apply_to_rows
from src.eda import claim_frame, line_frame
from src.trend import TREND_FACTORS, annual_trend, row_capped_target, year_effects

pd.set_option("display.width", 220)
claims = load_claims("w1")
addr = load_table("addresses")[["address_id", "region", "flood_zone", "property_type"]]
cfac = count_factors(claims)
YEARS3, YEARS4 = [2021, 2022, 2023], [2021, 2022, 2023, 2024]


def trends(s):
    return f"2021-23 {annual_trend(s, YEARS3):+.3f} | 2021-24 {annual_trend(s, YEARS4):+.3f}"


print("=== 1. motor by peril (frequency count-developed per peril; severity capped 100k, not IBNER-adjusted)")
rows = line_frame("motor", "w1", claims).merge(addr, on="address_id", how="left")
cf = claim_frame("motor", rows, claims)
ok = cf[cf["in_period"] & ~cf["declined"] & (cf["net"] > 0)]
factors = TREND_FACTORS["motor"]
out = []
for peril, g in ok.groupby("peril"):
    rows["n_p"] = rows["row_id"].map(g.groupby("row_id").size()).fillna(0)
    rows["freq_p"] = rows["n_p"] / rows["exposure_used"]
    fe = year_effects(rows, factors, "freq_p", "exposure_used", "poisson").set_index("policy_year")["mix_adjusted"]
    f = cfac[(cfac.line == "motor") & (cfac.peril == peril)].set_index("policy_year")["count_dev_factor"]
    fe = fe * (f / f.loc[2021]).reindex(fe.index).fillna(1.0)
    c = g[["claim_id", "row_id", "policy_year", "net"]].merge(rows[["row_id"] + factors], on="row_id")
    c["sev"] = c["net"].clip(upper=100_000)
    se = year_effects(c, factors, "sev", None, "gamma").set_index("policy_year")["mix_adjusted"]
    out.append({"peril": peril, "claims": len(g), "freq_2024": round(fe.loc[2024], 3), "freq_trend": trends(fe),
                "sev_2024": round(se.loc[2024], 3), "sev_trend": trends(se)})
print(pd.DataFrame(out).to_string(index=False))
del rows, cf, ok

print("\n=== 2. buildings: dry summers vs subsidence by year")
rows = line_frame("buildings", "w1", claims)
w = load_table("weather")
rows = rows.merge(w[["address_id", "year", "dry_summer_index", "rainfall_mm", "freezing_days", "wind_exposure_index"]]
                  .rename(columns={"year": "policy_year"}), on=["address_id", "policy_year"], how="left")
cf = claim_frame("buildings", rows, claims)
sub = cf[cf["in_period"] & ~cf["declined"] & (cf["peril"] == "subsidence")]
f = cfac[(cfac.line == "buildings") & (cfac.peril == "subsidence")].set_index("policy_year")["count_dev_factor"]
y = rows.groupby("policy_year").apply(lambda d: pd.Series({
    "exposure": d.exposure_used.sum(),
    "dry_summer_mean": (d.dry_summer_index * d.exposure_used).sum() / d.exposure_used.sum(),
    "rainfall_mean": (d.rainfall_mm * d.exposure_used).sum() / d.exposure_used.sum(),
    "freezing_mean": (d.freezing_days * d.exposure_used).sum() / d.exposure_used.sum(),
    "wind_mean": (d.wind_exposure_index * d.exposure_used).sum() / d.exposure_used.sum()}), include_groups=False)
y["subs_reported"] = sub.groupby("policy_year").size()
y["subs_freq_reported"] = y["subs_reported"] / y["exposure"]
y["subs_freq_developed"] = y["subs_freq_reported"] * f.reindex(y.index)
print(y.round(4).to_string())
print("weather years available:", sorted(w["year"].unique()), "| 2025 mean dry_summer (all addresses):",
      round(w.loc[w.year == 2025, "dry_summer_index"].mean(), 4))
rows["dry_q"] = pd.qcut(rows["dry_summer_index"], 5, duplicates="drop").astype(str)
sub_rows = sub.groupby("row_id").size()
rows["n_sub"] = rows["row_id"].map(sub_rows).fillna(0)
m = rows[rows.policy_year <= 2023].groupby("dry_q").agg(exposure=("exposure_used", "sum"), n=("n_sub", "sum"))
m["freq"] = m["n"] / m["exposure"]
print("subsidence frequency by dry_summer_index quintile (2021-23, mature):"); print(m.round(5).to_string())
del rows, cf, w

print("\n=== 3. burning-cost trend with vs without sum_insured in the GLM (capped, developed)")
for line, cap in (("buildings", 100_000), ("contents", 50_000)):
    rows = line_frame(line, "w1", claims).merge(addr, on="address_id", how="left")
    cf = claim_frame(line, rows, claims)
    yf = year_dev_factors(develop(peril_year_table(cf), cfac, 1.0))
    rows["target_cap"] = row_capped_target(rows, cf, cap)
    rows["pp"] = apply_to_rows(rows, line, yf, col="target_cap") / rows["exposure_used"]
    for label, fs in (("with SI", TREND_FACTORS[line]), ("without SI", [x for x in TREND_FACTORS[line] if x != "sum_insured"])):
        be = year_effects(rows, fs, "pp", "exposure_used", "tweedie").set_index("policy_year")["mix_adjusted"]
        print(f"{line:9s} {label:10s} effects {be.round(3).tolist()} | {trends(be)}")
    del rows, cf
