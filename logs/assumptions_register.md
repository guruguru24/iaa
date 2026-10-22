# Assumptions register

| # | Assumption | Line | Low | Central | High | Basis / evidence | Sensitivity (D², total cost) | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| A1 | Claims trend 2024 → 2025 (frequency × severity), annual | all | bldg +4.5% / cont +3% / motor +1% | bldg +6% / cont +4% / motor +3% | bldg +8% / cont +5% / motor +6% | PRELIMINARY. Mix-adjusted GLM year effects (scripts/view_development_trend.py): bc trend 2021-23/2021-24 bldg +4.6%/+6.9%, cont +3.7%/+3.4%, motor +0.8%/+1.5%. Motor line-level severity flat vs peril severities +6–8% → peril analysis pending; SI in GLM may absorb inflation | | G3 | Open — follow-ups running |
| A2 | Development to ultimate for immature years (IBNR count by peril × IBNER on open claims) | all | IBNER 0.95 | IBNER 1.0 | IBNER 1.10 (BI, subsidence 1.30) | Count factors from report-lag CDF (2021–22). 2024 year dev factor motor 1.181/1.207/1.324, bldg 1.088/1.108/1.176, cont 1.023/1.032/1.051. IBNER not estimable (single valuation). Bldg 2024 developed level ~8% above 2021–23 trend line (subsidence factor 1.64) | | G3 | Open |
| A3 | Large-loss threshold and loading (motor sensitivity: CL30973963 at 11.89M vs 1.19M) | all | | | | Tail analysis (§5); decision #6 | | G2 | Open — threshold pending |
| A4 | Test-row exposure basis: test rows are 2025 periods, partial-year allowed (min 0.0027), exposure = (days+1)/365 exactly; prediction = expected cost for the row's exposure | all | | (days+1)/365 | | Test file inspection 2026-10-20 (0 mismatches) | | G1 | Verified, pending approval |
| A5 | ~~Excess applied per claim~~ → amounts already net of excess (official description, DQ15); excess not subtracted | all | | vol+comp | | Claims < excess observed; no excess field in claims | | G1 | Pending approval |
| A6 | Open claims: incurred_to_date used as current estimate (case reserve), before development step | all | | incurred | | Closed: paid = incurred 100% | | G1/G3 | Pending approval |
