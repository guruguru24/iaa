# Assumptions register

| # | Assumption | Line | Low | Central | High | Basis / evidence | Sensitivity (D², total cost) | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| A1 | Claims trend 2024 → 2025 (frequency × severity) | all | | | | Mix-adjusted GLM year effects | | G3 | Open |
| A2 | Development to ultimate for immature years | all | | | | Reporting delay / development analysis | | G3 | Open |
| A3 | Large-loss threshold and loading | all | | | | Tail analysis | | G2 | Open |
| A4 | Test-row exposure basis: test rows are 2025 periods, partial-year allowed (min 0.0027), exposure = (days+1)/365 exactly; prediction = expected cost for the row's exposure | all | | (days+1)/365 | | Test file inspection 2026-10-20 (0 mismatches) | | G1 | Verified, pending approval |
| A5 | Excess applied per claim = voluntary + compulsory excess of the row containing the loss date | all | | vol+comp | | Claims < excess observed; no excess field in claims | | G1 | Pending approval |
| A6 | Open claims: incurred_to_date used as current estimate (case reserve), before development step | all | | incurred | | Closed: paid = incurred 100% | | G1/G3 | Pending approval |
