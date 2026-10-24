# 07 — Office premium build-up (reference_policies, Week 2)

Judged by the panel (not on Kaggle). For each of the 20 `reference_policies` (emerging, 2025), show how the office premium is built
from our predicted risk premium using `expenses` (emerging column; developed column for comparison).

## Inputs
- Risk premium (RP) = our model prediction for the row (row total, CU).
- `expenses` items: fixed expense per policy (CU), variable expense (share of premium), commission by channel (share of premium),
  premium tax (share of premium), catastrophe reinsurance (share of **property** risk premium → buildings/contents only), target profit margin (share of premium).
- Confirm from the file: whether premium tax applies on top of the net premium or within it; whether fixed expense scales with exposure. Record as assumptions.

## Standard formula (to confirm against the `unit` column)
```
loaded_cost   = RP + cat_RI_share × RP_property  + fixed_expense_per_policy × (exposure? see assumption)
office_premium_ex_tax = loaded_cost / (1 − variable_expense − commission[channel] − profit_margin)
office_premium = office_premium_ex_tax × (1 + premium_tax)      # if tax is added on top
```
If tax is expressed as a share of the gross premium, include it in the denominator instead. Document the choice.

## Report output
- One table: 20 rows × [row_id, line, channel, exposure, RP, cat RI, fixed, variable, commission, profit, tax, office premium, written_premium (current pricing), ratio].
- Comment on large gaps vs `written_premium` (where current pricing seems under/over adequate).
- One worked example shown step by step.

## Checks (pytest)
- Office premium ≥ RP; increases monotonically with each loading; channel commission applied correctly; cat RI only on property lines.
- Reproducible from `src/office_premium.py` with values read from `expenses` (no hard-coded numbers).
