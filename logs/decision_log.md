# Decision log

| # | Date | Gate | Decision | Options considered | Rationale / evidence | Approved by |
|---|---|---|---|---|---|---|
| 1 | 2026-10-19 | — | Enter solo; Python stack; temporal validation (2021–23 → 2024) as primary model-selection criterion | Random K-fold; Kaggle LB | Mirrors 2025 forecast; avoids LB overfitting | S. Fujita |
| 2 | 2026-10-19 | — | Fit Tweedie p=1.5 objective as main loss | Freq×sev; MSE | Aligns with scoring metric | S. Fujita (to confirm after validation) |
| 3 | 2026-10-22 | G1 | Target definition W1: per-claim net = max(incurred_to_date − recoveries, 0), excess NOT subtracted (amounts already net of excess); repudiated excluded (1,711, zero amounts); claims whose loss_date falls in no row excluded (414 claims, 1.79M incurred, 0.24% of net — all before the policy's first row); open claims at current incurred, development to ultimate handled separately (G3); match claim→row on policy_id and loss_date in inclusive [period_start, period_end] | Subtract excess (AI's earlier DQ2 reading); allow negative net; keep out-of-period claims | Official data description + target__excess_check (DQ15); reconciliation diff 0 in all 12 line×year cells; 163,150/163,150 claims covered; outputs/reports/w1_eda/target__*.csv | S. Fujita |
| 4 | 2026-10-22 | G2 | Cleaning rules (train and test identically, raw values kept in *_raw with *_fixed flags): (i) exposure = (days+1)/365 from dates (14,204 train rows; test already consistent); (ii) sum_insured > 50× train median divided by 1,000 (bldg 1,395/370, cont 1,402/360 train/test rows); (iii) construction_type case-normalised, 'T' (2021–22) and 'W' (2023–24) mapped to timber; (iv) renewal_decision dropped | Keep raw exposure; exclude flagged rows; keep W/T as separate levels | scripts/check_preprocess.py: raw exposure understates burning cost ~26% (motor 407 vs 547); flagged rows have normal date-based frequency/BC; SI/1000 quantiles and premium match normal rows, grey zone 0 rows; T/W never co-occur and volumes match. Open: W severity +37% vs T (real shift vs large fires) → scripts/check_timber_bi.py | S. Fujita |

## Submissions
| Date | Market | File | Git hash | Validation D² (pooled) | Kaggle public score | Approved by |
|---|---|---|---|---|---|---|
