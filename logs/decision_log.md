# Decision log

| # | Date | Gate | Decision | Options considered | Rationale / evidence | Approved by |
|---|---|---|---|---|---|---|
| 1 | 2026-10-19 | — | Enter solo; Python stack; temporal validation (2021–23 → 2024) as primary model-selection criterion | Random K-fold; Kaggle LB | Mirrors 2025 forecast; avoids LB overfitting | S. Fujita |
| 2 | 2026-10-19 | — | Fit Tweedie p=1.5 objective as main loss | Freq×sev; MSE | Aligns with scoring metric | S. Fujita (to confirm after validation) |

## Submissions
| Date | Market | File | Git hash | Validation D² (pooled) | Kaggle public score | Approved by |
|---|---|---|---|---|---|---|
