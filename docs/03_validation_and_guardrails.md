# 03 — Validation, diagnostics and guardrails

## 1. Validation design
- **Primary**: train 2021–2023 → validate 2024 (temporal hold-out, mirrors 2021–2024 → 2025).
- **Secondary**: rolling origin (2021→2022, 2021–22→2023, 2021–23→2024) to test stability.
- **Tertiary**: grouped K-fold by policy (no policy split across folds) for hyper-parameters only.
- Final fit on 2021–2024 with chosen settings.

## 2. Metrics and diagnostics (per line and pooled)
| Check | Pass criterion (initial, revisit) |
|---|---|
| Tweedie D² (p=1.5) | > baseline; stable across rolling folds |
| A/E overall | 0.95–1.05 on validation |
| A/E by decile of prediction (lift / double-lift) | monotonic, no decile outside 0.8–1.2 without explanation |
| A/E by key factor level and by year | no systematic pattern |
| Gini / Lorenz | report, compare models |
| Calibration plot | close to diagonal |
| Relativity sense check | signs/shapes actuarially plausible; human reviews |
| Prediction distribution train vs test | no unexplained shift (PSI < 0.2 flag) |

## 3. Automated tests (`pytest`)
- `test_data.py`: schema, dtypes, key uniqueness, non-negative exposure, date order
- `test_target.py`: target reconstruction on a hand-made toy case (excess, recovery, declined, out-of-period)
- `test_features.py`: no target leakage columns; unseen levels handled; deterministic output
- `test_models.py`: predictions finite and ≥ 0; prediction scales linearly with exposure; reproducible with seed
- `test_metrics.py`: Tweedie D² matches scikit-learn; constant prediction ≈ 0
- `test_submission.py`: all test row_ids present once, header exact, no NaN, ≥ 0, total level within expected band

Run: `pytest -q` before every commit that changes `src/` and before every submission.

## 4. Pre-submission gate (all must be true)
- [ ] `pytest` green
- [ ] Validation report regenerated from current code (`outputs/reports/`)
- [ ] Total predicted cost per line vs 2024 actual × trend: within agreed band
- [ ] Row count = sample_submission row count
- [ ] Human (owner) approved → record in `logs/decision_log.md` with submission file hash

## 5. Guardrails on AI output
- AI-written code: must have a test or a reproducible check before use.
- AI-claimed facts about data: re-run the query; paste evidence in the AI log.
- AI-suggested assumptions (trend, IBNR, caps): treat as proposals; human decides with sensitivity analysis.
- AI suggestions that conflict with actuarial principle (e.g. using claim info as feature) → rejected and logged.
- Second-opinion review: ask a separate AI session to critique key code/model choices; log findings.
