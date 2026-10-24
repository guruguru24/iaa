# CLAUDE.md — Project instructions for AI assistants

This repository is a solo entry to the **IAA AI Hackathon: Actuarial Pricing Across Markets with AI Assistance**.
Owner / responsible actuary: Suguru Fujita. The AI assistant supports; the human signs off.

Read this file before any work. Then read `docs/00_challenge_brief.md` and `docs/data_description.md`
(Part A = official data description; Part B = actuarial hypotheses on target construction, IBNR, leakage, fairness).

Data location (git-ignored): `data/raw/w1/` = `developed` folder (Part 1), `data/raw/w2/` = `emerging` folder (Part 2).
Read all `*_id` columns as string.

Week 2 (from 2026-10-24): emerging market, history 2023–2024 only, **2025 test mostly new business from a broker**;
customers, climate, regulation and costs differ. Follow `docs/05_week2_market_shift.md` and `docs/data_description.md` Part C (column differences:
no exact age/credit/mileage/telematics; gender allowed; occupation free text; regions E01–E06; data not cleaned).
**Join tables only within the same market; IDs collide across markets.** Week 2 leaderboard = emerging rows only.
Also required: office premium build-up for 20 `reference_policies` using `expenses` → `docs/07_office_premium.md`.

## 1. The problem in one paragraph

Predict, for each policy row in 2025, the **expected ultimate claim cost** (risk premium, not office premium)
for three lines: private motor, home buildings, home contents. Training data: synthetic, developed market,
policy years 2021–2024. Target = final cost of all claims with loss date in the row's period,
**net of policy excess, net of recoveries, excluding declined claims**. Week 2 adds a new market dataset;
we must detect what changed, adapt, and state remaining uncertainty.

Metric: **Tweedie D² (power = 1.5)**, higher is better. Submission: one CSV `row_id,prediction`, all three
lines together, predictions ≥ 0.

## 2. Non-negotiable rules

1. **Target definition first.** Never model a target column until its construction has been verified against
   the definition above (excess, recoveries, declined claims, ultimate vs paid/incurred, loss date in period).
   Record the check in `logs/data_quality_log.md`.
2. **Exposure matters.** Rows may have partial-year exposure. Model per unit exposure (offset / weight) and
   predict for the row's 2025 exposure. Confirm how the test rows define their period.
3. **No leakage.** No feature derived from claim outcomes, claim counts of the same period, or post-inception
   information. Test set rows are never used for fitting, target encoding or scaling statistics.
4. **Temporal validation is the selection criterion.** Primary: train 2021–2023, validate 2024.
   Kaggle public leaderboard is a sanity check only — never the model-selection criterion.
5. **Level matters as much as ranking.** D² punishes total-level error. Always report A/E (actual/expected)
   overall and by line, year and key segments. Trend/inflation to 2025 is an explicit, documented assumption.
6. **Reproducibility.** Fixed seeds (`configs/config.yaml`), pinned `requirements.txt`, scripts runnable end to end
   (`python -m src.pipeline`). No manual edits to outputs.
7. **Every AI contribution that affects the result is logged** in `logs/ai_usage_log.md`, including how it was
   verified and whether it was accepted, modified or rejected. Rejected AI suggestions are valuable evidence — log them.
8. **Human review gates** (see `docs/04_governance.md`): the AI must stop and ask the human before
   (a) finalising the target definition, (b) choosing trend/inflation and IBNR assumptions,
   (c) excluding or capping data, (d) final model choice per line, (e) any Kaggle submission,
   (f) Week 2 adaptation strategy.
9. **Never fabricate.** If a number is not computed from data in this repo, say so. State uncertainty plainly.

## 3. Standard workflow (follow in order)

| Step | What | Where | Guide |
|---|---|---|---|
| 0 | Load & schema check, data dictionary | `notebooks/01_eda_*.ipynb`, `src/data.py` | `docs/01_eda_checklist.md` |
| 1 | Target reconstruction & verification | `src/target.py` | `docs/01_eda_checklist.md` §2 |
| 2 | Actuarial EDA (frequency, severity, exposure, development, trend) | notebooks | `docs/01_eda_checklist.md` |
| 3 | Features | `src/features.py` | `docs/02_modelling_standards.md` |
| 4 | Models: baseline → Tweedie GLM → GBM → blend | `src/models.py` | `docs/02_modelling_standards.md` |
| 5 | Validation & diagnostics | `src/validation.py`, `tests/` | `docs/03_validation_and_guardrails.md` |
| 6 | 2025 projection (trend, level calibration) | `src/projection.py` | `docs/02_modelling_standards.md` §5 |
| 7 | Submission build + checks | `src/submission.py` | `docs/03_validation_and_guardrails.md` §4 |
| 8 | Week 2: market-shift analysis & adaptation | `src/drift.py` | `docs/05_week2_market_shift.md` |

## 4. Coding conventions

- Python 3.11+, pandas, numpy, scikit-learn, statsmodels/glum for GLM, lightgbm for GBM, shap, pytest.
- Logic in `src/` as importable functions; notebooks only call `src/` and show results.
- Every function that transforms data has a pytest test in `tests/`.
- Line is a parameter (`line in {"motor","buildings","contents"}`), never copy-paste code per line.
- Market is a parameter too (`market in {"w1","w2"}`) so Week 2 reuses the pipeline.
- Small, frequent git commits with meaningful messages: `feat:`, `fix:`, `eda:`, `model:`, `docs:`, `test:`.
  Mention AI involvement in the commit body when relevant (`AI-assisted: ...; verified by ...`).

## 5. How the AI should behave here

- Act as a **challenger**, not only a generator: for every modelling step, name at least one way it could be wrong.
- Show evidence (numbers, plots, tests) rather than assertions.
- When unsure about actuarial judgement (trend, IBNR, large-loss treatment, credibility), present options with
  trade-offs and **ask** — do not decide silently.
- After each working session, append to `logs/ai_usage_log.md` and `logs/decision_log.md`.
- Write submission-facing text (report, video script) in English; chat with the owner in Japanese.
