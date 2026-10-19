# 00 — Challenge brief (source of truth)

Summarised from the organisers' text. If the official text changes, update here and note it in `logs/decision_log.md`.

## Scenario
Pricing team of an established insurer in developed markets, reviewing technical pricing for three lines.
Models must be **technically sound, adaptable, explainable, practical**. The solution must demonstrate an
**AI-assisted actuarial workflow**; the claim model itself need not be AI (GLM, ML or both).

Part 1 title: *Create a pricing model for a novel market as well as a pricing tool that can be used by an Actuarial team.*

## Week 1 — Build the pricing approach
- Synthetic data, developed market: private motor, home buildings, home contents; policy years 2021–2024.
- Predict expected ultimate claim cost per policy row for 2025.
- Free choice: EDA/prep, techniques, workflow structure, where AI helps, testing/validation.

## Week 2 — Expansion
- New dataset (link at end of Week 1), possibly different characteristics and practical challenges.
- Show: what changed, how we responded, what uncertainty remains.

## Target definition
Ultimate cost of all claims with loss date in the row's period, **net of excess**, **net of recoveries**,
**excluding declined claims**. Fictitious currency. Risk premium, not premium charged.

## Submission file
One CSV for `test_motor`, `test_buildings`, `test_contents`:
```
row_id,prediction
M02409311,583.47
B01834522,412.10
```
Predictions ≥ 0. Metric: Tweedie D², power 1.5 (vs constant baseline). Kaggle leaderboard in each part.

## Evaluation weights
| Category | Weight | Key requirements |
|---|---|---|
| Pricing performance & adaptation | 40% | Accuracy on Week 1 & 2; appropriate choices and adaptation; data quality, limitations, uncertainty |
| Effective use of AI | 25% | AI across EDA, features, modelling, diagnostics, testing; demonstrated improvement; rigorous verification of AI output |
| Trust, validation & guardrails | 20% | Validation checks, automated tests, diagnostics; governance, reproducibility, human review/escalation |
| Communication & documentation | 15% | Video demo + written report; structured, version-controlled repo |

## Final submission (end of week 2)
1. Team name, member names
2. Week 1 prediction file
3. Week 2 prediction file
4. Video demo — **only first 5 minutes graded**: approach, results, AI workflow, change for new market
5. Write-up — **only first 5 pages graded**: approach, decisions, AI use, assumptions, limitations, adaptation, guardrails
6. Code (scripts/notebooks) under version control, reproducible

Review: quantitative scoring + qualitative review; **AI tools screen video, docs and code first**; panel decides.

## Timeline
- 2026-10-19: 4 days left in Week 1 (exact deadline: TO CONFIRM)
- Week 2 data: released end of Week 1
