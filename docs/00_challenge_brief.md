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

## Week 2 — The new (emerging) market  *(official text, updated 2026-10-24)*
- Insurer expands into an **emerging market**. It has written business there for **two years (2023, 2024)**.
- At the start of 2025 it takes on a **large portfolio from a broker → most 2025 policies are new business**.
- Data: `emerging` folder of the competition dataset; `developed` folder = Part 1 data for reference.
- Predict 2025 claim cost for every row in the three emerging-market test files.
- The new market will not necessarily behave like Part 1: **less data; customers, climate, regulation and costs differ**.
- Show: how we **identify what has changed**, **decide what to carry over from Part 1**, and **communicate remaining uncertainty**.
- Week 2 leaderboard: submit predictions for the **emerging-market test data only**.

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

## Prizes (for reference)
1st CAD $500, 2nd CAD $100, 3rd CAD $50; top 3 invited to IAA webinars. Only members or students of a Full or
Associate Member Association of the IAA are eligible. **Prizes are decided by the judging panel on the evaluation
criteria above, not on either leaderboard alone.**

## Timeline
- 2026-10-19: Week 1 started (4 days left at that point)
- 2026-10-24: Week 2 started; 7 days to final close (exact deadline and submission form link: TO CONFIRM)
