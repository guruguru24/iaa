# AI usage log

Evidence for evaluation category 2 (Effective use of AI, 25%). One row per meaningful AI interaction.
Outcome: **Accepted** / **Modified** / **Rejected**. Impact: quality / efficiency / challenge / reliability.

| # | Date | Stage | Tool | Request (short) | AI output (short) | Verification | Outcome | Impact | Link |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-19 | Planning | Claude (Cowork) | Summarise challenge brief and scoring | Brief, priorities, 4-day plan | Human checked against official text | Accepted | Efficiency | docs/00_challenge_brief.md |
| 2 | 2026-10-19 | Planning | Claude (Cowork) | Implication of Tweedie D² p=1.5 metric | Align objective to Tweedie p=1.5; level calibration critical | Will confirm with metric test + validation | Accepted (to verify) | Quality | docs/02_modelling_standards.md |
| 3 | 2026-10-19 | Governance | Claude (Cowork) | Draft governance, review gates, EDA & validation checklists | docs/01–05, CLAUDE.md | Human review of each checklist | Pending review | Reliability | docs/ |

| 4 | 2026-10-20 | EDA §1–2 | Claude Code | Day 1: schema/integrity checks and target reconstruction | src/data.py, src/target.py, tests/test_target.py; DQ1–DQ14 | Toy-case pytest (9 passed); claim-level vs row-level reconciliation diff 0.00 in all line×year cells; every DQ number from re-runnable query | Pending human review (G1) | Quality / reliability | outputs/reports/w1_target_reconciliation.csv |
| 5 | 2026-10-20 | Target G1 | Claude Code | Re-check target definition against official data_description.md | AI's DQ2 inference (amounts gross of excess, from 13–27% claims below excess) contradicted by official text (paid net of excess) | Official text + closed paid == incurred 100% | **Rejected** (AI's earlier inference); target.py fixed, tests updated (14 passed) | Reliability | logs/data_quality_log.md DQ15 |
| 6 | 2026-10-22 | G3 evidence | Claude Code | Development to ultimate and mix-adjusted trend | src/development.py, src/trend.py; preliminary trend ranges | Synthetic mix-shift unit test; owner ran on W1; AI challenged own results (motor line vs peril severity conflict; SI absorbing inflation; bldg 2024 outlier) and wrote follow-up checks | Pending (G3) | Quality / challenge | scripts/check_trend_followups.py |
| 7 | 2026-10-22 | Features / models | Claude Code | Feature build, train-fitted encoder, constant + Tweedie GLM, temporal validation | src/features.py, dataset.py, models.py, validation.py; scripts/run_validation_glm.py | pytest 31 passed (encoder unseen levels, exposure scaling, loading restores level, leakage columns excluded); synthetic end-to-end run incl. test frame | Pending owner run | Quality / reliability | outputs/reports/w1_validation_glm.csv |
| 8 | 2026-10-22 | Models (rung 4) | Claude Code | LightGBM Tweedie vs GLM | First GBM under-predicted level (validation A/E motor 1.06, bldg 1.26, cont 1.22) → D² below GLM though Gini similar; AI had not added a level balance step. Fix: training-total balance correction + tuning | Owner-run temporal validation; unit test (balanced total = training total) | **Modified** (AI omission caught by validation) | Reliability | outputs/reports/w1_validation_20261022_2240.csv |

## Notable cases (for report/video)
Record here the best examples where AI (a) found something the human missed, (b) was wrong and was caught,
(c) saved significant time. Include before/after evidence.

- 2026-10-22 (AI wrong, caught): AI's first LightGBM had no level balancing; temporal validation showed A/E 1.06–1.26 and lower D² than the GLM despite equal Gini. Balance correction added before any model choice.
- 2026-10-22 (AI found): motor 2025-trended levels look like a decline (685→627) but converge to ~630–640 once restated to the 2024 risk mix (mix effect ≈ −7% by 2024) — supports applying the mix-adjusted trend on top of a rating-factor model.
- 2026-10-22 (AI found): crude motor frequency fall (−4.7% by 2024) is almost entirely portfolio mix — mix-adjusted frequency is flat (0.995).
- 2026-10-20 (AI wrong, caught): AI inferred claims were gross of excess from data patterns and subtracted excess; the official data description says paid is net of excess → double deduction caught before any modelling (DQ2 → DQ15). Lesson: check documentation before inferring definitions from data.
- 2026-10-20 (AI found): train exposure is partly recorded in days (raw exposure sum +33–39%); using the raw exposure would have understated burning cost by ~26%. Evidence: DQ2, DQ7.
- 
