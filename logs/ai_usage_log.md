# AI usage log

Evidence for evaluation category 2 (Effective use of AI, 25%). One row per meaningful AI interaction.
Outcome: **Accepted** / **Modified** / **Rejected**. Impact: quality / efficiency / challenge / reliability.

| # | Date | Stage | Tool | Request (short) | AI output (short) | Verification | Outcome | Impact | Link |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-19 | Planning | Claude (Cowork) | Summarise challenge brief and scoring | Brief, priorities, 4-day plan | Human checked against official text | Accepted | Efficiency | docs/00_challenge_brief.md |
| 2 | 2026-10-19 | Planning | Claude (Cowork) | Implication of Tweedie D² p=1.5 metric | Align objective to Tweedie p=1.5; level calibration critical | Will confirm with metric test + validation | Accepted (to verify) | Quality | docs/02_modelling_standards.md |
| 3 | 2026-10-19 | Governance | Claude (Cowork) | Draft governance, review gates, EDA & validation checklists | docs/01–05, CLAUDE.md | Human review of each checklist | Pending review | Reliability | docs/ |

| 4 | 2026-10-20 | EDA §1–2 | Claude Code | Day 1: schema/integrity checks and target reconstruction | src/data.py, src/target.py, tests/test_target.py; DQ1–DQ14 | Toy-case pytest (9 passed); claim-level vs row-level reconciliation diff 0.00 in all line×year cells; every DQ number from re-runnable query | Pending human review (G1) | Quality / reliability | outputs/reports/w1_target_reconciliation.csv |

## Notable cases (for report/video)
Record here the best examples where AI (a) found something the human missed, (b) was wrong and was caught,
(c) saved significant time. Include before/after evidence.

- 2026-10-20 (AI found): claims are gross of excess and train exposure is partly recorded in days (raw exposure sum +33–39%); using the raw exposure would have understated burning cost by ~26%. Evidence: DQ2, DQ7.
- 
