# 05 — Week 2: market-shift analysis and adaptation playbook

Purpose: show *what changed*, *how we responded*, *what uncertainty remains* (explicitly graded).

## 1. Detect what changed (run before any modelling)
- [ ] Schema diff: new/missing/renamed columns, changed category levels, units/currency
- [ ] Definition diff: exposure, claim status codes, excess, recoveries, development maturity
- [ ] Volume: rows, exposure, claims per line (credibility)
- [ ] Distribution shift per feature (PSI, KS, level frequency tables)
- [ ] Outcome shift: frequency, severity, burning cost, tail, by line and year
- [ ] Relationship shift: apply Week 1 model to Week 2 training data → A/E overall and by factor (where does it break?)
- [ ] Data quality issues specific to the new market

## 2. Adaptation options (choose with evidence; human gate G6)
| Situation | Option |
|---|---|
| Same relationships, different level | Re-calibrate base level (+ trend) only |
| Some factors differ | Week 1 model as offset + refit selected factors on Week 2 |
| Limited Week 2 data | Credibility blend: Z × Week2 model + (1−Z) × Week1-transferred model |
| Rich Week 2 data | Refit full pipeline; Week 1 model as feature/prior; compare |
| New features | Add to GBM; check stability; GLM only if credible |
| Missing features | Refit without them; quantify lost lift |

Compare all candidates on Week 2 temporal validation, same as Week 1.

## 3. Communicate uncertainty
- Low/central/high for level assumptions; D² and total-cost sensitivity
- Segments with thin data flagged
- What we would do with more time/data

## 4. Evidence for the video/report
- Before/after table: Week 1 model on Week 2 data vs adapted model
- One chart of the biggest shift found
- Decision log entries G6
