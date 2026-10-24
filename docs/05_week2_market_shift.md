# 05 — Week 2: emerging-market shift analysis and adaptation playbook

Graded explicitly: **identify what changed**, **decide what to carry over from Part 1**, **communicate remaining uncertainty**.

## 0. Known facts from the brief (2026-10-24)
| Fact | Implication |
|---|---|
| Only 2023–2024 history | Very little mature data; 2024 immature (IBNR/IBNER) → effectively one mature year |
| Most 2025 policies are new business from a broker | Test mix ≠ training mix: `new_business`, `renewal_count`, `ncb_years`, `channel`, `days_quote_to_inception`, `discount_applied`, `written_premium` shift; renewal-driven effects may not transfer |
| Customers, climate, regulation, costs differ | Relativities, weather response, allowed rating factors, severity level and inflation may all differ |
| Less data | Credibility is the core tool; Part 1 model as prior/complement |
| `developed` folder provided for reference | Joint modelling and transfer are allowed and expected |
| Columns differ (no exact age, credit, mileage, telematics; gender allowed; free-text occupation; regions E01–E06; construction codes) — see `data_description.md` Part C | Transfer via a **harmonised developed model** on common features; quantify lift lost |
| Data not cleaned; IDs collide across markets | Full EDA §1 on emerging; never join across markets; prefix IDs if stacking |
| Leaderboard = emerging test only | Score only on emerging rows; but report keeps Week 1 for comparison |

## 1. Detect what changed (Day 1 — before modelling)
- [ ] File/schema diff vs developed: new/missing/renamed columns, changed levels, units, currency, ID formats
- [ ] Data description diff: read the emerging description line by line vs `docs/data_description.md`; record every difference
- [ ] Definition diff: exposure, claim status codes, excess (levels, compulsory amounts), recoveries, valuation date
- [ ] Regulation: rating factors absent/forbidden (e.g. credit, age, gender proxies) or capped
- [ ] Volume and credibility: rows, exposure, claims by line × peril × year
- [ ] Portfolio mix: train (2023–24) vs test (2025) — especially new business, channel=broker, NCB, renewal_count, region, sum insured
- [ ] Feature distribution shift: PSI/KS per feature, train-vs-test and developed-vs-emerging; adversarial validation (classifier train vs test) to find what drives the shift
- [ ] Outcome shift: frequency, severity, burning cost, tail, peril mix, report delay, development pattern vs developed market
- [ ] Climate: weather variable ranges vs developed (extrapolation outside the developed range?)
- [ ] Data quality issues specific to emerging data → `logs/data_quality_log.md` (market = w2)

## 2. Diagnose transfer (Day 2)
- [ ] Score the final Week 1 model on emerging 2023–2024 → A/E overall, by line, peril, year, factor level, new vs renewal
- [ ] Separate **level** (overall A/E, severity scale) from **shape** (relativities that break)
- [ ] Refit simple emerging-only GLM; compare relativities factor by factor with developed (with CIs) → "carry over / adjust / refit / drop" table

## 3. Adaptation candidates (Day 3–4; human gate G6)
| # | Candidate | When appropriate |
|---|---|---|
| C0 | Emerging-only baseline (constant / simple GLM) | Benchmark |
| C1 | Developed model + level recalibration (by line, maybe peril) | Shape transfers, level differs |
| C2 | Developed model as offset + refit selected factors on emerging | Some factors differ |
| C3 | Credibility blend: Z × emerging model + (1−Z) × recalibrated developed model; Z by line/segment from Bühlmann-type or exposure-based rule | Thin data |
| C4 | Joint model on developed + emerging with market indicator and market × factor interactions (GLM) or `market` feature (GBM), emerging weighted | Enough emerging data for interactions |
| C5 | GBM on emerging with developed prediction as feature / init_score | Non-linear differences |

Feature-set changes (carried over from the previous playbook version): **new features** → add to the GBM and check
stability; use in the GLM only if credible. **Missing features** → refit without them and quantify the lost lift.

Selection: validation train 2023 → validate 2024 (developed-to-ultimate where possible), plus **new-business-only subset** scores
(test is mostly new business). Prefer the simplest candidate within noise of the best. Bootstrap D² to show the noise band.

## 4. 2025 projection
- Level: emerging-specific trend/inflation (cost differences); development to ultimate from emerging triangles if credible, otherwise developed patterns scaled — document.
- New-business loading: check in developed and emerging data whether new business runs worse than renewals (selection effect); apply if supported.
- Broker portfolio: if the test portfolio's features look unlike the 2023–24 book, flag the segments where predictions extrapolate.

## 5. Communicate uncertainty
- Low/central/high for each level assumption (trend, development, new-business effect, credibility weight) → effect on total cost and D²
- Range of totals by line; segments with thin data or extrapolated features flagged
- Explicit list: what we carried over from Part 1, what we changed, and why
- What we would do with more time/data

## 6. Evidence for the video/report
- Table: Week 1 model on emerging vs adapted candidates (D² overall and new business only, A/E)
- One chart of the biggest shift found
- Carry-over table (factor → carry / adjust / refit / drop)
- Decision log entries G6; AI-usage log entries for Week 2 (e.g. AI-proposed shift checks, verified)
