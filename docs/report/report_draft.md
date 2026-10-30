# Actuarial Pricing Across Markets with AI Assistance — Solo entry (Suguru Fujita)

*Final version (v1.0, submitted 2026-10-30). The Week 2 private score is published after the close. Every number below is traceable
to `logs/` or `outputs/reports/`. Graded scope: first 5 pages.*

## 1. Executive summary
- **Task.** Expected 2025 ultimate claim cost (risk premium) per policy row for private motor, home buildings and home
  contents; metric Tweedie D² (p = 1.5). Week 1: developed market. Week 2: an emerging market with two years of
  history, whose 2025 test book is mostly new business from a broker.
- **Approach.** Target rebuilt from claims and reconciled to the cent; development to ultimate (cost-based IBNR,
  Bornhuetter-Ferguson override for immature bodily injury); mix-adjusted trend; per-line 50/50 blend of a Tweedie GLM
  (explainability anchor) and a regularised LightGBM Tweedie, with large-loss caps + loading and a credibility-weighted
  experience (bonus-malus) modifier. In Week 2 the rating shape was carried over from the developed market and the
  level re-fitted on emerging data, with a new-business loading.
- **Results.** Week 1 temporal validation (train 2021–23 → validate 2024, pooled): constant 0.082 → GLM 0.137 →
  blend 0.139; **private D² 0.13363, rank 30** (public 0.12480). The private score was close to our temporal-validation
  estimate, consistent with selecting models on validation rather than on the leaderboard. Week 2 validation (train
  2023 → validate 2024 on emerging): motor 0.082, buildings 0.102, contents 0.102; public 0.15691 (private score published after the close).
- **Office premium.** For the 20 emerging reference policies, the office premium built from our risk premium is a
  median 2.7× the current written premium, mostly because of the emerging expense structure (§5).
- **AI workflow.** Claude Code built and challenged every step; the human owner signed off 38 logged decisions at six
  review gates. Several AI errors were caught by tests, validation and the official data description before they
  reached a submission (§3).

## 2. Pricing performance & adaptation
### 2.1 Week 1 — developed market
**Data and target (gate G1).** Target per claim = max(incurred − recoveries, 0); amounts are already net of excess
(official description, confirmed on data: 1% quantile of closed claims ≈ 13–41 CU at every excess level). Declined
claims (1,711) and claims outside any row period (414; 0.24% of cost) excluded. All 163,150 claims covered;
claim-level vs row-level reconciliation difference 0.00 in all 12 line × year cells.

**Data quality (gate G2).** Train exposure recomputed from dates (14,204 rows had days or ~1.4% inflation; raw exposure
would understate burning cost by ~26%); sum insured ×1,000 entry errors corrected with the train median rule (train and
test, ~0.2%); construction type relabel T (2021–22) → W (2023–24) unified as timber, with a timber risk shift flagged.
One open bodily-injury claim at 11.89M (next largest 1.12M) treated as a genuine large loss with a sensitivity at 1.19M.

**Development and trend (gate G3).** Single valuation date (31 Dec 2024), so incurred development (IBNER) is not
estimable; cost-based reporting-delay factors per line × peril (2024: motor 1.32, buildings 1.12, contents 1.04,
central). Bodily-injury severity for 2023–24 falls as the open share rises (49%/80%), so it is replaced by the
2021–22 severity trended (+3% p.a.). Trend from mix-adjusted GLM year effects and per-peril analysis:

| Annual trend | Low | Central | High |
|---|---|---|---|
| Motor | 3% | 5% | 7% |
| Buildings | 4% | 5% | 7% |
| Contents | 2.5% | 3.5% | 4.5% |

Insight: motor's crude frequency fall is portfolio mix (mix-adjusted frequency flat); physical-damage severity
+6–10% p.a. Leaderboard probes of low/high trend were within noise of central, suggesting level is not the main source of error.

**Model ladder (temporal validation, 2024 fold).**

| Rung | Pooled D² | Pooled A/E |
|---|---|---|
| Constant per line | 0.0820 | 0.98 |
| Tweedie GLM (capped + loading) | 0.1370 | 0.97 |
| GLM + LightGBM blend | 0.1388 | 0.99 |

Rolling origin (→2022, →2023, →2024): the blend beats the GLM in 8 of 9 line × year folds. Level sensitivity: the
D²-optimal global scale is 0.96–1.03 (gain ≤ 0.0002), suggesting the remaining error lies mostly in ranking.

**Adopted vs rejected (pre-registered rule: +0.001 D² in BOTH 2023 and 2024 folds).**
Adopted: caps 100k/100k/25k; contents GBM c7 (+0.0018/+0.0017). Rejected: recency weighting (motor −0.0037 in 2023),
written premium as a feature, raw make_model, prior-claims features v1/v2, stacking (unstable weights).
Leaderboard-informed (documented as such): long-run instead of same-year weather (+0.0011 LB; also the principled
choice — realised weather is unknown at pricing); near-pass buildings c7 and experience modifier (b3, public +0.0004,
the two effects adding up; a pre-registered re-test on the final model confirmed the modifier as a near-pass: +0.0002 to +0.0011 in all six cells, only motor 2024 ≥ 0.001). Final selection: validated model (b) plus b3 (Kaggle scores the better of the two); step-by-step scores in Appendix A.

**Limitations.** 2024 "actuals" are developed estimates; IBNER assumed 1.0 (±10–30% on long-tail perils); single-year
validation folds; motor validation A/E 0.95–0.98; credit band is a fairness-sensitive factor.

### 2.2 Week 2 — emerging market (gate G6)
**What changed.** The 2025 test book is 76% new business (train 14%; mean renewal count 1.8 vs 6.9). Adversarial
validation separates train from test with AUC 0.86, and `new_business` carries 66–83% of that signal. Against the
developed market (2023–24), motor frequency is +35% and severity −45%; contents frequency −15%, severity −10%;
buildings is volatile (on-levelled burning cost 569 in 2023, 348 in 2024). Reporting delays and open-claim shares
match the developed market. The schema is thinner: no exact age, credit band, mileage or telematics. The data
had the same anomaly types as in Week 1, which we cleaned with the same rules (decision #26). Free-text occupation
was mapped by rule to the 20 developed levels, with 0 unmapped.

**Week 1 model on Week 2 data.** The developed model on harmonised features over-predicts emerging 2023–24 costs:
A/E motor 0.71, buildings 0.89, contents 0.86, so the **level does not transfer**. The **shape mostly does**:
comparing factor relativities between the markets, 31 of 53 motor factors (19/47 buildings, 23/47 contents)
can be carried over or have little effect in either market, and only 3 (11, 7) need refitting. Harmonising the features
costs motor −0.008 D² on developed validation; buildings and contents are almost unchanged.

**What was carried over and what changed.**

| Element | Week 2 choice | Evidence |
|---|---|---|
| Rating shape | Motor, contents: harmonised developed blend × emerging level (C1). Buildings: joint developed + emerging fit with a market feature, emerging rows ×5 (C4) | Emerging-only GLMs fail (D² motor 0.013, buildings/contents < 0); buildings C4 0.102 vs C1 0.077 (bootstrap band of the gain +0.003 to +0.051) |
| Level | Re-fitted on emerging 2023–24 (motor ×0.71, contents ×0.86); buildings: credibility Z = 0.5 between the joint fit and emerging actuals | Buildings swings between years; full credibility (≈ 490) and joint fit (≈ 398) bracket the central 446 |
| Development | Developed factors by line × peril | Same reporting delays and open shares |
| Trend | Developed central (provisional) | Two years are too short to estimate it |
| New business | Loading 1.25 on new-business rows (range 1.0–1.5) | See below |
| Other | Caps and 50/50 blend kept; GBM averaged over 5 seeds (Week 1 lesson). Not used: gender (no lift; fairness), self-employed flag, region factor (rule not met, #34) | Bootstrap rule on train 2023 → validate 2024 |

**New-business loading: the main judgement.** Training data are mostly renewals, so the new-business effect is
estimated from little evidence. Validation A/E for new business was inconsistent (2024: motor 0.91, buildings
0.57, contents 1.30). We therefore used pre-registered leaderboard probes. Against a loading of 1.0, a loading
of 0.9 scored −0.0066 and 1.25 scored +0.0058. To check whether this was a general level effect, we probed a
uniform uplift with the same total; it scored 0.0010 below NB 1.25, so we read the effect as specific to new
business. We kept 1.25 and did not chase the apparent optimum near 1.28. This choice is **largely
leaderboard-informed**: most of the gain (+0.0047 of +0.0058) is shared with a general level increase, so the
emerging 2025 level itself may be understated.

**Predicted 2025 burning cost** (per exposure year): motor 562, buildings 446, contents 99, compared with
on-levelled emerging 2023/2024 values of 488/491, 569/348 and 98/93. The motor increase reflects the younger,
mostly new-business test mix.

**Remaining uncertainty.**

| Item | Low | Central | High |
|---|---|---|---|
| NB loading | 1.0 | 1.25 | 1.5 |
| Buildings level (BC) | 398 | 446 | 490 |
| Trend | Developed low | Developed central | Developed high |

The table leaves out four smaller sources of uncertainty:
- We have only two years of emerging history.
- Development factors, including IBNER, are borrowed from the developed market.
- The emerging business may keep changing as the broker book matures.
- The comparison-site commission is blank in `expenses`, although about 5% of policies use that channel (DQ26).

## 3. Effective use of AI
**Workflow.** The AI (Claude Code) drafted code, checks and analyses. CLAUDE.md instructed it to act as a
challenger, to stop at six human gates (target, data fixes, assumptions, model choice, submission, Week 2
adaptation), and to log every contribution (`logs/ai_usage_log.md`, 17 entries; `logs/decision_log.md`,
38 decisions).

**Notable cases.**
1. *AI error, caught by documentation:* the AI inferred claims were gross of excess and subtracted it again; the
   official description said otherwise → corrected before modelling (would have understated cost by ~16.8M a year).
2. *AI omission, caught by validation:* the first LightGBM was not level-balanced (A/E 1.06–1.26); fixed with a
   training-total balance step.
3. *AI design flaw, caught by the two-fold rule:* prior-claims v1 coded unobservable history (2021) as zero claims —
   better in 2024, worse in 2023 in every line.
4. *Pre-agreed rule in action:* the AI's recency-weighting idea looked good on 2024 alone (+0.0016) but failed 2023 (−0.0037).
5. *AI probe design (Week 2):* because 76% of test rows are new business, a better score from the NB loading could
   simply mean the overall level was too low. The AI proposed a uniform-level probe with the same total, which
   separated the two explanations before we committed to the loading.
6. *AI errors, caught by run checks (Week 2):* the AI had trial-run only one line, so a column missing in emerging
   motor slipped through. In a second case, an output filter hid a crash. Both were fixed before submission, and
   every run now records its exit status.

**Verification.** Every AI-written transform has a unit test (65 tests, incl. leakage property tests); every claim about
the data was re-run by the owner on full data; owner-run scripts print compact evidence.

## 4. Trust, validation & guardrails
- **Validation design:** temporal (train 2021–23 → 2024) plus rolling origin; Week 2 train 2023 → 2024 on emerging
  with bootstrap bands, and a check on the new-business subset. Acceptance rules were written before results were
  seen. The leaderboard was used only as a sanity or tie-break check, and each probe was pre-registered with its
  expected outcome.
- **Diagnostics:** A/E by decile and segment across folds (persistent drifts identified and tested), level sensitivity,
  per-peril level checks; escalation triggers logged (e.g. motor 2023 A/E 1.13 explained by one claim); Week 2
  adversarial validation and PSI between train and test and between markets.
- **Leakage controls:** only information known at inception; prior claims counted only if reported before the row
  starts. Policy and customer IDs collide across markets, so the pipeline never joins across markets and prefixes
  IDs in joint fits.
- **Reproducibility:** `python -m src.pipeline --market w1 --model blend` and `python -m src.pipeline_w2`; seeds and
  assumptions in `configs/config.yaml`. Rebuilds on later days were byte-identical to the submitted files (first
  submission sha256 72be5f20…, final b3 b1e69ab9…, built by two different scripts). Git tags `w1-final` and
  `w2-final`.

## 5. Pricing tool & communication
**Pricing tool.** A Streamlit tool (`app/pricing_tool.py`) loads the same fitted models as the submission (test-row
totals match the final file exactly in every line). For a single policy it returns:
- the expected 2025 cost;
- the GLM and GBM views;
- the experience factor;
- a GLM breakdown (base rate × factor relativities × exposure).

A batch mode prices a CSV and returns it for download, and an About page shows the model, data and assumption
versions. Where GLM and GBM disagree for a policy, the team sees both numbers and can check before relying on the
blend.

**Office premium (Week 2, 20 reference policies).** We took our risk premium (RP) and read every rate from `expenses`
(`src/office_premium.py`). Loaded cost = RP + cat reinsurance (12% of RP, buildings and contents only) + fixed expense
85 CU. The office premium = loaded cost / (1 − variable 15% − commission by channel − premium tax 8% − profit 8%).
For example, B00229821 (buildings, broker):
- RP 325.54
- Cat reinsurance 39.07
- Fixed expense 85.00
- Loaded cost 449.61
- Premium-based loadings 51%
- **Office premium 917.57**, 3.2× its current premium of 285.60

The components add back to the office premium for every policy. Across the 20 policies:
- **The office premium is a median 2.7× the written premium.**
- RP alone is close to the written premium for motor (median 1.02) and buildings (1.20), and well below it for
  contents (0.48).
- The gap therefore comes mostly from the expenses. Broker business carries 51% premium-based loadings, and the
  85 CU fixed expense is close to a typical contents premium (about 100 CU).
- Our reading is that current emerging premiums do not cover expenses and profit, rather than that our risk premium
  is too high. This is a question about the profitability of the broker book.

**Sensitivity of the total.**
- Premium tax added on top instead of inside the premium: −6.4%.
- Fixed expense pro-rata to exposure: −2.2%.
- Both: −8.5%.

One outlier, B00217584, has an RP of 1,790 against a written premium of 323. It combines several flood-related
features: 0.6 m above river level, 1.7 km from the river, flood zone missing, mixed construction, built in 1933.
Current pricing may not reflect the flood exposure or the missing flood zone (DQ27).

## 6. Limitations & next steps
- **Week 1 lessons carried into Week 2:** average GBMs over seeds by default (applied: +0.0013 median gain on
  emerging contents); build final candidates early; keep the pre-registered rules.
- **Week 2 lessons:**
  - When the test mix differs this much, a single feature (new business) dominates the shift, and two years of
    history cannot pin down its effect. Monitoring actual-to-expected for the new broker book in 2025 should
    replace the leaderboard-informed loading as soon as data arrive.
  - The emerging trend and IBNER should be re-estimated once a third year is available.
- **Open items:**
  - Experience modifier for emerging motor: on emerging validation the gain is +0.0019 (median), with the bootstrap 5th
    percentile just above zero (+0.0001); new business is treated as history unknown. Not used in the Week 2 finals;
    to be confirmed on 2025 renewals.
  - Per-peril models (frequency learned, severity barely).
  - Longer development history for IBNER.
  - Fairness review of credit band before any production use (gender was not used in Week 2).
  - Commission for the comparison-site channel in the emerging market (DQ26).

---
## Appendix A (not graded). Week 1 leaderboard path

| Submission (public / private) | Change vs previous | Public |
|---|---|---|
| First blend | — | 0.12293 |
| + long-run weather | +0.0011 | 0.12401 |
| + contents GBM c7 (validated) | +0.0004 | 0.12438 |
| + buildings c7 + experience modifier (final, private 0.13363) | +0.0004 | 0.12480 |

Other probes (within the pre-registered ±0.0005 noise band unless stated): trend low 0.12275, trend high 0.12256, motor loading at 1.19M 0.12307,
same-year weather + contents c7 0.12322; after the deadline, 5-seed GBM averaging on b3: public 0.12497 / private 0.13403.

## Appendix B (not graded). Week 2 leaderboard path

| Submission | Public |
|---|---|
| C1/C4, buildings credibility 0.5, NB loading 1.0 | 0.15122 |
| NB 0.9 (probe) | 0.14458 |
| NB 1.25 (probe; final candidate 2) | 0.15697 |
| Uniform level ×1.2014, same total as NB 1.25 (probe) | 0.15596 |
| NB 1.5 (probe) | 0.15332 |
| Final: NB 1.25 + 5-seed GBM (final candidate 1) | 0.15691 |
