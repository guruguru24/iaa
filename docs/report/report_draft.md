# Actuarial Pricing Across Markets with AI Assistance — Solo entry (Suguru Fujita)

*Draft v0.3 (Week 1 final results in; Week 2 placeholders marked ⟨W2⟩). Every number below is traceable to `logs/` or
`outputs/reports/`. Graded scope: first 5 pages.*

## 1. Executive summary
- **Task.** Expected 2025 ultimate claim cost (risk premium) per policy row for private motor, home buildings and home
  contents; metric Tweedie D² (p = 1.5).
- **Approach.** Target rebuilt from claims and reconciled to the cent; development to ultimate (cost-based IBNR,
  Bornhuetter-Ferguson override for immature bodily injury); mix-adjusted trend; per-line 50/50 blend of a Tweedie GLM
  (explainability anchor) and a regularised LightGBM Tweedie, with large-loss caps + loading and a credibility-weighted
  experience (bonus-malus) modifier.
- **Results.** Temporal validation (train 2021–23 → validate 2024, pooled): constant 0.082 → GLM 0.137 → blend 0.139.
  Week 1 final: **private D² 0.13363, rank 30** (public 0.12480; first submission 0.12293). The private score was close
  to our temporal-validation estimate, while the public board read lower; this is consistent with our choice to
  select models on temporal validation rather than on the leaderboard. ⟨W2: Week 2 validation and LB⟩
- **AI workflow.** Claude Code built and challenged every step; the human owner signed off 23 logged decisions at five
  review gates. Several AI errors were caught by tests, validation and the official data description before they reached a
  submission (§3).

## 2. Pricing performance & adaptation
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
the two effects adding up). Final selection: validated model (b) plus b3 (Kaggle scores the better of the two); step-by-step scores in Appendix A.

**Limitations and uncertainty.** 2024 "actuals" are developed estimates; IBNER assumed 1.0 (±10–30% on long-tail
perils); single-year validation folds; motor validation A/E 0.95–0.98; credit band is a fairness-sensitive factor.
⟨W2: what changed, adaptation (gate G6), remaining uncertainty⟩

## 3. Effective use of AI
**Workflow.** The AI (Claude Code) drafted code, checks and analyses; it was instructed (CLAUDE.md) to act as a
challenger, to stop at five human gates (target, data fixes, assumptions, model choice, submission), and to log every
contribution (`logs/ai_usage_log.md`, 11 entries; `logs/decision_log.md`, 23 decisions).

**Notable cases.**
1. *AI error, caught by documentation:* the AI inferred claims were gross of excess and subtracted it again; the
   official description said otherwise → corrected before modelling (would have understated cost by ~16.8M a year).
2. *AI omission, caught by validation:* the first LightGBM was not level-balanced (A/E 1.06–1.26); fixed with a
   training-total balance step.
3. *AI design flaw, caught by the two-fold rule:* prior-claims v1 coded unobservable history (2021) as zero claims —
   better in 2024, worse in 2023 in every line.
4. *AI finding:* motor's apparent frequency decline was mostly portfolio mix; the remaining gap appeared to be ranking rather than level.
5. *Pre-agreed rule in action:* the AI's recency-weighting idea looked good on 2024 alone (+0.0016) but failed 2023 (−0.0037).

**Verification.** Every AI-written transform has a unit test (45 tests, incl. leakage property tests); every claim about
the data was re-run by the owner on full data; owner-run scripts print compact evidence.

## 4. Trust, validation & guardrails
- **Validation design:** temporal (train 2021–23 → 2024) plus rolling origin; acceptance rules written before results;
  leaderboard used only as a sanity/tie-break check, each probe pre-registered with its expected outcome.
- **Diagnostics:** A/E by decile and segment across folds (persistent drifts identified and tested), level sensitivity,
  per-peril level checks; escalation triggers logged (e.g. motor 2023 A/E 1.13 explained by one claim).
- **Leakage controls:** only information known at inception; prior claims counted only if reported before the row starts.
- **Reproducibility:** `python -m src.pipeline --market w1 --model blend`; seeds and assumptions in `configs/config.yaml`;
  rebuilds on later days were byte-identical to the submitted files (first submission sha256 72be5f20…, final b3
  b1e69ab9…, built by two different scripts); git tag `w1-final`.

## 5. Pricing tool & communication
⟨Streamlit tool: policy inputs → expected cost, GLM factor breakdown, model/data/assumption versions; batch CSV mode.⟩

## 6. Limitations & next steps
- **Week 1 lessons carried into Week 2:** average GBMs over seeds by default (a late 5-seed build scored private
  0.13403 vs 0.13363); build final candidates early (each full build took 20–40 min); keep the two-fold rule.
- **Open items:** pre-registered re-test of the experience modifier on the final model (deferred for time);
  per-peril models (frequency learned, severity barely); longer development history for IBNER; fairness review of
  credit band before any production use. ⟨W2 learnings⟩

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
