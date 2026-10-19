# 02 — Modelling standards

## 1. Model ladder (always build bottom-up, keep every rung for comparison)
| Rung | Model | Purpose |
|---|---|---|
| 0 | Constant burning cost per line (× exposure) | Metric baseline (D² ≈ 0) |
| 1 | One-way relativities / simple GLM | Sanity, explainability anchor |
| 2 | Tweedie GLM, p = 1.5, log link, log(exposure) offset | Main actuarial model; transparent relativities |
| 3 | Frequency (Poisson) × Severity (Gamma) GLMs | Diagnostic: where does signal come from? |
| 4 | LightGBM `objective=tweedie`, `tweedie_variance_power=1.5`, exposure via init_score or weight | Capture non-linearity/interactions |
| 5 | Blend (e.g. weighted GLM + GBM, or GBM on GLM offset) | Final, if it beats rung 2/4 on temporal CV |

Rule: a more complex rung is adopted only if it improves temporal-validation D² **and** passes diagnostics
(stable A/E by segment, sensible monotonic effects, no leakage). Record the decision in `logs/decision_log.md`.

## 2. Loss function alignment
- Score is Tweedie deviance p = 1.5 → fit with p = 1.5 (optionally tune 1.3–1.7 and check).
- Report D² per line and pooled (submission is pooled across lines).
- Predictions strictly ≥ 0 (log link guarantees > 0).

## 3. Features
- Start from raw rating factors; band continuous variables for GLM (or splines); keep raw for GBM.
- Thin categorical levels: group or regularise (ridge/credibility). Unseen test levels → map to "other"/base.
- Allowed derived features: age, vehicle/property age, sum insured bands, ratios available at inception.
- Forbidden: anything using claim information of the same or later period; target encoding without out-of-fold scheme.
- Monotonic constraints in GBM where actuarially expected (e.g. sum insured ↑ → cost ↑), verified by data.

## 4. Large losses
- Option A: fit on capped losses + spread excess-over-cap as a loading (by line, possibly by sum insured band).
- Option B: fit uncapped Tweedie (p = 1.5 is fairly robust).
- Choose by temporal CV; document threshold choice (human gate).

## 5. Projection to 2025
Prediction_2025 = model relativities × base level × trend factor × development adjustment
- Trend: from mix-adjusted year effects (GLM year factor) — extrapolate with judgement, show a range.
- Development: if 2024 (or later years) immature, gross up training targets or include year factor and project.
- Level calibration: check that Σ predictions on 2024 validation ≈ Σ actual (A/E ≈ 1) before projecting.
- All three are assumptions in `logs/assumptions_register.md` with low/central/high values.

## 6. Explainability
- GLM: relativity tables per factor, with CIs.
- GBM: SHAP summary, dependence plots for top factors; compare with GLM relativities (disagreements = discussion points).
- One-page "pricing factor card" per line for the pricing tool.

## 7. Pricing tool (Part 1 deliverable)
- Input: policy characteristics (+ exposure) → output: expected cost, frequency, severity, and factor breakdown.
- Shows model version, data version, and assumption values used.
- Batch mode (CSV in → CSV out) and simple UI (Streamlit).
