# 01 — Actuarial EDA checklist

Goal: understand the data as an actuary, not just as a data scientist. Each item ends with a recorded finding
in `logs/data_quality_log.md` (issue) or `logs/assumptions_register.md` (assumption). Tick items in the EDA notebook.

## 1. Structure and integrity
- [ ] List all files, row counts, columns, dtypes; build `docs/data_dictionary.md`
- [ ] Keys: `row_id` unique; policy ↔ claim join keys; orphan claims; policies with multiple rows (renewals, MTAs)
- [ ] Row prefix vs line (M/B/C?) consistent
- [ ] Missing values by column × line × year; is missingness informative?
- [ ] Impossible values: negative exposure/amounts, ages out of range, sum insured 0, dates out of order
- [ ] Duplicates (exact and near-duplicate)
- [ ] Train vs test schema and category levels (unseen levels in test?)

## 2. Target reconstruction (must pass before modelling)
- [ ] Identify claim status fields (open/closed/declined/reopened)
- [ ] Gross → net: subtract excess (per claim? capped at claim size?), subtract recoveries (salvage, subrogation, reinsurance? — confirm only policy-level recoveries)
- [ ] Exclude declined claims; check nil claims (closed at zero) treatment
- [ ] Loss date within row period; claims with loss date outside period
- [ ] Paid vs incurred vs ultimate: is a development/ultimate field given? If only paid/incurred, measure maturity by policy year
- [ ] Reconcile: Σ(claim-level net ultimate) = Σ(row target) per line × year
- [ ] Negative net amounts after recoveries? Decide floor rule (human gate)

## 3. Exposure
- [ ] Exposure definition (earned years?); distribution; partial-year rows
- [ ] Test rows' exposure: full year or variable? Prediction must scale with it
- [ ] Exposure by year (portfolio growth/shrink)

## 4. Frequency, severity, burning cost
For each line × policy year:
- [ ] Claim frequency per exposure, % rows with claims (zero-inflation)
- [ ] Average severity, severity distribution (log scale), quantiles 50/90/99/99.9
- [ ] Burning cost (target / exposure), loss ratio vs premium if premium given
- [ ] By peril / claim type if available (e.g. motor own damage vs third party; buildings weather vs escape of water)

## 5. Large losses and tail
- [ ] Top 20 claims per line; share of total cost from top 1% claims
- [ ] Catastrophe/event clustering (many claims same date/region → weather events)
- [ ] Decide candidate threshold for capping + large-loss loading (human gate)

## 6. Development and IBNR
- [ ] Reporting delay distribution (report date − loss date)
- [ ] Is the latest year (2024) less developed? Compare frequency/severity of 2024 vs earlier at same maturity
- [ ] If immature: chain-ladder or ratio-based development factors by line; record as assumption

## 7. Trend and inflation
- [ ] Frequency and severity trend by year (per exposure, mix-adjusted if possible)
- [ ] Sum insured / vehicle value inflation
- [ ] Mix shift over years (new segments, channel, region)
- [ ] Candidate 2025 trend factors with ranges (human gate)

## 8. One-way and two-way relativities
- [ ] Observed burning cost by each rating factor (with exposure and claim counts shown)
- [ ] Credibility of thin levels; banding candidates for continuous variables
- [ ] Key interactions (age × vehicle power, region × property age, etc.)
- [ ] Correlation/collinearity among features

## 9. Cross-line view
- [ ] Same customer / address across buildings and contents? Shared features?
- [ ] Line-level differences in data quality

## 10. AI usage in EDA (log each)
- Ask AI to propose additional checks → verify each against data
- Ask AI to explain anomalies → confirm with a query or plot before believing
- Ask AI to draft the data dictionary → human checks each field description
