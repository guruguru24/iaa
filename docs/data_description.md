# Data description (developed = Week 1; emerging = Week 2 differences in Part C)

Part A is the organisers' description, kept faithful. Part B is our actuarial interpretation: **hypotheses to verify
against the data**, not facts. Verified results go to `logs/data_quality_log.md` / `logs/assumptions_register.md`.

---

# Part A — Official description

## Files (Parquet)
| File | One row per | Description |
|---|---|---|
| customers | Customer | Customer details, shared across all their policies |
| addresses | Address | Location and property characteristics |
| weather | Address and year | Annual weather and climate measures for each address |
| policies_motor | Policy period | Training data for private motor, 2021 to 2024 |
| policies_buildings | Policy period | Training data for home buildings, 2021 to 2024 |
| policies_contents | Policy period | Training data for home contents, 2021 to 2024 |
| claims | Claim | All reported claims on the training policies, as at 31 December 2024 |
| test_motor | Policy period | 2025 motor policies to predict |
| test_buildings | Policy period | 2025 buildings policies to predict |
| test_contents | Policy period | 2025 contents policies to predict |
| sample_submission | Test row | A submission in the correct format |

## How the tables link
- `customer_id` links policies to customers.
- `address_id` links policies to addresses and weather.
- `policy_id` links claims to policies. A policy keeps the same `policy_id` across years, so **match each claim to a policy row using its `loss_date`**.
- `row_id` identifies each test row in the submission.
- A customer can hold motor, buildings and contents policies, and several customers can share an address.

## Policy periods
Each policy row covers one policy for one calendar year, or part of a year. A policy has more than one row in a year
if it changed mid-term (e.g. new vehicle, change of address). Rows end early when a policy is cancelled or lapses.
`exposure` is the length of the row in years.

## Data types
Identifiers (`customer_id`, `address_id`, `policy_id`, `row_id`, `claim_id`) are **text**. Read them as text, not numbers.

## customers
| Column | Description |
|---|---|
| customer_id | Customer identifier |
| surname | Customer surname (anonymised) |
| date_of_birth | Date of birth |
| occupation | Occupation |
| marital_status | single, married, divorced, widowed |
| household_size | Number of people in the household |
| credit_band | Credit score band, A (best) to E |
| prior_claims_5y | Claims in the five years before joining, as declared by the customer |
| licence_points | Penalty points on the driving licence |
| years_at_address | Years the customer has lived at their address |
| customer_since_year | Year the customer first took out a policy |
| email_domain | Email provider |
| marketing_consent | Consented to marketing |
| document_preference | email, post or app |
| previous_insurer | Insurer before joining |
| campaign_code | Marketing campaign at acquisition, if any |
| branch_id | Servicing branch |
| marketing_segment | Segment assigned by the marketing team |

## addresses
| Column | Description |
|---|---|
| address_id | Address identifier |
| region | Region code, R01 to R12 |
| urban_density | rural, suburban, urban, dense_urban |
| flood_zone | Flood risk zone, 1 (lowest) to 5 (highest) |
| distance_to_river_km | Distance to the nearest river |
| elevation_m | Elevation above the local river level, in metres |
| soil_type | clay, sand, loam, chalk, rock |
| year_built | Year the property was built |
| property_type | detached, semi_detached, terraced, flat |
| roof_type | pitched or flat |
| basement | Property has a basement |
| trees_within_5m | Trees within 5 m of the property |
| listed_building | Protected (listed) building |
| bedrooms | Number of bedrooms |
| energy_rating | Energy efficiency rating, A (best) to G |
| solar_panels | Solar panels fitted |

## weather
| Column | Description |
|---|---|
| address_id | Address identifier |
| year | Calendar year |
| hail_days | Days with hail in the year |
| wind_exposure_index | Exposure to damaging wind, 0 to 10 |
| freezing_days | Days below freezing |
| dry_summer_index | Summer dryness, 0 (wet) to 1 (very dry) |
| wildfire_index | Wildfire risk, 0 to 1 |
| rainfall_mm | Annual rainfall |
| mean_temperature_c | Annual mean temperature |

## Columns in all policy and test files
| Column | Description |
|---|---|
| row_id | Row identifier |
| policy_id | Policy identifier, constant across years |
| customer_id | Policyholder |
| address_id | Insured address (for motor, the policyholder's home address) |
| period_start, period_end | First and last day covered by the row |
| exposure | Length of the row in years |
| voluntary_excess | Excess chosen by the customer: 0, 250, 500 or 1,000 CU |
| compulsory_excess | Excess set by the insurer |
| channel | Sales channel: direct, comparison_site, broker |
| payment_method | annual or monthly |
| missed_payments_12m | Missed instalments in the previous 12 months |
| new_business | First policy year with the insurer |
| renewal_count | Number of previous renewals |
| days_quote_to_inception | Days between quote and the start of cover |
| discount_applied | Discount given on the premium |
| written_premium | Premium charged under the insurer's current pricing |
| customer_service_contacts | Contacts with customer service |
| renewal_decision | **Training files only**: renewed, lapsed_by_customer, cancelled, declined_by_insurer, mid_term_change |

## Motor columns
| Column | Description |
|---|---|
| driver_age | Age of the main driver at the start of the row |
| years_licensed | Years the main driver has held a licence |
| ncb_years | No-claims bonus years |
| vehicle_value_group | Vehicle value group, 1 (lowest) to 20 |
| vehicle_age | Vehicle age in years |
| make_model | Make and model code |
| engine_size_cc | Engine size |
| body_type | hatchback, saloon, estate, suv, coupe |
| transmission | manual or automatic |
| seats | Number of seats |
| vehicle_colour | Colour |
| fuel_type | petrol, diesel, hybrid, electric |
| driver_assist | Fitted with driver-assist features such as automatic braking |
| cover_type | comprehensive, third_party_fire_theft, third_party_only |
| annual_mileage | Declared annual mileage |
| overnight_parking | garage, driveway, street |
| vehicle_use | social, commuting, business |
| named_young_driver | A driver under 25 is named on the policy |
| telematics_opt_in | Customer opted in to a telematics device |
| telematics_score | Driving score from the device, 1 to 100 (blank if not opted in) |

## Buildings columns
| Column | Description |
|---|---|
| policyholder_age | Age at the start of the row |
| sum_insured | Buildings sum insured |
| construction_type | brick, stone, timber, non-standard |
| occupancy | owner_occupied or let |
| days_unoccupied | Longest period the property was unoccupied |
| smoke_alarms | Smoke alarms fitted |
| accidental_damage_cover | Optional accidental damage cover taken |
| mortgage | Property is mortgaged |

## Contents columns
| Column | Description |
|---|---|
| policyholder_age | Age at the start of the row |
| sum_insured | Contents sum insured |
| tenure | owner or renter |
| security_level | standard, locks, alarm_and_locks |
| high_value_items | Number of declared high-value items |
| accidental_damage_cover | Optional accidental damage cover taken |
| works_from_home | Policyholder works from home |
| lodgers | Lodgers or students in the household |
| bicycle_cover | Optional bicycle cover taken |

## claims
| Column | Description |
|---|---|
| claim_id | Claim identifier |
| policy_id | Policy the claim was made on |
| line | motor, buildings, contents |
| peril | Cause of loss (see below) |
| loss_date | Date of the incident |
| report_date | Date the claim was reported |
| status | closed, open, repudiated (declined) |
| paid_to_date | Amount paid so far, **net of excess** |
| incurred_to_date | Paid to date plus the current case estimate for open claims |
| recoveries | Amounts recovered from third parties or salvage |
| repairer_id | Repairer used (motor damage claims) |
| third_party_id | Third party involved (motor third-party claims) |

**Perils** — Motor: own_damage, third_party_prop, bodily_injury, theft, glass_hail.
Buildings: storm, flood, escape_of_water, subsidence, fire, accidental_damage.
Contents: theft, escape_of_water, accidental_damage, flood.

Claims are valued at **31 December 2024**. Some 2024 claims had not been reported by then (**IBNR**), and open claims may still develop (**IBNER**).

## sample_submission
| Column | Description |
|---|---|
| row_id | Row identifier from the test files |
| prediction | Expected claim cost for the row, in CU (must be zero or above) |

One file covering every row in all three test files.

## Glossary
| Term | Meaning |
|---|---|
| CU | Currency unit (fictional) |
| Exposure | Time on cover, in years |
| Excess | The part of each claim the policyholder pays |
| NCB | No-claims bonus: years without a claim |
| Sum insured | The maximum the policy pays |
| Incurred | Paid plus outstanding case estimate |
| Ultimate | The final cost of claims once all are reported and settled |
| Repudiated | A claim the insurer declined to pay |
| Comparison site | A price-comparison website |
| Telematics | A device that records driving behaviour |

---

# Part B — Actuarial interpretation and hypotheses to verify

## B1. Target construction (gate G1)
Proposed per-claim net cost (to verify on data):
```
net_cost = incurred_to_date - recoveries        # status in {closed, open}
net_cost = 0                                    # status == repudiated (excluded)
```
Then sum by policy row, matching claim → row on `policy_id` AND `period_start <= loss_date <= period_end`.

Checks before approval:
- [ ] `paid_to_date` is already net of excess → confirm `incurred_to_date` is too (is incurred ≥ paid? any claims below excess with 0 paid?)
- [ ] Is excess = voluntary + compulsory? Do small claims show cost ≈ 0 when gross < excess?
- [ ] Are recoveries already netted inside paid/incurred? (check claims with recoveries > 0: incurred − recoveries < 0?) — decide floor at 0 per claim or not
- [ ] Repudiated claims: any non-zero paid/incurred? Excluded regardless
- [ ] Claims whose loss_date matches no row (gaps, outside period) or more than one row (overlaps) → log
- [ ] Claims on other lines' policy_ids? Confirm `line` matches the policy file
- [ ] Reconcile Σ net_cost by line × year: claim-level vs row-level

## B2. Development to ultimate (gate G3)
- Valuation date 31 Dec 2024 → **2024 (and part of 2023) is immature**: missing IBNR and open-claim development.
- Plan: build report-delay and incurred-development triangles by line (and peril if credible) using loss/report dates;
  estimate IBNR count and IBNER factors (chain-ladder style), then either
  (a) gross up 2023–2024 targets to ultimate, or (b) use as weights/offset in modelling.
- Long-tail perils: motor bodily_injury, buildings subsidence (likely slow reporting and development) — treat separately.
- Validation subtlety: validating on 2024 actuals compares against **immature** values → compare with developed estimates or accept known bias, and document.

## B3. Matching test periods
- Test rows are 2025 policy periods with their own `exposure`; prediction = expected cost for that row (scale with exposure).
- `renewal_decision` is absent in test (outcome information).
- Weather is by address × year. *Corrected after checking the data (DQ14):* the hypothesis "2025 weather is not given"
  was wrong — the weather table **does contain 2025** (developed 2021–2025, emerging 2023–2025). Those values are the
  realised 2025 weather, which is unknown when the 2025 policies are priced, so using them would leak future events.
  Decision #19: both training and test use address-level **long-run averages over the training years only**
  (developed 2021–2024; emerging 2023–2024; config `weather_years` / `w2.train_years`); same-year weather is kept only
  as a sensitivity.

## B4. Leakage and feature-use review
| Feature | Risk | Default treatment |
|---|---|---|
| renewal_decision | Outcome after the period; not in test | **Exclude** |
| weather (same year) | Concurrent events, not known at pricing | Use averages / lagged; test both, document |
| customer_service_contacts | May include claim-related contacts in the period | Check correlation with claims; likely exclude or use lagged |
| missed_payments_12m | "previous 12 months" — check timing | Keep if measured at row start |
| written_premium, discount_applied | Current pricing output; strong proxy; circular | Analyse as benchmark (A/E vs premium); model **with and without**, decide at G4 |
| ncb_years, prior_claims_5y | Legit rating factors (history before period) | Keep |
| telematics_score | Missing when not opted in (informative) | Keep with opt-in flag |
| repairer_id, third_party_id | Claim-level, post-loss | **Exclude** |
| report_date, status | Post-loss | Only for development analysis |

## B5. Fairness / governance flags
- `surname`, `email_domain`, `marketing_segment`, `campaign_code`, `branch_id`, `vehicle_colour`: little causal rationale;
  some are possible proxies for protected characteristics (e.g. surname → ethnicity). Default **exclude** `surname`;
  test others for real lift and stability; document decisions (gate G4).
- `date_of_birth`/age, `occupation`, `credit_band`, `marital_status`: commonly used but regulated in some markets → note in report.

## B6. Useful derived features (hypotheses)
- Ages at row start from `date_of_birth`; property age = year − `year_built`; tenure = year − `customer_since_year`.
- Multi-line holding flags (customer has motor/buildings/contents); number of customers at the address.
- Sum insured per bedroom; vehicle value × cover type; mileage bands; NCB × age.
- Excess total = voluntary + compulsory (directly reduces net cost).
- Address-level long-run weather averages (hail, wind, freezing, rainfall, dry summer for subsidence).
- Peril-specific views for buildings/contents (flood ↔ flood_zone/elevation/distance_to_river; subsidence ↔ soil_type/trees/dry_summer; escape_of_water ↔ freezing_days/days_unoccupied/property age).

## B7. Modelling implications
- Consider **peril-level models** for buildings and contents (different drivers), summed to row level; compare with a single Tweedie model.
- Mid-term changes create multiple rows per year → split exposure correctly; grouped CV by `policy_id`.
- Same customer/address across lines → possible shared risk signal (cross-line features), but keep CV grouped by customer to avoid optimistic validation.

---

# Part C — Emerging market (Week 2): official differences and implications

Source: updated official dataset description (2026-10-24). Folder layout: `developed/` (Part 1, 2021–2024) and
`emerging/` (Part 2, 2023–2024). **Part 2 predicts the emerging market only.**

## C1. Official statements (new or changed)
- Data comes "as from an insurer's operational systems: in several linked tables, and **not cleaned**".
- Target explicitly **includes claims not yet reported or settled** (IBNR + IBNER) — confirms ultimate basis.
- Prediction is a **total for the row, not an annual rate** (exposure 0.5 = half a year).
- **The policy excess applies to all perils, including third-party perils.**
- **Join tables only within the same market folder.** customer/address/policy IDs can repeat across markets for unrelated records.
  Never join developed to emerging. Test `row_id`s are unique across both markets.
- sample_submission is CSV (emerging); other files Parquet.

| | Developed | Emerging |
|---|---|---|
| Training years | 2021–2024 | 2023–2024 |
| Weather years | 2021–2024 | 2023–2024 |
| Regions | R01–R12 | E01–E06 |
| Claims valued at | 31 Dec 2024 | 31 Dec 2024 |

Verified on data: the weather files also contain a 2025 year in both markets (realised weather; not used, see B3).

## C2. Column differences
| Table | Column | Developed | Emerging |
|---|---|---|---|
| customers | date_of_birth | yes | **absent** |
| customers | gender | absent | **M/F — may be used for pricing in emerging** |
| customers | occupation | categorical | **free text as entered** (needs cleaning/mapping) |
| customers | credit_band | A–E | **absent** |
| addresses | region | R01–R12 | **E01–E06** (no mapping to developed) |
| motor | driver_age | exact age | **absent** → `driver_age_band`: 17-24, 25-34, 35-49, 50-64, 65+ |
| motor | annual_mileage | yes | **absent** (not collected) |
| motor | telematics_opt_in / telematics_score | yes | **absent** (not offered) |
| buildings, contents | policyholder_age | exact age | **absent** → `policyholder_age_band` (same bands) |
| buildings | construction_type | brick, stone, timber, non-standard | **BRK, STN, WOD, MIX** (MIX = mixed or non-standard) |
All other columns are described as the same in both markets (values/distributions may still differ — verify).

## C3. New files (emerging folder)
| File | One row per | Content |
|---|---|---|
| expenses | Cost item | `item`, `developed`, `emerging` (blank if n/a), `unit` (CU / share of premium / share of property risk premium). Items: fixed expense per policy, variable expense, commission by channel, premium tax, catastrophe reinsurance, target profit margin |
| reference_policies | Policy | 20 emerging 2025 test policies (+ `line`). **Report must show the office premium build-up from our risk premium for each** — judged by panel, not on Kaggle. See `docs/07_office_premium.md` |

## C4. Implications (hypotheses to verify)
1. **Harmonised feature set for transfer**: build a developed model on features available in both markets only
   (age → same bands; construction mapped brick→BRK, stone→STN, timber→WOD, non-standard→MIX; drop credit, mileage, telematics, exact age;
   region excluded or replaced by address-level physical/weather features). Quantify the lift lost vs the full Week 1 model.
2. **Region** cannot transfer (different codes) → explain geography with flood_zone, elevation, distance_to_river, soil, urban_density,
   weather; region effect in emerging estimated with credibility (6 regions, 2 years).
3. **Gender** is permitted in emerging: test for lift and stability; document the regulatory difference and the decision (gate G4).
   Developed has no gender → no prior; credibility-weight toward 1.0.
4. **Occupation free text**: normalise (case, spaces, typos, synonyms) → map to developed occupation list where possible; log mapping
   quality (AI can propose the mapping — **human verifies a sample**, log in AI usage log).
5. **"Not cleaned"**: expect inconsistent codes, typos, out-of-range values, duplicates, date issues → run the full EDA checklist §1 on emerging.
6. **Excess on third-party perils**: target construction must not add excess back for TP perils; check paid/incurred vs excess on TP claims in both markets.
7. **Combining markets**: if stacked for a joint model, prefix IDs with market (`dev_`, `emg_`) and keep a `market` column; grouped CV keys per market.
8. **Weather**: only 2023–2024 → address-level 2-year averages; compare ranges with developed (extrapolation risk for climate-driven perils).
9. **Development**: only accident years 2023–2024 in emerging → development patterns from developed (by line/peril) probably needed;
   check emerging report delays vs developed before borrowing (gate G3).
10. **New business 2025**: broker portfolio → check test mix (channel, new_business, renewal_count, NCB) vs training; validate on new-business subset.
