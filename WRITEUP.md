# Estimating Vehicle CO2 Emissions for Financed-Emissions Accounting

*A proof-of-concept exploring emissions estimation for motor-vehicle loan portfolios*

---

## Context

Under PCAF (Partnership for Carbon Accounting Financials), a lender attributes a
share of each financed vehicle's emissions to its own balance sheet. For motor
vehicles the attributed figure depends on an emissions-per-kilometre input for
each car. In practice a portfolio contains vehicles whose certified emissions are
missing, unreliable, or — for newer powertrains — not a meaningful single number
at all.

This project is an independent, public-data proof-of-concept that asks a narrow
question: **given only vehicle characteristics, how well can we estimate a car's
CO2 emissions, and where does that approach break down?** It is built entirely on
the European Environment Agency's public monitoring dataset of CO2 emissions from
new passenger cars. It uses no proprietary or portfolio data.

## Headline finding

The right method is not one model — it is three, because a vehicle fleet contains
three physically different regimes:

| Regime | Example fuel types | Method | Why |
|---|---|---|---|
| Combustion (ICE) | petrol, diesel, lpg | Machine-learning model | Emissions track the hardware; features determine the target |
| Zero-tailpipe | electric, hydrogen | Assign 0 g/km | Zero by definition; prediction adds nothing |
| Plug-in hybrid (PHEV) | petrol/electric, diesel/electric | Certified value + real-world correction | Emissions depend on driver charging behaviour, which no vehicle dataset can observe |

The value of the project is less the model than the **map of where a model helps
and where it does not** — which is the question a financed-emissions methodology
has to answer for every vehicle in the book.

## The combustion model

For petrol/diesel-type vehicles, CO2 is close to a physical function of fuel burned,
which in turn tracks engine size, power, mass, and body category. A gradient-boosted
model (XGBoost) trained on distinct vehicle specifications estimates certified WLTP
CO2 with the following honest, held-out performance:

| Metric | Value |
|---|---|
| MAE | 8.3 g CO2/km |
| RMSE | 14.0 g CO2/km |
| R² | 0.909 |
| MAE as % of mean emissions | ~5% |

Methodology notes that matter for trust:

- **Honest evaluation.** Model and preprocessing were selected by cross-validation
  on the training data only; the reported figures come from a test set never used
  in selection. The reported number is an unbiased estimate, not a fitted one.
- **De-duplication to specifications.** The raw dataset is ~10.8M registration
  events. These were reduced to distinct specification combinations before splitting,
  so identical cars cannot leak across the train/test boundary.
- **Data-integrity filtering.** A cluster of combustion rows carried an implausible
  placeholder emissions value (~57 g/km repeated) alongside missing engine power —
  a systematic reporting gap, not real measurements. These were removed so they
  neither train nor score the model.

## Why plug-in hybrids need a different approach

The single most important finding is that PHEVs cannot be estimated the same way,
and it is worth being explicit about why, because it is the exact issue that makes
PHEVs a known headache in emissions accounting.

A PHEV's real emissions depend on **what fraction of its kilometres are driven on
battery versus engine** — its electric-driving share. That share is a property of
the *driver* (how often they plug in, how far they drive), not of the *car*. Two
identical vehicles with different owners genuinely emit differently.

Because a certificate still needs a single number, the type-approval test applies
an assumed electric-driving share (the "utility factor"). This is why a large-engine
PHEV can be certified at implausibly low emissions: the test assumes it runs mostly
electric. Published real-world evidence (ICCT, 2022–2024) finds real-world fuel
consumption of PHEVs runs materially higher than the label — on the order of
40–70% higher — because actual charging is less frequent than the test assumes.

Consequently, for PHEVs this project does **not** train a model to predict the true
value (the true value is unobservable from vehicle data). Instead it takes the
certified figure as a known, systematically biased input and applies a documented
real-world **correction factor**, scaled by electric range where available
(longer range → certificate closer to reality → smaller correction), with flat and
group-median fallbacks when inputs are missing. The correction magnitudes are drawn
from published studies, not fitted to the registry — because the registry contains
only certified values, so the real-world gap cannot be derived from it.

This is deliberately more honest than a model that would appear more sophisticated:
it reports a bias-corrected estimate rather than a confident prediction of something
the data cannot support.

## Removing PHEVs sharpened the combustion model — a quantified check

Training the combustion model on combustion-only vehicles (versus all fuel types
mixed together) improved it, and the *shape* of the improvement confirms the
diagnosis:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| All fuel types | 18.7 | 10.1 | 0.923 |
| Combustion-only | 14.0 | 8.3 | 0.909 |

RMSE fell proportionally more than MAE (roughly 25% versus 17%). Because RMSE
penalises large errors more heavily, this is direct evidence that the plug-in
hybrids were the source of the large-error tail — exactly as a worst-error
diagnostic had suggested. The small decline in R² is expected and not a
regression in quality: removing near-zero-emission vehicles reduces the spread of
the target, so an equally good model explains a smaller share of a smaller variance.

## Scope and limitations

Stated plainly, because naming limitations is part of the method:

- **Powertrain scope.** The ML model applies to conventional combustion vehicles.
  Zero-tailpipe vehicles are assigned 0; PHEVs use the corrected-certificate
  approach rather than the model.
- **Certified vs. real-world.** For combustion vehicles the model targets the
  *certified* WLTP figure. Real-world emissions are known to exceed certified
  values; a portfolio application would apply a real-world adjustment consistent
  with the approach used for PHEVs.
- **PHEV correction is population-based.** The correction reflects average
  real-world behaviour from published fleets, not any individual vehicle's usage.
  Closing that gap would require behavioural or telemetry data (charging frequency,
  trip distances) that is not present in vehicle registries.
- **No portfolio data used or implied.** This is a public-data feasibility study.
  Applying it to a specific book would require that book's own vehicle attributes
  and would surface additional data-quality questions.

## What a next step could look like

- Retrain the combustion model targeting real-world-adjusted emissions rather than
  certified, for consistency with the PHEV branch.
- Build a dedicated PHEV model using electric range and energy-consumption features
  to estimate the *plausible envelope*, reported with explicit uncertainty.
- Integrate the estimator into the PCAF attribution formula end-to-end, taking
  loan-to-value and annual distance as the remaining inputs.

---

*Independent project built on public EEA data. Not affiliated with or commissioned
by any institution. Shared as a good-faith exploration of the estimation problem.*




# Estimating the Financed Emissions of a Car-Loan Portfolio
 
A two-model system that estimates the CO₂ a lender should book against its
auto-loan book — and, just as importantly, how good that estimate is.
 
- **Model A** predicts per-vehicle emissions (gCO₂/km) from the handful of
  attributes a loan record actually carries.
- **Model B** turns those per-vehicle numbers into portfolio financed emissions
  under the PCAF accounting standard, and scores their data quality.
The headline finding is methodological, not a leaderboard number: **a car fleet is
three estimation problems, not one**, and a machine-learning model of the first
regime is not a redundant regression — joined to a loan book it is a *documented
PCAF data-quality upgrade*, the exact tool a consumer lender needs to move an auto
portfolio off a blunt money-based estimate and onto a defensible one.
 
---
 
## 1. The problem, and why anyone cares
 
When a bank finances a car, it becomes partly responsible — under carbon
accounting rules — for the CO₂ that car emits over its life. This is *financed
emissions*: Scope 3, Category 15. Three overlapping regimes now require lenders to
measure and report it: the PCAF Global GHG Accounting and Reporting Standard (the
methodology), CSRD / ESRS E1 (the EU disclosure mandate), and the SBTi Financial
Institutions Net-Zero standard (the target-setting).
 
PCAF's formula for a single motor-vehicle loan is simple:
 
```
financed emissions = attribution factor × vehicle emissions
attribution factor = outstanding amount / vehicle value at origination
vehicle emissions  = annual distance (km) × emission factor (gCO₂/km)
```
 
Here is the catch that creates the whole project. **A lender knows the loan side
and not the vehicle side.** From its book it knows the outstanding balance and the
origination value, so the attribution factor is trivial. What it usually does *not*
have joined to each loan is the car's real gCO₂/km, and often not the exact
make/model or how far it is driven. Without help, it can only report at the lowest
PCAF data-quality tier — score 5, where emissions are guessed from the loan amount
alone. That is a weak, hard-to-defend number.
 
PCAF grades this explicitly on a 1–5 ladder (1 = measured, 5 = economic proxy).
Closing the gap from "loan amount only" to "vehicle-specific estimate" is a *data
quality* problem, and that is precisely where a predictive model earns its place.
 
**Real-world anchor.** This is a live gap, not a toy. BNP Paribas Personal Finance
— a large European consumer lender — publicly discloses its *operational* emissions
but has not published a *financed* auto-emissions figure, and has run a
data-science collaboration with the Institut Louis Bachelier since 2019 that
includes work on net-zero scenarios for its automotive portfolio. Its group parent
reports an auto-financing intensity of ~141 gCO₂/km (2025, target 115–136 by 2030);
its leasing arm Arval reports ~105 gCO₂/km; Santander Consumer Finance reports
~133 gCO₂/vkm at a PCAF data-quality score of 2.7. The peers who *have* disclosed
give us benchmark numbers to sanity-check against. (See references; the specifics
of any one firm's internal tooling are not public — this project reconstructs the
*general* problem those disclosures imply.)
 
---
 
## 2. Model A — per-vehicle emissions, by regime
 
The core idea is that emissions relate to vehicle hardware in completely different
ways depending on the powertrain, so one model cannot serve the whole fleet:
 
| Regime            | Fuel types                          | Method                                   |
|-------------------|-------------------------------------|------------------------------------------|
| Combustion (ICE)  | petrol, diesel, lpg, e85, ng        | ML model (XGBoost, chosen by CV RMSE)    |
| Zero-tailpipe     | electric, hydrogen                  | Assigned 0 gCO₂/km (tank-to-wheel)       |
| Plug-in hybrid    | petrol/electric, diesel/electric    | Certified value + real-world correction  |
 
A three-way router sends each vehicle to the right method. Two decisions are worth
defending:
 
- **Plug-in hybrids are excluded from the ML model, not hidden.** Their real
  emissions depend on how often the driver charges — behaviour that no vehicle
  specification dataset can observe. Modelling them as if hardware determined
  emissions is what produces the large-error tail; removing them and correcting
  their optimistic certified value separately is the honest fix. Diagnosing this
  (the combustion-only model cut RMSE ~25% versus an all-fuels model,
  proportionally more than MAE — the signature of a heavy tail being removed) was
  the key EDA finding.
- **No target leakage.** Fuel consumption (L/100km) is near-perfectly collinear
  with CO₂ by physics; including it would inflate accuracy trivially. It is
  excluded. The model predicts gCO₂/km from mass, engine capacity, power, fuel
  type and transmission — the attributes a loan/registration record plausibly
  carries.
**Result (held-out combustion vehicles):** MAE ~8.3 gCO₂/km (~5% of the mean),
R² ~0.91. *(Update with your final `02_clean.ipynb` run.)* An R² of ~0.91 rather
than ~0.99 is itself evidence the model is doing real work rather than reading a
consumption column back to itself.
 
Model selection is by cross-validated RMSE across a decision-tree / random-forest /
XGBoost bake-off; the test set is touched once, as an honest final estimate.
 
---
 
## 3. Model B — PCAF accounting layer
 
Model B is deliberately decoupled from Model A: it consumes a frame that already
carries a `co2_estimate` column and never re-derives per-km emissions, so the
regime logic is decided once. It contributes three things.
 
**Attribution.** `outstanding ÷ origination value`, capped at 1 (a lender cannot
be booked for more than 100% of a car's emissions, even on an under-water loan).
Origination value is fixed at loan start, so the reported figure does not swing
with second-hand resale prices.
 
**Two PCAF methods, side by side.** The same portfolio is costed two ways to make
the data-quality upgrade visible:
 
- *Physical* (PCAF data option 2) — attribution × Model-A gCO₂/km × assumed annual
  km. This is the score-3 estimate.
- *Economic* (PCAF data option 3) — attribution × origination value × an emission
  factor per euro. This is the score-5 fallback, and it is exactly what Model A
  exists to replace.
**Data-quality scoring.** Each loan records *how* its per-km figure was obtained
(`measured_fuel` → `certified_specific` → `modelled_specific` → `segment_average`
→ `economic`), which maps to a PCAF score 1–5. The portfolio then carries an
exposure-weighted average score, and mileage — the second unknown on the vehicle
side — is handled as a documented lookup rather than an ML target, because a loan
record carries no signal about how far a *specific* driver goes.
 
---
 
## 4. Results
 
On an illustrative **synthetic** loan book (a stand-in for real securitisation
data — see limitations), the two methods diverge sharply on the *same* exposure:
 
| Basis                                  | Absolute financed | Weighted PCAF score |
|----------------------------------------|-------------------|---------------------|
| Baseline — loan amount only (economic) | 7,459 tCO₂e/yr    | 5.0                 |
| With Model A — modelled specs (physical) | 4,221 tCO₂e/yr  | 3.0                 |
 
Same loans, same money — a **43% swing** in the headline number and a data-quality
score moving 5.0 → 3.0. That gap *is* the argument for the model: the economic
fallback is not merely lower-quality, it materially misstates the total.
 
The physics-grounded intensity comes out at ~144 gCO₂/km, which lands right on the
independent peer benchmarks (BNP Paribas group 141, Santander 133) — a reassuring
external sanity check that the aggregation is sound.
 
**Sensitivity.** Scaling the mileage assumption ±20% moves the absolute total from
~3,380 to ~5,070 tCO₂e/yr. Mileage is the dominant assumption, so the project
reports a band, not a point — the same discipline a methodology team applies.
 
*(All Model-B figures above are on synthetic data and are illustrative of the
method, not a claim about any real portfolio. Re-run `03_portfolio.ipynb` with
`DEMO = False` and a real book to produce reportable numbers.)*
 
---
 
## 5. Data sources
 
- **EEA — CO₂ emissions from new passenger cars** (Regulation (EU) 2019/631
  monitoring dataset). Every new EU car with mass, engine capacity, power, fuel
  type and WLTP CO₂. The training set for Model A. Open.
- **ADEME Carlabelling** — French homologation data (technical specs, CO₂,
  pollutants). French-specific complement. Open via data.gouv.fr.
- **Mileage** — French national average ~11,600 km/yr, fuel-type-conditioned
  (ADEME / national mobility survey). The dominant Model-B assumption.
- **Emission factors** — ADEME Base Carbone (France; includes well-to-wheel);
  DEFRA / IEA for grid factors.
- **Loan-level data (future)** — US Reg-AB auto-ABS (SEC EDGAR, free) or European
  DataWarehouse auto-ABS templates, to replace the synthetic book.
---
 
## 6. Design decisions and honest limitations
 
The judgment calls are the point of the project, so they are stated, not hidden:
 
- **WLTP vs real-world.** Combustion CO₂ is homologation (WLTP), which understates
  real-world emissions by roughly 20–40%. The current headline is homologation-based
  for peer comparability; a real-world correction factor is a natural extension.
- **EV convention.** Battery-electric vehicles are 0 gCO₂/km *tank-to-wheel*, the
  PCAF/peer convention. A well-to-wheel toggle (grid emissions) is included and is
  small on France's low-carbon grid, but it makes explicit that tailpipe zero is a
  convention, not a life-cycle truth.
- **Plug-in hybrids are corrected, not measured.** With no observable charging
  behaviour, their figure is a plausibility-checked adjustment of the certified
  value, not a validated prediction. This is a known, bounded weakness.
- **Mileage is an assumption, not a measurement.** It is the largest lever on the
  total and is reported as a sensitivity band.
- **The loan book is synthetic.** Attribution mechanics are real; the specific
  euro amounts are illustrative until real ABS data is wired in.
- **The economic emission factor is illustrative.** In production it would be a
  PCAF-published asset-class factor; here it is a transparent, auditable stand-in.
---
 
## 7. Reproducibility
 
```
├── 01_exploration.ipynb   # EDA, the worst-error diagnostic, the PHEV finding
├── 02_clean.ipynb         # Model A: clean pipeline, bake-off, saved model
├── 03_portfolio.ipynb     # Model B: PCAF accounting on a portfolio
├── src/
│   ├── emissions.py        # Model A logic (router, ICE model, PHEV correction)
│   └── portfolio.py        # Model B logic (attribution, PCAF methods, scoring)
└── models/ice_model.pkl    # trained combustion model
```
 
`03_portfolio.ipynb` runs standalone with `DEMO = True` (synthesises specs, needs
neither the EEA data nor the trained model), so a reviewer can execute the full
accounting layer end-to-end. Set `DEMO = False` to use the real Model-A output.
 
---
 
## 8. What I'd do next
 
1. **Wire in real loan-level data** (US Reg-AB auto-ABS) to replace the synthetic
   book — the single biggest credibility upgrade.
2. **Add a real-world correction** for combustion vehicles, citing the EEA's
   real-world-gap monitoring.
3. **Model the data-quality tiers explicitly** — train Model A on progressively
   fewer features (exact spec → make/model → segment) to quantify how much
   estimation accuracy each PCAF score buys.

   SHAP on the ICE model to turn feature importance into a stakeholder-readable story; a temporal split (train on older registration years, test on newer) which is more honest than random for a deployment scenario; a scan for other sentinel/placeholder values beyond the ~57 g/km cluster; and a lookup-table baseline ("model vs. join-to-certified-then-segment-average") to prove the model earns its place.

   Your PHEV "lift" of 1.25–1.7 scaled by electric range is currently invented. That's the most attackable thing in the notebook. Ground it in a published real-world utility factor — the EEA has released OBFCM on-board fuel-consumption data showing certified PHEV figures understate real emissions by a large multiple. Cite that instead of a hand-chosen constant, and the whole PHEV branch goes from "plausible" to "sourced."

   our model trains on EEA data where mass/capacity/power are always present. On a real loan book, which of those are actually captured at origination? Test degraded inputs: what's the RMSE if you only have make/model/fuel/year? If accuracy collapses when the rich features go missing, the model won't transfer — and "why it can't just be looked up" is exactly a missingness problem, so this makes your narrative coherent end-to-end

   Uncertainty quantification. A number feeding regulatory reporting needs error bars. Quantile loss (LightGBM/GBM) or conformal prediction gives per-vehicle intervals; then propagate them to the portfolio total. Bonus: it lets you honestly flag that PHEVs and the "electric = 0" assumption carry more model risk than the ICE estimates.
---
 
## References
 
- PCAF, *The Global GHG Accounting and Reporting Standard for the Financial
  Industry*, Part A — motor vehicle loans asset class.
- EU, *ESRS E1 (Climate change)* under the Corporate Sustainability Reporting
  Directive; Scope 3 Category 15.
- Science Based Targets initiative, *Financial Institutions Net-Zero Standard*
  (2025).
- European Environment Agency, *Monitoring of CO₂ emissions from passenger cars —
  Regulation (EU) 2019/631*.
- ADEME, *Carlabelling* and *Base Carbone*.
- BNP Paribas Personal Finance & Institut Louis Bachelier, data-science
  collaboration (public communications, 2019–).
- BNP Paribas Group, sustainable-finance progress figures (auto-financing
  intensity). Arval, CSRD sustainability report (fleet intensity). Santander
  Consumer Finance, FY2024 climate disclosure (auto-lending intensity, PCAF score).
*Peer figures are used only as external sanity checks; this project reconstructs
the general PCAF problem and does not reproduce any firm's proprietary method.*