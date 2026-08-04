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
