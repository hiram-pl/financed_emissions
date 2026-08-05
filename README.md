# Vehicle CO2 Emissions Estimator

Estimating passenger-vehicle CO2 emissions from vehicle specifications — and
knowing when *not* to use a machine-learning model to do it.

**[Live demo](#)** · **[Methodology write-up](WRITEUP.md)**

> Replace the live-demo link with your Streamlit URL once deployed.

---

## TL;DR

A fleet of cars is not one estimation problem — it's three. This project routes
each vehicle to the right method:

- **Combustion cars (petrol/diesel):** an XGBoost model predicts CO2 from engine
  size, power, mass, and category. Held-out **MAE 8.3 g/km (~5% of mean), R² 0.91**.
- **Electric / hydrogen:** assigned **0 g/km** — zero tailpipe by definition.
- **Plug-in hybrids:** the certified emissions figure is corrected using published
  real-world adjustment factors, because a PHEV's true emissions depend on driver
  charging behaviour that no vehicle dataset can observe.

The interesting part isn't the model — it's the finding that **one third of the
problem shouldn't be modelled at all.**

## Why this project

I was initially inspired to create this project from a talk from the Chief Sustainability Officer at BNP Paribas Personal Finance. I found this problem to be a natural data science project, because banks reporting *financed emissions* under PCAF
need a CO2 figure for every vehicle they finance, including ones with missing or
unreliable data. This is a public-data proof-of-concept for that estimation step.
Full reasoning is in the [methodology write-up](WRITEUP.md).

## Key results

| Model | RMSE | MAE | R² | MAE % of mean |
|---|---|---|---|---|
| All fuel types mixed | 18.7 | 10.1 | 0.923 | 6.8% |
| **Combustion-only (final)** | **14.0** | **8.3** | **0.909** | **5.0%** |

Scoping the model to combustion vehicles cut RMSE by ~25% — more than MAE fell —
confirming that plug-in hybrids were the source of the large-error tail. The full
diagnostic story is in the exploratory notebook.

## How it works

```
                        ┌─────────────────────┐
   vehicle record  ──▶  │   fuel type?        │
                        └─────────┬───────────┘
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
       electric / hydrogen    petrol/diesel        */electric
              │                   │                    │
          0 g/km            XGBoost model     certified value ×
                            (specs → CO2)     real-world correction
```

## Repo structure

```
financed-emissions/
├── README.md                  # you are here
├── WRITEUP.md                 # methodology / bank-facing memo
├── requirements.txt
├── src/
│   └── emissions.py           # all reusable logic: correction, router, training
├── notebooks/
│   ├── 01_exploration.ipynb   # the full journey: EDA, diagnostics, dead ends, decisions
│   └── 02_clean.ipynb         # concise: load → train → route → results
├── models/
│   └── ice_model.pkl          # trained combustion-vehicle pipeline
└── app.py                     # Streamlit demo
```

The two notebooks serve different purposes on purpose: `01_exploration` documents
*how* every decision was reached (including the worst-error diagnostic that
uncovered the PHEV problem); `02_clean` is the straight-line version that reproduces
the final results. All shared logic lives in `src/emissions.py` — the notebooks and
the app import from it rather than redefining anything.

## Run it locally

```bash
git clone https://github.com/<you>/financed-emissions.git
cd financed-emissions
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the models, run `notebooks/02_clean.ipynb` end to end; it writes the
trained pipeline to `models/ice_model.pkl`.

## Data

European Environment Agency — *Monitoring of CO2 emissions from passenger cars*
(public dataset). ~10.8M registration events, reduced to distinct vehicle
specifications for modelling. No proprietary data is used.

## What I'd do next

- Retrain the combustion model against real-world-adjusted emissions for
  consistency with the PHEV branch.
- Add a dedicated PHEV model using electric-range features to narrow the
  uncertainty band.
- Wire the estimator into the full PCAF attribution formula.

## Tech

Python · pandas · scikit-learn · XGBoost · Streamlit

---

*Independent portfolio project. Built on public data; not affiliated with any
institution.*


# Financed-emissions demo (Streamlit)
 
Ties the two models together on the shared modules:
- **Model A** (`src/emissions.py`) — per-vehicle gCO₂/km via the three-regime router
- **Model B** (`src/portfolio.py`) — PCAF financed emissions + data-quality score
## Expected repo layout
```
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── src/
│   ├── emissions.py
│   └── portfolio.py
└── models/
    └── ice_model.pkl
```
 
## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
 
## Deploy (clickable link)
Push to GitHub → share.streamlit.io → **New app** → point at `app.py`.
The vehicle tab runs the real trained pipeline; the portfolio tab uses a
synthetic loan book (figures illustrative of the PCAF method).
