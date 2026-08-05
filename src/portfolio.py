"""
portfolio.py  —  Model B: the PCAF financed-emissions layer.

Model A (emissions.py) answers a *physics* question:
    "What does this vehicle emit, in gCO2 per km?"

Model B answers an *accounting* question:
    "Of that, how many tonnes does the lender book against its loan book,
     and how good is the estimate?"

The two are deliberately decoupled. This module consumes a dataframe that
already carries a `co2_estimate` column (whatever produced it — Model A's
three-regime router, a raw certified value, or a segment average) and turns
it into per-loan financed emissions plus the PCAF metrics a bank reports.

PCAF motor-vehicle-loans mechanics (Global GHG Accounting & Reporting
Standard, Part A, motor vehicle loans asset class):

    attribution factor = outstanding amount / value of vehicle at origination
    vehicle emissions  = annual distance (km) * emission factor (gCO2/km)
    financed emissions = attribution factor * vehicle emissions

Everything below the SEPARATOR is an *assumption*, not a fact. Those are the
knobs a methodology team argues about, so they live at the top, named and
citable, rather than buried as magic numbers in a formula.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# ASSUMPTIONS  (edit / defend / run sensitivity on these — they drive the total)
# ----------------------------------------------------------------------------

# Fuel-type-conditioned annual distance, km/year. This is the SECOND unknown on
# the vehicle side (the first, gCO2/km, is Model A's job). It is handled as a
# lookup, not an ML target, because a loan record carries no signal about how
# far *this* driver will go — only population averages exist. French national
# average is ~11,600 km/yr; diesels historically drive further (the reason
# people bought them), petrol less. These are deliberately conservative,
# reconcile roughly to the national mean, and are the single biggest lever on
# the absolute total — always report a sensitivity band around them.
DEFAULT_MILEAGE = {
    "petrol":          9_000,
    "diesel":         14_000,
    "lpg":            11_000,
    "e85":            10_000,
    "ng":             12_000,
    "electric":       12_000,
    "hydrogen":       12_000,
    "petrol/electric": 13_000,   # PHEV
    "diesel/electric": 15_000,   # PHEV
}
_FALLBACK_MILEAGE = 11_600       # national average, for any unmapped fuel type

# Score-5 "economic" emission factor: tonnes CO2e per euro of vehicle value per
# year, used when NOTHING but the loan amount is known. Illustrative — in
# production replace with a PCAF-published economic factor for the asset class.
# Derivation shown so it is auditable: an "average" financed car ~ 2.0 tCO2/yr
# tailpipe at an origination value ~ EUR 30,000 -> 2.0 / 30000.
DEFAULT_ECONOMIC_FACTOR = 2.0 / 30_000        # tCO2e per EUR per year

# Optional well-to-wheel factor for battery-electric vehicles, gCO2/km. Tank-to-
# wheel (tailpipe) is 0 by construction; well-to-wheel charges grid emissions.
# France is ~0.15 kWh/km * ~55 gCO2/kWh (very low-carbon grid) ~ 8 gCO2/km.
# OFF by default so the headline stays tailpipe-comparable to PCAF/peers; flip
# it on to show you know tailpipe != full life-cycle.
DEFAULT_EV_WTW_GCO2_KM = 8.0

# ----------------------------------------------------------------------------
# 1. Attribution factor
# ----------------------------------------------------------------------------

def attribution_factor(outstanding, origination_value):
    """Outstanding / value-at-origination, PCAF-capped to [0, 1].

    Origination value is fixed at loan start (like PCAF mortgages/CRE) so the
    reported number does not swing with second-hand resale prices. Capped at 1
    because a lender cannot be attributed more than 100% of a car's emissions,
    even on an under-water loan where outstanding briefly exceeds value.
    """
    out = np.asarray(outstanding, dtype="float64")
    val = np.asarray(origination_value, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(val > 0, out / val, 0.0)
    return np.clip(af, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 2. Annual distance
# ----------------------------------------------------------------------------

def annual_mileage(fuel_types, table=None):
    """Map each fuel type to an assumed annual distance (km/yr)."""
    table = DEFAULT_MILEAGE if table is None else table
    ft = pd.Series(fuel_types).astype(str).str.lower()
    return ft.map(table).fillna(_FALLBACK_MILEAGE).to_numpy(dtype="float64")


# ----------------------------------------------------------------------------
# 3. Per-loan financed emissions — two PCAF methods
# ----------------------------------------------------------------------------

def financed_emissions_physical(df, mileage_table=None,
                                ev_wtw=False, ev_wtw_gco2_km=DEFAULT_EV_WTW_GCO2_KM):
    """Physical method (PCAF data option 2): grounded in vehicle emissions.

    Requires columns: co2_estimate (gCO2/km, from Model A), Ft (fuel type),
    outstanding_amount, origination_value.

    Returns annual financed emissions per loan in tCO2e:
        attribution * (gCO2/km * km/yr) / 1e6
    """
    af = attribution_factor(df["outstanding_amount"], df["origination_value"])
    km = annual_mileage(df["Ft"], mileage_table)
    gco2_km = df["co2_estimate"].to_numpy(dtype="float64").copy()

    if ev_wtw:
        is_ev = df["Ft"].astype(str).str.lower().isin(["electric"]).to_numpy()
        gco2_km = np.where(is_ev & (gco2_km == 0.0), ev_wtw_gco2_km, gco2_km)

    vehicle_tco2 = gco2_km * km / 1e6          # g -> t
    return af * vehicle_tco2


def financed_emissions_economic(df, economic_factor=DEFAULT_ECONOMIC_FACTOR):
    """Economic method (PCAF data option 3, the score-5 fallback).

    Uses ONLY money: attribution * origination_value * (tCO2e per EUR per yr).
    This is what you are stuck with when the vehicle is unknown — it is what
    Model A exists to replace. Requires outstanding_amount, origination_value.
    """
    af = attribution_factor(df["outstanding_amount"], df["origination_value"])
    val = df["origination_value"].to_numpy(dtype="float64")
    return af * val * economic_factor


# ----------------------------------------------------------------------------
# 4. PCAF data-quality score  (1 = best, 5 = worst)
# ----------------------------------------------------------------------------
#
# `emissions_basis` records HOW each loan's per-km number was obtained, which
# is exactly what PCAF grades. This is the column that turns Model A from a
# redundant regression into a documented score upgrade.

_BASIS_TO_SCORE = {
    "measured_fuel":      1,   # metered real fuel/energy use of the vehicle
    "certified_specific": 2,   # this exact vehicle's certified gCO2/km  + est. km
    "modelled_specific":  3,   # Model A predicts gCO2/km from partial specs + est. km
    "segment_average":    4,   # segment/type average emissions + est. km
    "economic":           5,   # loan amount only, economic factor
}

def pcaf_score(emissions_basis):
    """Vectorised map from basis label to PCAF data-quality score (1-5)."""
    s = pd.Series(emissions_basis).astype(str)
    return s.map(_BASIS_TO_SCORE).fillna(5).to_numpy(dtype="int64")


# ----------------------------------------------------------------------------
# 5. Portfolio roll-up
# ----------------------------------------------------------------------------

def portfolio_summary(df, emissions_col="financed_tco2", score_col="pcaf_score"):
    """Aggregate a scored, per-loan frame into the metrics a bank reports.

    absolute_tco2e       : sum of financed emissions (tCO2e / yr)
    econ_intensity        : tCO2e per EUR-million outstanding
    phys_intensity_gco2km : exposure-weighted average gCO2/km (peer-comparable)
    wavg_pcaf_score       : outstanding-weighted PCAF data-quality score
    """
    total_out = df["outstanding_amount"].sum()
    absolute = df[emissions_col].sum()

    # exposure weights (by outstanding); fall back to equal weights if all zero
    w = df["outstanding_amount"].to_numpy(dtype="float64")
    w = w / w.sum() if w.sum() > 0 else np.full(len(df), 1 / len(df))

    phys_intensity = float(np.average(df["co2_estimate"].to_numpy(dtype="float64"),
                                      weights=w)) if len(df) else float("nan")
    wavg_score = float(np.average(df[score_col].to_numpy(dtype="float64"),
                                  weights=w)) if len(df) else float("nan")

    return {
        "n_loans":               int(len(df)),
        "total_outstanding_eur": float(total_out),
        "absolute_tco2e":        float(absolute),
        "econ_intensity_tco2e_per_eurM": float(absolute / (total_out / 1e6))
                                         if total_out else float("nan"),
        "phys_intensity_gco2_km": phys_intensity,
        "wavg_pcaf_score":        wavg_score,
    }


# ----------------------------------------------------------------------------
# 6. Synthetic loan book (for the demo / when no real ABS data is available)
# ----------------------------------------------------------------------------

def make_synthetic_portfolio(car_data, n_loans=5000, seed=0):
    """Attach a plausible loan to each of n_loans sampled vehicle specs.

    NOT modelling — just a stand-in so Model B runs end-to-end. Origination
    value scales loosely with engine size + power + mass (bigger cars cost
    more); loan age drives amortisation so outstanding < origination. Swap this
    out for real loan-level data (US Reg-AB auto-ABS, European DataWarehouse)
    when available; the rest of the pipeline does not change.
    """
    rng = np.random.default_rng(seed)
    n = min(n_loans, len(car_data))
    veh = car_data.sample(n, random_state=seed).reset_index(drop=True).copy()

    # crude price proxy, EUR: base + power + displacement + mass, with noise
    price = (
        8_000
        + 120.0 * veh.get("Ep (KW)", pd.Series(90, index=veh.index)).fillna(90)
        + 4.0  * veh.get("Ec (cm3)", pd.Series(1500, index=veh.index)).fillna(1500)
        + 6.0  * veh.get("M (kg)", pd.Series(1400, index=veh.index)).fillna(1400)
    )
    price *= rng.normal(1.0, 0.12, n).clip(0.6, 1.6)
    origination_value = price.round(-2).clip(lower=6_000)

    # loan age in years -> fraction repaid on a ~5yr straight-line schedule
    age_years = rng.uniform(0, 5, n)
    frac_remaining = (1 - age_years / 5).clip(0.05, 1.0)
    # financed ~80-100% of value at origination, then amortised
    ltv0 = rng.uniform(0.80, 1.00, n)
    outstanding = (origination_value * ltv0 * frac_remaining).round(-1)

    veh["loan_id"] = np.arange(n)
    veh["origination_value"] = origination_value.to_numpy()
    veh["outstanding_amount"] = outstanding.to_numpy()
    veh["loan_age_years"] = age_years.round(2)
    return veh
