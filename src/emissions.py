"""
emissions.py — Vehicle CO2 emissions estimation.

Single source of truth for the project's reusable logic:

  data access      : disco()
  data preparation : load_car_data(), split_data(), clean_placeholders(),
                     filter_ice()
  ICE model        : build_pipelines(), train_ice_model(),
                     save_model(), load_model()
  PHEV correction  : correction_factor(), estimate_phev_emissions()
  routing          : estimate_co2(), estimate_co2_frame()

Notebooks and the Streamlit app import from here rather than redefining anything.
"""

from __future__ import annotations

import requests
import numpy as np
import pandas as pd
from urllib.parse import quote

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TBL = "[CO2Emission].[latest].[co2cars_2025Pv31]"

# Model features (kept minimal and physically meaningful).
FEATURES = ["Ft", "Cr", "M (kg)", "Ec (cm3)", "Ep (KW)"]
CAT_COLS = ["Ft", "Cr"]
NUM_COLS = ["M (kg)", "Ec (cm3)", "Ep (KW)"]
TARGET = "Ewltp (g/km)"

# Fuel-type regimes.
ICE_FUELS = ["petrol", "diesel", "lpg", "e85", "ng"]
ZERO_CO2_FUELS = ("electric", "hydrogen")   # BEV + fuel-cell -> zero tailpipe

SEED = 1


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def disco(sql: str, p: int = 1, n: int = 1000) -> pd.DataFrame:
    """Run a SQL query against the EEA Discodata API and return a DataFrame.

    Args:
        sql: SQL query string.
        p:   page number.
        n:   rows per page.
    """
    # Everything after ? is the query string; quote() escapes spaces so the
    # SQL doesn't break the URL.
    url = (
        f"https://discodata.eea.europa.eu/sql?query={quote(sql)}"
        f"&p={p}&nrOfHits={n}"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return pd.DataFrame(j["results"])


def load_car_data(n: int = 160000, include_zr: bool = True) -> pd.DataFrame:
    """Load distinct vehicle specifications with target (and electric range).

    De-duplicating to distinct specs before any split prevents identical
    vehicles leaking across the train/test boundary. `Zr` (electric range) is
    only populated for electrified vehicles and is needed by the PHEV branch.
    """
    cols = f"[{TARGET}], Ft, Cr, [M (kg)], [Ec (cm3)], [Ep (KW)]"
    if include_zr:
        cols += ", [Zr]"
    return disco(f"SELECT DISTINCT {cols} FROM {TBL}", n=n)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size: float = 0.30):
    """Drop missing-target rows, then split into train/validation X and y.

    X is restricted to the model FEATURES so that extra columns present for the
    router (e.g. Zr electric range) do NOT leak into the model as passthrough
    features. Splitting before EDA/feature engineering avoids leaking test
    information into decisions made on the training data.
    """
    df = df.dropna(subset=[TARGET])
    y = df[TARGET]
    X = df[FEATURES]
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def clean_placeholders(X: pd.DataFrame, y: pd.Series):
    """Remove combustion rows carrying a placeholder emissions value.

    A cluster of petrol/diesel rows report an impossible sub-70 g/km value
    (dominated by a repeated ~57 g/km, usually alongside missing engine power).
    These are a systematic reporting gap, not real measurements, so they must
    not train or score the model. Returns cleaned (X, y) and the keep mask.
    """
    placeholder = X["Ft"].isin(["petrol", "diesel"]) & (y < 70)
    keep = ~placeholder
    return X[keep], y[keep], keep


def filter_ice(X: pd.DataFrame, y: pd.Series):
    """Restrict to combustion-only vehicles (the ICE model's domain).

    Excludes plug-in hybrids (unobservable usage), pure electric and hydrogen
    (zero-tailpipe, router-handled), and unknown fuels.
    """
    mask = X["Ft"].isin(ICE_FUELS)
    return X[mask], y[mask]


# ---------------------------------------------------------------------------
# ICE model
# ---------------------------------------------------------------------------

def _make_prep(cat_encoder, num_transformer) -> ColumnTransformer:
    """Preprocessor: encode categoricals, impute numerics, pass through rest."""
    return ColumnTransformer(
        transformers=[
            ("cat", cat_encoder, CAT_COLS),
            ("num", num_transformer, NUM_COLS),
        ],
        remainder="passthrough",
    )


def build_pipelines():
    """Return (pipelines, param_grids) for the three-model bake-off."""
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    onehot = OneHotEncoder(handle_unknown="ignore")
    imp_mean = SimpleImputer(strategy="mean")
    imp_median = SimpleImputer(strategy="median")

    pipelines = {
        "decision_tree": Pipeline([
            ("prep", _make_prep(ordinal, imp_median)),
            ("model", DecisionTreeRegressor(random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", _make_prep(ordinal, imp_median)),
            ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("prep", _make_prep(onehot, imp_median)),
            ("model", XGBRegressor(random_state=SEED, n_jobs=-1)),
        ]),
    }

    param_grids = {
        "decision_tree": {
            "prep__cat": [ordinal, onehot],
            "prep__num": [imp_mean, imp_median],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 5, 20],
        },
        "random_forest": {
            "prep__cat": [ordinal, onehot],
            "prep__num": [imp_mean, imp_median],
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__max_features": ["sqrt", 1.0],
        },
        "xgboost": {
            "prep__cat": [ordinal, onehot],
            "prep__num": [imp_mean, imp_median, "passthrough"],
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 6, 10],
            "model__learning_rate": [0.05, 0.1],
        },
    }
    return pipelines, param_grids


def train_ice_model(train_X, train_y, val_X, val_y, cv: int = 5):
    """Tune all three models on ICE data; return (best_pipeline, results).

    Winner is chosen by cross-validated RMSE (never by the test set, which is
    reported only as an honest final estimate). Each model reports RMSE, MAE,
    and R^2 on the held-out validation set.
    """
    pipelines, param_grids = build_pipelines()
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    results = {}
    for name, pipe in pipelines.items():
        search = RandomizedSearchCV(
            pipe, param_grids[name],
            cv=cv, scoring=scoring, refit="rmse", n_jobs=-1,
        )
        search.fit(train_X, train_y)

        i = search.best_index_
        cvr = search.cv_results_
        preds = search.predict(val_X)

        results[name] = {
            "cv_rmse": -cvr["mean_test_rmse"][i],
            "test_rmse": root_mean_squared_error(val_y, preds),
            "test_mae": mean_absolute_error(val_y, preds),
            "test_r2": r2_score(val_y, preds),
            "best_params": search.best_params_,
            "estimator": search.best_estimator_,
        }

    best_name = min(results, key=lambda k: results[k]["cv_rmse"])
    return results[best_name]["estimator"], results


def save_model(model, path: str = "models/ice_model.pkl") -> None:
    """Persist a trained pipeline to disk."""
    joblib.dump(model, path)


def load_model(path: str = "models/ice_model.pkl"):
    """Load a trained pipeline from disk."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# PHEV correction
# ---------------------------------------------------------------------------

# Central real-world / label gap for PHEVs (ICCT 2022-2024). Documented
# approximation, NOT fitted to this registry — the registry holds only certified
# values, so the real-world gap cannot be derived from it.
FLAT_CORRECTION = 1.45


def correction_factor(zr) -> float:
    """Real-world correction multiplier as a function of electric range (km).

    Longer electric range -> the certificate is closer to reality -> smaller
    correction. Unknown range falls back to the flat factor. Magnitudes reflect
    the direction/scale reported by ICCT/SAE, not a fit to this data.
    """
    if zr is None or (isinstance(zr, float) and np.isnan(zr)):
        return FLAT_CORRECTION
    if zr >= 60:
        return 1.25
    if zr >= 40:
        return 1.40
    if zr >= 20:
        return 1.55
    return 1.70


def estimate_phev_emissions(ewltp, zr, phev_median: float) -> float:
    """Tiered PHEV estimate.

    best -> certified value x range-conditional correction
    ok   -> certified value x flat correction (electric range missing)
    last -> group-median certified value x flat correction (certificate missing)
    """
    has_ewltp = ewltp is not None and not (isinstance(ewltp, float) and np.isnan(ewltp))
    has_zr = zr is not None and not (isinstance(zr, float) and np.isnan(zr))

    if has_ewltp and has_zr:
        return correction_factor(zr) * ewltp
    if has_ewltp:
        return FLAT_CORRECTION * ewltp
    return FLAT_CORRECTION * phev_median


def phev_median_ewltp(df: pd.DataFrame) -> float:
    """Median certified emissions among true PHEVs, for the last-resort tier."""
    is_phev = df["Ft"].str.contains("/electric", case=False, na=False)
    return df.loc[is_phev, TARGET].median()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def estimate_co2(row, ice_model, phev_median: float) -> float:
    """Estimate WLTP CO2 (g/km) for a single vehicle row by routing on fuel type.

    electric/hydrogen -> 0 ; any '/electric' hybrid -> corrected certificate ;
    everything else (combustion) -> the ICE model.
    """
    ft = str(row["Ft"]).lower()

    if ft in ZERO_CO2_FUELS:
        return 0.0

    if "/electric" in ft:
        return estimate_phev_emissions(
            row.get(TARGET), row.get("Zr"), phev_median
        )

    X_one = pd.DataFrame([{c: row.get(c) for c in FEATURES}])
    return float(ice_model.predict(X_one)[0])


def estimate_co2_frame(df: pd.DataFrame, ice_model, phev_median: float) -> pd.Series:
    """Vectorised convenience wrapper: apply estimate_co2 across a DataFrame."""
    return df.apply(lambda r: estimate_co2(r, ice_model, phev_median), axis=1)