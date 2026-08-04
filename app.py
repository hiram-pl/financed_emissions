"""
app.py — Vehicle CO2 emissions estimator (Streamlit / Hugging Face Spaces).

Demonstrates the three-regime router: combustion vehicles go to an ML model,
zero-tailpipe vehicles are assigned 0, and plug-in hybrids have their certified
emissions corrected for real-world usage. All logic is imported from
emissions.py — this file is only the interface.
"""

import numpy as np
import pandas as pd
import streamlit as st

import emissions as em


# ---------------------------------------------------------------------------
# Model loading (cached so it loads once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Load the trained ICE pipeline. Returns None if the file is missing."""
    try:
        return em.load_model("models/ice_model.pkl")
    except Exception:
        return None


# A reasonable stand-in for the PHEV last-resort median if we can't compute it
# live (the app doesn't hit the database). Update to your dataset's value.
PHEV_MEDIAN_FALLBACK = 40.0


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vehicle CO2 Estimator", page_icon="🚗")

st.title("🚗 Vehicle CO2 Emissions Estimator")
st.markdown(
    "Estimates WLTP CO2 emissions (g/km) from vehicle specifications. "
    "The method **routes each vehicle by fuel type**, because a fleet contains "
    "three physically different regimes — see the note at the bottom."
)

model = get_model()
if model is None:
    st.warning(
        "⚠️ Trained model file not found (`models/ice_model.pkl`). "
        "Combustion estimates are disabled until the model is added to the repo. "
        "Zero-tailpipe and plug-in-hybrid estimates still work."
    )

# --- Fuel type drives everything ---
fuel = st.selectbox(
    "Fuel type",
    ["petrol", "diesel", "lpg", "e85", "ng",
     "electric", "hydrogen",
     "petrol/electric", "diesel/electric"],
)

is_zero = fuel in em.ZERO_CO2_FUELS
is_phev = "/electric" in fuel


# ---------------------------------------------------------------------------
# Adaptive form: the inputs shown depend on the regime
# ---------------------------------------------------------------------------

if is_zero:
    st.info(
        f"**{fuel.title()}** vehicles have zero tailpipe CO2 by definition — "
        "no prediction needed."
    )
    if st.button("Estimate"):
        st.metric("Estimated CO2", "0 g/km")

elif is_phev:
    st.markdown(
        "Plug-in hybrids need their **certified** emissions value, because the "
        "true figure depends on how much the car is actually driven on battery — "
        "something vehicle specs cannot capture. We correct the certificate for "
        "real-world usage."
    )
    ewltp = st.number_input(
        "Certified WLTP CO2 (g/km)", min_value=0.0, max_value=500.0, value=50.0
    )
    zr = st.number_input(
        "Electric range (km) — optional, improves the correction",
        min_value=0.0, max_value=200.0, value=50.0,
    )
    zr_val = zr if zr > 0 else np.nan

    if st.button("Estimate"):
        est = em.estimate_phev_emissions(ewltp, zr_val, PHEV_MEDIAN_FALLBACK)
        factor = em.correction_factor(zr_val)
        st.metric("Estimated real-world CO2", f"{est:.0f} g/km")
        st.caption(
            f"Certified {ewltp:.0f} g/km × {factor:.2f} real-world correction "
            f"(electric range {zr:.0f} km)."
        )

else:  # combustion
    st.markdown("Enter the vehicle's specifications:")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", ["M1", "M1G"])
        mass = st.number_input("Mass (kg)", min_value=400.0, max_value=4000.0, value=1500.0)
    with col2:
        engine_cc = st.number_input("Engine capacity (cm³)", min_value=600.0, max_value=8000.0, value=1600.0)
        power_kw = st.number_input("Engine power (kW)", min_value=8.0, max_value=1200.0, value=100.0)

    if st.button("Estimate"):
        if model is None:
            st.error("Model unavailable — cannot estimate combustion vehicles.")
        else:
            row = {
                "Ft": fuel, "Cr": category,
                "M (kg)": mass, "Ec (cm3)": engine_cc, "Ep (KW)": power_kw,
            }
            est = em.estimate_co2(row, model, PHEV_MEDIAN_FALLBACK)
            st.metric("Estimated CO2", f"{est:.0f} g/km")


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

with st.expander("How does this work?"):
    st.markdown(
        """
A vehicle fleet is not one estimation problem — it's three:

- **Combustion (petrol, diesel, lpg, e85, ng):** emissions track the hardware, so
  a machine-learning model predicts CO2 from engine size, power, mass, and category.
- **Zero-tailpipe (electric, hydrogen):** zero by definition — assigned, not predicted.
- **Plug-in hybrid (petrol/electric, diesel/electric):** the certified figure is
  systematically optimistic because the test assumes more electric driving than
  happens in practice. We take the certificate and apply a real-world correction,
  scaled by electric range.

The interesting finding of the project is that **one third of the problem should
not be modelled at all** — the right move is to recognise the regime and use the
right tool for each.
        """
    )
    st.caption(
        "Built on public European Environment Agency data. "
        "Independent project; not affiliated with any institution."
    )
