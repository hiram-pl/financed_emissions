"""
app.py — Financed-emissions demo (Streamlit).
 
Ties the two models together on top of the shared modules:
  - Model A  (src/emissions.py) : per-vehicle gCO2/km via the three-regime router
  - Model B  (src/portfolio.py) : PCAF financed emissions + data-quality score
 
Run locally:   streamlit run app.py
Deploy:        push to GitHub, then Streamlit Community Cloud -> New app.
"""
 
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
 
# make src/ importable whether the app is run from repo root or elsewhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
import emissions as em
import portfolio as pf
 
# ----------------------------------------------------------------------------
# constants / theming
# ----------------------------------------------------------------------------
PEERS = {"Arval": 105, "Santander": 133, "BNP Paribas group": 141}
PHEV_MEDIAN_FALLBACK = 32.0   # only used if a PHEV certificate is missing
MODEL_PATHS = ["models/ice_model.pkl", "ice_model.pkl",
               os.path.join(os.path.dirname(__file__), "models", "ice_model.pkl")]
 
ARCH = {  # representative specs per fuel type: (mass, cc, kW, ewltp, zr)
    "petrol": (1250, 1400, 85, None, None), "diesel": (1650, 1900, 110, None, None),
    "lpg": (1300, 1500, 80, None, None),    "e85": (1450, 1700, 95, None, None),
    "ng": (1400, 1500, 85, None, None),     "electric": (1800, 0, 150, None, None),
    "hydrogen": (1900, 0, 120, None, None),
    "petrol/electric": (1700, 1600, 95, 30, 50),
    "diesel/electric": (1800, 1900, 120, 35, 55),
}
MIXES = {
    "2020 mix": {"petrol": .50, "diesel": .30, "lpg": .03, "e85": .01, "ng": .02,
                 "electric": .05, "petrol/electric": .06, "diesel/electric": .02, "hydrogen": .001},
    "2025 mix": {"petrol": .38, "diesel": .17, "lpg": .03, "e85": .02, "ng": .02,
                 "electric": .20, "petrol/electric": .13, "diesel/electric": .03, "hydrogen": .002},
    "EV-heavy": {"petrol": .22, "diesel": .08, "lpg": .02, "e85": .01, "ng": .01,
                 "electric": .45, "petrol/electric": .17, "diesel/electric": .03, "hydrogen": .01},
}
 
 
def co2_color(v: float) -> str:
    """Hex on a teal->amber->red scale over 0..300 gCO2/km (color encodes value)."""
    t = max(0.0, min(1.0, v / 300))
    a, b, c = (47, 191, 166), (231, 185, 77), (229, 106, 69)
    if t < 0.55:
        u = t / 0.55; col = [a[i] + (b[i] - a[i]) * u for i in range(3)]
    else:
        u = (t - 0.55) / 0.45; col = [b[i] + (c[i] - b[i]) * u for i in range(3)]
    return "#%02x%02x%02x" % tuple(int(x) for x in col)
 
 
# ----------------------------------------------------------------------------
# cached compute
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return em.load_model(p)
    st.error("Could not find models/ice_model.pkl. Place the trained model there.")
    st.stop()
 
 
@st.cache_data(show_spinner="Scoring vehicles with the trained model…")
def scored_specs(mix_key: str, n: int = 600) -> pd.DataFrame:
    """Synthetic vehicle specs, each scored by the REAL model + router."""
    ice = get_model()
    mix = MIXES[mix_key]
    p = np.array(list(mix.values()), dtype=float); p /= p.sum()
    rng = np.random.default_rng(1)
    fuels = rng.choice(list(mix), size=n, p=p)
    rows = []
    for f in fuels:
        m, ec, ep, ew, zr = ARCH[f]
        jit = lambda x, pr: None if x is None else float(x * (1 + (rng.random() - .5) * pr))
        rows.append({"Ft": f, "Cr": "M1", "M (kg)": jit(m, .15), "Ec (cm3)": jit(ec, .2),
                     "Ep (KW)": jit(ep, .25), "Ewltp (g/km)": ew, "Zr": zr})
    car = pd.DataFrame(rows)
    car["co2_estimate"] = em.estimate_co2_frame(car, ice, PHEV_MEDIAN_FALLBACK)
    return pf.make_synthetic_portfolio(car, n_loans=n, seed=1)
 
 
def cost_portfolio(book: pd.DataFrame, mileage_factor: float) -> dict:
    table = {k: v * mileage_factor for k, v in pf.DEFAULT_MILEAGE.items()}
    b = book.copy()
    b["financed_tco2"] = pf.financed_emissions_physical(b, mileage_table=table)
    b["pcaf_score"] = pf.pcaf_score(["modelled_specific"] * len(b))
    s = pf.portfolio_summary(b)
    s["economic_tco2e"] = float(pf.financed_emissions_economic(b).sum())
    return s
 
 
# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Financed emissions — car-loan portfolio",
                   page_icon="🌿", layout="wide")
 
st.markdown("""
<style>
  .stApp { background:#0C1618; }
  h1,h2,h3,h4 { font-family:"Space Grotesk",sans-serif; letter-spacing:-.01em; }
  .big { font-family:"IBM Plex Mono",monospace; font-weight:600; font-size:64px; line-height:1; }
  .rung { display:inline-block; text-align:center; padding:10px 0; width:19%; margin-right:1%;
          border:1px solid #22414A; border-radius:9px; background:#0C1618; color:#5D787E;
          font-family:"IBM Plex Mono",monospace; font-size:12px; }
  .rung b { display:block; font-size:18px; }
  .r5 { border-color:#E56A45; color:#E56A45; background:rgba(229,106,69,.08); }
  .r3 { border-color:#2FBFA6; color:#2FBFA6; background:rgba(47,191,166,.10); }
  .cap { color:#5D787E; font-family:"IBM Plex Mono",monospace; font-size:11px; }
</style>
""", unsafe_allow_html=True)
 
st.markdown("###### PCAF · SCOPE 3 CATEGORY 15 · MOTOR-VEHICLE LOANS")
st.title("Financed emissions of a car-loan book")
st.caption("Model A predicts each vehicle's gCO₂/km (the real trained XGBoost model); "
           "Model B turns that into portfolio financed emissions under PCAF and scores the data quality.")
 
tab_v, tab_p = st.tabs(["Estimate a vehicle", "Cost a portfolio"])
 
# ---------------- Model A ----------------
with tab_v:
    left, right = st.columns([1, 1], gap="large")
    with left:
        ft = st.selectbox("Fuel type", list(ARCH.keys()),
                          format_func=lambda x: x.replace("/electric", " plug-in hybrid").title())
        is_phev = "/electric" in ft
        is_zero = ft in ("electric", "hydrogen")
        row = {"Ft": ft, "Cr": "M1"}
        if not is_phev and not is_zero:
            row["Cr"] = st.radio("Category", ["M1", "M1G"], horizontal=True,
                                 format_func=lambda x: f"{x} · {'passenger' if x=='M1' else 'off-road'}")
            row["M (kg)"]   = st.slider("Mass (kg)", 800, 2800, 1400, 10)
            row["Ec (cm3)"] = st.slider("Engine capacity (cm³)", 600, 4000, 1600, 10)
            row["Ep (KW)"]  = st.slider("Engine power (kW)", 30, 320, 90, 5)
        elif is_phev:
            row["Ewltp (g/km)"] = st.slider("Certified WLTP (g/km)", 10, 90, 30, 1)
            row["Zr"]           = st.slider("Electric range (km)", 0, 100, 50, 1)
        else:
            st.info("Battery-electric and hydrogen are assigned 0 g/km (tank-to-wheel) by the router.")
 
    co2 = em.estimate_co2(row, get_model(), PHEV_MEDIAN_FALLBACK)
    with right:
        st.markdown(f'<div class="big" style="color:{co2_color(co2)}">{co2:.0f}'
                    f'<span style="font-size:18px;color:#8AA3A8"> gCO₂/km</span></div>',
                    unsafe_allow_html=True)
        if is_zero:
            st.write("**Zero-tailpipe.** Router assigns 0 — no model call.")
        elif is_phev:
            cf = em.correction_factor(row["Zr"])
            st.write(f"**Plug-in hybrid.** Certified {row['Ewltp (g/km)']} g/km × "
                     f"{cf:.2f} real-world correction (range {row['Zr']} km).")
        else:
            st.write("**Combustion.** Predicted by the trained XGBoost model from "
                     "mass, engine, power, fuel and category.")
        st.progress(min(1.0, co2 / 300))
 
# ---------------- Model B ----------------
with tab_p:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        mix_key = st.radio("Fleet composition", list(MIXES.keys()), index=1, horizontal=True)
    with c2:
        mil = st.slider("Annual mileage assumption", 0.8, 1.2, 1.0, 0.05,
                        help="The dominant Model-B assumption. Report a band, not a point.")
 
    book = scored_specs(mix_key)
    s = cost_portfolio(book, mil)
    delta = (s["absolute_tco2e"] / s["economic_tco2e"] - 1) * 100
 
    m1, m2, m3 = st.columns(3)
    m1.metric("Loan-amount only · PCAF 5", f"{s['economic_tco2e']:,.0f} tCO₂e/yr",
              help="Economic fallback: emissions guessed from the loan amount.")
    m2.metric("With Model A · PCAF 3", f"{s['absolute_tco2e']:,.0f} tCO₂e/yr",
              f"{delta:+.0f}% vs the blunt method", delta_color="inverse")
    m3.metric("Portfolio intensity", f"{s['phys_intensity_gco2_km']:.0f} gCO₂/km")
 
    # intensity vs peers
    df = pd.DataFrame({"who": ["Your book"] + list(PEERS),
                       "gco2km": [s["phys_intensity_gco2_km"]] + list(PEERS.values())})
    df["kind"] = ["you"] + ["peer"] * len(PEERS)
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("gco2km:Q", title="gCO₂/km"),
        y=alt.Y("who:N", sort="-x", title=None),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["you", "peer"],
                        range=["#2FBFA6", "#3a5a62"]), legend=None))
        .properties(height=170))
    st.altair_chart(chart, width="stretch")
 
    # PCAF ladder
    labels = {1: "metered", 2: "exact spec", 3: "Model A", 4: "segment", 5: "loan $ only"}
    rungs = "".join(
        f'<div class="rung {"r5" if n==5 else "r3" if n==3 else ""}"><b>{n}</b>{labels[n]}</div>'
        for n in range(1, 6))
    st.markdown(rungs, unsafe_allow_html=True)
    st.markdown('<div class="cap">1 · measured &nbsp;→&nbsp; 5 · economic proxy</div>',
                unsafe_allow_html=True)
 
    st.caption(f"Same loans, same exposure: the two methods differ by {abs(delta):.0f}%. "
               "Model A moves the book from a weighted PCAF score of 5 to 3 — and only then is "
               "a gCO₂/km intensity reportable. Portfolio is a synthetic loan book; figures are "
               "illustrative of the method. Peer numbers are external reference points.")
 
st.divider()
st.caption("Vehicle estimates run the real trained pipeline from src/emissions.py. "
           "Built on PCAF motor-vehicle-loans methodology.")