import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
ARTIFACT_DIR = Path(__file__).parent / "artifacts_gpr_k"
PREP_PATH = ARTIFACT_DIR / "preprocessor_fitted.joblib"
MODEL_PATH = ARTIFACT_DIR / "gpr_k_model.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"

MC_NS_DEFAULT = 5000
EPS = 1e-12

# Fixed ranges (display + validation bounds for numeric inputs)
RANGES = {
    "Soil_pH": (3.0, 10.0),
    "Chloride Content (mg/kg)": (0.3, 11400.0),
    "Soil_Resistivity (Ω·cm)": (80.0, 44000.0),
    "Sulphate_Content (mg/kg)": (6.9, 21800.0),
    "Moisture_Content (%)": (1.7, 261.4),
}

# Categorical options (must match training labels to avoid "unknown" categories)
SOIL_TYPES = ["GT","CL","SM","ML","SP","CH","GP","SW","OL","SC"]
WATER_TABLE = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]
FOREIGN_INCL = ["None", "Shreded wood", "Cinder", "Flyash"]  # keep exact spelling you provided
FILL_MATERIAL = [0, 1]

AGES_HORIZON = [10, 30, 50, 70, 80]


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    if not PREP_PATH.exists() or not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts. Ensure the repo contains:\n"
            f"- {PREP_PATH}\n- {MODEL_PATH}\n- {META_PATH}\n"
        )
    prep = joblib.load(PREP_PATH)
    gpr = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return prep, gpr, meta


# =========================
# MONTE CARLO HELPERS
# =========================
def rtruncnorm(mean, sd, low, high, size, seed=42, max_iter=5_000_000):
    rng = np.random.default_rng(seed)
    out = np.empty(size, dtype=float)
    filled = 0
    it = 0
    while filled < size:
        batch = min(size - filled, 100000)
        draw = rng.normal(mean, sd, size=batch)
        draw = draw[(draw >= low) & (draw <= high)]
        k = len(draw)
        if k > 0:
            out[filled:filled+k] = draw[:k]
            filled += k
        it += batch
        if it > max_iter:
            raise RuntimeError("TruncNormal rejection did not converge. Check sd/bounds.")
    return out


def mc_TL_from_k(mu_k, sd_k, ages, T_used, T0,
                mu_n, mu_beta,
                n_bounds=(0.4, 0.7), beta_bounds=(0.02, 0.04),
                Ns=5000, seed=42, shared_n_beta=False):
    """
    k ~ Normal(mu_k, sd_k) from ML (GPR). We clip k at EPS to keep TL physical.
    n, beta ~ TruncNormal with bounds; bounds treated as ~95% to infer sigma.
    TL = k * t^n * exp(beta*(T - T0))

    Returns:
      df with mean/sd and 68/95 bands (mean±sd, mean±2sd).
    """
    rng = np.random.default_rng(seed)
    mu_k = float(mu_k)
    sd_k = float(max(sd_k, EPS))

    nL, nU = n_bounds
    bL, bU = beta_bounds
    sigma_n    = (nU - nL) / (2.0 * 1.96)
    sigma_beta = (bU - bL) / (2.0 * 1.96)

    ages = np.asarray(ages, float)
    out_rows = []

    for t_age in ages:
        # sample k
        z = rng.standard_normal(Ns)
        k_s = mu_k + sd_k * z
        k_s = np.maximum(k_s, EPS)

        # sample n, beta
        if shared_n_beta:
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed+1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed+2)
        else:
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed+1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed+2)

        time_fac = np.power(max(float(t_age), EPS), n_draw)
        temp_fac = np.exp(b_draw * (float(T_used) - float(T0)))

        TL_s = k_s * time_fac * temp_fac

        TL_mean = float(np.mean(TL_s))
        TL_sd   = float(np.std(TL_s, ddof=1))

        lo68 = max(TL_mean - 1.0*TL_sd, 0.0)
        hi68 = TL_mean + 1.0*TL_sd
        lo95 = max(TL_mean - 2.0*TL_sd, 0.0)
        hi95 = TL_mean + 2.0*TL_sd

        out_rows.append({
            "Age (yr)": int(t_age),
            "TL_mean (mm)": TL_mean,
            "TL_sd (mm)": TL_sd,
            "TL_lo68 (mm)": lo68,
            "TL_hi68 (mm)": hi68,
            "TL_lo95 (mm)": lo95,
            "TL_hi95 (mm)": hi95,
        })

    return pd.DataFrame(out_rows)


def count_missing_ml_features(row_dict, expected_cols):
    miss = 0
    for c in expected_cols:
        v = row_dict.get(c, None)
        if v is None:
            miss += 1
        elif isinstance(v, float) and np.isnan(v):
            miss += 1
    return miss


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Steel Pile Corrosion: k → TL", layout="wide")
st.title("Steel Pile Corrosion — Predict k (GPR) and propagate Thickness Loss (MC)")

prep, gpr, meta = load_artifacts()
expected_cols = meta["expected_raw_columns"]

# constants used in MC
T0 = float(meta["constants"]["T0"])
mu_n = float(meta["constants"]["mu_n"])
mu_beta = float(meta["constants"]["mu_beta"])

# Sidebar settings
st.sidebar.header("Monte Carlo settings")
Ns = st.sidebar.number_input("MC samples (Ns)", min_value=1000, max_value=50000, value=MC_NS_DEFAULT, step=1000)

st.sidebar.header("Temperature")
st.sidebar.write(f"Default T0 = **{T0} °C** (used if Temperature is NA)")
temp_na = st.sidebar.checkbox("Temperature: NA (use default 10°C)", value=False)
if temp_na:
    T_used = T0
else:
    T_used = st.sidebar.number_input("Site temperature T (°C)", value=T0, step=1.0)

st.sidebar.header("Uncertainty bounds")
nL = st.sidebar.number_input("n lower bound", value=0.4, step=0.01, format="%.2f")
nU = st.sidebar.number_input("n upper bound", value=0.7, step=0.01, format="%.2f")
bL = st.sidebar.number_input("beta lower bound", value=0.02, step=0.001, format="%.3f")
bU = st.sidebar.number_input("beta upper bound", value=0.04, step=0.001, format="%.3f")
shared = st.sidebar.checkbox("Shared n,beta across samples (advanced)", value=False)

st.markdown("### Required inputs")
st.write("**Age is required.** Temperature is optional (if NA, default 10°C).")

st.markdown("### ML feature inputs (up to **2** can be NA)")
st.caption("Only the 9 ML features below are allowed to be missing; KNN imputation is applied using the saved fitted preprocessor.")

with st.form("input_form"):
    c0, c1, c2, c3 = st.columns([0.9, 1.2, 1.2, 1.2])

    # Required age for a single-age TL result
    age_now = c0.number_input("Age (yr) [required]", min_value=1, max_value=80, value=10, step=1)

    user_row = {}

    # numeric inputs with NA option
    def num_input(block_col, label, rmin, rmax, default):
        na = block_col.checkbox(f"{label}: NA", value=False, key=f"na_{label}")
        block_col.caption(f"Range: {rmin} to {rmax}")
        if na:
            return np.nan
        return block_col.number_input(label, min_value=float(rmin), max_value=float(rmax), value=float(default))

    user_row["Soil_pH"] = num_input(c1, "Soil_pH", *RANGES["Soil_pH"], default=7.0)
    user_row["Chloride Content (mg/kg)"] = num_input(c1, "Chloride Content (mg/kg)", *RANGES["Chloride Content (mg/kg)"], default=200.0)
    user_row["Soil_Resistivity (Ω·cm)"] = num_input(c1, "Soil_Resistivity (Ω·cm)", *RANGES["Soil_Resistivity (Ω·cm)"], default=5000.0)

    user_row["Sulphate_Content (mg/kg)"] = num_input(c2, "Sulphate_Content (mg/kg)", *RANGES["Sulphate_Content (mg/kg)"], default=100.0)
    user_row["Moisture_Content (%)"] = num_input(c2, "Moisture_Content (%)", *RANGES["Moisture_Content (%)"], default=15.0)

    # categorical inputs with NA option
    def cat_input(block_col, label, options, default):
        na = block_col.checkbox(f"{label}: NA", value=False, key=f"na_{label}")
        if na:
            return np.nan
        return block_col.selectbox(label, options, index=options.index(default) if default in options else 0)

    user_row["Soil Type"] = cat_input(c3, "Soil Type", SOIL_TYPES, default="CL")
    user_row["Location wrt Water Table"] = cat_input(c3, "Location wrt Water Table", WATER_TABLE, default="Above WaterTable")
    user_row["Foreign_Inclusion_Type"] = cat_input(c3, "Foreign_Inclusion_Type", FOREIGN_INCL, default="None")

    na_fill = c3.checkbox("Is_Fill_Material: NA", value=False, key="na_Is_Fill_Material")
    if na_fill:
        user_row["Is_Fill_Material"] = np.nan
    else:
        user_row["Is_Fill_Material"] = c3.selectbox("Is_Fill_Material", FILL_MATERIAL, index=0)

    submitted = st.form_submit_button("Run prediction + Monte Carlo")

if submitted:
    # Enforce missing <=2 on ML features only
    miss = count_missing_ml_features(user_row, expected_cols)
    if miss > 2:
        st.error(f"Too many missing ML inputs: {miss}. Maximum allowed is 2.")
        st.stop()

    # Build one-row df in correct column order
    X_in = pd.DataFrame([{c: user_row.get(c, np.nan) for c in expected_cols}])

    # Predict k distribution
    try:
        X_tr = prep.transform(X_in)  # uses saved KNNImputer/Scaler/OHE
        mu_k_arr, sd_k_arr = gpr.predict(np.asarray(X_tr, float), return_std=True)
        mu_k = float(mu_k_arr[0])
        sd_k = float(max(sd_k_arr[0], EPS))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Prediction complete.")

    st.subheader("Predicted k distribution (GPR)")
    st.write({"mu_k": mu_k, "sd_k": sd_k, "Temperature_used(°C)": float(T_used)})

    # --- Single age TL ---
    single_df = mc_TL_from_k(
        mu_k=mu_k, sd_k=sd_k,
        ages=[int(age_now)],
        T_used=float(T_used), T0=float(T0),
        mu_n=mu_n, mu_beta=mu_beta,
        n_bounds=(float(nL), float(nU)),
        beta_bounds=(float(bL), float(bU)),
        Ns=int(Ns),
        seed=42,
        shared_n_beta=bool(shared),
    )
    st.subheader("Thickness Loss at input age (Monte Carlo)")
    st.dataframe(single_df, use_container_width=True)

    # --- Horizon table TL ---
    horizon_df = mc_TL_from_k(
        mu_k=mu_k, sd_k=sd_k,
        ages=AGES_HORIZON,
        T_used=float(T_used), T0=float(T0),
        mu_n=mu_n, mu_beta=mu_beta,
        n_bounds=(float(nL), float(nU)),
        beta_bounds=(float(bL), float(bU)),
        Ns=int(Ns),
        seed=123,
        shared_n_beta=bool(shared),
    )
    st.subheader(f"Thickness Loss horizon table (ages = {AGES_HORIZON})")
    st.dataframe(horizon_df, use_container_width=True)

    # --- Plot TL vs Age with uncertainty bands ---
    st.subheader("Thickness Loss vs Age (mean + 68% and 95% bands)")

    ages = horizon_df["Age (yr)"].values
    mean = horizon_df["TL_mean (mm)"].values
    lo68 = horizon_df["TL_lo68 (mm)"].values
    hi68 = horizon_df["TL_hi68 (mm)"].values
    lo95 = horizon_df["TL_lo95 (mm)"].values
    hi95 = horizon_df["TL_hi95 (mm)"].values

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ages, mean, linewidth=2, label="Mean TL")
    plt.fill_between(ages, lo95, hi95, alpha=0.20, label="95% (mean ± 2σ)")
    plt.fill_between(ages, lo68, hi68, alpha=0.35, label="68% (mean ± 1σ)")
    plt.xlabel("Age (yr)")
    plt.ylabel("Thickness Loss (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig)

    # --- Diagnostic: k sample histogram ---
    st.subheader("Diagnostic: k samples used (Normal(mu_k, sd_k), clipped at EPS)")
    rng = np.random.default_rng(999)
    k_samples = np.maximum(mu_k + sd_k * rng.standard_normal(int(Ns)), EPS)

    fig2 = plt.figure(figsize=(10, 4))
    plt.hist(k_samples, bins=40, edgecolor="black", alpha=0.8)
    plt.xlabel("k")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    st.pyplot(fig2)