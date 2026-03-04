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

# McMaster theme
MCMAROON = "#7A003C"
LABELBLUE = "#1f5fbf"

# Fixed ranges (display + validation bounds for numeric inputs)
RANGES = {
    "Soil_pH": (3.0, 10.0),
    "Chloride Content (mg/kg)": (0.3, 11400.0),
    "Soil_Resistivity (Ω·cm)": (80.0, 44000.0),
    "Sulphate_Content (mg/kg)": (6.9, 21800.0),
    "Moisture_Content (%)": (1.7, 261.4),
}

# Categorical options (must match training labels to avoid "unknown" categories)
SOIL_TYPES = ["GT", "CL", "SM", "ML", "SP", "CH", "GP", "SW", "OL", "SC"]
WATER_TABLE = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]
FOREIGN_INCL = ["None", "Shreded wood", "Cinder", "Flyash"]
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
            out[filled:filled + k] = draw[:k]
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
    k ~ Normal(mu_k, sd_k) from ML (GPR). Clip k at EPS to keep TL physical.
    n, beta ~ TruncNormal with bounds; bounds treated as ~95% interval to infer sigma.
    TL = k * t^n * exp(beta*(T - T0))

    Returns df with mean/sd and CI bounds (mean±sd, mean±2sd), clipped at 0 for lower bounds.
    """
    rng = np.random.default_rng(seed)
    mu_k = float(mu_k)
    sd_k = float(max(sd_k, EPS))

    nL, nU = n_bounds
    bL, bU = beta_bounds
    sigma_n = (nU - nL) / (2.0 * 1.96)
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
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed + 1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed + 2)
        else:
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed + 1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed + 2)

        time_fac = np.power(max(float(t_age), EPS), n_draw)
        temp_fac = np.exp(b_draw * (float(T_used) - float(T0)))

        TL_s = k_s * time_fac * temp_fac

        TL_mean = float(np.mean(TL_s))
        TL_sd = float(np.std(TL_s, ddof=1))

        lo68 = max(TL_mean - TL_sd, 0.0)
        hi68 = TL_mean + TL_sd
        lo95 = max(TL_mean - 2.0 * TL_sd, 0.0)
        hi95 = TL_mean + 2.0 * TL_sd

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


def fmt_ci(lo, hi, nd=3):
    return f"{lo:.{nd}f} – {hi:.{nd}f}"


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Predicting corrosion-induced thickness loss", layout="wide")

# CSS: larger text + blue labels + boxed panels
st.markdown(
    f"""
    <style>
      .stApp {{
        font-size: 18px;
      }}
      h1, h2, h3 {{
        color: {MCMAROON};
      }}
      .panel {{
        border: 1px solid rgba(0,0,0,0.15);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(250,250,250,0.7);
      }}
      .panel-title {{
        color: {MCMAROON};
        font-weight: 800;
        font-size: 22px;
        margin-bottom: 8px;
      }}
      label, .stMarkdown p strong {{
        color: {LABELBLUE} !important;
        font-weight: 700 !important;
      }}
      .subtle {{
        color: rgba(0,0,0,0.65);
        font-size: 16px;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Predicting corrosion-induced thickness loss in buried steel pile</h1>", unsafe_allow_html=True)
st.markdown(
    rf"""
    <div class="subtle">
    Estimate soil aggressiveness factor <b>k</b> using ML and propagate Thickness Loss using Monte Carlo simulations:
    </div>
    """,
    unsafe_allow_html=True
)
st.latex(r"TL(t,T)=k\, t^n \exp\left[\beta\,(T-T_0)\right]")

prep, gpr, meta = load_artifacts()
expected_cols = meta["expected_raw_columns"]

T0 = float(meta["constants"]["T0"])
mu_n = float(meta["constants"]["mu_n"])
mu_beta = float(meta["constants"]["mu_beta"])

st.markdown(
    "<div class='panel-title'>Input Parameters [Up to two unknowns allowed to be imputed by kNN]</div>",
    unsafe_allow_html=True
)

# Layout: left (inputs) and right (MC settings)
left, right = st.columns([2.3, 1.1], gap="large")

with st.form("input_form"):
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        cL1, cL2 = st.columns(2, gap="large")

        # -------- required Age + optional Temperature ----------
        with cL1:
            age_now = st.number_input("Age (yr)", min_value=1, max_value=80, value=10, step=1)

        with cL2:
            temp_na = st.checkbox("Temperature: NA [Default 10°C]", value=True)
            if temp_na:
                T_used = T0
                st.number_input("Temperature (°C)", value=float(T0), disabled=True)
            else:
                T_used = st.number_input("Temperature (°C)", value=float(T0), step=1.0)

        # -------- ML features ----------
        user_row = {}

        def num_input(block_col, label, rmin, rmax, default):
            na = block_col.checkbox(f"{label}: NA", value=False, key=f"na_{label}")
            block_col.caption(f"Range: {rmin} to {rmax}")
            if na:
                return np.nan
            return block_col.number_input(label, min_value=float(rmin), max_value=float(rmax), value=float(default))

        def cat_input(block_col, label, options, default):
            na = block_col.checkbox(f"{label}: NA", value=False, key=f"na_{label}")
            if na:
                return np.nan
            idx = options.index(default) if default in options else 0
            return block_col.selectbox(label, options, index=idx)

        # numeric
        user_row["Soil_pH"] = num_input(cL1, "Soil_pH", *RANGES["Soil_pH"], default=7.0)
        user_row["Chloride Content (mg/kg)"] = num_input(cL1, "Chloride Content (mg/kg)", *RANGES["Chloride Content (mg/kg)"], default=200.0)
        user_row["Soil_Resistivity (Ω·cm)"] = num_input(cL1, "Soil_Resistivity (Ω·cm)", *RANGES["Soil_Resistivity (Ω·cm)"], default=5000.0)
        user_row["Sulphate_Content (mg/kg)"] = num_input(cL2, "Sulphate_Content (mg/kg)", *RANGES["Sulphate_Content (mg/kg)"], default=100.0)
        user_row["Moisture_Content (%)"] = num_input(cL2, "Moisture_Content (%)", *RANGES["Moisture_Content (%)"], default=15.0)

        # categorical
        user_row["Soil Type"] = cat_input(cL1, "Soil Type (USCS)", SOIL_TYPES, default="CL")
        user_row["Location wrt Water Table"] = cat_input(cL2, "Location wrt Water Table", WATER_TABLE, default="Above WaterTable")
        user_row["Foreign_Inclusion_Type"] = cat_input(cL1, "Foreign_Inclusion_Type", FOREIGN_INCL, default="None")

        # binary
        na_fill = cL2.checkbox("Is_Fill_Material: NA", value=False, key="na_Is_Fill_Material")
        if na_fill:
            user_row["Is_Fill_Material"] = np.nan
        else:
            user_row["Is_Fill_Material"] = cL2.selectbox("Is_Fill_Material", FILL_MATERIAL, index=0)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Monte Carlo Settings</div>", unsafe_allow_html=True)

        Ns = st.number_input("Sample size (Ns)", min_value=1000, max_value=50000, value=MC_NS_DEFAULT, step=1000)

        st.markdown("<b>Bounds</b>", unsafe_allow_html=True)
        nL = st.number_input("n lower bound", value=0.40, step=0.01, format="%.2f")
        nU = st.number_input("n upper bound", value=0.70, step=0.01, format="%.2f")

        bL = st.number_input("β lower bound", value=0.020, step=0.001, format="%.3f")
        bU = st.number_input("β upper bound", value=0.040, step=0.001, format="%.3f")

        shared = st.checkbox("Shared n, β across samples", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Run prediction + Monte Carlo")


if submitted:
    # enforce <=2 missing among 9 ML features only
    miss = count_missing_ml_features(user_row, expected_cols)
    if miss > 2:
        st.error(f"Too many unknown ML inputs: {miss}. Maximum allowed is 2.")
        st.stop()

    # build input row in expected order
    X_in = pd.DataFrame([{c: user_row.get(c, np.nan) for c in expected_cols}])

    # Predict k distribution from GPR
    try:
        X_tr = prep.transform(X_in)  # saved fitted KNN/scaler/OHE
        mu_k_arr, sd_k_arr = gpr.predict(np.asarray(X_tr, float), return_std=True)
        mu_k = float(mu_k_arr[0])
        sd_k = float(max(sd_k_arr[0], EPS))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Single-age TL
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
    ).iloc[0]

    mean_TL = float(single_df["TL_mean (mm)"])
    sd_TL = float(single_df["TL_sd (mm)"])

    st.markdown("## Output")

    # Predicted k
    st.markdown(
        rf"**Predicted k from ML:**  $\mu_k={mu_k:.6f}$   and   $\sigma_k={sd_k:.6f}$"
    )

    # TL at age
    st.markdown(
        rf"**Thickness loss at Input Age ({int(age_now)} years):** "
        rf"${mean_TL:.3f} \pm {sd_TL:.3f}$ (68% CI)  ;  "
        rf"${mean_TL:.3f} \pm {2.0*sd_TL:.3f}$ (95% CI)"
    )

    # Horizon table
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

    out_tbl = pd.DataFrame({
        "Age (yr)": horizon_df["Age (yr)"].astype(int),
        "Mean_TL (mm)": horizon_df["TL_mean (mm)"],
        "TL_sd (mm)": horizon_df["TL_sd (mm)"],
        "TL (68% CI)": [fmt_ci(a, b, nd=3) for a, b in zip(horizon_df["TL_lo68 (mm)"], horizon_df["TL_hi68 (mm)"])],
        "TL (95% CI)": [fmt_ci(a, b, nd=3) for a, b in zip(horizon_df["TL_lo95 (mm)"], horizon_df["TL_hi95 (mm)"])],
    })

    st.markdown("### Thickness Loss across Time")
    st.dataframe(out_tbl, use_container_width=True)

    # Plot
    ages = horizon_df["Age (yr)"].values
    mean = horizon_df["TL_mean (mm)"].values
    lo68 = horizon_df["TL_lo68 (mm)"].values
    hi68 = horizon_df["TL_hi68 (mm)"].values
    lo95 = horizon_df["TL_lo95 (mm)"].values
    hi95 = horizon_df["TL_hi95 (mm)"].values

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ages, mean, linewidth=2, label="Mean TL")
    plt.fill_between(ages, lo95, hi95, alpha=0.20, label="95% CI")
    plt.fill_between(ages, lo68, hi68, alpha=0.35, label="68% CI")
    plt.xlabel("Age (yr)")
    plt.ylabel("Thickness Loss (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig)
