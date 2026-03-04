import json
from pathlib import Path
from io import BytesIO

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
MCYELLOW = "#FDB515"
LABELBLUE = "#1f5fbf"

# Ranges (bounds + tooltips)
RANGES = {
    "Soil_pH": (3.0, 10.0),
    "Chloride Content (mg/kg)": (0.3, 11400.0),
    "Soil_Resistivity (Ω·cm)": (80.0, 44000.0),
    "Sulphate_Content (mg/kg)": (6.9, 21800.0),
    "Moisture_Content (%)": (1.7, 261.4),
}

# Steps
STEPS = {
    "Soil_pH": 0.1,
    "Chloride Content (mg/kg)": 50.0,
    "Soil_Resistivity (Ω·cm)": 100.0,
    "Sulphate_Content (mg/kg)": 100.0,
    "Moisture_Content (%)": 10.0,
    "Temperature (°C)": 1.0,
}

# Categoricals
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
    k ~ Normal(mu_k, sd_k). Clip k at EPS to keep TL physical.
    n,beta ~ TruncNormal using bounds as ~95% interval.
    TL = k * t^n * exp(beta*(T - T0))
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
        z = rng.standard_normal(Ns)
        k_s = np.maximum(mu_k + sd_k * z, EPS)

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
            "Age": int(t_age),
            "Mean_TL (mm)": TL_mean,
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


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title="Predicting corrosion-induced thickness loss", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ font-size: 20px; }}

      .title {{
        color: {MCMAROON};
        font-weight: 900;
        font-size: 48px;
        margin-bottom: 6px;
      }}
      .subtitle {{
        color: {MCMAROON};
        font-size: 24px;
        font-weight: 400;
        line-height: 1.25;
        margin-bottom: 6px;
      }}
      .sectiontitle {{
        color: {MCMAROON};
        font-size: 30px;
        font-weight: 900;
        margin-top: 16px;
        margin-bottom: 10px;
      }}
      .panel {{
        border: 2px solid rgba(0,0,0,0.18);
        border-radius: 14px;
        padding: 16px 18px;
        background: rgba(250,250,250,0.90);
      }}

      /* Feature header (blue) */
      .feat {{
        color: {LABELBLUE};
        font-weight: 600;
        font-size: 18px;
        margin-top: 8px;
        margin-bottom: 4px;
      }}

      /* Inputs */
      .stNumberInput input,
      .stSelectbox div[data-baseweb="select"] {{
        font-size: 16px !important;
      }}
      .stCheckbox p {{
        font-size: 16px !important;
      }}

      /* MC panel labels must match left */
      .mc-panel label {{
        font-size: 18px !important;
        font-weight: 600 !important;
        color: {LABELBLUE} !important;
      }}
      .mc-panel .stCheckbox p {{
        font-size: 18px !important;
      }}

      /* Yellow run button */
      div.stButton > button {{
        background: rgba(253,181,21,0.75) !important;
        border: 2px solid {MCMAROON} !important;
        color: {MCMAROON} !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 12px !important;
      }}
      div.stButton > button:hover {{
        background: rgba(253,181,21,0.95) !important;
      }}

      .outline {{
        font-size: 24px;
        line-height: 1.35;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>Predicting corrosion-induced thickness loss in buried steel pile</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Estimate soil aggressiveness factor k using ML and propagate Thickness Loss using Monte Carlo simulations:</div>",
    unsafe_allow_html=True
)
st.latex(r"TL(t,T)=k\, t^n \exp\left[\beta\,(T-T_0)\right]")

st.markdown(
    "<div class='sectiontitle'>Input Parameters [Up to two unknowns allowed to be imputed by kNN]</div>",
    unsafe_allow_html=True
)

prep, gpr, meta = load_artifacts()
expected_cols = meta["expected_raw_columns"]
T0 = float(meta["constants"]["T0"])
mu_n = float(meta["constants"]["mu_n"])
mu_beta = float(meta["constants"]["mu_beta"])


def feature_header(text: str):
    st.markdown(f"<div class='feat'>{text}</div>", unsafe_allow_html=True)


def checkbox_unknown(key, default=False, disabled=False):
    return st.checkbox("Unknown (NA)", value=default, key=key, disabled=disabled)


def num_input_no_label(value_key, minv, maxv, default, step, disabled=False, help_text=""):
    return st.number_input(
        label="",
        value=float(default),
        min_value=float(minv),
        max_value=float(maxv),
        step=float(step),
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed",
        help=help_text
    )


def select_input_no_label(value_key, options, default_idx=0, disabled=False):
    return st.selectbox(
        label="",
        options=options,
        index=default_idx,
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed"
    )


left, right = st.columns([2.4, 1.1], gap="large")

with st.form("input_form"):
    # LEFT PANEL
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        user_row = {}

        # AGE
        with c1:
            feature_header("Age (yr)")
            age_na = checkbox_unknown("na_age", default=False)
            age_now = st.number_input(
                label="",
                min_value=1,
                max_value=80,
                value=10,
                step=1,
                key="age_value",
                label_visibility="collapsed"
            )

        # TEMPERATURE
        with c2:
            feature_header("Temperature (°C)")
            temp_na = checkbox_unknown("na_temp", default=True)
            if temp_na:
                T_used = float(T0)
                _ = num_input_no_label("temp_value", -50.0, 60.0, float(T0), STEPS["Temperature (°C)"], disabled=True)
            else:
                T_used = float(num_input_no_label("temp_value", -50.0, 60.0, float(T0), STEPS["Temperature (°C)"]))

        # Soil_pH
        with c1:
            feature_header("Soil_pH")
            na = checkbox_unknown("na_Soil_pH", default=False)
            if na:
                _ = num_input_no_label("val_Soil_pH", *RANGES["Soil_pH"], 7.0, STEPS["Soil_pH"], disabled=True,
                                       help_text=f"Range: {RANGES['Soil_pH'][0]} to {RANGES['Soil_pH'][1]}")
                user_row["Soil_pH"] = np.nan
            else:
                user_row["Soil_pH"] = float(num_input_no_label("val_Soil_pH", *RANGES["Soil_pH"], 7.0, STEPS["Soil_pH"],
                                                               help_text=f"Range: {RANGES['Soil_pH'][0]} to {RANGES['Soil_pH'][1]}"))

        # Chloride
        with c2:
            feature_header("Chloride Content (mg/kg)")
            na = checkbox_unknown("na_Chloride", default=False)
            if na:
                _ = num_input_no_label("val_Chloride", *RANGES["Chloride Content (mg/kg)"], 200.0, STEPS["Chloride Content (mg/kg)"], disabled=True,
                                       help_text=f"Range: {RANGES['Chloride Content (mg/kg)'][0]} to {RANGES['Chloride Content (mg/kg)'][1]}")
                user_row["Chloride Content (mg/kg)"] = np.nan
            else:
                user_row["Chloride Content (mg/kg)"] = float(num_input_no_label("val_Chloride", *RANGES["Chloride Content (mg/kg)"], 200.0, STEPS["Chloride Content (mg/kg)"],
                                                                                help_text=f"Range: {RANGES['Chloride Content (mg/kg)'][0]} to {RANGES['Chloride Content (mg/kg)'][1]}"))

        # Resistivity
        with c1:
            feature_header("Soil_Resistivity (Ω·cm)")
            na = checkbox_unknown("na_Res", default=False)
            if na:
                _ = num_input_no_label("val_Res", *RANGES["Soil_Resistivity (Ω·cm)"], 5000.0, STEPS["Soil_Resistivity (Ω·cm)"], disabled=True,
                                       help_text=f"Range: {RANGES['Soil_Resistivity (Ω·cm)'][0]} to {RANGES['Soil_Resistivity (Ω·cm)'][1]}")
                user_row["Soil_Resistivity (Ω·cm)"] = np.nan
            else:
                user_row["Soil_Resistivity (Ω·cm)"] = float(num_input_no_label("val_Res", *RANGES["Soil_Resistivity (Ω·cm)"], 5000.0, STEPS["Soil_Resistivity (Ω·cm)"],
                                                                               help_text=f"Range: {RANGES['Soil_Resistivity (Ω·cm)'][0]} to {RANGES['Soil_Resistivity (Ω·cm)'][1]}"))

        # Sulphate
        with c2:
            feature_header("Sulphate_Content (mg/kg)")
            na = checkbox_unknown("na_Sul", default=False)
            if na:
                _ = num_input_no_label("val_Sul", *RANGES["Sulphate_Content (mg/kg)"], 100.0, STEPS["Sulphate_Content (mg/kg)"], disabled=True,
                                       help_text=f"Range: {RANGES['Sulphate_Content (mg/kg)'][0]} to {RANGES['Sulphate_Content (mg/kg)'][1]}")
                user_row["Sulphate_Content (mg/kg)"] = np.nan
            else:
                user_row["Sulphate_Content (mg/kg)"] = float(num_input_no_label("val_Sul", *RANGES["Sulphate_Content (mg/kg)"], 100.0, STEPS["Sulphate_Content (mg/kg)"],
                                                                                help_text=f"Range: {RANGES['Sulphate_Content (mg/kg)'][0]} to {RANGES['Sulphate_Content (mg/kg)'][1]}"))

        # Moisture
        with c1:
            feature_header("Moisture_Content (%)")
            na = checkbox_unknown("na_Moist", default=False)
            if na:
                _ = num_input_no_label("val_Moist", *RANGES["Moisture_Content (%)"], 15.0, STEPS["Moisture_Content (%)"], disabled=True,
                                       help_text=f"Range: {RANGES['Moisture_Content (%)'][0]} to {RANGES['Moisture_Content (%)'][1]}")
                user_row["Moisture_Content (%)"] = np.nan
            else:
                user_row["Moisture_Content (%)"] = float(num_input_no_label("val_Moist", *RANGES["Moisture_Content (%)"], 15.0, STEPS["Moisture_Content (%)"],
                                                                            help_text=f"Range: {RANGES['Moisture_Content (%)'][0]} to {RANGES['Moisture_Content (%)'][1]}"))

        # Soil Type
        with c2:
            feature_header("Soil Type (USCS)")
            na = checkbox_unknown("na_SoilType", default=False)
            if na:
                _ = select_input_no_label("val_SoilType", SOIL_TYPES, default_idx=1, disabled=True)
                user_row["Soil Type"] = np.nan
            else:
                user_row["Soil Type"] = select_input_no_label("val_SoilType", SOIL_TYPES, default_idx=1)

        # Water table
        with c1:
            feature_header("Location wrt Water Table")
            na = checkbox_unknown("na_WT", default=False)
            if na:
                _ = select_input_no_label("val_WT", WATER_TABLE, default_idx=0, disabled=True)
                user_row["Location wrt Water Table"] = np.nan
            else:
                user_row["Location wrt Water Table"] = select_input_no_label("val_WT", WATER_TABLE, default_idx=0)

        # Foreign inclusion
        with c2:
            feature_header("Foreign_Inclusion_Type")
            na = checkbox_unknown("na_Foreign", default=False)
            if na:
                _ = select_input_no_label("val_Foreign", FOREIGN_INCL, default_idx=0, disabled=True)
                user_row["Foreign_Inclusion_Type"] = np.nan
            else:
                user_row["Foreign_Inclusion_Type"] = select_input_no_label("val_Foreign", FOREIGN_INCL, default_idx=0)

        # Fill material
        with c1:
            feature_header("Is_Fill_Material")
            na = checkbox_unknown("na_Fill", default=False)
            if na:
                _ = select_input_no_label("val_Fill", FILL_MATERIAL, default_idx=0, disabled=True)
                user_row["Is_Fill_Material"] = np.nan
            else:
                user_row["Is_Fill_Material"] = select_input_no_label("val_Fill", FILL_MATERIAL, default_idx=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT PANEL
    with right:
        st.markdown("<div class='panel mc-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='sectiontitle' style='margin-top:0;color:{MCMAROON};'>Monte Carlo Settings</div>", unsafe_allow_html=True)

        Ns = st.number_input("Sample size (Ns)", min_value=1000, max_value=50000, value=MC_NS_DEFAULT, step=1000)

        nL = st.number_input("n lower bound", value=0.40, step=0.01, format="%.2f")
        nU = st.number_input("n upper bound", value=0.70, step=0.01, format="%.2f")

        bL = st.number_input("β lower bound", value=0.020, step=0.001, format="%.3f")
        bU = st.number_input("β upper bound", value=0.040, step=0.001, format="%.3f")

        shared = st.checkbox("Shared n, β across samples", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Run predictions + Monte Carlo")


# =========================
# OUTPUT
# =========================
if submitted:
    if st.session_state.get("na_age", False):
        st.error("Age is required. Please keep Age: Unknown (NA) unchecked.")
        st.stop()

    miss = count_missing_ml_features(user_row, expected_cols)
    if miss > 2:
        st.error(f"Too many unknown ML inputs: {miss}. Maximum allowed is 2.")
        st.stop()

    X_in = pd.DataFrame([{c: user_row.get(c, np.nan) for c in expected_cols}])

    try:
        X_tr = prep.transform(X_in)
        mu_k_arr, sd_k_arr = gpr.predict(np.asarray(X_tr, float), return_std=True)
        mu_k = float(mu_k_arr[0])
        sd_k = float(max(sd_k_arr[0], EPS))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    age_now = int(st.session_state["age_value"])

    single = mc_TL_from_k(
        mu_k=mu_k, sd_k=sd_k,
        ages=[age_now],
        T_used=float(T_used), T0=float(T0),
        mu_n=mu_n, mu_beta=mu_beta,
        n_bounds=(float(nL), float(nU)),
        beta_bounds=(float(bL), float(bU)),
        Ns=int(Ns),
        seed=42,
        shared_n_beta=bool(shared),
    ).iloc[0]

    mean_TL = float(single["Mean_TL (mm)"])
    sd_TL = float(single["TL_sd (mm)"])

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
        "Age": horizon_df["Age"].astype(int),
        "Mean_TL (mm)": horizon_df["Mean_TL (mm)"].round(3),
        "TL_sd (mm)": horizon_df["TL_sd (mm)"].round(3),
        "TL (68% CI)": [fmt_ci(a, b, 3) for a, b in zip(horizon_df["TL_lo68 (mm)"], horizon_df["TL_hi68 (mm)"])],
        "TL (95% CI)": [fmt_ci(a, b, 3) for a, b in zip(horizon_df["TL_lo95 (mm)"], horizon_df["TL_hi95 (mm)"])],
    })

    st.markdown("<div class='sectiontitle'>Output</div>", unsafe_allow_html=True)

    out_left, out_right = st.columns([1.2, 1.0], gap="large")

    with out_left:
        st.markdown("<div class='outline'><b>Predicted k from ML</b></div>", unsafe_allow_html=True)
        st.latex(rf"\mu_k={mu_k:.6f}\qquad \sigma_k={sd_k:.6f}")

        st.markdown(f"<div class='outline'><b>Thickness loss at Input Age ({age_now} years)</b></div>", unsafe_allow_html=True)
        st.latex(rf"\text{{68\% CI: }} {mean_TL:.3f}\pm{sd_TL:.3f}\qquad \text{{95\% CI: }} {mean_TL:.3f}\pm{2.0*sd_TL:.3f}")

        st.markdown("<div class='sectiontitle'>Thickness Loss across Time</div>", unsafe_allow_html=True)
        st.dataframe(out_tbl, use_container_width=True, hide_index=True)

        csv_bytes = out_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Download TL table (CSV)", data=csv_bytes, file_name="thickness_loss_table.csv", mime="text/csv")

    with out_right:
        ages = horizon_df["Age"].values
        mean = horizon_df["Mean_TL (mm)"].values
        lo68 = horizon_df["TL_lo68 (mm)"].values
        hi68 = horizon_df["TL_hi68 (mm)"].values
        lo95 = horizon_df["TL_lo95 (mm)"].values
        hi95 = horizon_df["TL_hi95 (mm)"].values

        fig = plt.figure(figsize=(10, 6))
        plt.plot(ages, mean, linewidth=2, label="Mean TL")
        plt.fill_between(ages, lo95, hi95, alpha=0.20, label="95% CI")
        plt.fill_between(ages, lo68, hi68, alpha=0.35, label="68% CI")
        plt.xlabel("Age (year)")
        plt.ylabel("Thickness Loss (mm)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        st.pyplot(fig)

        png_bytes = fig_to_png_bytes(fig)
        st.download_button("Download plot (PNG)", data=png_bytes, file_name="thickness_loss_plot.png", mime="image/png")

