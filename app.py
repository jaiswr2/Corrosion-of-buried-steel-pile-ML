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

# Theme
MCMAROON = "#7A003C"
MCYELLOW = "#FDB515"
LABELBLUE = "#1f5fbf"
BLACKTXT = "#111111"

# Plot theme requested
PLOT_MEAN = "cornflowerblue"
PLOT_BAND = "deeppink"

# Ranges (bounds + tooltips) -- UI names (keep UI the same)
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

# Categoricals (UI)
SOIL_TYPES = ["GT", "CL", "SM", "ML", "SP", "CH", "GP", "SW", "OL", "SC"]
WATER_TABLE = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]
FOREIGN_INCL = ["None", "Shreded wood", "Cinder", "Flyash"]
FILL_MATERIAL = [0, 1]

AGES_HORIZON = [10, 30, 50, 70, 80]

# UI ML keys (for NA counting rule)
UI_ML_COLS = [
    "Soil_pH",
    "Chloride Content (mg/kg)",
    "Soil_Resistivity (Ω·cm)",
    "Sulphate_Content (mg/kg)",
    "Moisture_Content (%)",
    "Soil Type",
    "Location wrt Water Table",
    "Foreign_Inclusion_Type",
    "Is_Fill_Material",
]


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


def mc_TL_from_k(
    mu_k, sd_k, ages, T_used, T0,
    mu_n, mu_beta,
    n_bounds=(0.4, 0.7), beta_bounds=(0.02, 0.04),
    Ns=5000, seed=42, shared_n_beta=False
):
    """
    k ~ Normal(mu_k, sd_k) from ML (GPR). Clip k at EPS to keep TL physical.
    n, beta ~ TruncNormal using bounds as ~95% interval.
    TL = k * t^n * exp(beta*(T - T0))

    Output: mean + predictive intervals (75% and 95%) from MC quantiles.
    """
    rng = np.random.default_rng(seed)
    mu_k = float(mu_k)
    sd_k = float(max(sd_k, EPS))

    nL, nU = n_bounds
    bL, bU = beta_bounds

    # treat bounds as approx 95% -> infer sigma
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
        
        q05 = float(np.quantile(TL_s, 0.05))
        q25 = float(np.quantile(TL_s, 0.25))
        q75 = float(np.quantile(TL_s, 0.75))
        q95 = float(np.quantile(TL_s, 0.95))
        
        out_rows.append({
            "Age": int(t_age),
            "Mean Thickness loss (mm)": TL_mean,
            "Q05 Thickness loss (mm)": q05,
            "Q25 Thickness loss (mm)": q25,
            "Q75 Thickness loss (mm)": q75,
            "Q95 Thickness loss (mm)": q95,
        })

    return pd.DataFrame(out_rows)


def count_missing_ml_features(row_dict, cols):
    miss = 0
    for c in cols:
        v = row_dict.get(c, None)
        if v is None:
            miss += 1
        elif isinstance(v, float) and np.isnan(v):
            miss += 1
    return miss


def fmt_pi(lo, hi, nd=3):
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
        font-size: 52px;
        margin-bottom: 6px;
      }}

      .subtitle {{
        color: {BLACKTXT};
        font-size: 30px;
        font-weight: 500;
        line-height: 1.25;
        margin-bottom: 6px;
      }}

      .sectiontitle {{
        color: {MCMAROON};
        font-size: 34px;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 0px;
      }}

      .sectionnote {{
        color: {BLACKTXT};
        font-size: 22px;
        font-weight: 500;
        margin-top: 4px;
        margin-bottom: 14px;
      }}

      .feat {{
        color: {LABELBLUE};
        font-weight: 700;
        font-size: 18px;
        margin-top: 8px;
        margin-bottom: 4px;
      }}

      div[data-testid="stNumberInput"] label,
      div[data-testid="stSelectbox"] label,
      div[data-testid="stCheckbox"] label {{
        font-size: 16px !important;
        font-weight: 600 !important;
        color: {BLACKTXT} !important;
      }}

      div[data-testid="stCheckbox"] p {{
        font-size: 16px !important;
      }}

      .stNumberInput input,
      .stSelectbox div[data-baseweb="select"] {{
        font-size: 16px !important;
      }}

      div.stButton > button,
      div[data-testid="stFormSubmitButton"] button {{
        background: {MCYELLOW} !important;
        border: 2px solid {MCMAROON} !important;
        color: {MCMAROON} !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        padding: 0.65rem 1.3rem !important;
        border-radius: 12px !important;
      }}
      div.stButton > button:hover,
      div[data-testid="stFormSubmitButton"] button:hover {{
        background: #ffd36a !important;
      }}

      .outline {{
        font-size: 24px;
        line-height: 1.35;
      }}

      .katex-display {{
        margin: 0.4em 0 0.2em 0 !important;
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

st.markdown("<div class='sectiontitle'>Input Parameters</div>", unsafe_allow_html=True)
st.markdown("<div class='sectionnote'>Up to two unknowns allowed to be imputed by kNN</div>", unsafe_allow_html=True)

prep, gpr, meta = load_artifacts()

# IMPORTANT: expected input names are now dataset-style (from artifacts)
expected_cols = meta["expected_raw_columns"]  # e.g., ['pH','Chloride (mg/kg)',...]
T0 = float(meta["constants"]["T0"])
mu_n = float(meta["constants"]["mu_n"])
mu_beta = float(meta["constants"]["mu_beta"])


def feature_header(text: str):
    st.markdown(f"<div class='feat'>{text}</div>", unsafe_allow_html=True)


def checkbox_unknown(key, default=False, disabled=False):
    return st.checkbox("Unknown (NA)", value=default, key=key, disabled=disabled)


def num_input_no_label(value_key, minv, maxv, default, step, disabled=False):
    return st.number_input(
        label="",
        value=float(default),
        min_value=float(minv),
        max_value=float(maxv),
        step=float(step),
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed",
    )


def select_input_no_label(value_key, options, default_idx=0, disabled=False):
    return st.selectbox(
        label="",
        options=options,
        index=default_idx,
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed",
    )


# Layout: inputs (left) and MC (right)
left, right = st.columns([2.4, 1.1], gap="large")

with st.form("input_form"):
    with left:
        try:
            left_box = st.container(border=True)
        except TypeError:
            left_box = st.container()

        with left_box:
            c1, c2 = st.columns(2, gap="large")
            user_row = {}

            # AGE (required)
            with c1:
                feature_header("Age (yr)")
                _age_na = checkbox_unknown("na_age", default=False)
                _ = st.number_input(
                    label="",
                    min_value=1,
                    max_value=200,
                    value=10,
                    step=1,
                    key="age_value",
                    label_visibility="collapsed",
                )

            # TEMPERATURE (optional; NA -> default T0)
            with c2:
                feature_header("Temperature (°C)")
                temp_na = checkbox_unknown("na_temp", default=True)
                if temp_na:
                    T_used = float(T0)
                    _ = num_input_no_label("temp_value", -50.0, 60.0, float(T0), STEPS["Temperature (°C)"], disabled=True)
                else:
                    T_used = float(num_input_no_label("temp_value", -50.0, 60.0, float(T0), STEPS["Temperature (°C)"]))

            # UI inputs (we will map to dataset columns later)
            with c1:
                feature_header("Soil_pH")
                na = checkbox_unknown("na_Soil_pH", default=False)
                user_row["Soil_pH"] = np.nan if na else float(num_input_no_label("val_Soil_pH", *RANGES["Soil_pH"], 7.0, STEPS["Soil_pH"], disabled=False if not na else True))

            with c2:
                feature_header("Chloride Content (mg/kg)")
                na = checkbox_unknown("na_Chloride", default=False)
                user_row["Chloride Content (mg/kg)"] = np.nan if na else float(num_input_no_label("val_Chloride", *RANGES["Chloride Content (mg/kg)"], 200.0, STEPS["Chloride Content (mg/kg)"], disabled=False if not na else True))

            with c1:
                feature_header("Soil_Resistivity (Ω·cm)")
                na = checkbox_unknown("na_Res", default=False)
                user_row["Soil_Resistivity (Ω·cm)"] = np.nan if na else float(num_input_no_label("val_Res", *RANGES["Soil_Resistivity (Ω·cm)"], 5000.0, STEPS["Soil_Resistivity (Ω·cm)"], disabled=False if not na else True))

            with c2:
                feature_header("Sulphate_Content (mg/kg)")
                na = checkbox_unknown("na_Sul", default=False)
                user_row["Sulphate_Content (mg/kg)"] = np.nan if na else float(num_input_no_label("val_Sul", *RANGES["Sulphate_Content (mg/kg)"], 100.0, STEPS["Sulphate_Content (mg/kg)"], disabled=False if not na else True))

            with c1:
                feature_header("Moisture_Content (%)")
                na = checkbox_unknown("na_Moist", default=False)
                user_row["Moisture_Content (%)"] = np.nan if na else float(num_input_no_label("val_Moist", *RANGES["Moisture_Content (%)"], 15.0, STEPS["Moisture_Content (%)"], disabled=False if not na else True))

            with c2:
                feature_header("Soil Type (USCS)")
                na = checkbox_unknown("na_SoilType", default=False)
                user_row["Soil Type"] = np.nan if na else select_input_no_label("val_SoilType", SOIL_TYPES, default_idx=1, disabled=False if not na else True)

            with c1:
                feature_header("Location wrt Water Table")
                na = checkbox_unknown("na_WT", default=False)
                user_row["Location wrt Water Table"] = np.nan if na else select_input_no_label("val_WT", WATER_TABLE, default_idx=0, disabled=False if not na else True)

            with c2:
                feature_header("Foreign_Inclusion_Type")
                na = checkbox_unknown("na_Foreign", default=False)
                user_row["Foreign_Inclusion_Type"] = np.nan if na else select_input_no_label("val_Foreign", FOREIGN_INCL, default_idx=0, disabled=False if not na else True)

            with c1:
                feature_header("Is_Fill_Material")
                na = checkbox_unknown("na_Fill", default=False)
                user_row["Is_Fill_Material"] = np.nan if na else select_input_no_label("val_Fill", FILL_MATERIAL, default_idx=0, disabled=False if not na else True)

    with right:
        try:
            right_box = st.container(border=True)
        except TypeError:
            right_box = st.container()

        with right_box:
            st.markdown(f"<div class='sectiontitle' style='margin-top:0;'>Monte Carlo Settings</div>", unsafe_allow_html=True)
            st.markdown("<div class='sectionnote' style='margin-bottom:10px;'> </div>", unsafe_allow_html=True)

            Ns = st.number_input("Sample size (Ns)", min_value=1000, max_value=50000, value=MC_NS_DEFAULT, step=1000)
            nL = st.number_input("n lower bound", value=0.40, step=0.01, format="%.2f")
            nU = st.number_input("n upper bound", value=0.70, step=0.01, format="%.2f")
            bL = st.number_input("β lower bound", value=0.020, step=0.001, format="%.3f")
            bU = st.number_input("β upper bound", value=0.040, step=0.001, format="%.3f")
            shared = st.checkbox("Shared n, β across samples", value=False)

    submitted = st.form_submit_button("Run predictions + Monte Carlo")


# =========================
# OUTPUT
# =========================
if submitted:
    if st.session_state.get("na_age", False):
        st.error("Age is required. Please keep Age: Unknown (NA) unchecked.")
        st.stop()

    # missing-count rule applies to the 9 ML inputs (UI), not the artifact column names
    miss = count_missing_ml_features(user_row, UI_ML_COLS)
    if miss > 2:
        st.error(f"Too many unknown ML inputs: {miss}. Maximum allowed is 2.")
        st.stop()

    # Map UI inputs -> dataset/raw columns expected by your saved preprocessor
    # expected raw columns (from your artifacts):
    # ['pH','Chloride (mg/kg)','Resistivity (Ω·cm)','Sulphate (mg/kg)',
    #  'Moisture (%)','Soil Type','Location wrt WaterTable','Foreign Inclusion','Fill Material']
    X_in = pd.DataFrame([{
        "pH": user_row.get("Soil_pH", np.nan),
        "Chloride (mg/kg)": user_row.get("Chloride Content (mg/kg)", np.nan),
        "Resistivity (Ω·cm)": user_row.get("Soil_Resistivity (Ω·cm)", np.nan),
        "Sulphate (mg/kg)": user_row.get("Sulphate_Content (mg/kg)", np.nan),
        "Moisture (%)": user_row.get("Moisture_Content (%)", np.nan),
        "Soil Type": user_row.get("Soil Type", np.nan),
        "Location wrt WaterTable": user_row.get("Location wrt Water Table", np.nan),
        "Foreign Inclusion": user_row.get("Foreign_Inclusion_Type", np.nan),
        "Fill Material": user_row.get("Is_Fill_Material", np.nan),
    }])

    # Ensure correct column order (safety)
    X_in = X_in.reindex(columns=expected_cols)

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

    mean_TL = float(single["Mean Thickness loss (mm)"])
    q75 = float(single["Q75 Thickness loss (mm)"])
    q95 = float(single["Q95 Thickness loss (mm)"])

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
        "Mean Thickness loss (mm)": horizon_df["Mean Thickness loss (mm)"].round(3),
        "75 Quantile": horizon_df["Q75 Thickness loss (mm)"].round(3),
        "95 Quantile": horizon_df["Q95 Thickness loss (mm)"].round(3),
    })

    st.markdown("<div class='sectiontitle'>Output</div>", unsafe_allow_html=True)

    out_left, out_right = st.columns([1.2, 1.0], gap="large")

    with out_left:
        st.markdown("<div class='outline'><b>Predicted k from ML</b></div>", unsafe_allow_html=True)
        st.latex(rf"\mu_k={mu_k:.6f}\qquad \sigma_k={sd_k:.6f}")

        st.markdown(f"<div class='outline'><b>Thickness loss at Input Age ({age_now} years)</b></div>", unsafe_allow_html=True)
        st.write(
            f"Mean Thickness loss (mm) = **{mean_TL:.3f}**   |   "
            f"75 Quantile = **{q75:.3f}**   |   "
            f"95 Quantile = **{q95:.3f}**"
        )

        st.markdown("<div class='sectiontitle'>Thickness Loss across Time</div>", unsafe_allow_html=True)
        st.dataframe(out_tbl, use_container_width=True, hide_index=True)

        csv_bytes = out_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Download TL table (CSV)", data=csv_bytes, file_name="thickness_loss_table.csv", mime="text/csv")

    with out_right:
        ages = horizon_df["Age"].values
        mean = horizon_df["Mean Thickness loss (mm)"].values
        
        lo50 = horizon_df["Q25 Thickness loss (mm)"].values
        hi50 = horizon_df["Q75 Thickness loss (mm)"].values
        
        lo90 = horizon_df["Q05 Thickness loss (mm)"].values
        hi90 = horizon_df["Q95 Thickness loss (mm)"].values

        PLOT_MEAN = "cornflowerblue"
        PLOT_BAND = "deeppink"
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(ages, mean, linewidth=2.5, color=PLOT_MEAN, label="Mean Thickness loss")
        
        plt.fill_between(ages, lo90, hi90, alpha=0.18, color=PLOT_BAND, label="90% band (5–95)")
        plt.fill_between(ages, lo50, hi50, alpha=0.35, color=PLOT_BAND, label="50% band (25–75)")
        
        plt.xlabel("Age (year)")
        plt.ylabel("Thickness loss (mm)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        st.pyplot(fig)

        png_bytes = fig_to_png_bytes(fig)
        st.download_button("Download plot (PNG)", data=png_bytes, file_name="thickness_loss_plot.png", mime="image/png")
