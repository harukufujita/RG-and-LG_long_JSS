# ------------------------------------------------------------
# 3-year RFS & OS Predictor (RSF models, 14 variables)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ────────────────────────────────────────────────────────────
# 0. Load models
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

rsf_rfs = load_model("rsf_final_14vars.pkl")       # RFSモデル
rsf_os  = load_model("rsf_final_os_14vars.pkl")    # OSモデル

# ────────────────────────────────────────────────────────────
# 1. Encoding maps
# ────────────────────────────────────────────────────────────
asa_map = {"1": 0, "2": 1, "≥3": 2}
surg_map = {"DG": 0, "PG": 1, "TG": 2}
recons_map = {"B-I": 0, "B-II": 1, "R-Y": 2, "Others": 3}
macro_map = {"type0": 0, "type1": 1, "type2": 2, "type3": 3, "type4": 4, "type5": 5}
v_map = {"v0": 0, "v1": 1}
histo_map = {"pap": 0, "tub": 1, "por": 2, "sig": 3, "muc": 4}
pt_map = {"pT0": 0, "pT1a": 1, "pT1b": 2, "pT2": 3, "pT3": 4, "pT4a": 5, "pT4b": 6}
pn_map = {"pN0": 0, "pN1": 1, "pN2": 2, "pN3a": 3, "pN3b": 4}
stage_map = {"0": 0, "IA": 1, "IB": 2, "IIA": 3, "IIB": 4, "IIIA": 5, "IIIB": 6, "IIIC": 7, "IV (CY1)": 8}

# ────────────────────────────────────────────────────────────
# 2. Stage determination function
# ────────────────────────────────────────────────────────────
def determine_stage(pt_key, pn_key):
    if pt_key == "pT0":
        return "0"
    if pt_key in ("pT1a", "pT1b"):
        return {"pN0": "IA", "pN1": "IB", "pN2": "IIA", "pN3a": "IIB", "pN3b": "IIIB"}.get(pn_key)
    if pt_key == "pT2":
        return {"pN0": "IB", "pN1": "IIA", "pN2": "IIB", "pN3a": "IIIA", "pN3b": "IIIB"}.get(pn_key)
    if pt_key == "pT3":
        return {"pN0": "IIA", "pN1": "IIB", "pN2": "IIIA", "pN3a": "IIIB", "pN3b": "IIIC"}.get(pn_key)
    if pt_key == "pT4a":
        return {"pN0": "IIB", "pN1": "IIIA", "pN2": "IIIA", "pN3a": "IIIB", "pN3b": "IIIC"}.get(pn_key)
    if pt_key == "pT4b":
        return {"pN0": "IIIA", "pN1": "IIIB", "pN2": "IIIB", "pN3a": "IIIC", "pN3b": "IIIC"}.get(pn_key)
    return None

# ────────────────────────────────────────────────────────────
# 2.5 Tumor diameter internal imputation map
# ────────────────────────────────────────────────────────────
PT_DIAM_IMPUTE = {
    "pT0": 10.0, "pT1a": 33.2, "pT1b": 35.2, "pT2": 39.8,
    "pT3": 59.5, "pT4a": 73.3, "pT4b": 79.0
}
def impute_diameter_from_pt(pt_key: str) -> float:
    return PT_DIAM_IMPUTE.get(pt_key, np.nan)

# ────────────────────────────────────────────────────────────
# 3. Title
# ────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:1rem;'>3-year RFS & OS Predictor</h3>",
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────
# 4. Input widgets
# ────────────────────────────────────────────────────────────
for key, default in [("height_str", ""), ("weight_str", ""), ("bmi", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

age_str = st.text_input("Age (years)", placeholder="e.g. 65")

height_str = st.text_input(
    "Height (cm) (at surgery)",
    key="height_str",
    placeholder="e.g. 160.0 — unnecessary if entering BMI directly",
    help="Height (at surgery, used for BMI calculation).",
)
weight_str = st.text_input(
    "Weight (kg) (at surgery)",
    key="weight_str",
    placeholder="e.g. 50.0 — unnecessary if entering BMI directly",
    help="Weight (at surgery, used for BMI calculation).",
)

# --- BMI計算ボタンをBMI入力欄の上に配置 ---
if st.button("Calculate BMI from Height & Weight"):
    try:
        h = float(st.session_state["height_str"])
        w = float(st.session_state["weight_str"])
        if h > 0:
            bmi_val = w / (h / 100) ** 2
            st.session_state.bmi = f"{bmi_val:.1f}"
            st.experimental_rerun()
        else:
            st.error("Height must be > 0.")
    except ValueError:
        st.error("Height and Weight must be numeric.")

# BMI入力欄
bmi_str = st.text_input(
    "BMI (kg/m²) (at surgery)",
    value=st.session_state.get("bmi", ""),
    key="bmi",
    help="BMI at surgery. Press the button above to calculate from Height and Weight, or enter directly.",
)

cea_str   = st.text_input("CEA (ng/mL)", placeholder="e.g. 4.5")
ca199_str = st.text_input("CA19-9 (U/mL)", placeholder="e.g. 25")

asa  = st.selectbox("ASA-PS", asa_map.keys(), index=None, placeholder="Select ASA-PS")
surg = st.selectbox("Surgical method", surg_map.keys(), index=None, placeholder="Select surgical method")

if surg == "PG":
    recon_options = ["R-Y", "Others"]
elif surg == "TG":
    recon_options = ["R-Y", "Others"]
else:
    recon_options = list(recons_map.keys())

recon = st.selectbox("Reconstruction", recon_options, index=None, placeholder="Select reconstruction")

macro = st.selectbox("Macroscopic type", macro_map.keys(), index=None, placeholder="Select macroscopic type")

diam = st.text_input(
    "Tumor diameter (mm)",
    placeholder="e.g. 45 — not required if unknown"
)

histo = st.selectbox("Histology", histo_map.keys(), index=None, placeholder="Select histology")
vcat  = st.selectbox("Vascular invasion (v)", v_map.keys(), index=None, placeholder="Select vascular invasion")
pt    = st.selectbox("Pathological T", pt_map.keys(), index=None, placeholder="Select pT")
pn    = st.selectbox("Pathological N", pn_map.keys(), index=None, placeholder="Select pN")

all_stage_options = list(stage_map.keys())
stage_options = all_stage_options
auto_stage = None
if pt and pn:
    auto_stage = determine_stage(pt, pn)
    if auto_stage:
        stage_options = [auto_stage, "IV (CY1)"]

stage = st.selectbox(
    "Pathological stage",
    stage_options,
    index=None,
    placeholder="Select stage",
    help="Auto-calculated from pT & pN when provided; you may also select 'IV (CY1)'."
)

# ────────────────────────────────────────────────────────────
# 5. Prediction
# ────────────────────────────────────────────────────────────
if st.button("Predict"):
    try:
        age = int(age_str)
        bmi = float(st.session_state["bmi"])
        cea = float(cea_str)
        ca199 = float(ca199_str)
    except ValueError:
        st.error("Age, BMI, CEA, and CA19-9 must be numeric.")
        st.stop()

    diam_val = None
    if diam and str(diam).strip():
        try:
            diam_val = float(diam)
        except ValueError:
            st.error("Tumor diameter must be numeric if entered.")
            st.stop()
    else:
        if pt:
            diam_val = impute_diameter_from_pt(pt)

    try:
        inp = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "cea_3": cea,
            "ca19_9_3": ca199,
            "asa_ps_2": asa_map[asa],
            "surgical_method2": surg_map[surg],
            "reconstruction2": recons_map[recon],
            "macro2": macro_map[macro],
            "diameter2": diam_val if diam_val is not None else np.nan,
            "histology2": histo_map[histo],
            "v2": v_map[vcat],
            "p_t_3": pt_map[pt],
            "p_n_3": pn_map[pn],
            "p_stage3": stage_map[stage] if stage else np.nan,
        }])
    except KeyError:
        st.error("Please complete the required categorical fields (e.g., ASA-PS, Surgical method, Reconstruction, Macroscopic type, Histology, V, pT, pN, Stage).")
        st.stop()

    try:
        inp_rfs = inp[rsf_rfs.feature_names_in_]
        inp_os  = inp[rsf_os.feature_names_in_]
    except Exception as e:
        st.error(f"Feature alignment error: {e}")
        st.stop()

    time_grid = np.arange(0, 37)
    surv_rfs = np.mean(
        [np.interp(time_grid, fn.x, fn.y) for fn in rsf_rfs.predict_survival_function(inp_rfs)],
        axis=0
    )
    surv_os = np.mean(
        [np.interp(time_grid, fn.x, fn.y) for fn in rsf_os.predict_survival_function(inp_os)],
        axis=0
    )

    # OS ≥ RFS を保証
    surv_os = np.maximum(surv_os, surv_rfs)

    rfs36 = float(surv_rfs[time_grid == 36]) * 100
    os36  = float(surv_os[time_grid == 36]) * 100

    st.success(f"Predicted 3-year RFS: **{rfs36:.1f}%**")
    st.success(f"Predicted 3-year OS:  **{os36:.1f}%**")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_grid, surv_rfs, lw=2, label="RFS")
    ax.plot(time_grid, surv_os,  lw=2, label="OS")
    ax.set_xlabel("Months after surgery")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 37, 6))
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

st.markdown(
    """
    <div style='margin-top:2rem; font-size:0.85em; color:gray;'>
    <b>Note:</b> This model was developed for patients with gastric or gastroesophageal junction 
    adenocarcinoma who underwent either robot-assisted or laparoscopic gastrectomy. 
    Cases with distant metastasis were excluded, but CY1 cases were included. 
    Patients with R2 resection were excluded.
    </div>
    """,
    unsafe_allow_html=True
)


