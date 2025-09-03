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
# 3. Title
# ────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:1rem;'>3-year RFS & OS Predictor</h3>",
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────
# 4. Input widgets（Height → Weight → BMI の順）
#    - (at surgery) の注釈をラベルに付与
#    - help で補足（Height/Weight は BMI 計算用、BMI は直接入力も可）
#    - BMI は on_change コールバックで自動反映（直接入力も可）
# ────────────────────────────────────────────────────────────

# session_state 初期化
for key, default in [("height_str", ""), ("weight_str", ""), ("bmi", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# Height/Weight 変更時に BMI を再計算して session_state["bmi"] を更新
def recompute_bmi():
    hs = st.session_state.get("height_str", "")
    ws = st.session_state.get("weight_str", "")
    try:
        h = float(hs)
        w = float(ws)
        if h > 0:
            bmi_val = w / (h / 100) ** 2
            st.session_state["bmi"] = f"{bmi_val:.1f}"
    except Exception:
        # 数値でなければ何もしない（既存BMIは保持）
        pass

age_str = st.text_input("Age (years)", placeholder="ex) 65")

height_str = st.text_input(
    "Height (cm) (at surgery)",
    key="height_str",
    placeholder="ex) 160.0",
    help="Height at surgery. Used for BMI calculation.",
    on_change=recompute_bmi,
)

weight_str = st.text_input(
    "Weight (kg) (at surgery)",
    key="weight_str",
    placeholder="ex) 50.0",
    help="Weight at surgery. Used for BMI calculation.",
    on_change=recompute_bmi,
)

# BMI は常に表示。Height/Weight 変更時は自動上書き。直接入力も可。
bmi_str = st.text_input(
    "BMI (kg/m²) (at surgery)",
    key="bmi",
    help="BMI at surgery. Automatically calculated from Height and Weight, but can also be entered directly.",
)

cea_str   = st.text_input("CEA (ng/mL)", placeholder="ex) 4.5")
ca199_str = st.text_input("CA19-9 (U/mL)", placeholder="ex) 25")

asa  = st.selectbox("ASA-PS", asa_map.keys(), index=None, placeholder="Select ASA-PS")
surg = st.selectbox("Surgical method", surg_map.keys(), index=None, placeholder="Select surgical method")

# 術式に応じて再建方法の選択肢を制御
if surg == "PG":
    recon_options = ["R-Y", "Others"]
elif surg == "TG":
    recon_options = ["R-Y", "Others"]
else:  # DG
    recon_options = list(recons_map.keys())

recon = st.selectbox("Reconstruction", recon_options, index=None, placeholder="Select reconstruction")

macro = st.selectbox("Macroscopic type", macro_map.keys(), index=None, placeholder="Select macroscopic type")
diam  = st.text_input("Tumor diameter (mm)", placeholder="ex) 45")
histo = st.selectbox("Histology", histo_map.keys(), index=None, placeholder="Select histology")
vcat  = st.selectbox("Vascular invasion (v)", v_map.keys(), index=None, placeholder="Select vascular invasion")
pt    = st.selectbox("Pathological T", pt_map.keys(), index=None, placeholder="Select pT")
pn    = st.selectbox("Pathological N", pn_map.keys(), index=None, placeholder="Select pN")

# pStage は常に表示。pT/pN が入れば候補を絞る（自動候補 + "IV (CY1)"）
all_stage_options = list(stage_map.keys())
stage_options = all_stage_options
auto_stage = None
if pt and pn:
    auto_stage = determine_stage(pt, pn)
    if auto_stage:
        stage_options = [auto_stage, "IV (CY1)"]

stage = st.selectbox(
    "Pathological stage (auto-calculated from pT & pN, or IV (CY1))",
    stage_options,
    index=None,
    placeholder="Select stage"
)

# ────────────────────────────────────────────────────────────
# 5. Prediction
# ────────────────────────────────────────────────────────────
if st.button("Predict"):
    try:
        age = int(age_str)
        h = float(st.session_state["height_str"])
        w = float(st.session_state["weight_str"])
        bmi = float(st.session_state["bmi"])  # 手入力 or 自動計算が反映済み
        cea = float(cea_str)
        ca199 = float(ca199_str)
        diam_val = float(diam)
    except ValueError:
        st.error("Age, Height, Weight, BMI, CEA, CA19-9, Diameter must be numeric.")
        st.stop()

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
            "diameter2": diam_val,
            "histology2": histo_map[histo],
            "v2": v_map[vcat],
            "p_t_3": pt_map[pt],
            "p_n_3": pn_map[pn],
            "p_stage3": stage_map[stage] if stage else np.nan,
        }])
    except KeyError:
        st.error("Please complete ASA-PS, Surgical method, Reconstruction, Macroscopic type, Histology, V, pT, pN, and Stage.")
        st.stop()

    # 学習時の特徴量順に合わせる
    try:
        inp_rfs = inp[rsf_rfs.feature_names_in_]
        inp_os  = inp[rsf_os.feature_names_in_]
    except Exception as e:
        st.error(f"Feature alignment error: {e}")
        st.stop()

    # 予測
    time_grid = np.arange(0, 37)
    surv_rfs = np.mean(
        [np.interp(time_grid, fn.x, fn.y) for fn in rsf_rfs.predict_survival_function(inp_rfs)],
        axis=0
    )
    surv_os = np.mean(
        [np.interp(time_grid, fn.x, fn.y) for fn in rsf_os.predict_survival_function(inp_os)],
        axis=0
    )

    rfs36 = float(surv_rfs[time_grid == 36]) * 100
    os36  = float(surv_os[time_grid == 36]) * 100

    st.success(f"Predicted 3-year RFS: **{rfs36:.1f}%**")
    st.success(f"Predicted 3-year OS:  **{os36:.1f}%**")

    # 図示
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


