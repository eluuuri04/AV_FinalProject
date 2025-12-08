import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import variables as vr  # <--- IMPORTED YOUR VARIABLES

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Explainability", page_icon="🧩", layout="wide")

# ================== GLOBAL STYLE ==================
st.markdown("""
<style>
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
.stTabs [role="tab"] { background: #e7f2ff; padding: 10px 22px; border-radius: 8px; font-weight: 600; margin-right: 10px; }
.stTabs [role="tab"][aria-selected="true"] { background: linear-gradient(135deg, #1E90FF, #00CED1); color: white; }
.stButton>button { background: linear-gradient(135deg, #1E90FF, #00CED1); color: white; border-radius: 8px; border: none; transition: .15s ease; }
.stButton>button:hover { transform: translateY(-2px); }
.main-header { font-size: 2.1rem; font-weight: 750; }
.section-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ================== HEADER + NAV ==================
top_col1, top_col2 = st.columns([1, 4])
with top_col1:
    if st.button("⬅️ Home"):
        st.switch_page("App.py")
with top_col2:
    st.markdown("<h1 class='main-header'>🔍 SHAP Explainability — Global & Local</h1>", unsafe_allow_html=True)
    st.write("Analyze feature impact on dropout probability (Global) and explain specific predictions (Local).")

# ================== CACHED FUNCTIONS ==================

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_static_data():
    if "data" in st.session_state and isinstance(st.session_state["data"], pd.DataFrame):
        return st.session_state["data"]
    return pd.read_csv("data.csv", sep=";")

def preprocess_data(df):
    rename_dict = {
        "Marital status": "marital", "Application mode": "app_mode", "Application order": "app_order",
        "Course": "course", "Daytime/evening attendance\t": "attendance", "Previous qualification": "prev_qual",
        "Previous qualification (grade)": "prev_grade", "Nacionality": "nationality", 
        "Mother's qualification": "mother_qual", "Father's qualification": "father_qual",
        "Mother's occupation": "mother_job", "Father's occupation": "father_job",
        "Admission grade": "admission_grade", "Displaced": "displaced",
        "Educational special needs": "special_needs", "Debtor": "debtor", "Tuition fees up to date": "fees",
        "Gender": "gender", "Scholarship holder": "scholarship", "Age at enrollment": "age",
        "International": "international", "Curricular units 1st sem (credited)": "cred_1",
        "Curricular units 1st sem (enrolled)": "enrolled_1", "Curricular units 1st sem (evaluations)": "evals_1",
        "Curricular units 1st sem (approved)": "approved_1", "Curricular units 1st sem (grade)": "grade_1",
        "Curricular units 1st sem (without evaluations)": "no_evals_1", "Curricular units 2nd sem (credited)": "cred_2",
        "Curricular units 2nd sem (enrolled)": "enrolled_2", "Curricular units 2nd sem (evaluations)": "evals_2",
        "Curricular units 2nd sem (approved)": "approved_2", "Curricular units 2nd sem (grade)": "grade_2",
        "Curricular units 2nd sem (without evaluations)": "no_evals_2", "Unemployment rate": "unemployment",
        "Inflation rate": "inflation", "GDP": "gdp", "Target": "target"
    }
    df = df.rename(columns=rename_dict)
    
    df_bin = df.copy()
    if "target" in df_bin.columns:
        df_bin["target_bin"] = df_bin["target"].replace({"Dropout": 0, "Graduate": 1, "Enrolled": 1})
        df_bin = df_bin.drop(columns=["target"])
        X = df_bin.drop(columns=["target_bin"])
        y = df_bin["target_bin"]
    else:
        X = df_bin
        y = None
    return X, y

def align_features(df, pipeline):
    for step in ["imputer", "scaler", "model"]:
        if hasattr(pipeline.named_steps[step], "feature_names_in_"):
            return df[list(pipeline.named_steps[step].feature_names_in_)]
    return df

@st.cache_data(show_spinner=False)
def compute_global_shap_sampled(_pipeline, X, sample_size=300):
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    imp = _pipeline.named_steps["imputer"]
    scl = _pipeline.named_steps["scaler"]
    model = _pipeline.named_steps["model"]
    
    X_tr = scl.transform(imp.transform(X_sample))
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_tr, check_additivity=False)
    
    arr = np.array(shap_values)
    if isinstance(shap_values, list):
        shap_out = shap_values[1]
    elif arr.ndim == 3:
        shap_out = arr[:, :, 1]
    else:
        shap_out = arr

    return shap_out, X_sample

# NEW: Helper function to map numbers to text for display
def get_readable_df(df_input):
    df_nice = df_input.copy()
    
    # Map for columns present in your models (handling both standard and _nc suffixes)
    mappings = {
        # Categories
        "marital": vr.marital_status, "marital_nc": vr.marital_status,
        "app_mode": vr.application_mode, "app_mode_nc": vr.application_mode,
        "course": vr.courses, "course_nc": vr.courses,
        "attendance": vr.attendance, "attendance_nc": vr.attendance,
        "prev_qual": vr.previous_qualification, "prev_qual_nc": vr.previous_qualification,
        "nationality": vr.nationalities, "nationality_nc": vr.nationalities,
        "gender": vr.gender, "gender_nc": vr.gender,
        
        # Qualifications
        "mother_qual": vr.mother_qual, "mother_qual_nc": vr.mother_qual,
        "father_qual": vr.fathers_qualification, "father_qual_nc": vr.fathers_qualification,
        
        # Jobs
        "mother_job": vr.mothers_occupation, "mother_job_nc": vr.mothers_occupation,
        "father_job": vr.fathers_occupation, "father_job_nc": vr.fathers_occupation,
        
        # Yes/No Fields
        "displaced": vr.yes_no, "displaced_nc": vr.yes_no,
        "special_needs": vr.yes_no, "special_nc": vr.yes_no,
        "scholarship": vr.yes_no, "scholarship_nc": vr.yes_no,
        "international": vr.yes_no, "international_nc": vr.yes_no,
        "debtor": vr.yes_no, "debtor_nc": vr.yes_no,
        "fees": vr.yes_no, "fees_nc": vr.yes_no,
    }

    for col, map_dict in mappings.items():
        if col in df_nice.columns:
            # Map values; if value not in dict, keep original number
            df_nice[col] = df_nice[col].map(map_dict).fillna(df_nice[col])
            
    return df_nice

# ================== INITIALIZATION ==================

raw_df = load_static_data()
X_raw, _ = preprocess_data(raw_df)

model_full = load_model("course_model.pkl")
model_nocourse = load_model("nocourse_model.pkl")

X_full = align_features(X_raw, model_full)
X_red = align_features(X_raw, model_nocourse)

if "shap_global_calculated" not in st.session_state:
    with st.spinner("🧠 Calculating Global Explainability..."):
        shap_full, X_full_sample = compute_global_shap_sampled(model_full, X_full, sample_size=300)
        shap_red, X_red_sample = compute_global_shap_sampled(model_nocourse, X_red, sample_size=300)
        
        st.session_state.update({
            "shap_full": shap_full, "X_full_sample": X_full_sample,
            "shap_red": shap_red, "X_red_sample": X_red_sample,
            "shap_global_calculated": True
        })

# ================== UI TABS ==================
tab_global, tab_local = st.tabs(["🌍 Global Explainability", "🎯 Local Explainability"])

# ----------------- GLOBAL -----------------
with tab_global:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🌍 Feature Importance (Global)")
    st.write("These plots show which features drive the model's decisions on average.")

    sub_t1, sub_t2 = st.tabs(["📚 With Course Performance", "🚫 Without Course Performance"])

    def plot_shap(shap_vals, X_dat):
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X_dat, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_vals, X_dat, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close(fig2)

    with sub_t1:
        plot_shap(st.session_state["shap_full"], st.session_state["X_full_sample"])
    with sub_t2:
        plot_shap(st.session_state["shap_red"], st.session_state["X_red_sample"])

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- LOCAL -----------------
with tab_local:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    if "last_model" not in st.session_state:
        st.warning("⚠️ No prediction found. Please go to the **Predictor** page first.")
        st.stop()
    
    last_model_name = st.session_state["last_model"]
    student_name = st.session_state.get("student_name", "Unknown Student")

    st.subheader(f"🎯 Local Explanation for: {student_name}")
    st.info(f"ℹ️ Explaining last prediction using model: **{last_model_name}**")

    if last_model_name == "course":
        pipeline = model_full
        X_input = st.session_state.get("X_df_course")
        prob = st.session_state.get("dropout_course")
    else:
        pipeline = model_nocourse
        X_input = st.session_state.get("X_df_nocourse")
        prob = st.session_state.get("dropout_nocourse")

    if X_input is None:
        st.error("Error retrieving prediction data.")
        st.stop()

    # Calculate SHAP
    imp = pipeline.named_steps["imputer"]
    scl = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    X_trans = scl.transform(imp.transform(X_input))
    explainer = shap.TreeExplainer(model)
    shap_values_single = explainer.shap_values(X_trans)
    
    if isinstance(shap_values_single, list):
        sv = shap_values_single[1][0]
        base_val = explainer.expected_value[1]
    elif np.array(shap_values_single).ndim == 3:
        sv = shap_values_single[0, :, 1]
        base_val = explainer.expected_value[1]
    else:
        sv = shap_values_single[0]
        base_val = explainer.expected_value

    exp = shap.Explanation(
        values=sv, base_values=base_val,
        data=X_input.iloc[0].values, feature_names=X_input.columns
    )

    col_viz1, col_viz2 = st.columns([2, 1])

    with col_viz1:
        st.markdown("#### Waterfall Plot")
        fig_water = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig_water)
        plt.close(fig_water)

    with col_viz2:
        st.markdown("#### Prediction Result")
        if prob is not None:
            color = "#28a745" if prob < 0.33 else "#ffc107" if prob < 0.66 else "#dc3545"
            st.markdown(
                f"<div style='background-color:{color}20; padding:15px; border-radius:10px; text-align:center;'>"
                f"<h2 style='color:{color}; margin:0;'>{prob:.2%}</h2>"
                f"<p style='margin:0;'><b>Dropout Probability</b></p></div>", 
                unsafe_allow_html=True
            )
        
        st.markdown("#### Input Data")
        # HERE IS THE CHANGE: We convert the data to readable text before showing it
        readable_df = get_readable_df(X_input)
        st.dataframe(readable_df.T, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)