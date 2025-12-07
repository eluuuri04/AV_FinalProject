import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import time

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Explainability", page_icon="🧩", layout="wide")

# ================== GLOBAL STYLE ==================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Tabs styling */
.stTabs [role="tab"] {
    background: #e7f2ff;
    padding: 10px 22px;
    border-radius: 8px;
    font-weight: 600;
    margin-right: 10px;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    transition: .15s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
}

.main-header {
    font-size: 2.1rem;
    font-weight: 750;
}

</style>
""", unsafe_allow_html=True)

# ================== HEADER + NAV ==================
top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    if st.button("⬅️ Home"):
        st.switch_page("App.py")

with top_col2:
    st.markdown("<h1 class='main-header'>🔍 SHAP Explainability — Global & Local</h1>", unsafe_allow_html=True)
    st.write("Analyze the impact of each feature on the probability of churn, both at a global and local level.")

# ================== HELPERS ==================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_binary_dataset(df):
    df_bin = df.copy()
    df_bin["target_bin"] = df_bin["target"].replace({
        "Dropout": 0, "Graduate": 1, "Enrolled": 1
    })
    df_bin = df_bin.drop(columns=["target"])
    X = df_bin.drop(columns=["target_bin"])
    y = df_bin["target_bin"]
    return X, y

def align_features(df, pipeline):
    for step in ["imputer", "scaler", "model"]:
        if hasattr(pipeline.named_steps[step], "feature_names_in_"):
            trained_cols = list(pipeline.named_steps[step].feature_names_in_)
            return df[trained_cols]
    return df

def compute_shap_global(pipeline, X):
    imp = pipeline.named_steps["imputer"]
    scl = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    X_tr = scl.transform(imp.transform(X))
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X_tr)
    arr = np.array(vals)
    if arr.ndim == 3:
        shap_ab = arr[:, :, 1]
        shap_no = arr[:, :, 0]
    elif isinstance(vals, list):
        shap_no = vals[0]
        shap_ab = vals[1]
    else:
        shap_ab = arr
        shap_no = -arr
    return shap_ab, shap_no

# ================== LOAD & CACHE GLOBAL SHAP ==================
if "global_data_loaded" not in st.session_state:
    df = pd.read_csv("data.csv", sep=";")

    rename_dict = {
        "Marital status": "marital",
        "Application mode": "app_mode",
        "Application order": "app_order",
        "Course": "course",
        "Daytime/evening attendance\t": "attendance",
        "Previous qualification": "prev_qual",
        "Previous qualification (grade)": "prev_grade",
        "Nacionality": "nationality",
        "Mother's qualification": "mother_qual",
        "Father's qualification": "father_qual",
        "Mother's occupation": "mother_job",
        "Father's occupation": "father_job",
        "Admission grade": "admission_grade",
        "Displaced": "displaced",
        "Educational special needs": "special_needs",
        "Debtor": "debtor",
        "Tuition fees up to date": "fees",
        "Gender": "gender",
        "Scholarship holder": "scholarship",
        "Age at enrollment": "age",
        "International": "international",
        "Curricular units 1st sem (credited)": "cred_1",
        "Curricular units 1st sem (enrolled)": "enrolled_1",
        "Curricular units 1st sem (evaluations)": "evals_1",
        "Curricular units 1st sem (approved)": "approved_1",
        "Curricular units 1st sem (grade)": "grade_1",
        "Curricular units 1st sem (without evaluations)": "no_evals_1",
        "Curricular units 2nd sem (credited)": "cred_2",
        "Curricular units 2nd sem (enrolled)": "enrolled_2",
        "Curricular units 2nd sem (evaluations)": "evals_2",
        "Curricular units 2nd sem (approved)": "approved_2",
        "Curricular units 2nd sem (grade)": "grade_2",
        "Curricular units 2nd sem (without evaluations)": "no_evals_2",
        "Unemployment rate": "unemployment",
        "Inflation rate": "inflation",
        "GDP": "gdp",
        "Target": "target"
    }
    df = df.rename(columns=rename_dict)
    X, y = build_binary_dataset(df)


    model_full = load_model("course_model.pkl")
    
    model_no_perf = load_model("nocourse_model.pkl")

    X_full = align_features(X, model_full)
    X_red = align_features(X, model_no_perf)

    with st.spinner("🧠 Calculanting SHAP global fot the two models..."):
        shap_full_ab, _ = compute_shap_global(model_full, X_full)
        shap_red_ab, _ = compute_shap_global(model_no_perf, X_red)

    st.session_state["X_full"] = X_full
    st.session_state["shap_full_ab"] = shap_full_ab

    st.session_state["X_red"] = X_red
    st.session_state["shap_red_ab"] = shap_red_ab

    st.session_state["global_data_loaded"] = True
    st.success("✅ Global SHAP values cached!")

# ================== TABS ==================
tab_global, tab_local = st.tabs(["🌍 Global Explainability", "🎯 Local Explainability"])

# ---------------- GLOBAL TAB ---------------- #
with tab_global:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🌍 Importància global de les característiques")
    
    shap.initjs()

    sub1, sub2 = st.tabs(["📚 With Course Performance", "🚫 Without Course Performance"])

    with sub1:
        st.markdown("#### 📚 Model with data about course performance")
        fig = plt.figure()
        shap.summary_plot(
            st.session_state["shap_full_ab"],
            st.session_state["X_full"],
            feature_names=st.session_state["X_full"].columns,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig); plt.close(fig)

        fig = plt.figure()
        shap.summary_plot(
            st.session_state["shap_full_ab"],
            st.session_state["X_full"],
            feature_names=st.session_state["X_full"].columns,
            show=False
        )
        st.pyplot(fig); plt.close(fig)

    with sub2:
        st.markdown("#### 🚫Model without data about course performance")
        fig = plt.figure()
        shap.summary_plot(
            st.session_state["shap_red_ab"],
            st.session_state["X_red"],
            feature_names=st.session_state["X_red"].columns,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig); plt.close(fig)

        fig = plt.figure()
        shap.summary_plot(
            st.session_state["shap_red_ab"],
            st.session_state["X_red"],
            feature_names=st.session_state["X_red"].columns,
            show=False
        )
        st.pyplot(fig); plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LOCAL TAB ---------------- #
with tab_local:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    if "student_name" in st.session_state and st.session_state.student_name.strip() != "":
        st.subheader(f"🎯 Local Explanation about the last prediction for {st.session_state.student_name}")
    else:
        st.subheader("🎯 Local Explanation about the last prediction")
    if "last_model" not in st.session_state:
        st.warning("⚠️ First has to be done a prediction. Go to page *Predictor*.")
        st.stop()

    last_used = st.session_state["last_model"]
    st.info(f"ℹ️ Using the last prediction of the model: **{last_used}**")

    if last_used == "course":
        X_df = st.session_state["X_df_course"]
        dropout = st.session_state["dropout_course"]
        pipeline = load_model("course_model.pkl")
    else:
        X_df = st.session_state["X_df_nocourse"]
        dropout = st.session_state["dropout_nocourse"]
        pipeline = load_model("nocourse_model.pkl")

    imp = pipeline.named_steps["imputer"]
    scl = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_trans = scl.transform(imp.transform(X_df))

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_trans)
    shap_arr = np.array(shap_raw)
    shap_instance = shap_arr[0, :, 1] if shap_arr.ndim == 3 else shap_arr[0]
    base = explainer.expected_value[1] if isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value

    exp_local = shap.Explanation(
        values=shap_instance,
        base_values=base,
        data=X_df.iloc[0, :].values,
        feature_names=X_df.columns
    )

    if dropout >= 0.5:
        st.error(f"⚠️ Dropout Risk: {dropout:.2f}")
    else:
        st.success(f"🎉 Probability to Continue: {1 - dropout:.2f}")

    shap.initjs()
    fig = plt.figure()
    shap.plots.waterfall(exp_local, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📄 Inputs used")
    st.dataframe(X_df)

    st.markdown("</div>", unsafe_allow_html=True)
