import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global SHAP Comparison", layout="wide")
st.title("🌍 Global SHAP – Amb i Sense Performance Acadèmica")

# ---------- HELPERS ----------
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
    return df  # fallback

def compute_shap(pipeline, X):
    imp = pipeline.named_steps["imputer"]
    scl = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    X_tr = scl.transform(imp.transform(X))
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X_tr)
    arr = np.array(vals)
    if arr.ndim == 3:
        return arr[:, :, 1], arr[:, :, 0]
    elif isinstance(vals, list):
        return vals[1], vals[0]
    else:
        return arr, -arr


# ---------- LOAD CSV ----------
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

st.write("Distribució de classes:")
st.write(y.value_counts())


# ---------- LOAD MODELS ----------
model_full = load_model("course_model.pkl")       # With performance
model_no_perf = load_model("nocourse_model.pkl")  # Without performance


# ---------- TABS ----------
tab1, tab2 = st.tabs(["📚 Amb Performance Acadèmica",
                      "🚫 Sense Performance Acadèmica"])


# 📚 TAB 1 — FULL MODEL
with tab1:
    st.header("📚 Global SHAP – Amb Performance Acadèmica")

    X_full = align_features(X, model_full)

    with st.spinner("Computant SHAP..."):
        shap_ab_full, _ = compute_shap(model_full, X_full)

    shap.initjs()
    st.subheader("Bar plot de importància")
    fig1 = plt.figure()
    shap.summary_plot(shap_ab_full, X_full, feature_names=X_full.columns,
                      plot_type="bar", show=False)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Beeswarm")
    fig2 = plt.figure()
    shap.summary_plot(shap_ab_full, X_full, feature_names=X_full.columns,
                      show=False)
    st.pyplot(fig2)
    plt.close(fig2)


# 🚫 TAB 2 — REDUCED MODEL
with tab2:
    st.header("🚫 Global SHAP – Sense Performance Acadèmica")

    X_red = align_features(X, model_no_perf)

    with st.spinner("Computant SHAP..."):
        shap_ab_red, _ = compute_shap(model_no_perf, X_red)

    shap.initjs()
    st.subheader("Bar plot de importància")
    fig3 = plt.figure()
    shap.summary_plot(shap_ab_red, X_red, feature_names=X_red.columns,
                      plot_type="bar", show=False)
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Beeswarm")
    fig4 = plt.figure()
    shap.summary_plot(shap_ab_red, X_red, feature_names=X_red.columns,
                      show=False)
    st.pyplot(fig4)
    plt.close(fig4)
