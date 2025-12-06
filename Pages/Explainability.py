# file: pages/02_global_shap_all_data.py  (or any name you want)
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global SHAP (all data)", layout="wide")
st.title("🌍 Global SHAP (totes les dades)")

# ---------- HELPERS -----------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def build_binary_dataset(df: pd.DataFrame):
    """
    Usa exactament el teu codi:

    df_bin["target_bin"] = df_bin["target"].replace({
        "Dropout": 0,
        "Graduate": 1,
        "Enrolled": 1
    })
    df_bin = df_bin.drop(["target"], axis = 1)

    I després separem X i y.
    """
    df_bin = df.copy()

    df_bin["target_bin"] = df_bin["target"].replace({
        "Dropout": 0,
        "Graduate": 1,
        "Enrolled": 1
    })

    df_bin = df_bin.drop(["target"], axis=1)

    # X = totes les columnes menys la diana binària
    X_bin = df_bin.drop(["target_bin"], axis=1)
    y_bin = df_bin["target_bin"]

    return X_bin, y_bin, df_bin


def compute_shap_for_pipeline(pipeline, X_bin):
    """
    Mateixa lògica que tenies al notebook:
    imputer + scaler + model d'arbres + TreeExplainer.
    """
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    tree_model = pipeline.named_steps["model"]

    # Transformem com ho veu el model
    X_transformed = scaler.transform(imputer.transform(X_bin))

    explainer = shap.TreeExplainer(tree_model)
    shap_vals = explainer.shap_values(X_transformed)

    shap_vals_arr = np.array(shap_vals)

    if shap_vals_arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_values_ab = shap_vals_arr[:, :, 1]  # classe 1
        shap_values_no = shap_vals_arr[:, :, 0]  # classe 0
    elif isinstance(shap_vals, list):
        shap_values_no = shap_vals[0]
        shap_values_ab = shap_vals[1]
    else:
        # cas binari clàssic: una sola matriu
        shap_values_ab = shap_vals_arr
        shap_values_no = -shap_vals_arr

    return shap_values_ab, shap_values_no


# ---------- MAIN --------------------------------------------------------------
# 1) Recuperar totes les dades de session_state
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


st.subheader("Dades originals (totes les files)")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# 2) Construir dataset binari segons el teu codi
st.markdown("### Transformació a target binària (`target_bin`)")
X_bin, y_bin, df_bin = build_binary_dataset(df)

st.write("Shape de `df_bin`:", df_bin.shape)
st.write("Shape de `X_bin` (features):", X_bin.shape)
st.write("Distribució de `target_bin`:")
st.write(y_bin.value_counts())

# 3) Carregar el model (pipeline amb imputer + scaler + model)
model_path = "course_model.pkl"
pipeline = load_model(model_path)

# 4) Calcular SHAP sobre **totes** les dades
with st.spinner("Calculant valors SHAP sobre totes les files..."):
    shap_values_ab, shap_values_no = compute_shap_for_pipeline(pipeline, X_bin)

# 5) Gràfic global SHAP (bar) per a la classe 1 (Abandona / o la que correspongui)
st.subheader("🔎 Importància global de les variables (classe: 1) – Bar plot")

shap.initjs()

fig_bar = plt.figure()
shap.summary_plot(
    shap_values_ab,
    X_bin,
    feature_names=X_bin.columns,
    plot_type="bar",
    show=False,
)
st.pyplot(fig_bar)
plt.close(fig_bar)

# 6) Beeswarm global per la mateixa classe
st.subheader("🐝 Distribució global SHAP (classe: 1) – Beeswarm")

fig_bee = plt.figure()
shap.summary_plot(
    shap_values_ab,
    X_bin,
    feature_names=X_bin.columns,
    show=False,
)
st.pyplot(fig_bee)
plt.close(fig_bee)

# 7) Opcional: veure també la classe 0
with st.expander("Mostra SHAP per a la classe 0"):
    fig_bar_no = plt.figure()
    shap.summary_plot(
        shap_values_no,
        X_bin,
        feature_names=X_bin.columns,
        plot_type="bar",
        show=False,
    )
    st.pyplot(fig_bar_no)
    plt.close(fig_bar_no)

    fig_bee_no = plt.figure()
    shap.summary_plot(
        shap_values_no,
        X_bin,
        feature_names=X_bin.columns,
        show=False,
    )
    st.pyplot(fig_bee_no)
    plt.close(fig_bee_no)
