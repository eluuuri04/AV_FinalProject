import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# ======================== CUSTOM STYLE ===========================
st.markdown("""
<style>
/* General */
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f7f9fc;
    color: #333333;
}

/* Tabs */
.stTabs [role="tablist"] {
    justify-content: center;
}
.stTabs [role="tab"] {
    background-color: #f0f8ff;
    border-radius: 8px;
    padding: 10px 20px;
    margin: 0 5px;
    font-weight: 600;
    color: #1E90FF;
    transition: all 0.25s ease;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
}

/* Headers */
h1 {
    font-size: 2.2rem;
    font-weight: 700;
}
h2 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1E90FF;
}
h3 {
    font-size: 1.3rem;
    font-weight: 500;
}

/* Cards */
.section-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
.section-card:hover {
    transform: translateY(-3px);
    transition: all 0.25s ease;
    box-shadow: 0 6px 12px rgba(0,0,0,0.12);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    transition: 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Feature badges */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    margin-right: 8px;
}
.badge-used {
    background: #1E90FF;
    color: white;
}
.badge-ignored {
    background: #FFA500;
    color: white;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ========================= LOAD MODELS =========================
with open("course_model.pkl", "rb") as f:
    course_model = pickle.load(f)
with open("nocourse_model.pkl", "rb") as f:
    nocourse_model = pickle.load(f)


# ========================= HELPERS ==========================
def show_optional(label, key, widget_fn, used_list, ignored_list, *args, **kwargs):
    ignore = st.checkbox(f"Ignore {label}", key=f"ignore_{key}")
    if ignore:
        ignored_list.append(label)
        return np.nan
    used_list.append(label)
    return widget_fn(label, key=key, *args, **kwargs)


# ========================= MAIN UI ==========================
st.title("📘 Student Dropout Predictor")
st.write("You may ignore ANY feature — the model will infer missing info.")

tab_course, tab_nocourse = st.tabs(["📚 With Performance", "🧩 Without Performance"])


# =========================================================
# TAB 1 — WITH COURSE PERFORMANCE
# =========================================================
with tab_course:
    used_features = []; ignored_features = []

    st.subheader("🧑 Personal & Academic")
    col1, col2 = st.columns(2)

    with col1:
        marital = show_optional("Marital Status", "marital", st.selectbox, used_features, ignored_features, [0,1,2])
        app_mode = show_optional("Application Mode", "app_mode", st.selectbox, used_features, ignored_features, [0,1,2,3])
        app_order = show_optional("Application Order", "app_order", st.number_input, used_features, ignored_features, value=1)
        course = show_optional("Course Code", "course", st.number_input, used_features, ignored_features, value=9003)
        admission_grade = show_optional("Admission Grade", "admission_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        attendance = show_optional("Attendance", "attendance", st.selectbox, used_features, ignored_features, [0,1])
        prev_qual = show_optional("Previous Qualification", "prev_qual", st.number_input, used_features, ignored_features, value=1)
        gender = show_optional("Gender", "gender", st.selectbox, used_features, ignored_features, [0,1])
        father_job = show_optional("Father Job", "father_job", st.number_input, used_features, ignored_features, value=3)
        displaced = show_optional("Displaced", "displaced", st.selectbox, used_features, ignored_features, [0,1])
        special_needs = show_optional("Special Needs", "special_needs", st.selectbox, used_features, ignored_features, [0,1])

    with col2:
        scholarship = show_optional("Scholarship", "scholarship", st.selectbox, used_features, ignored_features, [0,1])
        age = show_optional("Age", "age", st.number_input, used_features, ignored_features, value=21)
        international = show_optional("International", "international", st.selectbox, used_features, ignored_features, [0,1])
        prev_grade = show_optional("Previous Grade", "prev_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        nationality = show_optional("Nationality", "nationality", st.number_input, used_features, ignored_features, value=1)
        mother_qual = show_optional("Mother Qualification", "mother_qual", st.number_input, used_features, ignored_features, value=1)
        father_qual = show_optional("Father Qualification", "father_qual", st.number_input, used_features, ignored_features, value=1)
        mother_job = show_optional("Mother Job", "mother_job", st.number_input, used_features, ignored_features, value=3)
        debtor = show_optional("Debtor", "debtor", st.selectbox, used_features, ignored_features, [0,1])
        fees = show_optional("Paid Fees", "fees", st.selectbox, used_features, ignored_features, [0,1])

    st.subheader("🎓 Course Performance")
    col1, col2 = st.columns(2)
    with col1:
        cred_1 = show_optional("Credits 1", "cred_1_c", st.number_input, used_features, ignored_features, value=0)
        enrolled_1 = show_optional("Enrolled 1", "enrolled_1_c", st.number_input, used_features, ignored_features, value=6)
        evals_1 = show_optional("Evaluations 1", "evals_1_c", st.number_input, used_features, ignored_features, value=6)
        approved_1 = show_optional("Approved 1", "approved_1_c", st.number_input, used_features, ignored_features, value=3)
        grade_1 = show_optional("Grade 1", "grade_1_c", st.number_input, used_features, ignored_features, value=9.5)
        no_evals_1 = show_optional("No Exams 1", "no_evals_1_c", st.number_input, used_features, ignored_features, value=0)
    with col2:
        cred_2 = show_optional("Credits 2", "cred_2_c", st.number_input, used_features, ignored_features, value=0)
        enrolled_2 = show_optional("Enrolled 2", "enrolled_2_c", st.number_input, used_features, ignored_features, value=6)
        evals_2 = show_optional("Evaluations 2", "evals_2_c", st.number_input, used_features, ignored_features, value=6)
        approved_2 = show_optional("Approved 2", "approved_2_c", st.number_input, used_features, ignored_features, value=3)
        grade_2 = show_optional("Grade 2", "grade_2_c", st.number_input, used_features, ignored_features, value=9.0)
        no_evals_2 = show_optional("No Exams 2", "no_evals_2_c", st.number_input, used_features, ignored_features, value=0)

    st.subheader("📉 Economic Factors")
    unemployment = show_optional("Unemployment Rate", "unemployment_c", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_c", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_c", st.number_input, used_features, ignored_features, value=1.0)

    if st.button("Predict 📊", key="predict_course"):
        X_df_course = pd.DataFrame([locals()]).filter(items=course_model.feature_names_in_)
        dropout = course_model.predict_proba(X_df_course)[0][1]

        st.session_state.update({
            "last_model":"course",
            "last_prediction":time.time(),
            "X_df_course":X_df_course,
            "dropout_course":dropout
        })

        st.success(f"Dropout Risk: {dropout:.2f}")
        st.progress(dropout)


# =========================================================
# TAB 2 — WITHOUT COURSE PERFORMANCE
# =========================================================
with tab_nocourse:
    used_features = []; ignored_features = []

    st.subheader("🧑 Personal & Academic")

    col1, col2 = st.columns(2)
    with col1:
        marital = show_optional("Marital Status", "marital_nc", st.selectbox, used_features, ignored_features, [0,1,2])
        app_mode = show_optional("Application Mode", "app_mode_nc", st.selectbox, used_features, ignored_features, [0,1,2,3])
        app_order = show_optional("Application Order", "app_order_nc", st.number_input, used_features, ignored_features, value=1)
        course = show_optional("Course Code", "course_nc", st.number_input, used_features, ignored_features, value=9003)
        admission_grade = show_optional("Admission Grade", "admission_grade_nc", st.number_input, used_features, ignored_features, value=95.0)
        attendance = show_optional("Attendance", "attendance_nc", st.selectbox, used_features, ignored_features, [0,1])
        prev_qual = show_optional("Previous Qualification", "prev_qual_nc", st.number_input, used_features, ignored_features, value=1)
        gender = show_optional("Gender", "gender_nc", st.selectbox, used_features, ignored_features, [0,1])
        father_job = show_optional("Father Job", "father_job_nc", st.number_input, used_features, ignored_features, value=3)
        displaced = show_optional("Displaced", "displaced_nc", st.selectbox, used_features, ignored_features, [0,1])
        special_needs = show_optional("Special Needs", "special_nc", st.selectbox, used_features, ignored_features, [0,1])
    with col2:
        scholarship = show_optional("Scholarship", "scholarship_nc", st.selectbox, used_features, ignored_features, [0,1])
        age = show_optional("Age", "age_nc", st.number_input, used_features, ignored_features, value=21)
        international = show_optional("International", "international_nc", st.selectbox, used_features, ignored_features, [0,1])
        prev_grade = show_optional("Previous Grade", "prev_grade_nc", st.number_input, used_features, ignored_features, value=95.0)
        nationality = show_optional("Nationality", "nationality_nc", st.number_input, used_features, ignored_features, value=1)
        mother_qual = show_optional("Mother Qualification", "mother_qual_nc", st.number_input, used_features, ignored_features, value=1)
        father_qual = show_optional("Father Qualification", "father_qual_nc", st.number_input, used_features, ignored_features, value=1)
        mother_job = show_optional("Mother Job", "mother_job_nc", st.number_input, used_features, ignored_features, value=3)
        debtor = show_optional("Debtor", "debtor_nc", st.selectbox, used_features, ignored_features, [0,1])
        fees = show_optional("Paid Fees", "fees_nc", st.selectbox, used_features, ignored_features, [0,1])
   
    
    st.subheader("📉 Economic Factors")
    unemployment = show_optional("Unemployment Rate", "unemployment_nc", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_nc", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_nc", st.number_input, used_features, ignored_features, value=1.0)

    if st.button("Predict 📊", key="predict_nocourse"):
        X_df_nocourse = pd.DataFrame([locals()]).filter(items=nocourse_model.feature_names_in_)
        dropout_nc = nocourse_model.predict_proba(X_df_nocourse)[0][1]

        st.session_state.update({
            "last_model":"nocourse",
            "last_prediction":time.time(),
            "X_df_nocourse":X_df_nocourse,
            "dropout_nocourse":dropout_nc
        })

        st.success(f"Dropout Risk: {dropout_nc:.2f}")
        st.progress(dropout_nc)
