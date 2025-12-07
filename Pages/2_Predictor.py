import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# ======================== PAGE CONFIG ===========================
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# ======================== GLOBAL STYLE ===========================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Subheaders */
h2 {
    font-size: 1.6rem;
    font-weight: 650;
    color: #1E90FF;
}

/* Tabs */
.stTabs [role="tab"] {
    background: #e7f2ff;
    padding: 12px 25px;
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
    border-radius: 10px;
    padding: 12px 22px;
    font-size: 16px;
    font-weight: 650;
    border: none;
    transition: .2s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.07);
}

.result-box {
    padding: 18px;
    border-radius: 10px;
    margin: 15px 0;
    text-align: center;
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
top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    if st.button("⬅️ Home"):
        st.switch_page("app.py")

with top_col2:
    st.title("⚙️ Student Dropout Predictor")
    st.write("Select known student features and compute dropout risk below 👇")


tab_course, tab_nocourse = st.tabs(["📚 With Course Performance", "🚫 Without Course Performance"])

# =================================================================================
# TAB 1 — WITH COURSE PERFORMANCE
# =================================================================================
with tab_course:
    used_features = []; ignored_features = []
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("🧑 Personal & Academic Background")
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

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📉 Course Performance")
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

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏛 Economic Indicators")

    unemployment = show_optional("Unemployment Rate", "unemployment_c", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_c", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_c", st.number_input, used_features, ignored_features, value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict 📊", key="predict_course"):
        X_df_course = pd.DataFrame([locals()]).filter(items=course_model.feature_names_in_)
        dropout = course_model.predict_proba(X_df_course)[0][1]

        st.session_state.update({
            "last_model": "course",
            "last_prediction": time.time(),
            "X_df_course": X_df_course,
            "dropout_course": dropout
        })

        color = "#28a745" if dropout < 0.33 else "#ffc107" if dropout < 0.66 else "#dc3545"
        label = "✔️ Likely to Continue, Dropout Risk" if dropout < 0.5 else "⚠️ High Dropout Risk"

        st.markdown(
            f"<div class='result-box' style='background:{color}20;'>"
            f"<h3 style='color:{color};'>{label}: {dropout:.2f}</h3></div>",
            unsafe_allow_html=True,
        )

# =================================================================================
# TAB 2 — WITHOUT COURSE PERFORMANCE
# =================================================================================
with tab_nocourse:
    used_features = []; ignored_features = []

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧑 Personal & Academic Background")
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

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏛 Economic Indicators")

    unemployment = show_optional("Unemployment Rate", "unemployment_nc", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_nc", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_nc", st.number_input, used_features, ignored_features, value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict 📊", key="predict_nocourse"):
        X_df_nocourse = pd.DataFrame([locals()]).filter(items=nocourse_model.feature_names_in_)
        dropout_nc = nocourse_model.predict_proba(X_df_nocourse)[0][1]

        st.session_state.update({
            "last_model": "nocourse",
            "last_prediction": time.time(),
            "X_df_nocourse": X_df_nocourse,
            "dropout_nocourse": dropout_nc
        })

        color = "#28a745" if dropout_nc < 0.33 else "#ffc107" if dropout_nc < 0.66 else "#dc3545"
        label = "✔️ Likely to Continue, Dropout Risk" if dropout_nc < 0.5 else "⚠️ High Dropout Risk"

        st.markdown(
            f"<div class='result-box' style='background:{color}20;'>"
            f"<h3 style='color:{color};'>{label}: {dropout_nc:.2f}</h3></div>",
            unsafe_allow_html=True,
        )
