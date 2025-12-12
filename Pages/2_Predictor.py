import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import variables as vr

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

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
    color: black
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


with open("course_model.pkl", "rb") as f:
    course_model = pickle.load(f)
with open("nocourse_model.pkl", "rb") as f:
    nocourse_model = pickle.load(f)


def show_optional(label, key, widget_fn, used_list, ignored_list, *args, **kwargs):
    ignore = st.checkbox(f"Ignore {label}", key=f"ignore_{key}")
    if ignore:
        ignored_list.append(label)
        return np.nan
    used_list.append(label)
    return widget_fn(label, key=key, *args, **kwargs)

top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    if st.button("⬅️ Home"):
        st.switch_page("App.py")

cols1, cols2 = st.columns([1,6])

with cols1:
    st.subheader("Student's Name:")

with cols2:
    st.text_input("",  key="name")

with top_col2:
    st.title("⚙️ Student Dropout Predictor")
    st.write("Select known student features and compute dropout risk below 👇")


tab_course, tab_nocourse = st.tabs(["📚 With Course Performance", "🚫 Without Course Performance"])

def show_optional_dict(label, key, used_list, ignored_list, options_dict):
    ignore = st.checkbox(f"Ignore {label}", key=f"ignore_{key}")
    if ignore:
        ignored_list.append(label)
        return np.nan
    used_list.append(label)
    # Mostrem les descripcions però retornem el codi
    choice = st.selectbox(label, list(options_dict.values()), key=key)
    # Buscar la clau corresponent
    code = [k for k, v in options_dict.items() if v == choice][0]
    return code


# With Course performance
with tab_course:
    used_features = []; ignored_features = []
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("🧑 Personal & Academic Background")
    col1, col2 = st.columns(2)

    with col1:
        marital = show_optional_dict("Marital Status", "marital", used_features, ignored_features, vr.marital_status)
        app_mode = show_optional_dict("Application Mode", "app_mode", used_features, ignored_features, vr.application_mode)
        app_order = show_optional("Application Order", "app_order", st.number_input, used_features, ignored_features, min_value=0, max_value=9, value=0)
        course = show_optional_dict("Course Code", "course", used_features, ignored_features, vr.courses)
        admission_grade = show_optional("Admission Grade", "admission_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        attendance = show_optional_dict("Attendance", "attendance", used_features, ignored_features, vr.attendance)
        prev_qual = show_optional_dict("Previous Qualification", "prev_qual", used_features, ignored_features, vr.previous_qualification)
        gender = show_optional_dict("Gender", "gender", used_features, ignored_features, vr.gender)
        father_job = show_optional_dict("Father Job", "father_job", used_features, ignored_features, vr.fathers_occupation)
        displaced = show_optional_dict("Displaced", "displaced", used_features, ignored_features, vr.displaced_map)
        special_needs = show_optional_dict("Special Needs", "special_needs", used_features, ignored_features, vr.special_needs_map)

    with col2:
        scholarship = show_optional_dict("Scholarship", "scholarship", used_features, ignored_features, vr.scholarship_map)
        age = show_optional("Age", "age", st.number_input, used_features, ignored_features, min_value = 0 , value=21)
        international = show_optional_dict("International", "international", used_features, ignored_features, vr.international_map)
        prev_grade = show_optional("Previous Grade", "prev_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        nationality = show_optional_dict("Nationality", "nationality", used_features, ignored_features, vr.nationalities)
        mother_qual = show_optional_dict("Mother Qualification", "mother_qual", used_features, ignored_features, vr.mother_qual)
        father_qual = show_optional_dict("Father Qualification", "father_qual", used_features, ignored_features, vr.fathers_qualification)
        mother_job = show_optional_dict("Mother Job", "mother_job", used_features, ignored_features, vr.mothers_occupation)
        debtor = show_optional_dict("Debtor", "debtor", used_features, ignored_features, vr.debtor_map)
        fees = show_optional_dict("Paid Fees", "fees", used_features, ignored_features, vr.fees_map)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📉 Course Performance")
    col1, col2 = st.columns(2)

    with col1:
        cred_1 = show_optional("Credits 1", "cred_1_c", st.number_input, used_features, ignored_features, min_value = 0, value=0)
        enrolled_1 = show_optional("Enrolled 1", "enrolled_1_c", st.number_input, used_features, ignored_features,min_value = 0, value=6)
        evals_1 = show_optional("Evaluations 1", "evals_1_c", st.number_input, used_features, ignored_features, min_value = 0, value=6)
        approved_1 = show_optional("Approved 1", "approved_1_c", st.number_input, used_features, ignored_features, min_value = 0,value=3)
        grade_1 = show_optional("Grade 1", "grade_1_c", st.number_input, used_features, ignored_features, min_value = 0.0 ,value=9.5)
        no_evals_1 = show_optional("No Exams 1", "no_evals_1_c", st.number_input, used_features, ignored_features,min_value = 0, value=0)

    with col2:
        cred_2 = show_optional("Credits 2", "cred_2_c", st.number_input, used_features, ignored_features, min_value = 0,value=0)
        enrolled_2 = show_optional("Enrolled 2", "enrolled_2_c", st.number_input, used_features, ignored_features, min_value = 0,value=6)
        evals_2 = show_optional("Evaluations 2", "evals_2_c", st.number_input, used_features, ignored_features,min_value = 0, value=6)
        approved_2 = show_optional("Approved 2", "approved_2_c", st.number_input, used_features, ignored_features,min_value = 0, value=3)
        grade_2 = show_optional("Grade 2", "grade_2_c", st.number_input, used_features, ignored_features, min_value = 0.0 ,value=9.0)
        no_evals_2 = show_optional("No Exams 2", "no_evals_2_c", st.number_input, used_features, ignored_features, min_value = 0,value=0)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏛 Economic Indicators")

    unemployment = show_optional("Unemployment Rate", "unemployment_c", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_c", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_c", st.number_input, used_features, ignored_features, value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict 📊", key="predict_course"):
        X_df_course = pd.DataFrame([locals()]).filter(items=course_model.feature_names_in_)
        dropout = 1 - course_model.predict_proba(X_df_course)[0][1]

        st.session_state.update({
            "last_model": "course",
            "last_prediction": time.time(),
            "X_df_course": X_df_course,
            "dropout_course": dropout,
            "student_name": st.session_state.name
        })

        color = "#28a745" if dropout < 0.33 else "#ffc107" if dropout < 0.66 else "#dc3545"
        label = "✔️ Likely to Continue, Dropout Risk" if dropout < 0.5 else "⚠️ High Dropout Risk"

        st.markdown(
            f"<div class='result-box' style='background:{color}20;'>"
            f"<h3 style='color:{color};'>{label}: {dropout:.2f}</h3></div>",
            unsafe_allow_html=True,
        )

# Without course performance
with tab_nocourse:
    used_features = []; ignored_features = []

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧑 Personal & Academic Background")
    col1, col2 = st.columns(2)

    with col1:
        marital = show_optional_dict("Marital Status", "marital_nc", used_features, ignored_features, vr.marital_status)
        app_mode = show_optional_dict("Application Mode", "app_mode_nc", used_features, ignored_features, vr.application_mode)
        app_order = show_optional("Application Order", "app_order_nc", st.number_input, used_features, ignored_features, min_value=0, max_value=9, value=0)
        course = show_optional_dict("Course Code", "course_nc", used_features, ignored_features, vr.courses)
        admission_grade = show_optional("Admission Grade", "admission_grade_nc", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        attendance = show_optional_dict("Attendance", "attendance_nc", used_features, ignored_features, vr.attendance)
        prev_qual = show_optional_dict("Previous Qualification", "prev_qual_nc", used_features, ignored_features, vr.previous_qualification)
        gender = show_optional_dict("Gender", "gender_nc", used_features, ignored_features, vr.gender)
        father_job = show_optional_dict("Father Job", "father_job_nc", used_features, ignored_features, vr.fathers_occupation)
        displaced = show_optional_dict("Displaced", "displaced_nc", used_features, ignored_features, vr.displaced_map)
        special_needs = show_optional_dict("Special Needs", "special_nc", used_features, ignored_features, vr.special_needs_map)

    with col2:
        scholarship = show_optional_dict("Scholarship", "scholarship_nc", used_features, ignored_features, vr.scholarship_map)
        age = show_optional("Age", "age_nc", st.number_input, used_features, ignored_features, min_value = 0, value=21)
        international = show_optional_dict("International", "international_nc", used_features, ignored_features, vr.international_map)
        prev_grade = show_optional("Previous Grade", "prev_grade_nc", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
        nationality = show_optional_dict("Nationality", "nationality_nc", used_features, ignored_features, vr.nationalities)
        mother_qual = show_optional_dict("Mother Qualification", "mother_qual_nc", used_features, ignored_features, vr.mother_qual)
        father_qual = show_optional_dict("Father Qualification", "father_qual_nc", used_features, ignored_features, vr.fathers_qualification)
        mother_job = show_optional_dict("Mother Job", "mother_job_nc", used_features, ignored_features, vr.mothers_occupation)
        debtor = show_optional_dict("Debtor", "debtor_nc", used_features, ignored_features, vr.debtor_map)
        fees = show_optional_dict("Paid Fees", "fees_nc", used_features, ignored_features, vr.fees_map)


    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏛 Economic Indicators")

    unemployment = show_optional("Unemployment Rate", "unemployment_nc", st.number_input, used_features, ignored_features, value=7.5)
    inflation = show_optional("Inflation Rate", "inflation_nc", st.number_input, used_features, ignored_features, value=1.5)
    gdp = show_optional("GDP Growth", "gdp_nc", st.number_input, used_features, ignored_features, value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict 📊", key="predict_nocourse"):
        X_df_nocourse = pd.DataFrame([locals()]).filter(items=nocourse_model.feature_names_in_)
        dropout_nc = 1- nocourse_model.predict_proba(X_df_nocourse)[0][1]

        st.session_state.update({
            "last_model": "nocourse",
            "last_prediction": time.time(),
            "X_df_nocourse": X_df_nocourse,
            "dropout_nocourse": dropout_nc,
            "student_name": st.session_state.name
        })

        color = "#28a745" if dropout_nc < 0.33 else "#ffc107" if dropout_nc < 0.66 else "#dc3545"
        label = "✔️ Likely to Continue, Dropout Risk" if dropout_nc < 0.5 else "⚠️ High Dropout Risk"

        st.markdown(
            f"<div class='result-box' style='background:{color}20;'>"
            f"<h3 style='color:{color};'>{label}: {dropout_nc:.2f}</h3></div>",
            unsafe_allow_html=True,
        )
