import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

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
    letter-spacing: 0.5px;
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
    animation: fadeInUp 0.6s ease;
}
.section-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    transition: all 0.25s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    transition: transform 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #00CED1, #1E90FF);
    box-shadow: 0 0 12px rgba(30,144,255,0.4);
}

/* Badges */
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


# Load trained models
with open("course_model.pkl", "rb") as f:
    course_model = pickle.load(f)

with open("nocourse_model.pkl", "rb") as f:
    nocourse_model = pickle.load(f)

st.title("📘 Student Dropout Predictor")
st.write("Select features to ignore — ignored features are passed to the model as NaN.")

def show_optional(label, key, widget_fn, used_list, ignored_list, *args, **kwargs):
    ignore = st.checkbox(f"Ignore {label}", key=f"ignore_{key}")
    if ignore:
        ignored_list.append(label)
        return np.nan
    used_list.append(label)
    return widget_fn(label, key=key, *args, **kwargs)

tab_course, tab_nocourse = st.tabs(["📚 With Course Performance", "🧩 Without Course Performance"])

# =======================================================================
# TAB 1 — WITH COURSE PERFORMANCE
# =======================================================================
with tab_course:
    used_features = []
    ignored_features = []

    with st.container():
        st.subheader("🧑 Personal & Academic Profile")
        col1, col2 = st.columns(2)

        with col1:
            marital = show_optional("💍 Marital Status", "marital", st.selectbox, used_features, ignored_features, [0,1,2])
            app_mode = show_optional("📄 Application Mode", "app_mode", st.selectbox, used_features, ignored_features, [0,1,2,3])
            app_order = show_optional("🔢 Application Order", "app_order", st.number_input, used_features, ignored_features, value=1)
            course = show_optional("📘 Course Code", "course", st.number_input, used_features, ignored_features, value=9003)
            admission_grade = show_optional("Admission Grade", "admission_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
            attendance = show_optional("📅 Attendance", "attendance", st.selectbox, used_features, ignored_features, [0,1])
            prev_qual = show_optional("🎓 Previous Qualification", "prev_qual", st.number_input, used_features, ignored_features, value=1)
            gender = show_optional("Gender", "gender", st.selectbox, used_features, ignored_features, [0,1])
            scholarship = show_optional("Scholarship", "scholarship", st.selectbox, used_features, ignored_features, [0,1])
            age = show_optional("Age", "age", st.number_input, used_features, ignored_features, value=21)
            international = show_optional("International", "international", st.selectbox, used_features, ignored_features, [0,1])



        with col2:
            prev_grade = show_optional("📊 Previous Grade", "prev_grade", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
            nationality = show_optional("🌍 Nationality", "nationality", st.number_input, used_features, ignored_features, value=1)
            mother_qual = show_optional("👩 Mother Qualification", "mother_qual", st.number_input, used_features, ignored_features, value=1)
            father_qual = show_optional("👨 Father Qualification", "father_qual", st.number_input, used_features, ignored_features, value=1)
            mother_job = show_optional("Mother Job", "mother_job", st.number_input, used_features, ignored_features, value=3)
            father_job = show_optional("Father Job", "father_job", st.number_input, used_features, ignored_features, value=3)
            displaced = show_optional("Displaced", "displaced", st.selectbox, used_features, ignored_features, [0,1])
            special_needs = show_optional("Special Needs", "special_needs", st.selectbox, used_features, ignored_features, [0,1])
            debtor = show_optional("Debtor", "debtor", st.selectbox, used_features, ignored_features, [0,1])
            fees = show_optional("Paid Fees", "fees", st.selectbox, used_features, ignored_features, [0,1])



        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.subheader("🎓 Course Performance (Required)")
        col1, col2 = st.columns(2)
        with col1:
            cred_1 = st.number_input("Credits 1", value=0)
            enrolled_1 = st.number_input("Enrolled 1", value=6)
            evals_1 = st.number_input("Evaluations 1", value=6)
            approved_1 = st.number_input("Approved 1", value=3)
            grade_1 = st.number_input("Grade 1", value=9.5)
        with col2:
            no_evals_1 = st.number_input("No Exams 1", value=0)
            cred_2 = st.number_input("Credits 2", value=0)
            enrolled_2 = st.number_input("Enrolled 2", value=6)
            evals_2 = st.number_input("Evaluations 2", value=6)
            approved_2 = st.number_input("Approved 2", value=3)
            grade_2 = st.number_input("Grade 2", value=9.0)
            no_evals_2 = st.number_input("No Exams 2", value=0)

    with st.container():
        st.subheader("📉 Economic Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            unemployment = show_optional("📈 Unemployment Rate", "unemployment", st.number_input, used_features, ignored_features, value=7.5)
        with col2:
            inflation = show_optional("💹 Inflation Rate", "inflation", st.number_input, used_features, ignored_features, value=1.5)
        with col3:
            gdp = show_optional("📊 GDP Growth", "gdp", st.number_input, used_features, ignored_features, value=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📌 Feature Summary")
    st.success(f"Used: {', '.join(used_features)}" if used_features else "Used: None")
    st.warning(f"Ignored: {', '.join(ignored_features)}" if ignored_features else "Ignored: None")

    if st.button("Predict 📊", key="predict_course"):
        data = {
            "marital": marital, "app_mode": app_mode, "app_order": app_order, "course": course,
            "attendance": attendance, "prev_qual": prev_qual, "prev_grade": prev_grade,
            "nationality": nationality, "mother_qual": mother_qual, "father_qual": father_qual,
            "mother_job": mother_job, "father_job": father_job, "admission_grade": admission_grade,
            "displaced": displaced, "special_needs": special_needs, "debtor": debtor, "fees": fees,
            "gender": gender, "scholarship": scholarship, "age": age, "international": international,
            "unemployment": unemployment, "inflation": inflation, "gdp": gdp,
            "cred_1": cred_1, "enrolled_1": enrolled_1, "evals_1": evals_1, "approved_1": approved_1,
            "grade_1": grade_1, "no_evals_1": no_evals_1,
            "cred_2": cred_2, "enrolled_2": enrolled_2, "evals_2": evals_2, "approved_2": approved_2,
            "grade_2": grade_2, "no_evals_2": no_evals_2
        }

        X_df = pd.DataFrame([data])
        X_df = X_df.reindex(columns=course_model.feature_names_in_)
        prob = course_model.predict_proba(X_df)[0]
        dropout = prob[1]

        st.markdown("<h3 style='font-size:1.3rem;font-weight:500;'>🎯 Result</h3>", unsafe_allow_html=True)
        st.progress(dropout)  # Barra visual del risc

        if dropout >= 0.5:
            st.error(f"⚠️ Dropout Risk: {dropout:.2f}")
        else:
            st.success(f"🎉 Likely to Continue: {1 - dropout:.2f}")

# =======================================================================
# TAB 2 — WITHOUT COURSE PERFORMANCE
# =======================================================================
with tab_nocourse:
    used_features = []
    ignored_features = []

    with st.container():
        st.subheader("🧑 Personal & Academic Profile")
        col1, col2 = st.columns(2)

        with col1:
            marital = show_optional("💍 Marital Status", "marital_nc", st.selectbox, used_features, ignored_features, [0,1,2])
            app_mode = show_optional("📄 Application Mode", "app_mode_nc", st.selectbox, used_features, ignored_features, [0,1,2,3])
            app_order = show_optional("🔢 Application Order", "app_order_nc", st.number_input, used_features, ignored_features, value=1)
            course = show_optional("📘 Course Code", "course_nc", st.number_input, used_features, ignored_features, value=9003)
            admission_grade = show_optional("Admission Grade", "admission_grade_nc", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
            attendance = show_optional("📅 Attendance", "attendance_nc", st.selectbox, used_features, ignored_features, [0,1])
            prev_qual = show_optional("🎓 Previous Qualification", "prev_qual_nc", st.number_input, used_features, ignored_features, value=1)
            gender = show_optional("Gender", "gender_nc", st.selectbox, used_features, ignored_features, [0,1])
            scholarship = show_optional("Scholarship", "scholarship_nc", st.selectbox, used_features, ignored_features, [0,1])
            age = show_optional("Age", "age_nc", st.number_input, used_features, ignored_features, value=21)
            international = show_optional("International", "international_nc", st.selectbox, used_features, ignored_features, [0,1])



        with col2:
            prev_grade = show_optional("📊 Previous Grade", "prev_grade_nc", st.number_input, used_features, ignored_features, min_value=0.0, max_value=100.0, value=95.0)
            nationality = show_optional("🌍 Nationality", "nationality_nc", st.number_input, used_features, ignored_features, value=1)
            mother_qual = show_optional("👩 Mother Qualification", "mother_qual_nc", st.number_input, used_features, ignored_features, value=1)
            father_qual = show_optional("👨 Father Qualification", "father_qual_nc", st.number_input, used_features, ignored_features, value=1)
            mother_job = show_optional("Mother Job", "mother_job_nc", st.number_input, used_features, ignored_features, value=3)
            father_job = show_optional("Father Job", "father_job_nc", st.number_input, used_features, ignored_features, value=3)
            displaced = show_optional("Displaced", "displaced_nc", st.selectbox, used_features, ignored_features, [0,1])
            special_needs = show_optional("Special Needs", "special_needs_nc", st.selectbox, used_features, ignored_features, [0,1])
            debtor = show_optional("Debtor", "debtor_nc", st.selectbox, used_features, ignored_features, [0,1])
            fees = show_optional("Paid Fees", "fees_nc", st.selectbox, used_features, ignored_features, [0,1])
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.subheader("📉 Economic Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            unemployment = show_optional("📈 Unemployment Rate", "unemployment_nc", st.number_input, used_features, ignored_features, value=7.5)
        with col2:
            inflation = show_optional("💹 Inflation Rate", "inflation_nc", st.number_input, used_features, ignored_features, value=1.5)
        with col3:
            gdp = show_optional("📊 GDP Growth", "gdp_nc", st.number_input, used_features, ignored_features, value=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📌 Feature Summary")
    st.success(f"Used: {', '.join(used_features)}" if used_features else "Used: None")
    st.warning(f"Ignored: {', '.join(ignored_features)}" if ignored_features else "Ignored: None")

    if st.button("Predict 📊", key="predict_nocourse"):
        data_nc = {
            "marital": marital, "app_mode": app_mode, "app_order": app_order, "course": course,
            "attendance": attendance, "prev_qual": prev_qual, "prev_grade": prev_grade,
            "nationality": nationality, "mother_qual": mother_qual, "father_qual": father_qual,
            "mother_job": mother_job, "father_job": father_job, "admission_grade": admission_grade,
            "displaced": displaced, "special_needs": special_needs, "debtor": debtor, "fees": fees,
            "gender": gender, "scholarship": scholarship, "age": age, "international": international,
            "unemployment": unemployment, "inflation": inflation, "gdp": gdp,
        }

        X_df_nc = pd.DataFrame([data_nc])
        X_df_nc = X_df_nc.reindex(columns=nocourse_model.feature_names_in_)
        prob = nocourse_model.predict_proba(X_df_nc)[0]
        dropout = prob[1]

        st.subheader("🎯 Result")
        st.progress(dropout)  # Barra visual del risc

        if dropout >= 0.5:
            st.error(f"⚠️ Dropout Risk: {dropout:.2f}")
        else:
            st.success(f"🎉 Likely to Continue: {1 - dropout:.2f}")
