import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="EDA — Data Exploration",
    page_icon="📊",
    layout="wide"
)

# ================== GLOBAL STYLE ==================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Header title */
.main-header {
    font-size: 2.1rem;
    font-weight: 750;
}

/* Metrics tweak */
[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 700;
    color: #1E90FF;
}

.stButton>button {
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    transition: .15s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# ================== TOP BAR ==================
top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    if st.button("⬅️ Home"):
        st.switch_page("App.py")

with top_col2:
    st.markdown("<h1 class='main-header'>📊 Data Exploration & Analysis</h1>", unsafe_allow_html=True)
    st.write("Visual exploration of the dataset to understand patterns and differences between students who drop out and those who continue.")

st.markdown("---")

# ================== LOAD DATA ==================
try:
    df = st.session_state["data"]
except KeyError:
    st.error("❌ Data not loaded. Please return to the home page to upload the dataset.")
    st.stop()

# ================== DATASET INFO ==================
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("📄 Dataset Information")
    st.write("These data were obtained from: " \
    "Valentim Realinho, Jorge Machado, Luís Baptista, & Mónica V. Martins. (2021). Predict students' dropout and academic success (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5777340")
    st.write("Preview of the first rows:")
    st.dataframe(df.head(5))

    st.markdown("</div>", unsafe_allow_html=True)

# ================== KPIs GLOBALS ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📌 Key Performance Indicators (KPIs)")

prev_grade = df["Previous qualification (grade)"].mean()
admission_grade = df["Admission grade"].mean()

no_enrolled_1 = (df["Curricular units 1st sem (enrolled)"] == 0).sum() #0 enrolled subjects
diff_1 = df["Curricular units 1st sem (enrolled)"] - df["Curricular units 1st sem (approved)"]
all_passed_1 = (diff_1 == 0).sum() - no_enrolled_1 #all enrolled passed
approved_1 = all_passed_1 / (len(df) - no_enrolled_1) * 100

no_enrolled_2 = (df["Curricular units 2nd sem (enrolled)"] == 0).sum() #0 enrolled subjects
diff_2 = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 2nd sem (approved)"]
all_passed_2 = (diff_2 == 0).sum() - no_enrolled_2 #all enrolled passed
approved_2 = all_passed_2 / (len(df) - no_enrolled_2) * 100

#Grades between 0-20
grade_1 = df["Curricular units 1st sem (grade)"].mean()
grade_2 = df["Curricular units 2nd sem (grade)"].mean()
enrolled_1 = df["Curricular units 1st sem (enrolled)"].mean()
enrolled_2 = df["Curricular units 2nd sem (enrolled)"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎓 Previous qualification (grade) (0-200)", f"{prev_grade:.2f}")
    st.metric("📝 Admission grade (0-200)", f"{admission_grade:.2f}")

with col2:
    st.metric("✅ Approved 1st sem (%)", f"{approved_1:.2f}%")
    st.metric("✅ Approved 2nd sem (%)", f"{approved_2:.2f}%")

with col3:
    st.metric("📈 Grade 1st sem (mean) (0-20)", f"{grade_1:.2f}")
    st.metric("📈 Grade 2nd sem (mean) (0-20)", f"{grade_2:.2f}")

with col4:
    st.metric("📚 Enrolled 1st sem (mean)", f"{enrolled_1:.2f}")
    st.metric("📚 Enrolled 2nd sem (mean)", f"{enrolled_2:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ================== PREPARE GROUPS & MAPPINGS ==================
df_drop = df[df['Target'] == 'Dropout']
df_no_drop = df[df['Target'] != 'Dropout']

category_mappings = {
    'Gender': {1: 'Male', 0: 'Female'},
    'Marital status': {
        1: 'Single', 2: 'Married', 3: 'Widower',
        4: 'Divorced', 5: 'Facto Union', 6: 'Separated'
    },
    'Displaced': {1: 'Yes', 0: 'No'},
    'Scholarship holder': {1: 'Yes', 0: 'No'},
    'Tuition fees up to date': {1: 'Yes', 0: 'No'},
    'Educational special needs': {1: 'Yes', 0: 'No'},
    'Daytime/evening attendance': {1: 'Daytime', 0: 'Evening'},
}

categorical_cols = [
    'Gender', 'Marital status', 'Displaced', 'Scholarship holder',
    'Tuition fees up to date', 'Educational special needs', 'Daytime/evening attendance\t'
]

kpi_columns = [
    'Admission grade',
    'Previous qualification (grade)',
    'Age at enrollment',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Inflation rate',
    'GDP'
]

color_palette = plt.cm.Set2.colors
palette = sns.color_palette("Set2", 2)
color_map = {"Dropout": palette[0], "No Dropout": palette[1]}

# ================== CATEGORICAL PIE CHARTS ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🥧 Dropout vs No Dropout — Categorical Variables")
st.write("Select categorical variables to view their distribution between dropout and non-dropout students.")

selected_cats = st.multiselect("Categorical variables:", categorical_cols)

for col_selected in selected_cats:
    if col_selected not in df.columns:
        st.warning(f"⚠️ The column {col_selected} does not exist in the DataFrame.")
        continue

    counts_drop = df_drop[col_selected].value_counts()
    counts_no_drop = df_no_drop[col_selected].value_counts()

    mapped_labels_drop = pd.Series(counts_drop.index.map(category_mappings.get(col_selected, {}).get))
    labels_drop = mapped_labels_drop.fillna(pd.Series(counts_drop.index)).tolist()

    mapped_labels_no_drop = pd.Series(counts_no_drop.index.map(category_mappings.get(col_selected, {}).get))
    labels_no_drop = mapped_labels_no_drop.fillna(pd.Series(counts_no_drop.index)).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(
        counts_drop,
        labels=labels_drop,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85,
        colors=color_palette[:len(counts_drop)]
    )
    axes[0].set_title(f'{col_selected} — Dropout')

    axes[1].pie(
        counts_no_drop,
        labels=labels_no_drop,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85,
        colors=color_palette[:len(counts_no_drop)]
    )
    axes[1].set_title(f'{col_selected} — No Dropout')

    plt.suptitle(f'Distribution of {col_selected}', fontsize=14)
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# ================== NUMERIC DISTRIBUTIONS ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📈 Numeric KPI Distributions (Dropout vs No Dropout)")
st.write("Explore numeric variable distributions and compare the behavior between both groups.")

selected_kpis = st.multiselect("Numeric variables:", kpi_columns)

for col in selected_kpis:
    if col not in df.columns:
        st.warning(f"⚠️ The column {col} does not exist in the DataFrame.")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    sns.histplot(
        data=df_drop,
        x=col,
        kde=True,
        ax=axes[0],
        stat="density",
        color=color_map["Dropout"],
        alpha=0.7
    )
    axes[0].set_title(f'{col} — Dropout')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Density')

    sns.histplot(
        data=df_no_drop,
        x=col,
        kde=True,
        ax=axes[1],
        stat="density",
        color=color_map["No Dropout"],
        alpha=0.7
    )
    axes[1].set_title(f'{col} — No Dropout')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Density')

    plt.suptitle(f'Distribution of {col} (Dropout vs No Dropout)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
