import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="EDA — Exploració de dades",
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
        st.switch_page("app.py")

with top_col2:
    st.markdown("<h1 class='main-header'>📊 Exploració i anàlisi de les dades</h1>", unsafe_allow_html=True)
    st.write("Exploració visual del conjunt de dades per entendre patrons i diferències entre estudiants que abandonen i els que continuen.")

st.markdown("---")

# ================== LOAD DATA ==================
try:
    df = st.session_state["data"]
except KeyError:
    st.error("❌ Les dades no estan carregades. Torna a la pantalla principal per carregar el fitxer.")
    st.stop()

# ================== DATASET INFO ==================
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("📄 Informació del conjunt de dades")
    st.write("Aqustes dades han estat obtingudes de: " \
    "Valentim Realinho, Jorge Machado, Luís Baptista, & Mónica V. Martins. (2021). Predict students' dropout and academic success (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5777340")
    st.write("Una mostra de les primeres files de les dades:")
    st.dataframe(df.head(5))

    st.markdown("</div>", unsafe_allow_html=True)

# ================== KPIs GLOBALS ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📌 Indicadors clau (KPIs)")

prev_grade = df["Previous qualification (grade)"].mean()
admission_grade = df["Admission grade"].mean()
approved_1 = df["Curricular units 1st sem (approved)"].count() / len(df) * 100
approved_2 = df["Curricular units 2nd sem (approved)"].count() / len(df) * 100
grade_1 = df["Curricular units 1st sem (grade)"].mean()
grade_2 = df["Curricular units 2nd sem (grade)"].mean()
enrolled_1 = df["Curricular units 1st sem (enrolled)"].mean()
enrolled_2 = df["Curricular units 2nd sem (enrolled)"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎓 Previous qualification (grade)", f"{prev_grade:.2f}")
    st.metric("📝 Admission grade", f"{admission_grade:.2f}")

with col2:
    st.metric("✅ Approved 1st sem (%)", f"{approved_1:.2f}%")
    st.metric("✅ Approved 2nd sem (%)", f"{approved_2:.2f}%")

with col3:
    st.metric("📈 Grade 1st sem (mean)", f"{grade_1:.2f}")
    st.metric("📈 Grade 2nd sem (mean)", f"{grade_2:.2f}")

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
st.subheader("🥧 Comparació Dropout vs No Dropout (variables categòriques)")
st.write("Selecciona variables categòriques per veure la seva distribució entre estudiants que abandonen i els que continuen.")

selected_cats = st.multiselect("Variables categòriques:", categorical_cols)

for col_selected in selected_cats:
    if col_selected not in df.columns:
        st.warning(f"⚠️ La columna {col_selected} no existeix al DataFrame.")
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
    axes[0].set_title(f'{col_selected} - Dropout')

    axes[1].pie(
        counts_no_drop,
        labels=labels_no_drop,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85,
        colors=color_palette[:len(counts_no_drop)]
    )
    axes[1].set_title(f'{col_selected} - No Dropout')

    plt.suptitle(f'Distribution of {col_selected}', fontsize=14)
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# ================== NUMERIC DISTRIBUTIONS ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📈 Distribució de KPIs numèriques (Dropout vs No Dropout)")
st.write("Explora les distribucions numèriques i compara el comportament entre els dos grups.")

selected_kpis = st.multiselect("KPIs numèriques:", kpi_columns)

for col in selected_kpis:
    if col not in df.columns:
        st.warning(f"⚠️ La columna {col} no existeix al DataFrame.")
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
    axes[0].set_title(f'{col} - Dropout')
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
    axes[1].set_title(f'{col} - No Dropout')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Density')

    plt.suptitle(f'Distribution of {col} (Dropout vs No Dropout)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
