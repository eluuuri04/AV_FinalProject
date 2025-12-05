import streamlit as st
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

# Configuració de la pàgina
st.set_page_config(
    page_title="EDA",
    page_icon="📄",
    layout="centered"
)
# Botó per tornar a la pantalla principal
if st.button("⬅️ Tornar a la pantalla principal"):
    st.switch_page()

# Comprovar si les dades existeixen
df = st.session_state["data"]


# Títol i descripció
st.title("📄 Exploració i anàlisi de les dades")
st.write("Un cop d'ull a les dades de (QUINES DADES SÓN).")
st.write("Aqustes dades han estat obtingudes de: " \
"Valentim Realinho, Jorge Machado, Luís Baptista, & Mónica V. Martins. (2021). Predict students' dropout and academic success (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5777340")

st.write("Un overview de les dades:")
st.write(df.head(5))


st.set_page_config(page_title="Dashboard KPIs", layout="wide")

st.title("📊 Dashboard de KPIs")

# --- Calculs ---
prev_grade = df["Previous qualification (grade)"].mean()
admission_grade = df["Admission grade"].mean()
approved_1 = df["Curricular units 1st sem (approved)"].count() / len(df) * 100
approved_2 = df["Curricular units 2nd sem (approved)"].count() / len(df) * 100
grade_1 = df["Curricular units 1st sem (grade)"].mean()
grade_2 = df["Curricular units 2nd sem (grade)"].mean()
enrolled_1 = df["Curricular units 1st sem (enrolled)"].mean()
enrolled_2 = df["Curricular units 2nd sem (enrolled)"].mean()

# --- Layout amb columnes ---
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


# Paleta de colors
color_palette = plt.cm.Set2.colors  

# Filtrar els dos grups
df_drop = df[df['Target'] == 'Dropout']
df_no_drop = df[df['Target'] != 'Dropout']

# Diccionaris de mapping
category_mappings = {
    'Gender': {1: 'Male', 0: 'Female'},
    'Marital status': {
        1: 'Single',
        2: 'Married',
        3: 'Widower',
        4: 'Divorced',
        5: 'Facto Union',
        6: 'Separated'
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

st.title("Comparació Dropout vs No Dropout per variables categòriques")

# Selector de variable
col_selected = st.selectbox("Selecciona una variable:", categorical_cols)

# Comptar valors per cada grup
counts_drop = df_drop[col_selected].value_counts()
counts_no_drop = df_no_drop[col_selected].value_counts()

# Map labels
mapped_labels_drop = pd.Series(counts_drop.index.map(category_mappings.get(col_selected, {}).get))
labels_drop = mapped_labels_drop.fillna(pd.Series(counts_drop.index)).tolist()

mapped_labels_no_drop = pd.Series(counts_no_drop.index.map(category_mappings.get(col_selected, {}).get))
labels_no_drop = mapped_labels_no_drop.fillna(pd.Series(counts_no_drop.index)).tolist()

# Crear gràfics amb colors macos
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(counts_drop, labels=labels_drop, autopct='%1.1f%%', startangle=140,
            pctdistance=0.85, colors=color_palette[:len(counts_drop)])
axes[0].set_title(f'{col_selected} - Dropout')

axes[1].pie(counts_no_drop, labels=labels_no_drop, autopct='%1.1f%%', startangle=140,
            pctdistance=0.85, colors=color_palette[:len(counts_no_drop)])
axes[1].set_title(f'{col_selected} - No Dropout')

plt.suptitle(f'Distribution of {col_selected}', fontsize=14)

# Mostrar a Streamlit
st.pyplot(fig)



# Suposem que la columna 'Target' és categòrica amb 2 valors: Dropout i No Dropout
target_col = 'Target'
targets = df[target_col].unique()

# Paleta de colors amb 2 tons macos
palette = sns.color_palette("Set2", len(targets))
color_map = dict(zip(targets, palette))
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
# Iterem per cada KPI numèrica
for col in kpi_columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Iterem pels dos subsets
    for i, target in enumerate(targets):
        subset = df[df[target_col] == target]
        sns.histplot(
            data=subset,
            x=col,
            kde=True,
            ax=axes[i],
            stat="density",
            color=color_map[target]
        )
        axes[i].set_title(f'{col} - {target}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')

    plt.suptitle(f'Distribution of {col} by {target_col}', fontsize=14)
    plt.tight_layout()
    plt.show()
