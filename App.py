import streamlit as st
import pandas as pd

# Configuració de la pàgina
st.set_page_config(
    page_title="Predictor d'abandonament universitari",
    page_icon="🎓",
    layout="wide"
)

# CSS personalitzat
st.markdown("""
<style>
/* Hero section */
.hero {
    text-align: center;
    padding: 50px 20px;
    background: linear-gradient(135deg, #1E90FF, #00CED1);
    border-radius: 15px;
    color: white;
}
.hero h1 {
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 10px;
}
.hero h2 {
    font-size: 28px;
    font-weight: 400;
    margin-top: 0;
    color: #FFD700;
}
.hero p {
    font-size: 20px;
    margin-top: 20px;
}

/* Cards */
.card {
    background-color: #f9f9f9;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.card:hover {
    transform: translateY(-5px);
}
.card h3 {
    margin-top: 15px;
    font-size: 22px;
    color: #1E90FF;
}
.card p {
    font-size: 16px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>🎓 Predictor d'abandonament universitari</h1>
    <h2>Projecte Final d'Analítica Visual</h2>
    <p>Explora, prediu i entén els factors clau de l'èxit acadèmic</p>
</div>
""", unsafe_allow_html=True)

st.write("\n")
# Cards amb funcionalitats
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📊 Anàlisi</h3>
        <p>Explora les dades històriques dels estudiants</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    if st.button("Entrar a Anàlisi"):
        st.switch_page("pages/EDA.py")

with col2:
    st.markdown("""
    <div class="card">
        <h3>⚙️ Predicció</h3>
        <p>Descobreix si un estudiant abandonarà o continuarà</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")

    if st.button("Entrar a Predicció"):
        st.switch_page("pages/Predictor.py")

with col3:
    st.markdown("""
    <div class="card">
        <h3>🪤 Explainability</h3>
        <p>Entén els factors més rellevants de cada predicció</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")

    if st.button("Entrar a Explainability"):
        st.switch_page("pages/Explainability.py")

# Footer amb autors
st.markdown("""
---
👨‍💻 **Autors:**  
- Uriel Cabañas Pedro (NIA: 269121)  
- Pau Colomer Coll (NIA: 268401)
""")

# Carregar CSV
df = pd.read_csv("data.csv", sep=";")
st.session_state["data"] = df
st.success("Fitxer carregat i guardat a session_state ✅")
