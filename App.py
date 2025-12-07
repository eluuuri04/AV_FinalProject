import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="University Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
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
    <h1>🎓 University Dropout Predictor</h1>
    <h2>Final Visual Analytics Project</h2>
    <p>Explore, predict, and understand the key factors behind academic success</p>
</div>
""", unsafe_allow_html=True)

st.write("\n")

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📊 Analysis</h3>
        <p>Explore historical student data</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    if st.button("Go to Analysis"):
        st.switch_page("pages/1_EDA.py")

with col2:
    st.markdown("""
    <div class="card">
        <h3>⚙️ Prediction</h3>
        <p>Predict whether a student will drop out or continue</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    if st.button("Go to Prediction"):
        st.switch_page("pages/2_Predictor.py")

with col3:
    st.markdown("""
    <div class="card">
        <h3>🪤 Explainability</h3>
        <p>Understand the most relevant factors for each prediction</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    if st.button("Go to Explainability"):
        st.switch_page("pages/3_Explainability.py")

# Footer with authors
st.markdown("""
---
👨‍💻 **Authors:**  
- Uriel Cabañas Pedro (ID: 269121)  
- Pau Colomer Coll (ID: 268401)
""")

# Load CSV
df = pd.read_csv("data.csv", sep=";")
st.session_state["data"] = df
st.success("File successfully loaded into session_state ✅")
