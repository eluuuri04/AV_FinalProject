import streamlit as st
import plotly.express as px
import pandas as pd

# Configuració de la pàgina
st.set_page_config(
    page_title="Demo Streamlit",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 Demo amb Streamlit")
st.write("Explora components moderns sense necessitat de streamlit-elements.")

# --- Editor de codi (simulat amb text_area) ---
st.subheader("Editor de codi")
code = st.text_area("Escriu el teu codi Python aquí:", "print('Hola Streamlit!')")
if st.button("Executar codi"):
    try:
        exec(code)
    except Exception as e:
        st.error(f"Error: {e}")

# --- Gràfic interactiu amb Plotly ---
st.subheader("Gràfic interactiu")
df = pd.DataFrame({
    "Categoria": ["A", "B", "C"],
    "Valors": [4, 7, 3]
})
fig = px.bar(df, x="Categoria", y="Valors", title="Exemple de bar chart")
st.plotly_chart(fig, use_container_width=True)

# --- Media Player (iframe YouTube) ---
st.subheader("Reproductor multimèdia")
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
