import streamlit as st
import pandas as pd


st.markdown("## <span style='color:deepskyblue'>Predictor d'abandonament universitari</span>", unsafe_allow_html=True)
st.markdown("## <span style='color:goldenrod'>Projecte Final d'Analítica Visual</span>", unsafe_allow_html=True)
st.markdown("#### <span style='color:mediumseagreen'>Benvingut/da!</span>", unsafe_allow_html=True)

st.markdown("""
**Nom:** Uriel Cabañas Pedro  
**NIA:** 269121
           
**Nom:** Pau Colomer Coll  
**NIA:** 268401

            
En aquesta aplicació web podràs:
- **Explorar** les dades històriques dels estudiants (DIR QUINS ESTUDIANTS SÓN).
- **Predir** si un estudiant concret abandonarà o seguirà els seus estudis.
- **Descobrir** quins són els factors, de la predicció, més rellevants de cada estudiant.
""")


if st.button("Anar a Predicció ⚙️"):
    st.switch_page("pages/Predictor.py")


# Carregar CSV només una vegada


df = pd.read_csv("data.csv", sep = ";")
st.session_state["data"] = df   # Guardar al session_state
st.success("Fitxer carregat i guardat a session_state ✅")
