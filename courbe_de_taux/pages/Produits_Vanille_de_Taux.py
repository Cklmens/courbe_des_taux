import numpy as np
import streamlit as st
import math as ma
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from scipy.stats import norm
from nelsonsiegelsvneson import NSS , NS
from splinecubique import SplineCubique
from interplolationpolynomial import InterpolationCubique,InterpolationSimple

st.set_page_config(
    page_title="Produits de taux",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.title("Produits vanille sur les taux d'intéret")
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.form("Forward Rate Agreement  "):
   st.subheader(" Valorisation du Forward Rate Agreement  ")
   st.write("Parametres: ")
   nominal=st.number_input("Le nominal",step=1)
   K=st.number_input("Taux fixe convenu", step=0.01)
   datenego=st.date_input("Date de négociation")
   datedeb=st.date_input("Date de début de contrat")
   datefin=st.date_input("Matuirté du contrat")
   position=st.selectbox("Position", ["Emprunteur","Prêteur"])
   
   submitted = st.form_submit_button("Submit")
  
   if (submitted):
     pass
   
with st.form(" Swap de Taux  "):
   st.subheader(" Valorisation Swap de Taux  ")


   submitted = st.form_submit_button("Submit")
  
   if (submitted):
     pass
   
with st.form("option sur les taux  "):
   st.subheader(" Valorisation du Forward Rate Agreement  ")


   submitted = st.form_submit_button("Submit")
  
   if (submitted):
     pass