import numpy as np
import pandas as pd
import streamlit as st
import math as ma
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from scipy.stats import norm
from nelsonsiegelsvneson import NSS , NS
from splinecubique import SplineCubique
from interplolationpolynomial import InterpolationCubique,InterpolationSimple
from modeledevasicek import VasicekModel
from modelecir import CIRModel

st.set_page_config(
    page_title="Modelisation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

st.title("Modélisation des Taux Zéros coupon ")
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
pre = os.path.dirname(os.path.realpath(__file__))
fname = "tauxjour2123.xlsx"#'taux_tresor_ex.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)
data["Days"]=(data["Date"]-data["Date"].iloc[0]).dt.total_seconds().astype(int)/(3600*24)

with st.form("Modélisation de la courbe de taux avec le modèle de Vasicek"):
    st.subheader("Modélisation de la courbe de taux avec le modèle de Vasicek")
    st.latex(r'''dr_t = a(b - r_t) \, dt + \sigma \, dW_t''')
    vas=VasicekModel(data["Date"],data["Taux Moyen"])
    datenego=st.date_input("Date d'estimation")
    date=pd.Timedelta(pd.Timestamp(datenego)-data["Date"].iloc[-1]).days/365
    submitted = st.form_submit_button("Submit")
    
    if (submitted):
      st.markdown("Estimation des paramètres du modèle de Vasicek")
      parms=vas.setParameters()
      st.write("La vitesse de retour à la moyenne est ",parms[0] )
      st.write('Le taux long terme est ', parms[1])
      st.write('la volatilité ', parms[2])
      st.markdown("Estimation des taux courts par du modèle de Vasicek")
      st.pyplot(vas.plotVasicek())
      st.markdown("Optimisation  de l'ecart entre les taux du modèle de Vasicek et le modèle deterministe le plus performant")
      opt=vas.optimizationModel(date)
      st.write('la prime de risque ', opt[0], " Erreur de l'optimisation", opt[1], opt[2] )
      st.write("RMSE: ",vas.tauxZCVasicek(date)[1])
      st.markdown("Prévision des taux zeros coupons par du modèle de Vasicek")
      st.pyplot(vas.predictionVasicek(date))

    
with st.form("Modélisation de la courbe de taux avec le modèle de Cox-Ingersoll-Ross"):
    st.subheader("Modélisation de la courbe de taux avec le modèle de Cox-Ingersoll-Ross")
    st.latex(r'''dr_t = a(b - r_t) \, dt + \sigma \sqrt{r_t} \, dW_t''')
    cir=CIRModel(data["Date"],data["Taux Moyen"])
    datenego=st.date_input("Date d'estimation")
    submitted = st.form_submit_button("Submit")
    date=pd.Timedelta(pd.Timestamp(datenego)-data["Date"].iloc[-1]).days/365

    if (submitted):
      st.markdown("Estimation des paramètres du modèle de Cox-Ingersoll-Ross")
      parms=cir.setparameters()
      st.write("La vitesse de retour à la moyenne est ",parms[0][0] )
      st.write('Le taux long terme est ', parms[1])
      st.write('la volatilité ', parms[2][0])
      st.markdown("Estimation des taux courts par du modèle deCox-Ingersoll-Ross")
      st.pyplot(cir.plotCIR())
      st.markdown("Optimisation  de l'ecart entre les taux du modèle de Cox-Ingersoll-Ross et le modèle deterministe le plus performant")
      opt=cir.optimizationRSS()
      st.write( " Erreur de l'optimisation", opt[1], opt[2])
      st.write("RMSE: ",cir.calculSigma())
      st.markdown("Prévision des taux zeros coupons par du modèle deCox-Ingersoll-Ross")
      st.pyplot(cir.plotCIRPred(date))







