import numpy as np
import streamlit as st
import math as ma
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from scipy.stats import norm
from nelsonsiegelsvneson import NSS , NS, NSSparms
from splinecubique import SplineCubique, SplineExponentiel
from interplolationpolynomial import InterpolationCubique,InterpolationSimple

st.set_page_config(
    page_title="Reconstruction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.title("Modèle de construction de la courbe des taux")
st.markdown(" Notre algorithme intelligent utilise les données scrappées sur le site de Bank Al Maghrib pour reconstruire la courbe zéro coupon de manière fiable"+
            "Explorez les diférrentes méthode de construction, ajustez les paramètres, et choissisez celui qui s'ajuste au mieux ")

st.set_option('deprecation.showPyplotGlobalUse', False)

with st.form("Présentation de la courbe de taux zéro coupon "):
   st.subheader("Présentation de la courbe de taux zéro coupon du" +str(InterpolationSimple().date))
   submitted = st.form_submit_button("Submit")
   #st.pyplot(InterpolationSimple().plotData())
   st.write(InterpolationSimple().data())
   st.pyplot(InterpolationSimple().plotData())
   if (submitted):
      st.write(InterpolationSimple().data())
      st.pyplot(InterpolationSimple().plotData())
      

st.header("Reconstruction de la courbe de taux zéro coupon du " +str(InterpolationSimple().date))
  
st.subheader("Interpolation polynomiale")
with st.form("Interpolation par les méthode déterministe"):
   st.subheader("Interpolation linéaire")
   st.latex(r''' r(t)= \dfrac{(t_i - t)*r(t_{i-1})+(t-t_{i-1})*r(t_i)}{t_i-t_{i-1}}''')
   mean=st.number_input("Maturité en jour",step=1,value=1)/360
   submitted = st.form_submit_button("Submit")
   inpol=InterpolationSimple()
   st.pyplot(inpol.plotInterpolationsimple())
   st.write("La valeur estimée par interpolation linéaire est ", inpol.interpolationsimple(mean))

   if (submitted):
      inpol=InterpolationSimple()
      st.pyplot(inpol.plotInterpolationsimple())
      st.write("La valeur estimée par interpolation linéaire est ", inpol.interpolationsimple(mean))


with st.form("Interpolation cubique"):
   st.subheader("Interpolation cubique")
   st.latex(r'''r(t)=at^{3}+bt^{2}+ct+d''')
   submitted = st.form_submit_button("Submit")
   mean=st.number_input("Maturité en jour",step=1,value=0)/360
   incub=InterpolationCubique()
   st.pyplot(incub.plotRegressionMultiple())
   st.pyplot(incub.plotInterpolationCubique())
   st.write("La valeur estimée par interpolation cubique est ", incub.regressionMultiple(mean))
   st.write("Erreur estimée par interpolationcubique par régression est ", 1-incub.regressionScore())
   if (submitted):
      incub=InterpolationCubique()
      st.pyplot(incub.plotRegressionMultiple())
      st.pyplot(incub.plotInterpolationCubique())
      st.write("La valeur estimée par interpolation cubique est ", incub.regressionMultiple(mean))
      st.write("Erreur estimée par interpolationcubique par régression est ", 1-incub.regressionScore())

st.header("Méthodes indirectes de reconstruction")  
       
with st.form("Interpolation spline cubique"):
   st.subheader("Interpolation  par spline cubique")
   st.latex(r'''B(s)= B_0(s)+ \sum_{i=0}^{n} (a_i -a_{i-1} )\max(s-s_i,0)^{3}''')
   submitted = st.form_submit_button("Submit")
   mean=st.number_input("Maturité en jour",step=1,value=1)/360
   spcub=SplineCubique()
   st.pyplot(spcub.plotsplineCubique())
   st.write("La valeur estimée par interpolation cubique est ", spcub.splineCubiqueTotal(mean))
   st.write("Erreur estimée par interpolationcubique par régression est ", spcub.parmsValue()[1])
   st.write("Convergence: ",spcub.parmsValue()[2],spcub.parmsValue()[3] )

   if (submitted):
      spcub=SplineCubique()
      st.pyplot(spcub.plotsplineCubique())
      st.write("La valeur estimée par interpolation cubique est ", spcub.splineCubiqueTotal(mean))
      st.write("Erreur estimée par interpolationcubique par régression est ", spcub.parmsValue()[1])
      st.write("Convergence: ",spcub.parmsValue()[2],spcub.parmsValue()[3] )


with st.form("Interpolation spline exponentiel"):
   st.subheader("Interpolation  par spline exponentiel")
   st.latex(r'''B(s)= B_0(s)+ \sum_{i=0}^{n} (a_i -a_{i-1} )\min(e^{-us}-e^{-us_i},0)^{3}''')
   submitted = st.form_submit_button("Submit")
   mean=st.number_input("Maturité en jour",step=1,value=1)/365
   spex=SplineExponentiel()
   st.pyplot(spex.plotsplineExponentiel())
   st.write("La valeur estimée par interpolation cubique est ", spex.splineExponentielTotal(mean))
   st.write("Erreur estimée par interpolationcubique par régression est ", spex.parmsValue()[1])
   if (submitted):
      spex=SplineExponentiel()
      st.pyplot(spex.plotsplineExponentiel())
      st.write("La valeur estimée par interpolation cubique est ", spex.splineExponentielTotal(mean))
      st.write("Erreur estimée par interpolationcubique par régression est ", spex.parmsValue()[1])


with st.form("Evolution"):
   st.subheader("Evolution des parametres des méthodes de Nelson Siegel Svensson")
   submitted = st.form_submit_button("Submit")
   st.pyplot(NSSparms().plotNSSparms())
   if (submitted):
      st.pyplot(NSSparms().plotNSSparms())

with st.form("NS"):
   st.subheader("Méthode de Nelson Siegel")
   st.latex(r''' R(t) = \beta_0 + \beta_1 \left( \frac{1 - e^{-t/\tau}}{t/\tau} \right) + \beta_2 \left( \frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau} \right)''')
   submitted = st.form_submit_button("Submit")
   mean=st.number_input("Maturité en jour",step=1,value=1)/360
   ns=NS()
   st.pyplot(ns.plotNSoptimisation())
   st.write("La valeur estimée par la méthode de Nelson Siegel est ", ns.NS_taux(mean))
   st.write("Erreur estimée par optimisation des parametres", ns.parmsValue()[1])
   st.write("Convergence optimisation: ",ns.parmsValue()[2],ns.parmsValue()[3] )
   st.pyplot(ns.plotNSregression())
   st.write("La valeur estimée par la méthode de Nelson Siegel est ", ns.NStauxRegression(mean))
   st.write("Erreur estimée par regression et optimisation des parametres", ns.optimiser()[1])
   st.write("Convergence regression: ",ns.optimiser()[2],ns.optimiser()[3] )

   if (submitted):
      ns=NS()
      st.pyplot(ns.plotNSoptimisation())
      st.write("La valeur estimée par la méthode de Nelson Siegel est ", ns.NS_taux(mean))
      st.write("Erreur estimée par optimisation des parametres", ns.parmsValue()[1])
      st.write("Convergence optimisation: ",ns.parmsValue()[2],ns.parmsValue()[3] )
      st.pyplot(ns.plotNSregression())
      st.write("La valeur estimée par la méthode de Nelson Siegel est ", ns.NStauxRegression(mean))
      st.write("Erreur estimée par regression et optimisation des parametres", ns.optimiser()[1])
      st.write("Convergence regression: ",ns.optimiser()[2],ns.optimiser()[3] )



with st.form("NSS"):
   st.subheader("Méthode de Nelson Siegel Svensson")
   st.latex(r'''R(t) = \beta_0 + \beta_1 \left( \frac{1 - e^{-t/\tau}}{t/\tau} \right) + \beta_2 \left( \frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau} \right) + \beta_3 \left( \frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2}  \right)''')
   submitted = st.form_submit_button("Submit")
   mean=st.number_input("Maturité en jour",step=1,value=1)/360
   nss=NSS()
   st.pyplot(nss.plotNSS())
   st.write("La valeur estimée par la méthode de Nelson Siegel est ", nss.NSS_taux(mean))
   st.write("Erreur estimée par optimisation des parametres", nss.parmsValue()[1])
   st.write("Convergence optimisation: ",nss.parmsValue()[2],nss.parmsValue()[3] )
   st.pyplot(nss.plotNSSregression())
   st.write("La valeur estimée par la méthode de Nelson Siegel est ", nss.NSS_Taux(mean))
   st.write("Erreur estimée par regression et optimisation des parametres", nss.optimiser()[1])
   st.write("Convergence regression: ",nss.optimiser()[2],nss.optimiser()[3] )
   if (submitted):
      nss=NSS()
      st.pyplot(nss.plotNSS())
      st.write("La valeur estimée par la méthode de Nelson Siegel est ", nss.NSS_taux(mean))
      st.write("Erreur estimée par optimisation des parametres", nss.parmsValue()[1])
      st.write("Convergence optimisation: ",nss.parmsValue()[2],nss.parmsValue()[3] )
      st.pyplot(nss.plotNSSregression())
      st.write("La valeur estimée par la méthode de Nelson Siegel est ", nss.NSS_Taux(mean))
      st.write("Erreur estimée par regression et optimisation des parametres", nss.optimiser()[1])
      st.write("Convergence regression: ",nss.optimiser()[2],nss.optimiser()[3] )







