import streamlit as stm 
  
stm.set_page_config(page_title = "Courbe de Taux") 
stm.title("Courbe de Taux Zéro coupon") 
stm.sidebar.success("Select Pages ") 
import streamlit as st



# Introduction
st.markdown("## Explorez les Tendances des Taux Zéro Coupon ")

# Fonctionnalités Principales
st.markdown("### Fonctionnalités Principales :")

st.markdown("1. **Reconstruction des Courbes :** \
    Visualisez les résultats obtenus avec différentes méthodes de reconstruction. \
    Comparez les courbes reconstruites pour identifier celle qui correspond le mieux à vos besoins et à vos données.")

st.markdown("2. **Modélisation Avancée :** \
    Explorez les modèles de Vasicek et de Cox-Ingersoll-Ross pour anticiper les tendances futures des taux zéro coupon. \
    Notre interface intuitive facilite la modélisation, vous permettant de prendre des décisions éclairées.")

st.markdown("3. **Comparaison Interactive :** \
    Comparez les courbes reconstruites avec les modèles prédictifs pour une compréhension complète des mouvements des taux zéro coupon. \
    Utilisez nos outils interactifs pour ajuster les paramètres et affiner vos prévisions.")

# Pourquoi [Nom de votre Application]
st.markdown("### Valeur ajoutée de l'application  :")

st.markdown("- **Simplicité d'Utilisation :** Notre interface conviviale permet une navigation aisée à travers les différentes méthodes de reconstruction et les modèles de prévision.")
st.markdown("- **Précision des Prévisions :** Les modèles de Vasicek et de Cox-Ingersoll-Ross vous offrent une précision maximale pour modéliser les taux zéro coupon à l'avenir.")
st.markdown("- **Analyse Personnalisée :** Personnalisez vos analyses en ajustant les paramètres des modèles pour répondre spécifiquement à vos besoins.")

# Commencez Votre Exploration
st.markdown("### Commencez Votre Exploration :")
start_button = st.button("Démarrer votre exploration")

