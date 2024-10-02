import streamlit as st
import pandas as pd
import uuid

st.set_page_config(page_title="CSV Loader", layout="wide")

st.title("Chargement de fichier CSV")

# Fonction pour charger le CSV
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Widget pour uploader le fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Charger le dataset
    df = load_csv(uploaded_file)
    
    # Réinitialiser le dataset et le nom du fichier dans la session lors du chargement d'un nouveau fichier
    st.session_state.dataset = df
    st.session_state.filename = uploaded_file.name
    st.session_state.original_dataset = df.copy()  # Réinitialiser aussi l'original
    st.session_state.dataset_id = str(uuid.uuid4())

    # Affichage du tableau de données
    st.header("Tableau de données")
    st.dataframe(df)

    # Récapitulatif des données
    st.header("Récapitulatif des données")
    st.write(df.describe())

elif 'dataset' in st.session_state:
    st.success(f"Fichier CSV chargé : {st.session_state.filename}")

    # Affichage du tableau de données
    st.header("Tableau de données")
    st.dataframe(st.session_state.dataset)

    # Récapitulatif des données
    st.header("Récapitulatif des données")
    st.write(st.session_state.dataset.describe())

    # Bouton pour réinitialiser le dataframe à sa version originale
    if st.button("Réinitialiser le dataset à sa forme initiale"):
        if 'original_dataset' in st.session_state:
            st.session_state.dataset = st.session_state.original_dataset.copy()  # Réinitialiser à la version originale
            st.session_state.dataset_id = str(uuid.uuid4())
            st.success("Le dataset a été réinitialisé à sa version initiale.")
            st.rerun()  # Recharger la page pour mettre à jour l'affichage
        else:
            st.warning("Aucun dataset original n'est disponible pour la réinitialisation.")

else:
    st.info("Veuillez charger un fichier CSV pour commencer.")
