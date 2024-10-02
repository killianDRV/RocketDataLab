from utils.functions_preprocessing import *


st.set_page_config(page_title="Prétraitement", layout="wide")
st.title("Prétraitement des données")

# Load NLP models
nlp_en, nlp_fr, stemmer_en, stemmer_fr = load_spacy_models()


# Vérifier si le dataset est dans la session
if 'dataset' in st.session_state and st.session_state.dataset is not None:
    df_temp = st.session_state.dataset.copy()  # Utilise la version modifiée du dataset

    st.markdown("\n")

    # Appel des fonctions
    df_temp = drop_na(df_temp)
    df_temp = fill_na(df_temp)
    df_temp = handle_outliers(df_temp)
    df_temp = remove_whitespace(df_temp)
    df_temp = remove_non_ascii(df_temp)
    df_temp = case_conversion(df_temp)
    df_temp = remove_punctuation(df_temp)
    df_temp = remove_stopwords(df_temp)
    df_temp = normalize_date_format(df_temp)
    df_temp = convert_column_types(df_temp)
    df_temp = convert_boolean_representation(df_temp)
    df_temp = convert_units(df_temp)
    df_temp = remove_duplicates(df_temp)
    df_temp = lemmatize_text(df_temp, nlp_en, nlp_fr)
    df_temp = stemming_text(df_temp, stemmer_en, stemmer_fr)


    # Affichage du dataset final après prétraitement
    st.markdown("\n")
    st.header("Dataset final après prétraitement")
    st.dataframe(df_temp)

    # Bouton pour enregistrer le dataset modifié dans la session
    if st.button("Enregistrer le dataframe pour la session"):
        st.session_state.dataset = df_temp  # Mettre à jour le dataframe modifié dans la session
        st.session_state.dataset_id = str(uuid.uuid4())
        st.success("Le dataframe a été enregistré pour cette session.")

    # Bouton de téléchargement
    csv = df_temp.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger le dataset",
        data=csv,
        file_name='dataset_pretraite.csv',
        mime='text/csv',
        key='download-csv'
    )

else:
    st.warning("Aucun dataset n'a été chargé. Veuillez retourner à la page d'accueil pour charger un fichier CSV.")