from utils.functions_analyse import *

st.set_page_config(page_title="Analyse de données", layout="wide")
st.title("Analyse de données")

# Vérifier si le dataset est dans la session
if 'dataset' in st.session_state and st.session_state.dataset is not None:
    df = st.session_state.dataset
    
    # Vérifier si le dataset a changé
    if 'last_dataset_id' not in st.session_state or st.session_state.last_dataset_id != st.session_state.dataset_id:
        reset_charts()
        st.session_state.last_dataset_id = st.session_state.dataset_id

    # Obtenir les colonnes par type
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = categorical_cols + numeric_cols

    # Initialisation des états de session pour les sélections
    init_session_states(numeric_cols, categorical_cols)

    # Affichage des graphiques
    if numeric_cols:
        create_histogram(df, numeric_cols)
        create_box_plot(df, numeric_cols)

    if len(numeric_cols) > 1:
        create_scatter_plot(df, numeric_cols)
        create_line_chart(df, numeric_cols)

    if numeric_cols:
        create_bar_chart(df, all_cols, numeric_cols)

    if categorical_cols:
        create_pie_chart(df, categorical_cols)

    if len(numeric_cols) > 1:
        create_correlation_matrix(df, numeric_cols)

else:
    st.warning("Aucun dataset n'a été chargé. Veuillez retourner à la page d'accueil pour charger un fichier CSV.")
