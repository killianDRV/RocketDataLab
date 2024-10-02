import streamlit as st
import pandas as pd
import numpy as np
import string
import json
import re
import pint
import requests
import spacy
from pathlib import Path
import uuid
from nltk.stem import PorterStemmer, SnowballStemmer


from utils.currencies import currencies


# Charger les stopwords à partir des fichiers JSON
def load_stopwords(language):
    if language == 'anglais':
        with open('utils/english_stopwords.json', 'r') as f:
            return json.load(f)
    elif language == 'français':
        with open('utils/french_stopwords.json', 'r') as f:
            return json.load(f)
    else:
        return []
    

def drop_na(df):
    st.header("Suppression des lignes avec des données manquantes")
    cols_to_drop_na = st.multiselect(
        "Sélectionnez les colonnes pour lesquelles vous souhaitez supprimer les lignes avec des valeurs manquantes :",
        df.columns.tolist(),
        key="cols_to_drop_na"
    )
    if cols_to_drop_na:
        original_shape = df.shape[0]
        df = df.dropna(subset=cols_to_drop_na)
        total_removed = original_shape - df.shape[0]
        st.success(f"{total_removed} lignes ont été supprimées.")
    return df

def fill_na(df):
    st.markdown("\n")
    st.header("Remplacement des valeurs manquantes")
    cols_to_fill = st.multiselect(
        "Sélectionnez les colonnes pour lesquelles vous souhaitez remplacer les valeurs manquantes :",
        df.columns.tolist()
    )
    for col_to_fill in cols_to_fill:
        st.markdown(f"### Remplacement pour la colonne : {col_to_fill}")
        fill_method = st.selectbox(
            f"Méthode de remplissage pour {col_to_fill} :",
            ["Moyenne (pour colonnes numériques)", "Médiane (pour colonnes numériques)", "Mode", "Valeur constante"],
            key=f"{col_to_fill}_method"
        )

        if fill_method == "Moyenne (pour colonnes numériques)":
            if pd.api.types.is_numeric_dtype(df[col_to_fill]):
                mean_value = df[col_to_fill].mean()
                df[col_to_fill].fillna(mean_value, inplace=True)
                st.write(f"Les valeurs manquantes de la colonne {col_to_fill} ont été remplacées par la moyenne ({mean_value}).")
            else:
                st.warning(f"La colonne {col_to_fill} n'est pas numérique. Impossible d'utiliser la moyenne.")

        elif fill_method == "Médiane (pour colonnes numériques)":
            if pd.api.types.is_numeric_dtype(df[col_to_fill]):
                median_value = df[col_to_fill].median()
                df[col_to_fill].fillna(median_value, inplace=True)
                st.write(f"Les valeurs manquantes de la colonne {col_to_fill} ont été remplacées par la médiane ({median_value}).")
            else:
                st.warning(f"La colonne {col_to_fill} n'est pas numérique. Impossible d'utiliser la médiane.")

        elif fill_method == "Mode":
            mode_value = df[col_to_fill].mode()[0]
            df[col_to_fill].fillna(mode_value, inplace=True)
            st.write(f"Les valeurs manquantes de la colonne {col_to_fill} ont été remplacées par la mode ({mode_value}).")

        elif fill_method == "Valeur constante":
            constant_value = st.text_input(f"Entrez la valeur constante pour la colonne {col_to_fill} :", key=f"{col_to_fill}_constant")
            if constant_value:
                if pd.api.types.is_integer_dtype(df[col_to_fill]):
                    if constant_value.isnumeric():
                        df[col_to_fill].fillna(round(constant_value), inplace=True)
                        print(df[col_to_fill].dtypes)
                    else:
                        df[col_to_fill].fillna(str(constant_value), inplace=True)
                        print(df[col_to_fill].dtypes)
                elif pd.api.types.is_float_dtype(df[col_to_fill]):
                    try:
                        df[col_to_fill].fillna(float(constant_value), inplace=True)
                        print(df[col_to_fill].dtypes)
                    except:
                        df[col_to_fill].fillna(str(constant_value), inplace=True)
                        print(df[col_to_fill].dtypes)
                else:
                    df[col_to_fill].fillna(str(constant_value), inplace=True)
                st.write(f"Les valeurs manquantes de la colonne {col_to_fill} ont été remplacées par la valeur constante ({constant_value}).")

    return df

def handle_outliers(df):
    st.markdown("\n")
    st.header("Gestion des valeurs aberrantes (outliers)", help="Les outliers sont des valeurs qui se trouvent en dehors de l'intervalle interquartile (1.5 * IQR).  \n[IQR = Quartile3 - Quartile1]")
    cols_for_outliers = st.multiselect(
        "Sélectionnez les colonnes pour gérer les valeurs aberrantes :",
        df.columns.tolist()
    )
    for col in cols_for_outliers:
        st.markdown(f"### Gestion des outliers pour la colonne : {col}")
        action = st.selectbox(
            f"Que souhaitez-vous faire pour {col} ?",
            ["Supprimer les outliers", "Remplacer par la moyenne", "Remplacer par la médiane"],
            key=f"outlier_action_{col}"
        )

        if action in ["Supprimer les outliers", "Remplacer par la moyenne", "Remplacer par la médiane"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"La colonne {col} n'est pas numérique. Impossible de traiter les outliers.")
                continue

        if action == "Supprimer les outliers":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            original_count = df.shape[0]
            df = df[(df[col] >= (q1 - 1.5 * iqr)) & (df[col] <= (q3 + 1.5 * iqr))]
            removed_count = original_count - df.shape[0]
            st.success(f"{removed_count} outliers ont été supprimés dans la colonne {col}.")

        elif action == "Remplacer par la moyenne":
            mean_value = df[col].mean()
            outliers = (df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | \
                        (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))
            df.loc[outliers | df[col].isna(), col] = mean_value
            st.write(f"Les outliers et les valeurs manquantes dans la colonne {col} ont été remplacés par la moyenne ({mean_value}).")

        elif action == "Remplacer par la médiane":
            median_value = df[col].median()
            outliers = (df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | \
                        (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))
            df.loc[outliers | df[col].isna(), col] = median_value
            st.write(f"Les outliers et les valeurs manquantes dans la colonne {col} ont été remplacés par la médiane ({median_value}).")

    return df

def remove_whitespace(df):
    st.markdown("\n")
    st.header("Suppression des espaces superflus", help="Supprime les espaces au début/fin ou les espaces multiples entre les mots.")
    cols_for_whitespace_removal = st.multiselect(
        "Sélectionnez les colonnes textuelles pour supprimer les espaces superflus :",
        [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    )
    if cols_for_whitespace_removal:
        for col in cols_for_whitespace_removal:
            whitespace_action = st.selectbox(
                f"Que souhaitez-vous faire pour la colonne {col} ?",
                ["Supprimer les espaces au début et à la fin", 
                    "Supprimer les espaces multiples entre les mots", 
                    "Supprimer les espaces au début/fin ET réduire les espaces multiples"],
                key=f"whitespace_action_{col}"
            )
            if whitespace_action == "Supprimer les espaces au début et à la fin":
                df[col] = df[col].str.strip()
            elif whitespace_action == "Supprimer les espaces multiples entre les mots":
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            elif whitespace_action == "Supprimer les espaces au début/fin ET réduire les espaces multiples":
                df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)

    return df

def remove_non_ascii(df):
    st.markdown("\n")
    st.header("Suppression des caractères non-ASCII ou spéciaux", help="Supprime les caractères qui ne sont pas dans l'ensemble ASCII.")
    cols_for_non_ascii_removal = st.multiselect(
        "Sélectionnez les colonnes textuelles pour supprimer les caractères non-ASCII :",
        [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    )
    if cols_for_non_ascii_removal:
        for col in cols_for_non_ascii_removal:
            # df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str).apply(lambda x: x if x == '' else re.sub(r'[^\x00-\x7F]+', '', x))

        st.success("Les caractères non-ASCII ont été supprimés avec succès.")
    return df

def case_conversion(df):
    st.markdown("\n")
    st.header("Conversion en minuscules/majuscules")
    cols_for_case_conversion = st.multiselect(
        "Sélectionnez les colonnes textuelles pour la conversion de casse :",
        [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    )
    case_option = st.radio("Choisissez la conversion :", ["Minuscules", "Majuscules"])
    if cols_for_case_conversion and case_option:
        for col in cols_for_case_conversion:
            if case_option == "Minuscules":
                df[col] = df[col].str.lower()
                st.success(f"La colonne {col} a été convertie en minuscules.")
            elif case_option == "Majuscules":
                df[col] = df[col].str.upper()
                st.success(f"La colonne {col} a été convertie en majuscules.")
    return df

def remove_punctuation(df):
    st.markdown("\n")
    st.header("Suppression de la ponctuation", help="Supprime les caractères de ponctuation, tels que !\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~.")
    cols_for_punctuation_removal = st.multiselect(
        "Sélectionnez les colonnes textuelles pour supprimer la ponctuation :",
        [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    )
    if cols_for_punctuation_removal:
        for col in cols_for_punctuation_removal:
            df[col] = df[col].str.replace(f"[{string.punctuation}]", "", regex=True)
        st.success("La ponctuation a été supprimée avec succès.")
    return df

def remove_stopwords(df):
    st.markdown("\n")
    st.header("Suppression des stopwords", help="Supprime les mots courants (stopwords) qui n'ajoutent pas de sens particulier au texte.  \nLanguage pris en compte: Français / Anglais")
    cols_for_stopwords_removal = st.multiselect(
        "Sélectionnez les colonnes textuelles pour supprimer les stopwords :",
        [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    )
    language = st.selectbox("Sélectionnez la langue des stopwords :", options=["anglais", "français"], key="stopwords_language")
    
    if cols_for_stopwords_removal:
        stop_words = load_stopwords(language)
        for col in cols_for_stopwords_removal:
            df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
        st.success("Les stopwords ont été supprimés avec succès.")
    return df


def is_valid_date_format(format_string):
    valid_directives = set('%aAbBcdHIjmMpSUwWxXyYzZ')
    pattern = r'%[aAbBcdHIjmMpSUwWxXyYzZ]'
    
    # Vérification des cas '%%' qui ne doivent pas être acceptés
    if '%%' in format_string:
        return False

    # Vérifier si tous les caractères % sont suivis d'une directive valide
    if not all(c in valid_directives for c in re.findall(r'%(.)', format_string)):
        return False
    
    # Vérifier si le format contient au moins une directive valide
    if not re.search(pattern, format_string):
        return False
    
    return True

def normalize_date_format(df):
    st.markdown("\n")
    st.header("Normalisation des formats de date", help="""%Y : Année avec le siècle (ex : 2024)  
        %y : Année sur deux chiffres (ex : 24 pour 2024)  
        %m : Mois sur deux chiffres (ex : 09 pour septembre)  
        %B : Nom complet du mois (ex : September)  
        %b : Nom abrégé du mois (ex : Sep)  
        %d : Jour du mois sur deux chiffres (ex : 25 pour le 25 septembre)  
        %A : Nom complet du jour de la semaine (ex : Monday)  
        %a : Nom abrégé du jour de la semaine (ex : Mon)  
        %H : Heure (24h) sur deux chiffres (ex : 14 pour 14h)  
        %I : Heure (12h) sur deux chiffres (ex : 02 pour 2h de l'après-midi)  
        %p : AM ou PM (ex : PM)  
        %M : Minutes sur deux chiffres (ex : 05 pour 5 minutes)  
        %S : Secondes sur deux chiffres (ex : 09 pour 9 secondes)  
        %f : Microsecondes (ex : 123456 pour 0.123456 secondes)  
        %j : Jour de l'année (ex : 268 pour le 25 septembre)  
        %U : Numéro de la semaine de l'année (dimanche comme premier jour de la semaine)  
        %W : Numéro de la semaine de l'année (lundi comme premier jour de la semaine)""")

    date_columns = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
    
    if not date_columns:
        st.warning("Aucune colonne de type date n'a été trouvée dans le dataset.")
        return df

    cols_to_normalize = st.multiselect(
        "Sélectionnez les colonnes de date à normaliser :",
        date_columns
    )

    if cols_to_normalize:
        for col in cols_to_normalize:
            st.subheader(f"Normalisation pour la colonne : {col}")

            input_format = st.text_input(f"Spécifiez le format d'entrée de la date pour {col} (ex: %Y-%m-%d %H:%M:%S pour 2024-02-23 14:30:00)", "%Y-%m-%d", key=f"input_{col}")

            output_format = st.text_input(f"Spécifiez le format de sortie de la date pour {col} (ex: %d/%m/%Y %H:%M)", "%Y-%m-%d", key=f"output_{col}")

            if st.button(f"Normaliser les dates pour {col}"):
                if not is_valid_date_format(output_format):
                    st.error(f"Le format de sortie '{output_format}' n'est pas valide. Veuillez utiliser des directives de format de date valides.")
                else:
                    try:
                        df[col] = pd.to_datetime(df[col], format=input_format)
                        df[col] = df[col].dt.strftime(output_format)
                        st.success(f"La colonne {col} a été normalisée avec succès.")
                    except Exception as e:
                        st.error(f"Erreur lors de la normalisation de {col} : {str(e)}  \nAssurez-vous que le format d'entrée correspond bien aux données.")

    return df


def convert_column_types(df):
    st.markdown("\n")
    st.header("Conversion de types de colonnes")

    cols_to_convert = st.multiselect(
        "Sélectionnez les colonnes pour changer le type de données :",
        df.columns.tolist()
    )

    type_options = ["String", "Integer", "Float", "Boolean"]

    conversions = {}
    for col in cols_to_convert:
        new_type = st.selectbox(f"Quel type pour la colonne {col} ?", type_options, key=f"type_{col}")
        conversions[col] = new_type

    for col, new_type in conversions.items():
        if new_type == "Integer":
            try:
                df[col] = df[col].astype(int)
                st.success(f"La colonne {col} a été convertie en entier.")
            except:
                st.warning(f"Erreur lors de la conversion de {col} en entier. Vérifiez les valeurs.")

        elif new_type == "Float":
            try:
                df[col] = df[col].astype(float)
                st.success(f"La colonne {col} a été convertie en float.")
            except:
                st.warning(f"Erreur lors de la conversion de {col} en float. Vérifiez les valeurs.")

        elif new_type == "Boolean":
            try:
                df[col] = df[col].map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}).fillna(df[col])
                df[col] = df[col].astype('bool')
                st.success(f"La colonne {col} a été convertie en booléen.")
            except:
                st.warning(f"Erreur lors de la conversion de {col} en booléen. Vérifiez les valeurs.")

    return df

def convert_boolean_representation(df):
    st.markdown("\n")
    st.header("Conversion de la représentation des booléens", help="Par défaut, seules les colonnes ne contenant pas de valeurs nulles et ayant des valeurs True ou False sont considérées comme étant de type booléen. Vous pouvez utiliser la conversion de types de colonnes pour transformer une colonne en type booléen.")
    
    boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()

    if not boolean_columns:
        st.warning("Aucune colonne booléenne n'a été trouvée dans le dataset.")
        return df

    cols_to_convert = st.multiselect(
        "Sélectionnez les colonnes booléennes à convertir :",
        boolean_columns
    )

    conversions = {}
    for col in cols_to_convert:
        conversion_type = st.selectbox(
            f"Choisissez le type de conversion pour {col} :",
            ["True/False", "Yes/No", "1/0"],
            key=f"conv_{col}"
        )
        conversions[col] = conversion_type

    # if st.button("Appliquer les conversions booléennes"):
    for col, conversion_type in conversions.items():
        if conversion_type == "True/False":
            df[col] = df[col].astype(bool)
        elif conversion_type == "Yes/No":
            df[col] = df[col].map({True: "Yes", False: "No"})
        elif conversion_type == "1/0":
            df[col] = df[col].astype(int)

        st.success(f"La colonne {col} a été convertie en mode '{conversion_type}' avec succès.")

    return df




def convert_units(df):
    st.markdown("\n")
    st.header("Conversion des unités")

    # Initialiser les outils de conversion
    ureg = pint.UnitRegistry()

    # Listes prédéfinies pour les autres unités
    distances = ["km", "m", "cm", "mm", "mile", "yard", "foot", "inch"]
    weights = ["kg", "g", "mg", "lb", "oz", "ton"]
    temperatures = ["Celsius", "Fahrenheit", "Kelvin"]

    # Sélection des colonnes à convertir
    cols_to_convert = st.multiselect(
        "Sélectionnez les colonnes pour la conversion d'unités :",
        df.select_dtypes(include=['number', 'datetime64']).columns.tolist()
    )

    for col in cols_to_convert:
        st.subheader(f"Conversion pour la colonne : {col}")

        # Sélection du type de conversion
        conversion_type = st.selectbox(
            f"Type de conversion pour {col}",
            ["Monnaie", "Distance", "Poids", "Température"],
            key=f"conv_type_{col}"
        )

        if conversion_type == "Monnaie":
            from_currency = st.selectbox(f"Devise d'origine pour {col}", currencies, key=f"from_curr_{col}")
            to_currency = st.selectbox(f"Devise cible pour {col}", currencies, key=f"to_curr_{col}")

            if from_currency and to_currency:
                try:
                    # Extraire le code des devises (avant l'espace et parenthèses)
                    from_currency_code = from_currency.split()[0]
                    to_currency_code = to_currency.split()[0]

                    # Requête directe à l'API ExchangeRate
                    response = requests.get(f'https://api.exchangerate-api.com/v4/latest/{from_currency_code}')

                    if response.status_code == 200:
                        data = response.json()
                        rate = data['rates'].get(to_currency_code)

                        if rate:
                            df[f"{col}_converti_{to_currency_code}"] = df[col] * rate
                            st.success(f"Conversion de {from_currency} à {to_currency} effectuée pour {col}")
                        else:
                            st.error(f"Erreur : taux de change non trouvé pour {to_currency}.")
                    else:
                        st.error(f"Erreur lors de la requête API : statut {response.status_code}.")
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de devise. {e}")

        elif conversion_type == "Distance":
            from_unit = st.selectbox(f"Unité d'origine pour {col}", distances, key=f"from_dist_{col}")
            to_unit = st.selectbox(f"Unité cible pour {col}", distances, key=f"to_dist_{col}")
            if from_unit and to_unit:
                try:
                    df[f"{col}_converti_{to_unit}"] = df[col].apply(lambda x: (x * ureg(from_unit)).to(to_unit).magnitude)
                    st.success(f"Conversion de {from_unit} à {to_unit} effectuée pour {col}")
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de distance. {e}")

        elif conversion_type == "Poids":
            from_unit = st.selectbox(f"Unité d'origine pour {col}", weights, key=f"from_weight_{col}")
            to_unit = st.selectbox(f"Unité cible pour {col}", weights, key=f"to_weight_{col}")
            if from_unit and to_unit:
                try:
                    df[f"{col}_converti_{to_unit}"] = df[col].apply(lambda x: (x * ureg(from_unit)).to(to_unit).magnitude)
                    st.success(f"Conversion de {from_unit} à {to_unit} effectuée pour {col}")
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de poids. {e}")

        elif conversion_type == "Température":
            from_scale = st.selectbox(f"Échelle d'origine pour {col}", temperatures, key=f"from_temp_{col}")
            to_scale = st.selectbox(f"Échelle cible pour {col}", temperatures, key=f"to_temp_{col}")
            if from_scale and to_scale:
                try:
                    # Fonction de conversion de température
                    def convert_temperature(value, from_scale, to_scale):
                        if from_scale == "Celsius":
                            if to_scale == "Fahrenheit":
                                return (value * 9/5) + 32
                            elif to_scale == "Kelvin":
                                return value + 273.15
                        elif from_scale == "Fahrenheit":
                            if to_scale == "Celsius":
                                return (value - 32) * 5/9
                            elif to_scale == "Kelvin":
                                return (value - 32) * 5/9 + 273.15
                        elif from_scale == "Kelvin":
                            if to_scale == "Celsius":
                                return value - 273.15
                            elif to_scale == "Fahrenheit":
                                return (value - 273.15) * 9/5 + 32
                        return value

                    df[f"{col}_converti_{to_scale}"] = df[col].apply(lambda x: convert_temperature(x, from_scale, to_scale))
                    st.success(f"Conversion de {from_scale} à {to_scale} effectuée pour {col}")
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de température. {e}")
    return df


def remove_duplicates(df):
    st.markdown("\n")
    st.header("Suppression des doublons")
    cols_to_check_duplicates = st.multiselect(
        "Sélectionnez les colonnes à vérifier pour supprimer les doublons :",
        df.columns.tolist()
    )
    if cols_to_check_duplicates:
        keep_option = st.selectbox("Que souhaitez-vous conserver ?", ["Première occurrence", "Dernière occurrence"])
        original_shape = df.shape[0]
        df = df.drop_duplicates(subset=cols_to_check_duplicates, keep="first" if keep_option == "Première occurrence" else "last")
        total_removed = original_shape - df.shape[0]
        st.success(f"{total_removed} doublons ont été supprimés.")
    return df


@st.cache_resource
def load_spacy_models():
    path_to_utils = Path('utils', 'spacy')
    nlp_en = spacy.util.load_model_from_path(path_to_utils / "en_core_web_sm-3.7.1")
    nlp_fr = spacy.util.load_model_from_path(path_to_utils / "fr_core_news_sm-3.7.0")
    stemmer_en = PorterStemmer()
    stemmer_fr = SnowballStemmer('french')
    return nlp_en, nlp_fr, stemmer_en, stemmer_fr


def lemmatize_text(df, nlp_en, nlp_fr):
    st.markdown("\n")
    st.header("Lemmatisation du texte")
    
    cols_to_lemmatize = st.multiselect(
        "Sélectionnez les colonnes de texte à lemmatiser :",
        df.select_dtypes(include=['object']).columns.tolist(),
        key="lemmatize_cols"
    )

    language_lem = st.selectbox(
        "Choisissez la langue du texte :",
        ["Anglais", "Français"],
        key="lemmatize_language"
    )

    if cols_to_lemmatize:
        for col in cols_to_lemmatize:
            if f"{col}_lemmatized" not in df.columns:
                if language_lem == "Anglais":
                    df[f"{col}_lemmatized"] = df[col].apply(lambda x: " ".join([token.lemma_ for token in nlp_en(str(x))]))
                else:
                    df[f"{col}_lemmatized"] = df[col].apply(lambda x: " ".join([token.lemma_ for token in nlp_fr(str(x))]))
        
        st.success(f"Lemmatisation '{language_lem}' effectuée pour les colonnes : {', '.join(cols_to_lemmatize)}")

    return df

def stemming_text(df, stemmer_en, stemmer_fr):
    st.markdown("\n")
    st.header("Stemming du texte")
    
    cols_to_stem = st.multiselect(
        "Sélectionnez les colonnes de texte à stemmer :",
        df.select_dtypes(include=['object']).columns.tolist(),
        key="stem_cols"
    )

    language_ste = st.selectbox(
        "Choisissez la langue du texte :",
        ["Anglais", "Français"],
        key="stem_language"
    )

    if cols_to_stem:
        for col in cols_to_stem:
            if f"{col}_stemmed" not in df.columns:
                if language_ste == "Anglais":
                    df[f"{col}_stemmed"] = df[col].apply(lambda x: " ".join([stemmer_en.stem(word) for word in str(x).split()]))
                else:
                    df[f"{col}_stemmed"] = df[col].apply(lambda x: " ".join([stemmer_fr.stem(word) for word in str(x).split()]))
        
        st.success(f"Stemming '{language_ste}' effectué pour les colonnes : {', '.join(cols_to_stem)}")

    return df
