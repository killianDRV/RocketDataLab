import streamlit as st
import plotly.express as px
import pandas as pd

def get_numeric_columns(df):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def init_session_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def reset_charts():
    for key in list(st.session_state.keys()):
        if key.endswith('_chart') or key.startswith('last_'):
            del st.session_state[key]

def update_session_state(key):
    st.session_state[key] = st.session_state[f"{key}_select"]

def init_session_states(numeric_cols, categorical_cols):
    init_session_state('hist_col', numeric_cols[0] if numeric_cols else None)
    init_session_state('box_col', numeric_cols[0] if numeric_cols else None)
    init_session_state('scatter_x', numeric_cols[0] if numeric_cols else None)
    init_session_state('scatter_y', numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None)
    init_session_state('bar_x', categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else None)
    init_session_state('bar_y', numeric_cols[0] if numeric_cols else None)
    init_session_state('line_x', numeric_cols[0] if numeric_cols else None)
    init_session_state('line_y', numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None)
    init_session_state('pie_col', categorical_cols[0] if categorical_cols else None)

def create_histogram(df, numeric_cols):
    st.header("Histogramme")
    hist_col = st.selectbox(
        "Choisissez une colonne numérique pour l'histogramme",
        numeric_cols,
        key="hist_col_select",
        index=numeric_cols.index(st.session_state.hist_col) if st.session_state.hist_col in numeric_cols else 0,
        on_change=lambda: update_session_state('hist_col')
    )
    if 'hist_chart' not in st.session_state or st.session_state.hist_col != st.session_state.get('last_hist_col'):
        fig = px.histogram(df, x=st.session_state.hist_col, title=f'Histogramme de {st.session_state.hist_col}')
        st.session_state.hist_chart = fig
        st.session_state.last_hist_col = st.session_state.hist_col
    st.plotly_chart(st.session_state.hist_chart, use_container_width=True)

def create_box_plot(df, numeric_cols):
    st.header("Boîte à moustaches")
    box_col = st.selectbox(
        "Choisissez une colonne numérique pour la boîte à moustaches",
        numeric_cols,
        key="box_col_select",
        index=numeric_cols.index(st.session_state.box_col) if st.session_state.box_col in numeric_cols else 0,
        on_change=lambda: update_session_state('box_col')
    )
    if 'box_chart' not in st.session_state or st.session_state.box_col != st.session_state.get('last_box_col'):
        fig = px.box(df, y=st.session_state.box_col, title=f'Boîte à moustaches de {st.session_state.box_col}')
        st.session_state.box_chart = fig
        st.session_state.last_box_col = st.session_state.box_col
    st.plotly_chart(st.session_state.box_chart, use_container_width=True)

def create_scatter_plot(df, numeric_cols):
    st.header("Nuage de points")
    col1, col2 = st.columns(2)
    with col1:
        scatter_x = st.selectbox(
            "Choisissez la colonne X pour le nuage de points",
            numeric_cols,
            key="scatter_x_select",
            index=numeric_cols.index(st.session_state.scatter_x) if st.session_state.scatter_x in numeric_cols else 0,
            on_change=lambda: update_session_state('scatter_x')
        )
    with col2:
        scatter_y = st.selectbox(
            "Choisissez la colonne Y pour le nuage de points",
            numeric_cols,
            key="scatter_y_select",
            index=numeric_cols.index(st.session_state.scatter_y) if st.session_state.scatter_y in numeric_cols else 0,
            on_change=lambda: update_session_state('scatter_y')
        )
    if 'scatter_chart' not in st.session_state or st.session_state.scatter_x != st.session_state.get('last_scatter_x') or st.session_state.scatter_y != st.session_state.get('last_scatter_y'):
        fig = px.scatter(df, x=st.session_state.scatter_x, y=st.session_state.scatter_y, title=f'Nuage de points : {st.session_state.scatter_x} vs {st.session_state.scatter_y}')
        st.session_state.scatter_chart = fig
        st.session_state.last_scatter_x = st.session_state.scatter_x
        st.session_state.last_scatter_y = st.session_state.scatter_y
    st.plotly_chart(st.session_state.scatter_chart, use_container_width=True)

def create_bar_chart(df, all_cols, numeric_cols):
    st.header("Diagramme en barres")
    col1, col2 = st.columns(2)
    with col1:
        bar_x = st.selectbox(
            "Choisissez la colonne X pour le diagramme en barres",
            all_cols,
            key="bar_x_select",
            index=all_cols.index(st.session_state.bar_x) if st.session_state.bar_x in all_cols else 0,
            on_change=lambda: update_session_state('bar_x')
        )
    with col2:
        bar_y = st.selectbox(
            "Choisissez la colonne Y (numérique) pour le diagramme en barres",
            numeric_cols,
            key="bar_y_select",
            index=numeric_cols.index(st.session_state.bar_y) if st.session_state.bar_y in numeric_cols else 0,
            on_change=lambda: update_session_state('bar_y')
        )
    if 'bar_chart' not in st.session_state or st.session_state.bar_x != st.session_state.get('last_bar_x') or st.session_state.bar_y != st.session_state.get('last_bar_y'):
        fig = px.bar(df, x=st.session_state.bar_x, y=st.session_state.bar_y, title=f'Diagramme en barres : {st.session_state.bar_x} vs {st.session_state.bar_y}')
        st.session_state.bar_chart = fig
        st.session_state.last_bar_x = st.session_state.bar_x
        st.session_state.last_bar_y = st.session_state.bar_y
    st.plotly_chart(st.session_state.bar_chart, use_container_width=True)

def create_line_chart(df, numeric_cols):
    st.header("Graphique linéaire")
    col1, col2 = st.columns(2)
    with col1:
        line_x = st.selectbox(
            "Choisissez la colonne X pour le graphique linéaire",
            numeric_cols,
            key="line_x_select",
            index=numeric_cols.index(st.session_state.line_x) if st.session_state.line_x in numeric_cols else 0,
            on_change=lambda: update_session_state('line_x')
        )
    with col2:
        line_y = st.selectbox(
            "Choisissez la colonne Y pour le graphique linéaire",
            numeric_cols,
            key="line_y_select",
            index=numeric_cols.index(st.session_state.line_y) if st.session_state.line_y in numeric_cols else 0,
            on_change=lambda: update_session_state('line_y')
        )
    if 'line_chart' not in st.session_state or st.session_state.line_x != st.session_state.get('last_line_x') or st.session_state.line_y != st.session_state.get('last_line_y'):
        fig = px.line(df, x=st.session_state.line_x, y=st.session_state.line_y, title=f'Graphique linéaire : {st.session_state.line_x} vs {st.session_state.line_y}')
        st.session_state.line_chart = fig
        st.session_state.last_line_x = st.session_state.line_x
        st.session_state.last_line_y = st.session_state.line_y
    st.plotly_chart(st.session_state.line_chart, use_container_width=True)

def create_pie_chart(df, categorical_cols):
    st.header("Diagramme circulaire")
    pie_col = st.selectbox(
        "Choisissez une colonne catégorielle pour le diagramme circulaire",
        categorical_cols,
        key="pie_col_select",
        index=categorical_cols.index(st.session_state.pie_col) if st.session_state.pie_col in categorical_cols else 0,
        on_change=lambda: update_session_state('pie_col')
    )
    if 'pie_chart' not in st.session_state or st.session_state.pie_col != st.session_state.get('last_pie_col'):
        value_counts = df[st.session_state.pie_col].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Diagramme circulaire de {st.session_state.pie_col}')
        st.session_state.pie_chart = fig
        st.session_state.last_pie_col = st.session_state.pie_col
    st.plotly_chart(st.session_state.pie_chart, use_container_width=True)

def create_correlation_matrix(df, numeric_cols):
    st.header("Matrice de corrélation")
    if 'corr_matrix' not in st.session_state:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                        title="Matrice de corrélation",
                        color_continuous_scale="RdBu_r")
        st.session_state.corr_matrix = fig
    st.plotly_chart(st.session_state.corr_matrix, use_container_width=True)
