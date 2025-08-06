"""
pages/2_Univariate_Analysis.py
Analisi univariata delle singole variabili del dataset Titanic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import PAGE_CONFIG, COLUMN_LABELS, COLOR_PALETTES
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, detect_outliers_iqr, handle_outliers
from src.components.charts import create_age_distribution_chart, create_survival_overview_chart
from src.components.univariate_charts import *
from src.utils.log import logger

# Logger per l'ingresso del file
logger.info(f"Caricamento pagina {__name__}")

# ----------------1. Configurazione pagina (da config.py)
def setup_page():
    """Configura la pagina Streamlit"""
    logger.info("Configurazione pagina Streamlit")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Caricamento e pulizia dati (da notebook sezione 2.1 e 3)
logger.info("Caricamento dati Titanic")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Impossibile caricare i dati Titanic")
    st.error("Impossibile caricare i dati")
    st.stop()

logger.info("Pulizia dati base")
df = clean_dataset_basic(df_original)
logger.info(f"Dati puliti. Shape: {df.shape}")

# ----------------3. Header pagina
logger.info("Setup header pagina")
st.title("Analisi Univariata - Variabili Singole")
st.markdown("Esplorazione dettagliata delle caratteristiche individuali di ogni variabile")

# ----------------4. Sidebar controlli
logger.info("Setup sidebar controlli")
with st.sidebar:
    st.header("Controlli Analisi")
    
    # Selezione variabile da analizzare
    numeric_variables = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col not in ['PassengerId']]
    categorical_variables = [col for col in df.select_dtypes(include=['object']).columns 
                           if col not in ['Name', 'Ticket']]
    
    analysis_type = st.selectbox(
        "Tipo di Analisi:",
        ["Panoramica Generale", "Variabili Numeriche", "Variabili Categoriche", "Focus su Età", "Focus su Sopravvivenza"]
    )
    logger.debug(f"Tipo analisi selezionato: {analysis_type}")
    
    # Opzioni visualizzazione
    show_statistics = st.checkbox("Mostra statistiche dettagliate", value=True)
    show_outliers = st.checkbox("Analisi outliers", value=True)
    logger.debug(f"Opzioni visualizzazione: stats={show_statistics}, outliers={show_outliers}")

# ----------------5. Panoramica Generale (da notebook sezione 4.1.1)
if analysis_type == "Panoramica Generale":
    logger.info("Avvio analisi panoramica generale")
    st.header("1. Panoramica Generale Dataset")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Passeggeri Totali", f"{len(df):,}")
    with col2:
        st.metric("Variabili Numeriche", len(numeric_variables))
    with col3:
        st.metric("Variabili Categoriche", len(categorical_variables))
    with col4:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Tasso Sopravvivenza", f"{survival_rate:.1f}%")
        logger.debug(f"Tasso sopravvivenza: {survival_rate:.1f}%")
    
    # ----------------6. Distribuzioni principali
    logger.info("Creazione distribuzioni principali")
    st.subheader("Distribuzioni Principali")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico distribuzione età")
        fig_age = create_age_distribution_detailed(df)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico distribuzione sopravvivenza")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)
    
    # ----------------7. Summary statistiche tutte variabili numeriche
    if show_statistics:
        logger.info("Calcolo statistiche descrittive")
        st.subheader("Statistiche Descrittive Variabili Numeriche")
        stats_df = df[numeric_variables].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

# ----------------8. Analisi Variabili Numeriche (da notebook sezione 4.2.1)
elif analysis_type == "Variabili Numeriche":
    logger.info("Avvio analisi variabili numeriche")
    st.header("2. Analisi Variabili Numeriche")
    
    # Selezione variabile specifica
    selected_var = st.selectbox(
        "Seleziona variabile numerica:",
        numeric_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    logger.debug(f"Variabile numerica selezionata: {selected_var}")
    
    if selected_var:
        st.subheader(f"Analisi dettagliata: {COLUMN_LABELS.get(selected_var, selected_var)}")
        
        # ----------------9. Statistiche base (da notebook sezione 4.2.1 - Age analysis)
        col1, col2, col3, col4 = st.columns(4)
        
        var_data = df[selected_var].dropna()
        
        with col1:
            mean_val = var_data.mean()
            st.metric("Media", f"{mean_val:.2f}")
        with col2:
            median_val = var_data.median()
            st.metric("Mediana", f"{median_val:.2f}")
        with col3:
            std_val = var_data.std()
            st.metric("Deviazione Standard", f"{std_val:.2f}")
        with col4:
            unique_val = var_data.nunique()
            st.metric("Valori Unici", f"{unique_val}")
        
        logger.debug(f"Statistiche {selected_var}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}, uniques={unique_val}")
        
        # ----------------10. Visualizzazioni multiple (da notebook sezione 4.2.1)
        logger.debug("Creazione grafici analisi numerica")
        fig = create_numerical_analysis_charts(df, selected_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # ----------------11. Analisi outliers (da notebook sezione 4.2.1 - Outlier detection)
        if show_outliers:
            logger.info("Analisi outliers")
            st.subheader("Analisi Outliers")
            
            outliers, lower_bound, upper_bound = detect_outliers_iqr(var_data)
            logger.debug(f"Outliers rilevati: {len(outliers)}, bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outliers Rilevati", len(outliers))
            with col2:
                st.metric("Limite Inferiore", f"{lower_bound:.2f}")
            with col3:
                st.metric("Limite Superiore", f"{upper_bound:.2f}")
            
            if len(outliers) > 0:
                st.write("**Valori outliers:**")
                outliers_df = pd.DataFrame({
                    'Indice': outliers.index,
                    'Valore': outliers.values
                }).sort_values('Valore')
                st.dataframe(outliers_df, use_container_width=True)

# ----------------12. Analisi Variabili Categoriche (da notebook sezione 4.2.2)
elif analysis_type == "Variabili Categoriche":
    logger.info("Avvio analisi variabili categoriche")
    st.header("3. Analisi Variabili Categoriche")
    
    # Selezione variabile categorica
    selected_cat_var = st.selectbox(
        "Seleziona variabile categorica:",
        categorical_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    logger.debug(f"Variabile categorica selezionata: {selected_cat_var}")
    
    if selected_cat_var:
        st.subheader(f"Analisi dettagliata: {COLUMN_LABELS.get(selected_cat_var, selected_cat_var)}")
        
        # ----------------13. Statistiche categoriche (da notebook sezione 4.2.2)
        value_counts = df[selected_cat_var].value_counts()
        value_props = df[selected_cat_var].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Categorie Uniche", len(value_counts))
        with col2:
            st.metric("Categoria Più Frequente", value_counts.index[0])
        with col3:
            st.metric("Frequenza Massima", f"{value_props.iloc[0]:.1f}%")
        
        logger.debug(f"Statistiche {selected_cat_var}: {len(value_counts)} categorie, top={value_counts.index[0]}, freq={value_props.iloc[0]:.1f}%")
        
        # ----------------14. Tabella frequenze
        freq_table = pd.DataFrame({
            'Categoria': value_counts.index,
            'Conteggio': value_counts.values,
            'Percentuale': value_props.values.round(1)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tabella Frequenze:**")
            st.dataframe(freq_table, use_container_width=True)
        
        with col2:
            # ----------------15. Visualizzazione categorica
            logger.debug("Creazione grafico analisi categorica")
            fig_cat = create_categorical_analysis_chart(df, selected_cat_var)
            st.plotly_chart(fig_cat, use_container_width=True)

# ----------------16. Focus su Età (da notebook sezione 4.2.1 - Age Analysis completa)
elif analysis_type == "Focus su Età":
    logger.info("Avvio analisi focus età")
    st.header("4. Analisi Approfondita dell'Età")
    
    # ----------------17. Statistiche età (da notebook sezione 4.2.1)
    age_data = df['Age'].dropna()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mean_age = age_data.mean()
        st.metric("Età Media", f"{mean_age:.1f} anni")
    with col2:
        median_age = age_data.median()
        st.metric("Età Mediana", f"{median_age:.1f} anni")
    with col3:
        min_age = age_data.min()
        st.metric("Età Minima", f"{min_age:.1f} anni")
    with col4:
        max_age = age_data.max()
        st.metric("Età Massima", f"{max_age:.1f} anni")
    with col5:
        std_age = age_data.std()
        st.metric("Deviazione Standard", f"{std_age:.1f} anni")
    
    logger.debug(f"Statistiche età: mean={mean_age:.1f}, median={median_age:.1f}, min={min_age:.1f}, max={max_age:.1f}, std={std_age:.1f}")
    
    # ----------------18. Visualizzazioni età multiple (da notebook sezione 4.2.1)
    logger.debug("Creazione grafici analisi età")
    fig_age_complete = create_age_complete_analysis(df)
    st.plotly_chart(fig_age_complete, use_container_width=True)
    
    # ----------------19. Gruppi di età (da notebook sezione 4.2.2.4)
    st.subheader("Analisi per Gruppi di Età")
    
    # Crea gruppi età
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 25, 40, 100], 
                            labels=['Bambini (0-12)', 'Giovani (13-25)', 'Adulti (26-40)', 'Anziani (41+)'])
    
    age_group_stats = df['Age_Group'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuzione per Gruppi:**")
        age_groups_df = pd.DataFrame({
            'Gruppo': age_group_stats.index,
            'Conteggio': age_group_stats.values,
            'Percentuale': (age_group_stats.values / len(df) * 100).round(1)
        })
        st.dataframe(age_groups_df, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico gruppi età")
        fig_age_groups = px.bar(
            x=age_group_stats.index,
            y=age_group_stats.values,
            title="Distribuzione per Gruppi di Età",
            labels={'x': 'Gruppo Età', 'y': 'Numero Passeggeri'}
        )
        st.plotly_chart(fig_age_groups, use_container_width=True)
    
    # ----------------20. Trattamento outliers età (da notebook sezione 4.2.1)
    if show_outliers:
        logger.info("Analisi outliers età")
        st.subheader("Gestione Outliers Età")
        
        outliers, lower_bound, upper_bound = detect_outliers_iqr(age_data)
        logger.debug(f"Outliers età rilevati: {len(outliers)}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Outliers rilevati:** {len(outliers)}")
            st.write(f"**Limite inferiore:** {lower_bound:.1f} anni")
            st.write(f"**Limite superiore:** {upper_bound:.1f} anni")
            
            if len(outliers) > 0:
                outliers_list = sorted(outliers.values)
                st.write("**Età outliers:**", outliers_list)
        
        with col2:
            # Opzioni trattamento outliers
            outlier_method = st.selectbox(
                "Metodo trattamento outliers:",
                ["Nessuno", "Rimozione", "Sostituzione con mediana", "Clipping ai limiti"]
            )
            logger.debug(f"Metodo trattamento outliers selezionato: {outlier_method}")
            
            if outlier_method != "Nessuno":
                if outlier_method == "Rimozione":
                    df_processed = handle_outliers(df, method='remove', columns=['Age'])
                elif outlier_method == "Sostituzione con mediana":
                    df_processed = handle_outliers(df, method='replace_median', columns=['Age'])
                else:  # Clipping
                    df_processed = handle_outliers(df, method='clip', columns=['Age'])
                
                st.write(f"**Dataset dopo trattamento:** {len(df_processed)} righe")
                logger.debug(f"Dati dopo trattamento outliers: {len(df_processed)} righe")
                
                # Confronto distribuzioni
                logger.debug("Creazione grafico confronto outliers")
                fig_comparison = create_outlier_comparison_chart(df, df_processed, 'Age')
                st.plotly_chart(fig_comparison, use_container_width=True)

# ----------------21. Focus su Sopravvivenza (da notebook sezione 4.2.2 - Survival Analysis)
elif analysis_type == "Focus su Sopravvivenza":
    logger.info("Avvio analisi focus sopravvivenza")
    st.header("5. Analisi Approfondita della Sopravvivenza")
    
    # ----------------22. Statistiche sopravvivenza (da notebook sezione 4.2.2)
    survival_stats = df['Survived'].value_counts()
    survival_props = df['Survived'].value_counts(normalize=True) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totale Passeggeri", len(df))
    with col2:
        survived = int(survival_stats[1])
        st.metric("Sopravvissuti", survived)
    with col3:
        died = int(survival_stats[0])
        st.metric("Morti", died)
    with col4:
        survival_rate = survival_props[1]
        st.metric("Tasso Sopravvivenza", f"{survival_rate:.1f}%")
    
    logger.debug(f"Statistiche sopravvivenza: total={len(df)}, survived={survived}, died={died}, rate={survival_rate:.1f}%")
    
    # ----------------23. Visualizzazioni sopravvivenza (da notebook sezione 4.2.2)
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico a torta sopravvivenza")
        fig_pie = create_survival_overview_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico a barre sopravvivenza")
        fig_bar = px.bar(
            x=['Non Sopravvissuti', 'Sopravvissuti'],
            y=[died, survived],
            title="Conteggio Sopravvivenza",
            color=['Non Sopravvissuti', 'Sopravvissuti'],
            color_discrete_map={'Non Sopravvissuti': COLOR_PALETTES['survival'][0], 
                               'Sopravvissuti': COLOR_PALETTES['survival'][1]}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ----------------24. Analisi distribuzione sopravvivenza
    st.subheader("Distribuzione Sopravvivenza nel Dataset")
    
    survival_interpretation = f"""
    **Interpretazione dei risultati:**
    
    - **Tasso di mortalità**: {survival_props[0]:.1f}% dei passeggeri non è sopravvissuto
    - **Tasso di sopravvivenza**: {survival_props[1]:.1f}% dei passeggeri è sopravvissuto
    - **Differenza**: La probabilità di morte era {survival_props[0]/survival_props[1]:.1f} volte superiore alla sopravvivenza
    
    Questo dato conferma la gravità del disastro del Titanic, dove la maggioranza dei passeggeri ha perso la vita.
    """
    
    st.markdown(survival_interpretation)

# ----------------25. Note metodologiche
with st.expander("Note Metodologiche"):
    st.markdown("""
    **Metodologia di analisi basata su:**
    
    - **Sezione 4.1.1 del notebook**: Statistiche descrittive generali
    - **Sezione 4.2.1 del notebook**: Analisi univariata dell'età
    - **Sezione 4.2.2 del notebook**: Analisi della sopravvivenza
    - **Sezione 4.1.1 del notebook**: Rilevamento e gestione outliers
    
    **Tecniche utilizzate:**
    - Statistiche descrittive (media, mediana, deviazione standard)
    - Visualizzazioni multiple (istogrammi, boxplot, grafici a torta)
    - Rilevamento outliers con metodo IQR
    - Analisi delle frequenze per variabili categoriche
    - Creazione di gruppi di età per analisi segmentata
    
    **Metriche calcolate:**
    - Percentili (25%, 50%, 75%)
    - Limiti per outliers (Q1-1.5*IQR, Q3+1.5*IQR)
    - Distribuzioni di frequenza
    - Tassi di sopravvivenza
    """)

logger.info(f"Pagina {__name__} completata con successo")