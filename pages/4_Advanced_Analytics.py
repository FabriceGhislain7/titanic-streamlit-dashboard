"""
pages/4_Advanced_Analytics.py
Analisi avanzate, correlazioni e feature engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, create_basic_features, handle_outliers
from src.components.advanced_charts import *
from src.utils.feature_engineering import *
from src.utils.statistical_analysis import *
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

# ----------------1. Configurazione pagina (da config.py)
def setup_page():
    """Configura la pagina Streamlit"""
    logger.info("Configurazione pagina Streamlit")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Caricamento e preparazione dati (da notebook sezioni 2.1, 3, e feature engineering)
logger.info("Caricamento dati Titanic")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Impossibile caricare i dati Titanic")
    st.error("Impossibile caricare i dati")
    st.stop()

logger.info("Pulizia dati base")
df = clean_dataset_basic(df_original)
logger.info("Creazione feature base")
df = create_basic_features(df)
logger.info(f"Dati preparati. Shape: {df.shape}")

# ----------------3. Header pagina
logger.info("Setup header pagina")
st.title("Analisi Avanzate e Feature Engineering")
st.markdown("Correlazioni approfondite, feature engineering e analisi statistiche avanzate")

# ----------------4. Sidebar controlli
logger.info("Setup sidebar controlli")
with st.sidebar:
    st.header("Controlli Analisi Avanzate")
    
    analysis_section = st.selectbox(
        "Sezione Analisi:",
        [
            "Matrice Correlazioni Avanzata",
            "Feature Engineering",
            "Analisi Outliers Avanzata", 
            "Analisi Statistica",
            "Pattern Mining",
            "Segmentazione Avanzata"
        ]
    )
    logger.debug(f"Sezione analisi selezionata: {analysis_section}")
    
    # Opzioni feature engineering
    st.subheader("Feature Engineering")
    create_title_feature = st.checkbox("Estrai titolo dal nome", value=True)
    create_deck_feature = st.checkbox("Estrai deck dalla cabina", value=True)
    create_fare_per_person = st.checkbox("Calcola tariffa per persona", value=True)
    logger.debug(f"Opzioni feature engineering: title={create_title_feature}, deck={create_deck_feature}, fare_per_person={create_fare_per_person}")
    
    # Opzioni outliers
    st.subheader("Gestione Outliers")
    outlier_method = st.selectbox(
        "Metodo outliers:",
        ["Nessuno", "Rimozione", "Clipping", "Sostituzione"],
        index=0
    )
    logger.debug(f"Metodo outliers selezionato: {outlier_method}")

# ----------------5. Applicazione feature engineering avanzate
logger.info("Applicazione feature engineering avanzate")
df_engineered = df.copy()

if create_title_feature:
    logger.debug("Estrazione titolo dal nome")
    df_engineered = extract_title_from_name(df_engineered)

if create_deck_feature:
    logger.debug("Estrazione deck dalla cabina")
    df_engineered = extract_deck_from_cabin(df_engineered)

if create_fare_per_person:
    logger.debug("Calcolo tariffa per persona")
    df_engineered = calculate_fare_per_person(df_engineered)

# Gestione outliers se richiesta
if outlier_method != "Nessuno":
    logger.info(f"Applicazione trattamento outliers: {outlier_method}")
    method_map = {
        "Rimozione": "remove",
        "Clipping": "clip", 
        "Sostituzione": "replace_median"
    }
    df_engineered = handle_outliers(df_engineered, method=method_map[outlier_method])
    logger.debug(f"Dati dopo trattamento outliers. Shape: {df_engineered.shape}")
else:
    logger.debug("Nessun trattamento outliers applicato")

# ----------------6. Matrice Correlazioni Avanzata (da notebook sezione 4.1.2)
if analysis_section == "Matrice Correlazioni Avanzata":
    logger.info("Avvio analisi correlazioni avanzate")
    st.header("1. Analisi Correlazioni Avanzate")
    
    # ----------------7. Correlazioni multiple (da notebook sezione 4.1.2 - Spearman correlation)
    logger.info("Creazione matrici correlazione multiple")
    st.subheader("Matrici di Correlazione Multiple")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione correlazione Pearson")
        # Correlazione Pearson
        st.write("**Correlazione Pearson (lineare):**")
        fig_pearson = create_correlation_matrix(df_engineered, method='pearson')
        st.plotly_chart(fig_pearson, use_container_width=True)
    
    with col2:
        logger.debug("Creazione correlazione Spearman")
        # Correlazione Spearman
        st.write("**Correlazione Spearman (monotonica):**")
        fig_spearman = create_correlation_matrix(df_engineered, method='spearman')
        st.plotly_chart(fig_spearman, use_container_width=True)
    
    # ----------------8. Top correlazioni con target (da notebook focus sopravvivenza)
    logger.info("Calcolo correlazioni con target")
    st.subheader("Correlazioni con Sopravvivenza")
    
    correlations_with_target = calculate_target_correlations(df_engineered, 'Survived')
    if correlations_with_target is not None:
        logger.debug("Visualizzazione top correlazioni")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Correlazioni Positive:**")
            st.dataframe(correlations_with_target.head(), use_container_width=True)
        
        with col2:
            logger.debug("Creazione grafico correlazioni target")
            # Grafico correlazioni
            fig_target_corr = create_target_correlation_chart(correlations_with_target)
            st.plotly_chart(fig_target_corr, use_container_width=True)
    else:
        logger.debug("Correlazioni con target non disponibili")
    
    # ----------------9. Correlazioni per categorie (approfondimento)
    logger.debug("Creazione correlazioni per categoria")
    st.subheader("Correlazioni per Categoria")
    
    # Correlazioni separate per genere
    fig_corr_by_gender = create_correlation_by_category(df_engineered, 'Sex')
    st.plotly_chart(fig_corr_by_gender, use_container_width=True)

# ----------------10. Feature Engineering (combinazione notebook + nuove features)
elif analysis_section == "Feature Engineering":
    logger.info("Avvio sezione feature engineering")
    st.header("2. Feature Engineering Avanzato")
    
    # ----------------11. Riepilogo features create
    logger.info("Creazione riepilogo features")
    st.subheader("Features Esistenti vs Nuove")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features Originali:**")
        original_features = [col for col in df.columns if col in df_original.columns]
        logger.debug(f"Features originali: {len(original_features)}")
        for feature in original_features:
            st.write(f"- {COLUMN_LABELS.get(feature, feature)}")
    
    with col2:
        st.write("**Features Engineered:**")
        new_features = [col for col in df_engineered.columns if col not in df_original.columns]
        logger.debug(f"Features engineered: {len(new_features)}")
        for feature in new_features:
            st.write(f"- {feature}")
    
    # ----------------12. Analisi nuove features (da notebook + aggiunte)
    logger.info("Analisi features engineered")
    st.subheader("Analisi Features Engineered")
    
    # Title analysis (da nome)
    if 'Title' in df_engineered.columns:
        logger.debug("Analisi feature Title")
        st.write("**Analisi Titolo (estratto dal Nome):**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title_survival = df_engineered.groupby('Title')['Survived'].agg(['count', 'mean']).round(3)
            title_survival.columns = ['Conteggio', 'Tasso_Sopravvivenza']
            st.dataframe(title_survival, use_container_width=True)
        
        with col2:
            fig_title = create_title_survival_analysis(df_engineered)
            st.plotly_chart(fig_title, use_container_width=True)
    else:
        logger.debug("Feature Title non disponibile")
    
    # Deck analysis (da cabina)
    if 'Deck' in df_engineered.columns:
        logger.debug("Analisi feature Deck")
        st.write("**Analisi Deck (estratto dalla Cabina):**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            deck_survival = df_engineered.groupby('Deck')['Survived'].agg(['count', 'mean']).round(3)
            deck_survival.columns = ['Conteggio', 'Tasso_Sopravvivenza']
            st.dataframe(deck_survival, use_container_width=True)
        
        with col2:
            fig_deck = create_deck_survival_analysis(df_engineered)
            st.plotly_chart(fig_deck, use_container_width=True)
    else:
        logger.debug("Feature Deck non disponibile")
    
    # ----------------13. Feature importance approximation
    logger.info("Calcolo importanza features")
    st.subheader("Importanza Features (Approssimata)")
    
    feature_importance = calculate_feature_importance_proxy(df_engineered, 'Survived')
    if feature_importance is not None:
        logger.debug("Visualizzazione importanza features")
        fig_importance = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        logger.debug("Importanza features non disponibile")

# ----------------14. Analisi Outliers Avanzata (da notebook sezione 4.1.1 estesa)
elif analysis_section == "Analisi Outliers Avanzata":
    logger.info("Avvio analisi outliers avanzata")
    st.header("3. Analisi Outliers Avanzata")
    
    # ----------------15. Rilevamento outliers multivariato
    logger.info("Rilevamento outliers multivariato")
    st.subheader("Rilevamento Outliers Multivariato")
    
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId']]
    logger.debug(f"Variabili numeriche disponibili: {len(numeric_cols)}")
    
    if len(numeric_cols) > 1:
        selected_vars = st.multiselect(
            "Seleziona variabili per analisi outliers:",
            numeric_cols,
            default=numeric_cols[:3],
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        logger.debug(f"Variabili selezionate per outliers: {selected_vars}")
        
        if len(selected_vars) >= 2:
            logger.debug("Creazione visualizzazioni outliers")
            # ----------------16. Scatter plot outliers (da notebook esteso)
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter_outliers = create_outliers_scatter_plot(df_engineered, selected_vars[0], selected_vars[1])
                st.plotly_chart(fig_scatter_outliers, use_container_width=True)
            
            with col2:
                # Boxplot comparison
                fig_outliers_comparison = create_outliers_comparison_boxplot(df_engineered, selected_vars)
                st.plotly_chart(fig_outliers_comparison, use_container_width=True)
        else:
            logger.debug("Numero insufficiente di variabili selezionate per outliers")
    else:
        logger.debug("Numero insufficiente di variabili numeriche per outliers")
    
    # ----------------17. Impatto outliers su correlazioni
    st.subheader("Impatto Outliers su Correlazioni")
    
    if outlier_method != "Nessuno":
        logger.info("Analisi impatto outliers su correlazioni")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prima del trattamento:**")
            corr_before = df.select_dtypes(include=[np.number]).corr()['Survived'].abs().sort_values(ascending=False)
            st.dataframe(corr_before.head(10), use_container_width=True)
        
        with col2:
            st.write("**Dopo il trattamento:**")
            corr_after = df_engineered.select_dtypes(include=[np.number]).corr()['Survived'].abs().sort_values(ascending=False)
            st.dataframe(corr_after.head(10), use_container_width=True)
    else:
        logger.debug("Nessun trattamento outliers per confronto correlazioni")

# ----------------18. Analisi Statistica (test statistici e distribuzioni)
elif analysis_section == "Analisi Statistica":
    logger.info("Avvio analisi statistica avanzata")
    st.header("4. Analisi Statistica Avanzata")
    
    # ----------------19. Test normalità (estensione analisi notebook)
    logger.info("Test di normalità")
    st.subheader("Test di Normalità")
    
    numeric_vars = [col for col in df_engineered.select_dtypes(include=[np.number]).columns 
                   if col not in ['PassengerId', 'Survived']]
    logger.debug(f"Variabili numeriche per test normalità: {len(numeric_vars)}")
    
    if numeric_vars:
        selected_var = st.selectbox(
            "Seleziona variabile per test normalità:",
            numeric_vars,
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        logger.debug(f"Variabile selezionata per normalità: {selected_var}")
        
        # ----------------20. Visualizzazioni normalità
        logger.debug("Creazione visualizzazioni normalità")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_normality = create_normality_test_plots(df_engineered, selected_var)
            st.plotly_chart(fig_normality, use_container_width=True)
        
        with col2:
            # Statistiche normalità
            logger.debug("Calcolo statistiche normalità")
            normality_stats = calculate_normality_statistics(df_engineered, selected_var)
            st.write("**Statistiche di Normalità:**")
            for stat_name, stat_value in normality_stats.items():
                st.metric(stat_name, f"{stat_value:.4f}")
    else:
        logger.debug("Nessuna variabile numerica disponibile per test normalità")
    
    # ----------------21. Analisi distribuzioni per gruppo
    logger.info("Confronto distribuzioni per gruppi")
    st.subheader("Confronto Distribuzioni per Gruppi")
    
    categorical_vars = ['Sex', 'Pclass']
    if 'Title' in df_engineered.columns:
        categorical_vars.append('Title')
    
    cat_var = st.selectbox("Variabile di raggruppamento:", categorical_vars)
    num_var = st.selectbox("Variabile numerica:", numeric_vars, index=1)
    logger.debug(f"Confronto distribuzioni: {num_var} per {cat_var}")
    
    fig_dist_comparison = create_distribution_comparison_by_group(df_engineered, num_var, cat_var)
    st.plotly_chart(fig_dist_comparison, use_container_width=True)

# ----------------22. Pattern Mining (ricerca pattern interessanti)
elif analysis_section == "Pattern Mining":
    logger.info("Avvio pattern mining")
    st.header("5. Pattern Mining e Insights")
    
    # ----------------23. Pattern sopravvivenza (combinazioni interessanti)
    logger.info("Ricerca pattern sopravvivenza")
    st.subheader("Pattern di Sopravvivenza")
    
    survival_patterns = discover_survival_patterns(df_engineered)
    if survival_patterns is not None:
        logger.debug("Visualizzazione pattern sopravvivenza")
        st.dataframe(survival_patterns, use_container_width=True)
    else:
        logger.debug("Pattern sopravvivenza non disponibili")
    
    # ----------------24. Anomalie interessanti
    logger.info("Ricerca anomalie interessanti")
    st.subheader("Casi Anomali Interessanti")
    
    # Trova passeggeri con caratteristiche inusuali
    anomalies = find_interesting_anomalies(df_engineered)
    if anomalies is not None:
        logger.debug("Visualizzazione anomalie")
        st.write("**Passeggeri con caratteristiche inusuali:**")
        st.dataframe(anomalies[['Name', 'Sex', 'Age', 'Pclass', 'Fare', 'Survived']], use_container_width=True)
    else:
        logger.debug("Anomalie interessanti non trovate")
    
    # ----------------25. Combinazioni rare ma significative
    logger.info("Ricerca combinazioni rare ma significative")
    st.subheader("Combinazioni Rare ma Significative")
    
    rare_combinations = find_rare_but_significant_combinations(df_engineered)
    if rare_combinations is not None:
        logger.debug("Visualizzazione combinazioni rare")
        for combination, stats in rare_combinations.items():
            st.write(f"**{combination}:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conteggio", stats['count'])
            with col2:
                st.metric("Tasso Sopravvivenza", f"{stats['survival_rate']:.1f}%")
            with col3:
                st.metric("Significatività", stats['significance'])
    else:
        logger.debug("Combinazioni rare non trovate")

# ----------------26. Segmentazione Avanzata (cluster analisi)
elif analysis_section == "Segmentazione Avanzata":
    logger.info("Avvio segmentazione avanzata")
    st.header("6. Segmentazione Avanzata")
    
    # ----------------27. Segmentazione basata su caratteristiche multiple
    logger.info("Creazione segmenti passeggeri")
    st.subheader("Segmenti di Passeggeri")
    
    segments = create_passenger_segments(df_engineered)
    if segments is not None:
        logger.debug("Analisi segmenti creati")
        # Aggiungi segmenti al dataframe
        df_with_segments = df_engineered.copy()
        df_with_segments['Segment'] = segments
        
        # Analisi segmenti
        segment_analysis = analyze_segments(df_with_segments)
        st.dataframe(segment_analysis, use_container_width=True)
        
        # ----------------28. Visualizzazione segmenti
        logger.debug("Creazione visualizzazioni segmenti")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_segments_survival = create_segments_survival_chart(df_with_segments)
            st.plotly_chart(fig_segments_survival, use_container_width=True)
        
        with col2:
            fig_segments_dist = create_segments_distribution_chart(df_with_segments)
            st.plotly_chart(fig_segments_dist, use_container_width=True)
    else:
        logger.debug("Segmenti passeggeri non creati")
    
    # ----------------29. Analisi RFM-style (Recency, Frequency, Monetary adapting to Titanic)
    logger.info("Creazione profili Age-Fare-Class")
    st.subheader("Profili Passeggeri (Age-Fare-Class)")
    
    afc_profiles = create_age_fare_class_profiles(df_engineered)
    if afc_profiles is not None:
        logger.debug("Analisi profili AFC")
        df_with_profiles = df_engineered.copy()
        df_with_profiles['Profile'] = afc_profiles
        
        profile_survival = df_with_profiles.groupby('Profile')['Survived'].agg(['count', 'mean']).round(3)
        profile_survival.columns = ['Conteggio', 'Tasso_Sopravvivenza']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(profile_survival, use_container_width=True)
        
        with col2:
            fig_profiles = create_profiles_chart(profile_survival)
            st.plotly_chart(fig_profiles, use_container_width=True)
    else:
        logger.debug("Profili AFC non creati")

# ----------------30. Summary e insights finali
logger.info("Creazione summary analisi avanzate")
st.header("Summary Analisi Avanzate")

# ----------------31. Metriche chiave dall'analisi avanzata
col1, col2, col3, col4 = st.columns(4)

with col1:
    original_features = len([col for col in df.columns if col in df_original.columns])
    st.metric("Features Originali", original_features)

with col2:
    engineered_features = len([col for col in df_engineered.columns if col not in df_original.columns])
    st.metric("Features Engineered", engineered_features)

with col3:
    if outlier_method != "Nessuno":
        outliers_handled = len(df) - len(df_engineered)
        st.metric("Outliers Gestiti", outliers_handled)
        logger.debug(f"Outliers gestiti: {outliers_handled}")
    else:
        st.metric("Outliers Gestiti", 0)
        logger.debug("Nessun outlier gestito")

with col4:
    data_quality_score = calculate_data_quality_score(df_engineered)
    st.metric("Data Quality Score", f"{data_quality_score:.1f}%")
    logger.debug(f"Data quality score: {data_quality_score:.1f}%")

# ----------------32. Note metodologiche
with st.expander("Note Metodologiche Avanzate"):
    st.markdown("""
    **Metodologie implementate:**
    
    **Correlazioni:**
    - Pearson (relazioni lineari)
    - Spearman (relazioni monotoniche)
    - Correlazioni per sottogruppi
    
    **Feature Engineering:**
    - Estrazione titolo da nomi (basato su pattern)
    - Estrazione deck da cabine
    - Calcolo tariffa per persona
    - Creazione variabili composite
    
    **Analisi Outliers:**
    - Metodo IQR univariato
    - Analisi multivariata
    - Impatto su correlazioni
    
    **Analisi Statistica:**
    - Test di normalità
    - Confronto distribuzioni
    - Significatività statistica
    
    **Pattern Mining:**
    - Ricerca combinazioni rare
    - Identificazione anomalie
    - Pattern di sopravvivenza
    
    **Segmentazione:**
    - Clustering basato su caratteristiche
    - Profili Age-Fare-Class
    - Analisi segmenti per sopravvivenza
    """)

logger.info(f"Pagina {__name__} completata con successo")