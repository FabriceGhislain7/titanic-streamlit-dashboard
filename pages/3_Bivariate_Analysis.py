"""
pages/3_Bivariate_Analysis.py
Analisi bivariata dei fattori che influenzano la sopravvivenza
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, create_basic_features
from src.components.bivariate_charts import *
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
st.title("Analisi Bivariata - Fattori di Sopravvivenza")
st.markdown("Esplorazione delle relazioni tra variabili e la sopravvivenza dei passeggeri")

# ----------------4. Sidebar controlli
logger.info("Setup sidebar controlli")
with st.sidebar:
    st.header("Controlli Analisi")
    
    analysis_focus = st.selectbox(
        "Focus Analisi:",
        [
            "Panoramica Generale", 
            "Classe e Sopravvivenza", 
            "Genere e Sopravvivenza",
            "Età e Sopravvivenza", 
            "Prezzo e Sopravvivenza", 
            "Famiglia e Sopravvivenza",
            "Analisi Combinata"
        ]
    )
    logger.debug(f"Focus analisi selezionato: {analysis_focus}")
    
    show_statistics = st.checkbox("Mostra statistiche dettagliate", value=True)
    show_interpretations = st.checkbox("Mostra interpretazioni", value=True)
    logger.debug(f"Opzioni visualizzazione: stats={show_statistics}, interpretations={show_interpretations}")

# ----------------5. Panoramica Generale (overview dei fattori principali)
if analysis_focus == "Panoramica Generale":
    logger.info("Avvio analisi panoramica generale")
    st.header("1. Panoramica Fattori di Sopravvivenza")
    
    # ----------------6. Metriche principali (da notebook sezioni multiple)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_survival = df['Survived'].mean() * 100
        st.metric("Sopravvivenza Generale", f"{overall_survival:.1f}%")
    
    with col2:
        female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
        st.metric("Sopravvivenza Donne", f"{female_survival:.1f}%")
    
    with col3:
        first_class_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100
        st.metric("Sopravvivenza 1ª Classe", f"{first_class_survival:.1f}%")
    
    with col4:
        child_survival = df[df['Age'] <= 12]['Survived'].mean() * 100
        st.metric("Sopravvivenza Bambini", f"{child_survival:.1f}%")
    
    logger.debug(f"Metriche calcolate: general={overall_survival:.1f}%, female={female_survival:.1f}%, first_class={first_class_survival:.1f}%, children={child_survival:.1f}%")
    
    # ----------------7. Grafici panoramica (da notebook sezioni 4.2.2.x)
    logger.info("Creazione grafici panoramica")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico sopravvivenza per classe")
        # Sopravvivenza per classe
        fig_class = create_survival_by_class_detailed(df)
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico sopravvivenza per genere")
        # Sopravvivenza per genere
        fig_gender = create_survival_by_gender_detailed(df)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # ----------------8. Heatmap correlazioni (da notebook sezione 4.1.2)
    logger.debug("Creazione heatmap correlazioni")
    st.subheader("Matrice di Correlazione")
    fig_corr = create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------9. Classe e Sopravvivenza (da notebook sezione 4.2.2.1 e 4.2.2.2)
elif analysis_focus == "Classe e Sopravvivenza":
    logger.info("Avvio analisi classe e sopravvivenza")
    st.header("2. Analisi Classe Passeggeri e Sopravvivenza")
    
    # ----------------10. Statistiche per classe (da notebook sezione 4.2.2.2)
    logger.info("Calcolo statistiche per classe")
    class_survival_stats = df.groupby('Pclass').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Age': 'mean',
        'Fare': 'mean'
    }).round(3)
    
    class_survival_stats.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza', 'Eta_Media', 'Prezzo_Medio']
    class_survival_stats = class_survival_stats.reset_index()
    logger.debug(f"Statistiche classe calcolate per {len(class_survival_stats)} classi")
    
    if show_statistics:
        logger.debug("Mostra statistiche per classe")
        st.subheader("Statistiche per Classe")
        
        # Tabella statistiche
        display_stats = class_survival_stats.copy()
        display_stats['Pclass'] = display_stats['Pclass'].map(VALUE_MAPPINGS['Pclass'])
        display_stats['Tasso_Sopravvivenza'] = (display_stats['Tasso_Sopravvivenza'] * 100).round(1)
        display_stats.columns = ['Classe', 'Sopravvissuti', 'Totale', 'Tasso (%)', 'Età Media', 'Prezzo Medio']
        
        st.dataframe(display_stats, use_container_width=True)
    
    # ----------------11. Visualizzazioni classe (da notebook sezione 4.2.2.2)
    logger.info("Creazione visualizzazioni classe")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico distribuzione classe")
        # Conteggi per classe
        fig_class_counts = create_class_distribution_analysis(df)
        st.plotly_chart(fig_class_counts, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico tassi sopravvivenza per classe")
        # Tassi di sopravvivenza
        fig_class_survival = create_survival_rates_by_class(df)
        st.plotly_chart(fig_class_survival, use_container_width=True)
    
    # ----------------12. Analisi dettagliata per classe
    logger.debug("Creazione analisi dettagliata classe")
    st.subheader("Analisi Dettagliata per Classe")
    fig_class_detailed = create_class_survival_detailed_analysis(df)
    st.plotly_chart(fig_class_detailed, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Mostra interpretazioni classe")
        st.markdown("""
        **Interpretazione Classe e Sopravvivenza:**
        
        - **1ª Classe**: Tasso di sopravvivenza più alto (~65%), accesso privilegiato alle scialuppe
        - **2ª Classe**: Tasso intermedio (~48%), posizione intermedia sulla nave
        - **3ª Classe**: Tasso più basso (~24%), difficoltà nell'accesso alle aree di evacuazione
        
        La classe sociale ha avuto un impatto significativo sulla sopravvivenza.
        """)

# ----------------13. Genere e Sopravvivenza (da notebook sezione 4.2.2.3)
elif analysis_focus == "Genere e Sopravvivenza":
    logger.info("Avvio analisi genere e sopravvivenza")
    st.header("3. Analisi Genere e Sopravvivenza")
    
    # ----------------14. Statistiche per genere (da notebook sezione 4.2.2.3)
    logger.info("Calcolo statistiche per genere")
    gender_survival_stats = df.groupby('Sex').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Age': 'mean',
        'Fare': 'mean'
    }).round(3)
    
    gender_survival_stats.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza', 'Eta_Media', 'Prezzo_Medio']
    gender_survival_stats = gender_survival_stats.reset_index()
    logger.debug(f"Statistiche genere calcolate per {len(gender_survival_stats)} generi")
    
    if show_statistics:
        logger.debug("Mostra statistiche per genere")
        st.subheader("Statistiche per Genere")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        female_stats = gender_survival_stats[gender_survival_stats['Sex'] == 'female'].iloc[0]
        male_stats = gender_survival_stats[gender_survival_stats['Sex'] == 'male'].iloc[0]
        
        with col1:
            st.metric("Donne Sopravvissute", f"{int(female_stats['Sopravvissuti'])}/{int(female_stats['Totale'])}")
        with col2:
            st.metric("Tasso Donne", f"{female_stats['Tasso_Sopravvivenza']*100:.1f}%")
        with col3:
            st.metric("Uomini Sopravvissuti", f"{int(male_stats['Sopravvissuti'])}/{int(male_stats['Totale'])}")
        with col4:
            st.metric("Tasso Uomini", f"{male_stats['Tasso_Sopravvivenza']*100:.1f}%")
    
    # ----------------15. Visualizzazioni genere (da notebook sezione 4.2.2.3)
    logger.info("Creazione visualizzazioni genere")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico confronto sopravvivenza per genere")
        # Confronto sopravvivenza per genere
        fig_gender_comparison = create_gender_survival_comparison(df)
        st.plotly_chart(fig_gender_comparison, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico distribuzione genere-classe")
        # Distribuzione per genere e classe
        fig_gender_class = create_gender_class_distribution(df)
        st.plotly_chart(fig_gender_class, use_container_width=True)
    
    # ----------------16. Analisi genere per classe
    logger.debug("Creazione analisi genere per classe")
    st.subheader("Sopravvivenza per Genere e Classe")
    fig_gender_class_survival = create_gender_class_survival_analysis(df)
    st.plotly_chart(fig_gender_class_survival, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Mostra interpretazioni genere")
        st.markdown("""
        **Interpretazione Genere e Sopravvivenza:**
        
        - **Regola "Donne e bambini prima"**: Chiaramente applicata con ~75% sopravvivenza femminile vs ~20% maschile
        - **Differenza drastica**: Il genere è stato il fattore più determinante per la sopravvivenza
        - **Variazioni per classe**: Anche tra le donne, la classe ha influenzato le probabilità
        
        Il protocollo di evacuazione ha privilegiato nettamente le donne.
        """)

# ----------------17. Età e Sopravvivenza (da notebook sezione 4.2.2.4)
elif analysis_focus == "Età e Sopravvivenza":
    logger.info("Avvio analisi età e sopravvivenza")
    st.header("4. Analisi Età e Sopravvivenza")
    
    # ----------------18. Creazione gruppi età (da notebook sezione 4.2.2.4)
    logger.info("Creazione gruppi età")
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 25, 40, 100], 
                            labels=['Bambini (0-12)', 'Giovani (13-25)', 'Adulti (26-40)', 'Anziani (41+)'])
    
    # Statistiche per gruppo età
    age_survival_stats = df.groupby('Age_Group').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    age_survival_stats.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza']
    age_survival_stats = age_survival_stats.reset_index()
    logger.debug(f"Statistiche età calcolate per {len(age_survival_stats)} gruppi")
    
    if show_statistics:
        logger.debug("Mostra statistiche per gruppo età")
        st.subheader("Statistiche per Gruppo di Età")
        st.dataframe(age_survival_stats, use_container_width=True)
    
    # ----------------19. Visualizzazioni età (da notebook sezione 4.2.2.4)
    logger.info("Creazione visualizzazioni età")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico distribuzione età per sopravvivenza")
        # Distribuzione età per sopravvivenza
        fig_age_dist = create_age_survival_distribution(df)
        st.plotly_chart(fig_age_dist, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico tassi per gruppo età")
        # Tassi per gruppo età
        fig_age_rates = create_age_group_survival_rates(df)
        st.plotly_chart(fig_age_rates, use_container_width=True)
    
    # ----------------20. Analisi età per genere (da notebook sezione 4.2.2.4)
    logger.debug("Creazione analisi età per genere")
    st.subheader("Sopravvivenza per Età e Genere")
    fig_age_gender = create_age_gender_survival_analysis(df)
    st.plotly_chart(fig_age_gender, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Mostra interpretazioni età")
        st.markdown("""
        **Interpretazione Età e Sopravvivenza:**
        
        - **Bambini favoriti**: ~58% sopravvivenza, applicazione "bambini prima"
        - **Giovani adulti**: Tasso più basso (~36%), molti uomini giovani
        - **Differenze di genere**: Persistenti in tutti i gruppi di età
        - **Anziani**: Difficoltà fisiche nell'evacuazione (~36% sopravvivenza)
        """)

# ----------------21. Prezzo e Sopravvivenza (da notebook sezione 4.2.2.5)
elif analysis_focus == "Prezzo e Sopravvivenza":
    logger.info("Avvio analisi prezzo e sopravvivenza")
    st.header("5. Analisi Prezzo Biglietto e Sopravvivenza")
    
    # ----------------22. Categorie prezzo (da notebook sezione 4.2.2.5)
    logger.info("Creazione categorie prezzo")
    df['Fare_Category'] = pd.qcut(df['Fare'], q=4, labels=['Basso', 'Medio', 'Alto', 'Molto Alto'], duplicates='drop')
    
    # Statistiche per categoria prezzo
    fare_survival_stats = df.groupby('Fare_Category').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Fare': ['min', 'max', 'mean']
    }).round(2)
    
    fare_survival_stats.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza', 'Prezzo_Min', 'Prezzo_Max', 'Prezzo_Medio']
    fare_survival_stats = fare_survival_stats.reset_index()
    logger.debug(f"Statistiche prezzo calcolate per {len(fare_survival_stats)} categorie")
    
    if show_statistics:
        logger.debug("Mostra statistiche per categoria prezzo")
        st.subheader("Statistiche per Categoria Prezzo")
        st.dataframe(fare_survival_stats, use_container_width=True)
    
    # ----------------23. Visualizzazioni prezzo (da notebook sezione 4.2.2.5)
    logger.info("Creazione visualizzazioni prezzo")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico sopravvivenza per categoria prezzo")
        # Sopravvivenza per categoria prezzo
        fig_fare_survival = create_fare_category_survival(df)
        st.plotly_chart(fig_fare_survival, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico distribuzione prezzi per sopravvivenza")
        # Distribuzione prezzi per sopravvivenza
        fig_fare_dist = create_fare_distribution_by_survival(df)
        st.plotly_chart(fig_fare_dist, use_container_width=True)
    
    # ----------------24. Correlazione prezzo-classe
    logger.debug("Creazione analisi prezzo-classe-sopravvivenza")
    st.subheader("Relazione Prezzo-Classe-Sopravvivenza")
    fig_fare_class = create_fare_class_survival_analysis(df)
    st.plotly_chart(fig_fare_class, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Mostra interpretazioni prezzo")
        st.markdown("""
        **Interpretazione Prezzo e Sopravvivenza:**
        
        - **Correlazione positiva**: Prezzo più alto = maggiore sopravvivenza
        - **Soglia critica**: Differenza significativa tra prezzi bassi e alti
        - **Proxy del status sociale**: Il prezzo riflette la posizione sociale
        - **Accesso alle risorse**: Biglietti costosi = cabine vicine alle scialuppe
        """)

# ----------------25. Famiglia e Sopravvivenza (da notebook sezione 4.2.2.6)
elif analysis_focus == "Famiglia e Sopravvivenza":
    logger.info("Avvio analisi famiglia e sopravvivenza")
    st.header("6. Analisi Famiglia e Sopravvivenza")
    
    # ----------------26. Statistiche famiglia (da notebook sezione 4.2.2.6)
    logger.info("Calcolo statistiche famiglia")
    family_survival_stats = df.groupby('Family_Size').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    family_survival_stats.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza']
    family_survival_stats = family_survival_stats.reset_index()
    logger.debug(f"Statistiche famiglia calcolate per {len(family_survival_stats)} dimensioni")
    
    if show_statistics:
        logger.debug("Mostra statistiche per dimensione famiglia")
        st.subheader("Statistiche per Dimensione Famiglia")
        st.dataframe(family_survival_stats, use_container_width=True)
    
    # ----------------27. Visualizzazioni famiglia (da notebook sezione 4.2.2.6)
    logger.info("Creazione visualizzazioni famiglia")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creazione grafico sopravvivenza per dimensione famiglia")
        # Sopravvivenza per dimensione famiglia
        fig_family_survival = create_family_size_survival(df)
        st.plotly_chart(fig_family_survival, use_container_width=True)
    
    with col2:
        logger.debug("Creazione grafico solo vs famiglia")
        # Solo vs famiglia
        fig_alone_family = create_alone_vs_family_analysis(df)
        st.plotly_chart(fig_alone_family, use_container_width=True)
    
    # ----------------28. Analisi dettagliata famiglia
    logger.debug("Creazione analisi composizione famiglia")
    st.subheader("Analisi Dettagliata Composizione Famiglia")
    fig_family_composition = create_family_composition_analysis(df)
    st.plotly_chart(fig_family_composition, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Mostra interpretazioni famiglia")
        st.markdown("""
        **Interpretazione Famiglia e Sopravvivenza:**
        
        - **Famiglie piccole-medie**: Migliore sopravvivenza (2-4 membri)
        - **Viaggiatori soli**: Sopravvivenza ridotta (~32%)
        - **Famiglie grandi**: Difficoltà logistiche nell'evacuazione
        - **Supporto reciproco**: Famiglie piccole si aiutano a vicenda
        """)

# ----------------29. Analisi Combinata (sintesi di tutti i fattori)
elif analysis_focus == "Analisi Combinata":
    logger.info("Avvio analisi combinata multi-fattore")
    st.header("7. Analisi Combinata Multi-Fattore")
    
    # ----------------30. Dashboard combinata
    logger.info("Creazione dashboard combinata")
    st.subheader("Dashboard Fattori Combinati")
    
    # Analisi multivariata
    logger.debug("Creazione grafico analisi multivariata")
    fig_combined = create_multivariate_survival_analysis(df)
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # ----------------31. Top fattori influenza
    logger.info("Calcolo ranking fattori influenza")
    st.subheader("Ranking Fattori di Influenza")
    
    factors_ranking = calculate_survival_factors_ranking(df)
    if factors_ranking is not None:
        logger.debug("Mostra ranking fattori")
        st.dataframe(factors_ranking, use_container_width=True)
    else:
        logger.debug("Ranking fattori non disponibile")
    
    # ----------------32. Scenari ottimali/pessimi
    logger.info("Creazione scenari ottimali/pessimi")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Mostra profilo ottimale")
        st.subheader("Profilo Sopravvivenza Ottimale")
        optimal_profile = {
            "Genere": "Donna",
            "Classe": "1ª Classe", 
            "Età": "Bambina o giovane donna",
            "Famiglia": "Piccola famiglia (2-4 membri)",
            "Prezzo": "Alto"
        }
        
        for key, value in optimal_profile.items():
            st.write(f"**{key}**: {value}")
    
    with col2:
        logger.debug("Mostra profilo critico")
        st.subheader("Profilo Sopravvivenza Critico")
        critical_profile = {
            "Genere": "Uomo",
            "Classe": "3ª Classe",
            "Età": "Adulto giovane",
            "Famiglia": "Solo o famiglia molto grande",
            "Prezzo": "Basso"
        }
        
        for key, value in critical_profile.items():
            st.write(f"**{key}**: {value}")

# ----------------33. Note metodologiche
with st.expander("Note Metodologiche"):
    st.markdown("""
    **Metodologia di analisi basata su:**
    
    - **Sezione 4.2.2.1 del notebook**: Analisi biglietti per classe
    - **Sezione 4.2.2.2 del notebook**: Sopravvivenza per classe
    - **Sezione 4.2.2.3 del notebook**: Sopravvivenza per genere
    - **Sezione 4.2.2.4 del notebook**: Sopravvivenza per gruppi di età
    - **Sezione 4.2.2.5 del notebook**: Sopravvivenza per prezzo biglietto
    - **Sezione 4.2.2.6 del notebook**: Sopravvivenza per dimensione famiglia
    - **Sezione 4.1.2 del notebook**: Matrice di correlazione
    
    **Analisi implementate:**
    - Tabelle di contingenza
    - Test chi-quadro (implicito nelle distribuzioni)
    - Analisi delle proporzioni
    - Correlazioni multiple
    - Segmentazione multivariata
    
    **Feature engineering applicato:**
    - Gruppi di età standardizzati
    - Categorie di prezzo per quartili
    - Dimensione famiglia (SibSp + Parch + 1)
    - Indicatori binari (Is_Alone)
    """)

logger.info(f"Pagina {__name__} completata con successo")