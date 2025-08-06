"""
src/utils/data_loader.py
Funzioni per il caricamento e gestione dei dati Titanic
"""

import pandas as pd
import streamlit as st
import os
import logging
from src.config import DATA_FILE, DATA_URL

logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

# ----------------1. Caricamento Dataset (da notebook sezione 2.1 - Structure of dataset)
@st.cache_data(ttl=3600)
def load_titanic_data():
    """
    Carica il dataset Titanic da file locale o URL remoto
    Implementa la logica di caricamento dal notebook sezione 2.1
    """
    logger.info("Esecuzione load_titanic_data")
    try:
        # Prova prima a caricare da file locale
        if os.path.exists(DATA_FILE):
            logger.debug(f"Trovato file locale: {DATA_FILE}")
            df = pd.read_csv(DATA_FILE)
            st.success(f"Dati caricati da file locale: {DATA_FILE}")
        else:
            # Carica da URL GitHub come nel notebook
            logger.debug(f"File locale non trovato, caricamento da URL: {DATA_URL}")
            df = pd.read_csv(DATA_URL)
            st.info("Dati caricati da repository GitHub")
        
        logger.info(f"Dataset caricato con {len(df)} righe e {len(df.columns)} colonne")
        return df
    
    except Exception as e:
        logger.error(f"Errore nel caricamento dati: {str(e)}")
        st.error(f"Errore nel caricamento dati: {str(e)}")
        return None

# ----------------2. Informazioni Dataset (da notebook sezione 2.1 - Dataset information)
def get_data_summary(df):
    """
    Calcola statistiche riassuntive del dataset
    Basato sull'analisi del notebook sezione 2.1
    """
    logger.info("Esecuzione get_data_summary")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    # Calcoli base dal notebook
    total_passengers = len(df)
    survived_count = df['Survived'].sum()
    died_count = total_passengers - survived_count
    survival_rate = (survived_count / total_passengers) * 100
    missing_values = df.isnull().sum().sum()
    
    logger.debug(f"Calcolate statistiche: {survived_count} sopravvissuti, {missing_values} valori mancanti")
    
    return {
        'rows': total_passengers,
        'columns': len(df.columns),
        'survived': survived_count,
        'died': died_count,
        'survival_rate': survival_rate,
        'missing_values': missing_values
    }

# ----------------3. Controllo Qualita Dati (da notebook sezione 2.2 - Missing values)
def get_missing_values_info(df):
    """
    Analizza i valori mancanti come nel notebook sezione 2.2
    """
    logger.info("Esecuzione get_missing_values_info")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Colonna': missing_counts.index,
        'Valori_Mancanti': missing_counts.values,
        'Percentuale': missing_percentages.values
    })
    
    result = missing_info[missing_info['Valori_Mancanti'] > 0].sort_values('Valori_Mancanti', ascending=False)
    logger.debug(f"Trovate {len(result)} colonne con valori mancanti")
    
    return result

# ----------------4. Controllo Duplicati (da notebook sezione 2.3 - Check duplicates)
def check_duplicates(df):
    """
    Verifica presenza duplicati come nel notebook sezione 2.3
    """
    logger.info("Esecuzione check_duplicates")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return 0
    
    duplicates = df.duplicated().sum()
    logger.debug(f"Trovati {duplicates} duplicati")
    
    return duplicates

# ----------------5. Preparazione Dati Base (da notebook sezione 3 - Data Cleaning)
def prepare_basic_dataset(df):
    """
    Applica pulizia base dei dati seguendo notebook sezione 3
    """
    logger.info("Esecuzione prepare_basic_dataset")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    # Copia del dataframe originale
    df_clean = df.copy()
    logger.debug("Creato copy del DataFrame")
    
    # Rimuovi righe duplicate (sezione 3.1)
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.debug(f"Rimossi {duplicates} duplicati")
        df_clean = df_clean.drop_duplicates()
    
    # Rimuovi colonne con troppi valori mancanti (sezione 3.2)
    if 'Cabin' in df_clean.columns:
        logger.debug("Rimossa colonna Cabin (77% valori mancanti)")
        df_clean = df_clean.drop('Cabin', axis=1)
    
    logger.info(f"Dataset pulito: {len(df_clean)} righe, {len(df_clean.columns)} colonne")
    return df_clean

# ----------------6. Dataset Statistics (da notebook sezione 4.1.1 - Descriptive Statistics)
def get_descriptive_statistics(df):
    """
    Calcola statistiche descrittive come nel notebook sezione 4.1.1
    """
    logger.info("Esecuzione get_descriptive_statistics")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    # Seleziona solo colonne numeriche
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    stats = df[numeric_columns].describe()
    logger.debug(f"Calcolate statistiche per {len(numeric_columns)} colonne numeriche")
    
    return stats

logger.info(f"Caricamento completato {__name__}")