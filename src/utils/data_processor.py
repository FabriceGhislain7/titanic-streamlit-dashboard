"""
src/utils/data_processor.py
Funzioni per l'elaborazione e pulizia dei dati
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from src.config import MISSING_VALUE_THRESHOLDS, OUTLIER_CONFIG

logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

# ----------------1. Pulizia Base Dataset (da notebook sezione 3 - Data Cleaning)
@st.cache_data
def clean_dataset_basic(df):
    """
    Applica pulizia base del dataset seguendo notebook sezione 3
    """
    logger.info("Esecuzione clean_dataset_basic")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    # Copia del dataframe originale
    df_clean = df.copy()
    logger.debug("Creato copy del DataFrame")
    
    # ----------------2. Rimozione duplicati (da notebook sezione 3.1)
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    if duplicates_removed > 0:
        logger.info(f"Rimosse {duplicates_removed} righe duplicate")
        st.info(f"Rimosse {duplicates_removed} righe duplicate")
    
    # ----------------3. Rimozione colonne con troppi missing (da notebook sezione 3.2)
    missing_threshold = MISSING_VALUE_THRESHOLDS['drop_column_threshold']
    logger.debug(f"Soglia missing values per rimozione colonne: {missing_threshold}")
    
    for column in df_clean.columns:
        missing_pct = df_clean[column].isnull().sum() / len(df_clean)
        if missing_pct > missing_threshold:
            df_clean = df_clean.drop(column, axis=1)
            logger.info(f"Rimossa colonna '{column}' ({missing_pct:.1%} valori mancanti)")
            st.info(f"Rimossa colonna '{column}' ({missing_pct:.1%} valori mancanti)")
    
    # ----------------4. Gestione Age missing values (da notebook sezione 3.3)
    if 'Age' in df_clean.columns:
        age_missing = df_clean['Age'].isnull().sum()
        if age_missing > 0:
            age_median = df_clean['Age'].median()
            df_clean.loc[df_clean['Age'].isnull(), 'Age'] = age_median
            logger.info(f"Sostituiti {age_missing} valori mancanti in 'Age' con mediana ({age_median:.1f})")
            st.info(f"Sostituiti {age_missing} valori mancanti in 'Age' con mediana ({age_median:.1f})")
    
    # ----------------5. Gestione Embarked missing values
    if 'Embarked' in df_clean.columns:
        embarked_missing = df_clean['Embarked'].isnull().sum()
        if embarked_missing > 0:
            embarked_mode = df_clean['Embarked'].mode()[0]
            df_clean.loc[df_clean['Embarked'].isnull(), 'Embarked'] = embarked_mode
            logger.info(f"Sostituiti {embarked_missing} valori mancanti in 'Embarked' con moda ('{embarked_mode}')")
            st.info(f"Sostituiti {embarked_missing} valori mancanti in 'Embarked' con moda ('{embarked_mode}')")
    
    logger.info(f"Dataset pulito: {len(df_clean)} righe, {len(df_clean.columns)} colonne")
    return df_clean

# ----------------6. Rilevamento Outliers (da notebook sezione 4.1.1 - Outlier detection)
def detect_outliers_iqr(series, lower_q=0.25, upper_q=0.75, multiplier=1.5):
    """
    Rileva outliers usando il metodo IQR dal notebook
    """
    logger.debug(f"Rilevamento outliers IQR per serie (q1={lower_q}, q3={upper_q}, m={multiplier})")
    Q1 = series.quantile(lower_q)
    Q3 = series.quantile(upper_q)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    logger.debug(f"Trovati {len(outliers)} outliers (bounds: {lower_bound:.2f}, {upper_bound:.2f})")
    return outliers, lower_bound, upper_bound

# ----------------7. Summary Outliers per Dataset (da notebook sezione 4.1.1)
@st.cache_data
def detect_outliers_summary(df):
    """
    Crea summary degli outliers per tutte le variabili numeriche
    Basato sull'analisi del notebook sezione 4.1.1
    """
    logger.info("Esecuzione detect_outliers_summary")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    logger.debug(f"Colonne numeriche analizzate: {numeric_cols}")
    
    outliers_summary = []
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            outliers, lower_bound, upper_bound = detect_outliers_iqr(
                df[col].dropna(),
                lower_q=OUTLIER_CONFIG['lower_quantile'],
                upper_q=OUTLIER_CONFIG['upper_quantile'],
                multiplier=OUTLIER_CONFIG['iqr_multiplier']
            )
            
            outliers_summary.append({
                'Variable': col,
                'Total_Values': df[col].notna().sum(),
                'Outliers_Count': len(outliers),
                'Outliers_Percentage': (len(outliers) / df[col].notna().sum()) * 100,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Min_Outlier': outliers.min() if len(outliers) > 0 else np.nan,
                'Max_Outlier': outliers.max() if len(outliers) > 0 else np.nan
            })
            logger.debug(f"Analizzata colonna {col}: {len(outliers)} outliers")
    
    logger.info(f"Generato summary per {len(outliers_summary)} colonne")
    return pd.DataFrame(outliers_summary)

# ----------------8. Gestione Outliers (da notebook - metodi di gestione outliers)
@st.cache_data
def handle_outliers(df, method='clip', columns=None):
    """
    Gestisce outliers usando vari metodi dal notebook
    """
    logger.info(f"Esecuzione handle_outliers (metodo: {method})")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns
        columns = [col for col in columns if col != 'PassengerId']
        logger.debug(f"Usate tutte le colonne numeriche: {columns}")
    
    for col in columns:
        if col in df_processed.columns:
            outliers, lower_bound, upper_bound = detect_outliers_iqr(df_processed[col].dropna())
            
            if len(outliers) > 0:
                logger.debug(f"Gestione {len(outliers)} outliers in {col} con metodo {method}")
                
                if method == 'clip':
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                    logger.debug(f"Clippati valori in {col} tra {lower_bound:.2f} e {upper_bound:.2f}")
                
                elif method == 'remove':
                    mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                    df_processed = df_processed[mask]
                    logger.debug(f"Rimosse {len(outliers)} righe con outliers in {col}")
                
                elif method == 'replace_median':
                    median_val = df_processed[col].median()
                    outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                    df_processed.loc[outlier_mask, col] = median_val
                    logger.debug(f"Sostituiti {outlier_mask.sum()} outliers in {col} con mediana {median_val:.2f}")
    
    logger.info(f"Outliers gestiti in {len(columns)} colonne")
    return df_processed

# ----------------9. Feature Engineering Base (preparazione per analisi successive)
def create_basic_features(df):
    """
    Crea feature base che saranno utilizzate nelle analisi successive
    """
    logger.info("Esecuzione create_basic_features")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    df_featured = df.copy()
    logger.debug("Creato copy del DataFrame")
    
    # Family Size (da notebook sezione 4.2.2.5)
    if 'SibSp' in df_featured.columns and 'Parch' in df_featured.columns:
        df_featured['Family_Size'] = df_featured['SibSp'] + df_featured['Parch'] + 1
        logger.debug("Creata feature 'Family_Size'")
    
    # Is Alone
    if 'Family_Size' in df_featured.columns:
        df_featured['Is_Alone'] = (df_featured['Family_Size'] == 1).astype(int)
        logger.debug("Creata feature 'Is_Alone'")
    
    # Age Groups (da notebook sezione 4.2.2.4)
    if 'Age' in df_featured.columns:
        df_featured['Age_Group'] = pd.cut(
            df_featured['Age'],
            bins=[0, 12, 25, 40, 100],
            labels=['Child', 'Young_Adult', 'Middle_Adult', 'Older_Adult'],
            include_lowest=True
        )
        logger.debug("Creata feature 'Age_Group'")
    
    # Fare Categories (da notebook sezione 4.2.2.4 - Fare analysis)
    if 'Fare' in df_featured.columns:
        df_featured['Fare_Category'] = pd.qcut(
            df_featured['Fare'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very_High'],
            duplicates='drop'
        )
        logger.debug("Creata feature 'Fare_Category'")
    
    logger.info(f"Create {len(df_featured.columns) - len(df.columns)} nuove features")
    return df_featured

# ----------------10. Validazione Qualità Dati
def validate_data_quality(df):
    """
    Valida la qualità generale del dataset
    """
    logger.info("Esecuzione validate_data_quality")
    if df is None:
        logger.warning("DataFrame vuoto in input")
        return None
    
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'completeness_score': ((df.count().sum() / (len(df) * len(df.columns))) * 100)
    }
    
    logger.info(f"Validazione completata: completeness_score={quality_report['completeness_score']:.2f}%")
    return quality_report

logger.info(f"Caricamento completato {__name__}")