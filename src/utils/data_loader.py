"""
src/utils/data_loader.py
Functions for loading and managing Titanic data
"""

import pandas as pd
import streamlit as st
import os
import logging
from src.config import DATA_FILE, DATA_URL

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Dataset Loading (from notebook section 2.1 - Structure of dataset)
@st.cache_data(ttl=3600)
def load_titanic_data():
    """
    Load Titanic dataset from local file or remote URL
    Implements loading logic from notebook section 2.1
    """
    logger.info("Executing load_titanic_data")
    try:
        # Try loading from local file first
        if os.path.exists(DATA_FILE):
            logger.debug(f"Found local file: {DATA_FILE}")
            df = pd.read_csv(DATA_FILE)
            st.success(f"Data loaded from local file: {DATA_FILE}")
        else:
            # Load from GitHub URL as in notebook
            logger.debug(f"Local file not found, loading from URL: {DATA_URL}")
            df = pd.read_csv(DATA_URL)
            st.info("Data loaded from GitHub repository")
        
        logger.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

# ----------------2. Dataset Information (from notebook section 2.1 - Dataset information)
def get_data_summary(df):
    """
    Calculate summary statistics of the dataset
    Based on notebook section 2.1 analysis
    """
    logger.info("Executing get_data_summary")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    # Basic calculations from notebook
    total_passengers = len(df)
    survived_count = df['Survived'].sum()
    died_count = total_passengers - survived_count
    survival_rate = (survived_count / total_passengers) * 100
    missing_values = df.isnull().sum().sum()
    
    logger.debug(f"Calculated statistics: {survived_count} survivors, {missing_values} missing values")
    
    return {
        'rows': total_passengers,
        'columns': len(df.columns),
        'survived': survived_count,
        'died': died_count,
        'survival_rate': survival_rate,
        'missing_values': missing_values
    }

# ----------------3. Data Quality Check (from notebook section 2.2 - Missing values)
def get_missing_values_info(df):
    """
    Analyze missing values as in notebook section 2.2
    """
    logger.info("Executing get_missing_values_info")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Values': missing_counts.values,
        'Percentage': missing_percentages.values
    })
    
    result = missing_info[missing_info['Missing_Values'] > 0].sort_values('Missing_Values', ascending=False)
    logger.debug(f"Found {len(result)} columns with missing values")
    
    return result

# ----------------4. Duplicate Check (from notebook section 2.3 - Check duplicates)
def check_duplicates(df):
    """
    Check for duplicates as in notebook section 2.3
    """
    logger.info("Executing check_duplicates")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return 0
    
    duplicates = df.duplicated().sum()
    logger.debug(f"Found {duplicates} duplicates")
    
    return duplicates

# ----------------5. Basic Data Preparation (from notebook section 3 - Data Cleaning)
def prepare_basic_dataset(df):
    """
    Apply basic data cleaning following notebook section 3
    """
    logger.info("Executing prepare_basic_dataset")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    # Copy of original dataframe
    df_clean = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Remove duplicate rows (section 3.1)
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.debug(f"Removed {duplicates} duplicates")
        df_clean = df_clean.drop_duplicates()
    
    # Remove columns with too many missing values (section 3.2)
    if 'Cabin' in df_clean.columns:
        logger.debug("Removed Cabin column (77% missing values)")
        df_clean = df_clean.drop('Cabin', axis=1)
    
    logger.info(f"Cleaned dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    return df_clean

# ----------------6. Dataset Statistics (from notebook section 4.1.1 - Descriptive Statistics)
def get_descriptive_statistics(df):
    """
    Calculate descriptive statistics as in notebook section 4.1.1
    """
    logger.info("Executing get_descriptive_statistics")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    stats = df[numeric_columns].describe()
    logger.debug(f"Calculated statistics for {len(numeric_columns)} numeric columns")
    
    return stats

logger.info(f"Loading completed {__name__}")