"""
src/utils/data_processor.py
Functions for data processing and cleaning
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from src.config import MISSING_VALUE_THRESHOLDS, OUTLIER_CONFIG

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Basic Dataset Cleaning (from notebook section 3 - Data Cleaning)
@st.cache_data
def clean_dataset_basic(df):
    """
    Apply basic dataset cleaning following notebook section 3
    """
    logger.info("Executing clean_dataset_basic")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    # Copy of original dataframe
    df_clean = df.copy()
    logger.debug("Created DataFrame copy")
    
    # ----------------2. Remove duplicates (from notebook section 3.1)
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        st.info(f"Removed {duplicates_removed} duplicate rows")
    
    # ----------------3. Remove columns with too many missing values (from notebook section 3.2)
    missing_threshold = MISSING_VALUE_THRESHOLDS['drop_column_threshold']
    logger.debug(f"Missing values threshold for column removal: {missing_threshold}")
    
    for column in df_clean.columns:
        missing_pct = df_clean[column].isnull().sum() / len(df_clean)
        if missing_pct > missing_threshold:
            df_clean = df_clean.drop(column, axis=1)
            logger.info(f"Removed column '{column}' ({missing_pct:.1%} missing values)")
            st.info(f"Removed column '{column}' ({missing_pct:.1%} missing values)")
    
    # ----------------4. Handle Age missing values (from notebook section 3.3)
    if 'Age' in df_clean.columns:
        age_missing = df_clean['Age'].isnull().sum()
        if age_missing > 0:
            age_median = df_clean['Age'].median()
            df_clean.loc[df_clean['Age'].isnull(), 'Age'] = age_median
            logger.info(f"Replaced {age_missing} missing values in 'Age' with median ({age_median:.1f})")
            st.info(f"Replaced {age_missing} missing values in 'Age' with median ({age_median:.1f})")
    
    # ----------------5. Handle Embarked missing values
    if 'Embarked' in df_clean.columns:
        embarked_missing = df_clean['Embarked'].isnull().sum()
        if embarked_missing > 0:
            embarked_mode = df_clean['Embarked'].mode()[0]
            df_clean.loc[df_clean['Embarked'].isnull(), 'Embarked'] = embarked_mode
            logger.info(f"Replaced {embarked_missing} missing values in 'Embarked' with mode ('{embarked_mode}')")
            st.info(f"Replaced {embarked_missing} missing values in 'Embarked' with mode ('{embarked_mode}')")
    
    logger.info(f"Cleaned dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    return df_clean

# ----------------6. Outlier Detection (from notebook section 4.1.1 - Outlier detection)
def detect_outliers_iqr(series, lower_q=0.25, upper_q=0.75, multiplier=1.5):
    """
    Detect outliers using IQR method from notebook
    """
    logger.debug(f"IQR outlier detection for series (q1={lower_q}, q3={upper_q}, m={multiplier})")
    Q1 = series.quantile(lower_q)
    Q3 = series.quantile(upper_q)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    logger.debug(f"Found {len(outliers)} outliers (bounds: {lower_bound:.2f}, {upper_bound:.2f})")
    return outliers, lower_bound, upper_bound

# ----------------7. Outliers Summary for Dataset (from notebook section 4.1.1)
@st.cache_data
def detect_outliers_summary(df):
    """
    Create outliers summary for all numerical variables
    Based on notebook section 4.1.1 analysis
    """
    logger.info("Executing detect_outliers_summary")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    logger.debug(f"Analyzed numerical columns: {numeric_cols}")
    
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
            logger.debug(f"Analyzed column {col}: {len(outliers)} outliers")
    
    logger.info(f"Generated summary for {len(outliers_summary)} columns")
    return pd.DataFrame(outliers_summary)

# ----------------8. Outlier Handling (from notebook - outlier handling methods)
@st.cache_data
def handle_outliers(df, method='clip', columns=None):
    """
    Handle outliers using various methods from notebook
    """
    logger.info(f"Executing handle_outliers (method: {method})")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns
        columns = [col for col in columns if col != 'PassengerId']
        logger.debug(f"Using all numerical columns: {columns}")
    
    for col in columns:
        if col in df_processed.columns:
            outliers, lower_bound, upper_bound = detect_outliers_iqr(df_processed[col].dropna())
            
            if len(outliers) > 0:
                logger.debug(f"Handling {len(outliers)} outliers in {col} with method {method}")
                
                if method == 'clip':
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                    logger.debug(f"Clipped values in {col} between {lower_bound:.2f} and {upper_bound:.2f}")
                
                elif method == 'remove':
                    mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                    df_processed = df_processed[mask]
                    logger.debug(f"Removed {len(outliers)} rows with outliers in {col}")
                
                elif method == 'replace_median':
                    median_val = df_processed[col].median()
                    outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                    df_processed.loc[outlier_mask, col] = median_val
                    logger.debug(f"Replaced {outlier_mask.sum()} outliers in {col} with median {median_val:.2f}")
    
    logger.info(f"Outliers handled in {len(columns)} columns")
    return df_processed

# ----------------9. Basic Feature Engineering (preparation for subsequent analyses)
def create_basic_features(df):
    """
    Create basic features that will be used in subsequent analyses
    """
    logger.info("Executing create_basic_features")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return None
    
    df_featured = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Family Size (from notebook section 4.2.2.5)
    if 'SibSp' in df_featured.columns and 'Parch' in df_featured.columns:
        df_featured['Family_Size'] = df_featured['SibSp'] + df_featured['Parch'] + 1
        logger.debug("Created 'Family_Size' feature")
    
    # Is Alone
    if 'Family_Size' in df_featured.columns:
        df_featured['Is_Alone'] = (df_featured['Family_Size'] == 1).astype(int)
        logger.debug("Created 'Is_Alone' feature")
    
    # Age Groups (from notebook section 4.2.2.4)
    if 'Age' in df_featured.columns:
        df_featured['Age_Group'] = pd.cut(
            df_featured['Age'],
            bins=[0, 12, 25, 40, 100],
            labels=['Child', 'Young_Adult', 'Middle_Adult', 'Older_Adult'],
            include_lowest=True
        )
        logger.debug("Created 'Age_Group' feature")
    
    # Fare Categories (from notebook section 4.2.2.4 - Fare analysis)
    if 'Fare' in df_featured.columns:
        df_featured['Fare_Category'] = pd.qcut(
            df_featured['Fare'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very_High'],
            duplicates='drop'
        )
        logger.debug("Created 'Fare_Category' feature")
    
    logger.info(f"Created {len(df_featured.columns) - len(df.columns)} new features")
    return df_featured

# ----------------10. Data Quality Validation
def validate_data_quality(df):
    """
    Validate overall dataset quality
    """
    logger.info("Executing validate_data_quality")
    if df is None:
        logger.warning("Empty DataFrame in input")
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
    
    logger.info(f"Validation completed: completeness_score={quality_report['completeness_score']:.2f}%")
    return quality_report

logger.info(f"Loading completed {__name__}")