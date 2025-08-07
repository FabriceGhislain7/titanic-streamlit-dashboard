"""
src/utils/feature_engineering.py
Functions for advanced feature engineering
"""

import pandas as pd
import numpy as np
import re
import logging
from sklearn.preprocessing import LabelEncoder
import streamlit as st

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Title Extraction from Name (name analysis extension)
def extract_title_from_name(df):
    """
    Extract title from passenger name
    Extends name analysis from notebook
    """
    logger.info("Executing extract_title_from_name")
    if df is None or 'Name' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Name' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Extract title using regex
    df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    logger.debug(f"Extracted {df_copy['Title'].nunique()} raw titles")
    
    # Group rare titles into common categories
    title_mapping = {
        'Mr': 'Mr',
        'Mrs': 'Mrs', 
        'Miss': 'Miss',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Mlle': 'Miss',
        'Countess': 'Royalty',
        'Ms': 'Miss',
        'Lady': 'Royalty',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Capt': 'Officer',
        'Sir': 'Royalty'
    }
    
    df_copy['Title'] = df_copy['Title'].map(title_mapping)
    df_copy['Title'].fillna('Other', inplace=True)
    
    unique_titles = df_copy['Title'].nunique()
    logger.info(f"Extracted {unique_titles} unique titles from names")
    st.info(f"Extracted {unique_titles} unique titles from names")
    
    return df_copy

# ----------------2. Deck Extraction from Cabin (ship position analysis)
def extract_deck_from_cabin(df):
    """
    Extract deck from cabin for ship position analysis
    """
    logger.info("Executing extract_deck_from_cabin")
    if df is None or 'Cabin' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Cabin' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Extract first letter of cabin as deck
    df_copy['Deck'] = df_copy['Cabin'].str[0]
    df_copy['Deck'].fillna('Unknown', inplace=True)
    
    # Group rare decks
    deck_counts = df_copy['Deck'].value_counts()
    rare_decks = deck_counts[deck_counts < 10].index
    df_copy.loc[df_copy['Deck'].isin(rare_decks), 'Deck'] = 'Other'
    
    unique_decks = df_copy['Deck'].nunique()
    logger.info(f"Extracted {unique_decks} decks from cabin")
    st.info(f"Extracted {unique_decks} decks from cabin")
    
    return df_copy

# ----------------3. Fare Per Person Calculation (economic analysis)
def calculate_fare_per_person(df):
    """
    Calculate fare per person considering family
    """
    logger.info("Executing calculate_fare_per_person")
    if df is None or 'Fare' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Fare' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Calculate family size if not exists
    if 'Family_Size' not in df_copy.columns:
        df_copy['Family_Size'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        logger.debug("Calculated missing Family_Size")
    
    # Calculate fare per person
    df_copy['Fare_Per_Person'] = df_copy['Fare'] / df_copy['Family_Size']
    
    # Handle divisions by zero or null values
    median_fare = df_copy['Fare_Per_Person'].median()
    df_copy['Fare_Per_Person'].fillna(median_fare, inplace=True)
    
    logger.info("Calculated fare per person based on family size")
    st.info("Calculated fare per person based on family size")
    
    return df_copy

# ----------------4. Advanced Age Groups Creation (from notebook age groups)
def create_advanced_age_groups(df):
    """
    Create more detailed age groups
    Extends age group analysis from notebook
    """
    logger.info("Executing create_advanced_age_groups")
    if df is None or 'Age' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Age' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Detailed age groups
    df_copy['Age_Group_Detailed'] = pd.cut(
        df_copy['Age'],
        bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
        labels=['Infant', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior', 'Elderly'],
        include_lowest=True
    )
    
    # Binary indicators for critical age groups
    df_copy['Is_Child'] = (df_copy['Age'] <= 12).astype(int)
    df_copy['Is_Senior'] = (df_copy['Age'] >= 60).astype(int)
    df_copy['Is_Adult_Prime'] = ((df_copy['Age'] >= 25) & (df_copy['Age'] <= 45)).astype(int)
    
    logger.debug("Created 4 new age-related features")
    return df_copy

# ----------------5. Advanced Name Analysis (name patterns)
def analyze_name_patterns(df):
    """
    Analyze patterns in names for feature engineering
    """
    logger.info("Executing analyze_name_patterns")
    if df is None or 'Name' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Name' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Name length
    df_copy['Name_Length'] = df_copy['Name'].str.len()
    
    # Number of words in name
    df_copy['Name_Word_Count'] = df_copy['Name'].str.split().str.len()
    
    # Nickname presence (parentheses)
    df_copy['Has_Nickname'] = df_copy['Name'].str.contains(r'\(.*\)').astype(int)
    
    # Nobility title presence
    nobility_pattern = r'(Count|Countess|Lady|Sir|Don|Dona|Jonkheer)'
    df_copy['Is_Nobility'] = df_copy['Name'].str.contains(nobility_pattern, case=False).astype(int)
    
    logger.debug("Created 4 new features from name patterns")
    return df_copy

# ----------------6. Interaction Features (variable combinations)
def create_interaction_features(df):
    """
    Create interaction features between variables
    """
    logger.info("Executing create_interaction_features")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Class-Gender interaction
    if 'Pclass' in df_copy.columns and 'Sex' in df_copy.columns:
        df_copy['Class_Sex'] = df_copy['Pclass'].astype(str) + '_' + df_copy['Sex']
        logger.debug("Created Class_Sex feature")
    
    # Age-Class interaction
    if 'Age' in df_copy.columns and 'Pclass' in df_copy.columns:
        df_copy['Age_Class_Ratio'] = df_copy['Age'] / df_copy['Pclass']
        logger.debug("Created Age_Class_Ratio feature")
    
    # Family-Class interaction
    if 'Family_Size' in df_copy.columns and 'Pclass' in df_copy.columns:
        df_copy['Family_Class_Score'] = df_copy['Family_Size'] * (4 - df_copy['Pclass'])
        logger.debug("Created Family_Class_Score feature")
    
    # Luxury traveler indicator
    if 'Fare' in df_copy.columns and 'Pclass' in df_copy.columns:
        fare_threshold = df_copy['Fare'].quantile(0.8)
        df_copy['Is_Luxury_Traveler'] = ((df_copy['Fare'] > fare_threshold) & (df_copy['Pclass'] == 1)).astype(int)
        logger.debug("Created Is_Luxury_Traveler feature")
    
    return df_copy

# ----------------7. Advanced Economic Features (fare analysis)
def create_economic_features(df):
    """
    Create advanced economic features
    """
    logger.info("Executing create_economic_features")
    if df is None or 'Fare' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Fare' column")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Fare relative to class
    if 'Pclass' in df_copy.columns:
        class_fare_mean = df_copy.groupby('Pclass')['Fare'].transform('mean')
        df_copy['Fare_Relative_To_Class'] = df_copy['Fare'] / class_fare_mean
        logger.debug("Created Fare_Relative_To_Class feature")
    
    # Fare percentile
    df_copy['Fare_Percentile'] = df_copy['Fare'].rank(pct=True)
    logger.debug("Created Fare_Percentile feature")
    
    # Economic indicators
    fare_q25 = df_copy['Fare'].quantile(0.25)
    fare_q75 = df_copy['Fare'].quantile(0.75)
    
    df_copy['Is_Economy_Fare'] = (df_copy['Fare'] <= fare_q25).astype(int)
    df_copy['Is_Premium_Fare'] = (df_copy['Fare'] >= fare_q75).astype(int)
    logger.debug("Created 2 binary economic features")
    
    return df_copy

# ----------------8. Advanced Family Features (from notebook family)
def create_advanced_family_features(df):
    """
    Create advanced family features
    Extends family analysis from notebook
    """
    logger.info("Executing create_advanced_family_features")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Detailed family type
    if 'SibSp' in df_copy.columns and 'Parch' in df_copy.columns:
        df_copy['Family_Type'] = 'Single'
        df_copy.loc[(df_copy['SibSp'] > 0) & (df_copy['Parch'] == 0), 'Family_Type'] = 'Couple'
        df_copy.loc[(df_copy['SibSp'] == 0) & (df_copy['Parch'] > 0), 'Family_Type'] = 'Parent'
        df_copy.loc[(df_copy['SibSp'] > 0) & (df_copy['Parch'] > 0), 'Family_Type'] = 'Full_Family'
        
        # Specific indicators
        df_copy['Has_Spouse'] = (df_copy['SibSp'] > 0).astype(int)
        df_copy['Has_Children'] = (df_copy['Parch'] > 0).astype(int)
        df_copy['Has_Siblings'] = (df_copy['SibSp'] > 1).astype(int)
        
        # Optimal family (2-4 members)
        if 'Family_Size' in df_copy.columns:
            df_copy['Is_Optimal_Family'] = ((df_copy['Family_Size'] >= 2) & (df_copy['Family_Size'] <= 4)).astype(int)
        
        logger.debug("Created 5 new family features")
    
    return df_copy

# ----------------9. Advanced Categorical Features
def create_advanced_categorical_features(df):
    """
    Create advanced versions of categorical features
    """
    logger.info("Executing create_advanced_categorical_features")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return df
    
    df_copy = df.copy()
    logger.debug("Created DataFrame copy")
    
    # Ordinal encoding for class (inverse for importance)
    if 'Pclass' in df_copy.columns:
        df_copy['Class_Rank'] = 4 - df_copy['Pclass']
        logger.debug("Created Class_Rank feature")
    
    # Binary gender
    if 'Sex' in df_copy.columns:
        df_copy['Is_Female'] = (df_copy['Sex'] == 'female').astype(int)
        logger.debug("Created Is_Female feature")
    
    # Embarkation port with logic
    if 'Embarked' in df_copy.columns:
        df_copy['Embarked_Wealth_Score'] = df_copy['Embarked'].map({'S': 2, 'C': 3, 'Q': 1}).fillna(2)
        logger.debug("Created Embarked_Wealth_Score feature")
    
    return df_copy

# ----------------10. Complete Feature Engineering
def apply_full_feature_engineering(df):
    """
    Apply all feature engineering in sequence
    """
    logger.info("Executing apply_full_feature_engineering")
    if df is None:
        logger.warning("Empty DataFrame in input")
        return df
    
    logger.info("Starting complete feature engineering...")
    st.info("Starting complete feature engineering...")
    
    # Complete pipeline
    df_engineered = df.copy()
    
    # 1. Features from names
    df_engineered = extract_title_from_name(df_engineered)
    df_engineered = analyze_name_patterns(df_engineered)
    
    # 2. Features from cabins
    df_engineered = extract_deck_from_cabin(df_engineered)
    
    # 3. Economic features
    df_engineered = calculate_fare_per_person(df_engineered)
    df_engineered = create_economic_features(df_engineered)
    
    # 4. Age features
    df_engineered = create_advanced_age_groups(df_engineered)
    
    # 5. Family features
    df_engineered = create_advanced_family_features(df_engineered)
    
    # 6. Categorical features
    df_engineered = create_advanced_categorical_features(df_engineered)
    
    # 7. Interaction features
    df_engineered = create_interaction_features(df_engineered)
    
    # Remove columns with too many NaN
    cols_removed = 0
    for col in df_engineered.columns:
        null_pct = df_engineered[col].isnull().sum() / len(df_engineered)
        if null_pct > 0.8:
            df_engineered.drop(col, axis=1, inplace=True)
            cols_removed += 1
            logger.warning(f"Removed column {col} (too many missing values)")
            st.warning(f"Removed column {col} (too many missing values)")
    
    new_features_count = len(df_engineered.columns) - len(df.columns)
    logger.info(f"Feature engineering completed! Added {new_features_count} new features, removed {cols_removed} columns")
    st.success(f"Feature engineering completed! Added {new_features_count} new features")
    
    return df_engineered

logger.info(f"Loading completed {__name__}")