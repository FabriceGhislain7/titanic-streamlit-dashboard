"""
src/utils/feature_engineering.py
Functions for advanced feature engineering.
"""

import logging
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")


def extract_title_from_name(df):
    """
    Extract the title from the passenger name.
    Extend the notebook-based name analysis.
    """
    logger.info("Running extract_title_from_name")
    if df is None or "Name" not in df.columns:
        logger.warning("DataFrame is empty or column 'Name' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    df_copy["Title"] = df_copy["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    logger.debug(f"Extracted {df_copy['Title'].nunique()} raw titles")

    title_mapping = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Mlle": "Miss",
        "Countess": "Royalty",
        "Ms": "Miss",
        "Lady": "Royalty",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Capt": "Officer",
        "Sir": "Royalty",
    }

    df_copy["Title"] = df_copy["Title"].map(title_mapping)
    df_copy["Title"].fillna("Other", inplace=True)

    unique_titles = df_copy["Title"].nunique()
    logger.info(f"Extracted {unique_titles} unique titles from names")
    st.info(f"Extracted {unique_titles} unique titles from names")

    return df_copy


def extract_deck_from_cabin(df):
    """
    Extract the deck from the cabin for ship-position analysis.
    """
    logger.info("Running extract_deck_from_cabin")
    if df is None or "Cabin" not in df.columns:
        logger.warning("DataFrame is empty or column 'Cabin' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    df_copy["Deck"] = df_copy["Cabin"].str[0]
    df_copy["Deck"].fillna("Unknown", inplace=True)

    deck_counts = df_copy["Deck"].value_counts()
    rare_decks = deck_counts[deck_counts < 10].index
    df_copy.loc[df_copy["Deck"].isin(rare_decks), "Deck"] = "Other"

    unique_decks = df_copy["Deck"].nunique()
    logger.info(f"Extracted {unique_decks} deck values from cabin data")
    st.info(f"Extracted {unique_decks} deck values from cabin data")

    return df_copy


def calculate_fare_per_person(df):
    """
    Calculate fare per person based on family size.
    """
    logger.info("Running calculate_fare_per_person")
    if df is None or "Fare" not in df.columns:
        logger.warning("DataFrame is empty or column 'Fare' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    if "Family_Size" not in df_copy.columns:
        df_copy["Family_Size"] = df_copy["SibSp"] + df_copy["Parch"] + 1
        logger.debug("Computed missing Family_Size")

    df_copy["Fare_Per_Person"] = df_copy["Fare"] / df_copy["Family_Size"]

    median_fare = df_copy["Fare_Per_Person"].median()
    df_copy["Fare_Per_Person"].fillna(median_fare, inplace=True)

    logger.info("Calculated fare per person based on family size")
    st.info("Calculated fare per person based on family size")

    return df_copy


def create_advanced_age_groups(df):
    """
    Create more detailed age groups.
    Extend the notebook age-group analysis.
    """
    logger.info("Running create_advanced_age_groups")
    if df is None or "Age" not in df.columns:
        logger.warning("DataFrame is empty or column 'Age' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    df_copy["Age_Group_Detailed"] = pd.cut(
        df_copy["Age"],
        bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
        labels=["Infant", "Child", "Teen", "Young_Adult", "Adult", "Middle_Age", "Senior", "Elderly"],
        include_lowest=True,
    )

    df_copy["Is_Child"] = (df_copy["Age"] <= 12).astype(int)
    df_copy["Is_Senior"] = (df_copy["Age"] >= 60).astype(int)
    df_copy["Is_Adult_Prime"] = ((df_copy["Age"] >= 25) & (df_copy["Age"] <= 45)).astype(int)

    logger.debug("Created 4 new age-related features")
    return df_copy


def analyze_name_patterns(df):
    """
    Analyze name patterns for feature engineering.
    """
    logger.info("Running analyze_name_patterns")
    if df is None or "Name" not in df.columns:
        logger.warning("DataFrame is empty or column 'Name' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    df_copy["Name_Length"] = df_copy["Name"].str.len()
    df_copy["Name_Word_Count"] = df_copy["Name"].str.split().str.len()
    df_copy["Has_Nickname"] = df_copy["Name"].str.contains(r"\(.*\)").astype(int)

    nobility_pattern = r"(Count|Countess|Lady|Sir|Don|Dona|Jonkheer)"
    df_copy["Is_Nobility"] = df_copy["Name"].str.contains(nobility_pattern, case=False).astype(int)

    logger.debug("Created 4 new features from name patterns")
    return df_copy


def create_interaction_features(df):
    """
    Create interaction features between variables.
    """
    logger.info("Running create_interaction_features")
    if df is None:
        logger.warning("Empty input DataFrame")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    if "Pclass" in df_copy.columns and "Sex" in df_copy.columns:
        df_copy["Class_Sex"] = df_copy["Pclass"].astype(str) + "_" + df_copy["Sex"]
        logger.debug("Created feature Class_Sex")

    if "Age" in df_copy.columns and "Pclass" in df_copy.columns:
        df_copy["Age_Class_Ratio"] = df_copy["Age"] / df_copy["Pclass"]
        logger.debug("Created feature Age_Class_Ratio")

    if "Family_Size" in df_copy.columns and "Pclass" in df_copy.columns:
        df_copy["Family_Class_Score"] = df_copy["Family_Size"] * (4 - df_copy["Pclass"])
        logger.debug("Created feature Family_Class_Score")

    if "Fare" in df_copy.columns and "Pclass" in df_copy.columns:
        fare_threshold = df_copy["Fare"].quantile(0.8)
        df_copy["Is_Luxury_Traveler"] = ((df_copy["Fare"] > fare_threshold) & (df_copy["Pclass"] == 1)).astype(int)
        logger.debug("Created feature Is_Luxury_Traveler")

    return df_copy


def create_economic_features(df):
    """
    Create advanced economic features.
    """
    logger.info("Running create_economic_features")
    if df is None or "Fare" not in df.columns:
        logger.warning("DataFrame is empty or column 'Fare' is missing")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    if "Pclass" in df_copy.columns:
        class_fare_mean = df_copy.groupby("Pclass")["Fare"].transform("mean")
        df_copy["Fare_Relative_To_Class"] = df_copy["Fare"] / class_fare_mean
        logger.debug("Created feature Fare_Relative_To_Class")

    df_copy["Fare_Percentile"] = df_copy["Fare"].rank(pct=True)
    logger.debug("Created feature Fare_Percentile")

    fare_q25 = df_copy["Fare"].quantile(0.25)
    fare_q75 = df_copy["Fare"].quantile(0.75)

    df_copy["Is_Economy_Fare"] = (df_copy["Fare"] <= fare_q25).astype(int)
    df_copy["Is_Premium_Fare"] = (df_copy["Fare"] >= fare_q75).astype(int)
    logger.debug("Created 2 binary economic features")

    return df_copy


def create_advanced_family_features(df):
    """
    Create advanced family features.
    Extend the notebook family analysis.
    """
    logger.info("Running create_advanced_family_features")
    if df is None:
        logger.warning("Empty input DataFrame")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    if "SibSp" in df_copy.columns and "Parch" in df_copy.columns:
        df_copy["Family_Type"] = "Single"
        df_copy.loc[(df_copy["SibSp"] > 0) & (df_copy["Parch"] == 0), "Family_Type"] = "Couple"
        df_copy.loc[(df_copy["SibSp"] == 0) & (df_copy["Parch"] > 0), "Family_Type"] = "Parent"
        df_copy.loc[(df_copy["SibSp"] > 0) & (df_copy["Parch"] > 0), "Family_Type"] = "Full_Family"

        df_copy["Has_Spouse"] = (df_copy["SibSp"] > 0).astype(int)
        df_copy["Has_Children"] = (df_copy["Parch"] > 0).astype(int)
        df_copy["Has_Siblings"] = (df_copy["SibSp"] > 1).astype(int)

        if "Family_Size" in df_copy.columns:
            df_copy["Is_Optimal_Family"] = ((df_copy["Family_Size"] >= 2) & (df_copy["Family_Size"] <= 4)).astype(int)

        logger.debug("Created 5 new family features")

    return df_copy


def create_advanced_categorical_features(df):
    """
    Create advanced versions of categorical features.
    """
    logger.info("Running create_advanced_categorical_features")
    if df is None:
        logger.warning("Empty input DataFrame")
        return df

    df_copy = df.copy()
    logger.debug("Created a DataFrame copy")

    if "Pclass" in df_copy.columns:
        df_copy["Class_Rank"] = 4 - df_copy["Pclass"]
        logger.debug("Created feature Class_Rank")

    if "Sex" in df_copy.columns:
        df_copy["Is_Female"] = (df_copy["Sex"] == "female").astype(int)
        logger.debug("Created feature Is_Female")

    if "Embarked" in df_copy.columns:
        df_copy["Embarked_Wealth_Score"] = df_copy["Embarked"].map({"S": 2, "C": 3, "Q": 1}).fillna(2)
        logger.debug("Created feature Embarked_Wealth_Score")

    return df_copy


def apply_full_feature_engineering(df):
    """
    Apply the full feature engineering sequence.
    """
    logger.info("Running apply_full_feature_engineering")
    if df is None:
        logger.warning("Empty input DataFrame")
        return df

    logger.info("Starting full feature engineering")
    st.info("Starting full feature engineering")

    df_engineered = df.copy()

    df_engineered = extract_title_from_name(df_engineered)
    df_engineered = analyze_name_patterns(df_engineered)
    df_engineered = extract_deck_from_cabin(df_engineered)
    df_engineered = calculate_fare_per_person(df_engineered)
    df_engineered = create_economic_features(df_engineered)
    df_engineered = create_advanced_age_groups(df_engineered)
    df_engineered = create_advanced_family_features(df_engineered)
    df_engineered = create_advanced_categorical_features(df_engineered)
    df_engineered = create_interaction_features(df_engineered)

    cols_removed = 0
    for col in df_engineered.columns:
        null_pct = df_engineered[col].isnull().sum() / len(df_engineered)
        if null_pct > 0.8:
            df_engineered.drop(col, axis=1, inplace=True)
            cols_removed += 1
            logger.warning(f"Removed column {col} (too many missing values)")
            st.warning(f"Removed column {col} (too many missing values)")

    new_features_count = len(df_engineered.columns) - len(df.columns)
    logger.info(
        f"Feature engineering completed. Added {new_features_count} new features and removed {cols_removed} columns"
    )
    st.success(f"Feature engineering completed. Added {new_features_count} new features")

    return df_engineered


logger.info(f"Loading completed {__name__}")
