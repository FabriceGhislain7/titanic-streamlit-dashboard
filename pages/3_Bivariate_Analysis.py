"""
pages/3_Bivariate_Analysis.py
Bivariate analysis of factors affecting survival.
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.components.bivariate_charts import *
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, create_basic_features

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")


def setup_page():
    """Configure the Streamlit page."""
    logger.info("Configuring the Streamlit page")
    st.set_page_config(**PAGE_CONFIG)


setup_page()

logger.info("Loading Titanic data")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load the data")
    st.stop()

logger.info("Applying basic data cleaning")
df = clean_dataset_basic(df_original)
logger.info("Creating basic features")
df = create_basic_features(df)
logger.info(f"Prepared data shape: {df.shape}")

logger.info("Setting up page header")
st.title("Bivariate Analysis - Survival Factors")
st.markdown("Exploration of the relationships between variables and passenger survival")

logger.info("Setting up sidebar controls")
with st.sidebar:
    st.header("Analysis Controls")

    analysis_focus = st.selectbox(
        "Analysis Focus:",
        [
            "General Overview",
            "Class and Survival",
            "Gender and Survival",
            "Age and Survival",
            "Fare and Survival",
            "Family and Survival",
            "Combined Analysis",
        ],
    )
    logger.debug(f"Selected analysis focus: {analysis_focus}")

    show_statistics = st.checkbox("Show detailed statistics", value=True)
    show_interpretations = st.checkbox("Show interpretations", value=True)
    logger.debug(f"Display options: stats={show_statistics}, interpretations={show_interpretations}")

if analysis_focus == "General Overview":
    logger.info("Starting general overview analysis")
    st.header("1. Survival Factors Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        overall_survival = df["Survived"].mean() * 100
        st.metric("Overall Survival", f"{overall_survival:.1f}%")

    with col2:
        female_survival = df[df["Sex"] == "female"]["Survived"].mean() * 100
        st.metric("Female Survival", f"{female_survival:.1f}%")

    with col3:
        first_class_survival = df[df["Pclass"] == 1]["Survived"].mean() * 100
        st.metric("1st Class Survival", f"{first_class_survival:.1f}%")

    with col4:
        child_survival = df[df["Age"] <= 12]["Survived"].mean() * 100
        st.metric("Child Survival", f"{child_survival:.1f}%")

    logger.debug(
        f"Computed metrics: overall={overall_survival:.1f}%, female={female_survival:.1f}%, "
        f"first_class={first_class_survival:.1f}%, child={child_survival:.1f}%"
    )

    logger.info("Creating overview charts")
    col1, col2 = st.columns(2)

    with col1:
        logger.debug("Creating survival by class chart")
        fig_class = create_survival_by_class_detailed(df)
        st.plotly_chart(fig_class, use_container_width=True)

    with col2:
        logger.debug("Creating survival by gender chart")
        fig_gender = create_survival_by_gender_detailed(df)
        st.plotly_chart(fig_gender, use_container_width=True)

    logger.debug("Creating correlation heatmap")
    st.subheader("Correlation Matrix")
    fig_corr = create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_focus == "Class and Survival":
    logger.info("Starting class and survival analysis")
    st.header("2. Passenger Class and Survival Analysis")

    logger.info("Calculating class statistics")
    class_survival_stats = df.groupby("Pclass").agg({"Survived": ["sum", "count", "mean"], "Age": "mean", "Fare": "mean"}).round(3)
    class_survival_stats.columns = ["Survivors", "Total", "Survival_Rate", "Average_Age", "Average_Fare"]
    class_survival_stats = class_survival_stats.reset_index()
    logger.debug(f"Calculated class statistics for {len(class_survival_stats)} classes")

    if show_statistics:
        st.subheader("Statistics by Class")
        display_stats = class_survival_stats.copy()
        display_stats["Pclass"] = display_stats["Pclass"].map(VALUE_MAPPINGS["Pclass"])
        display_stats["Survival_Rate"] = (display_stats["Survival_Rate"] * 100).round(1)
        display_stats.columns = ["Class", "Survivors", "Total", "Rate (%)", "Average Age", "Average Fare"]
        st.dataframe(display_stats, use_container_width=True)

    logger.info("Creating class visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_class_counts = create_class_distribution_analysis(df)
        st.plotly_chart(fig_class_counts, use_container_width=True)

    with col2:
        fig_class_survival = create_survival_rates_by_class(df)
        st.plotly_chart(fig_class_survival, use_container_width=True)

    st.subheader("Detailed Class Analysis")
    fig_class_detailed = create_class_survival_detailed_analysis(df)
    st.plotly_chart(fig_class_detailed, use_container_width=True)

    if show_interpretations:
        st.markdown(
            """
            **Class and Survival Interpretation:**

            - **1st Class**: Highest survival rate (~65%), with privileged access to lifeboats
            - **2nd Class**: Intermediate survival rate (~48%), with a middle position on the ship
            - **3rd Class**: Lowest survival rate (~24%), with more difficulty reaching evacuation areas

            Social class had a significant impact on survival.
            """
        )

elif analysis_focus == "Gender and Survival":
    logger.info("Starting gender and survival analysis")
    st.header("3. Gender and Survival Analysis")

    logger.info("Calculating gender statistics")
    gender_survival_stats = df.groupby("Sex").agg({"Survived": ["sum", "count", "mean"], "Age": "mean", "Fare": "mean"}).round(3)
    gender_survival_stats.columns = ["Survivors", "Total", "Survival_Rate", "Average_Age", "Average_Fare"]
    gender_survival_stats = gender_survival_stats.reset_index()
    logger.debug(f"Calculated gender statistics for {len(gender_survival_stats)} groups")

    if show_statistics:
        st.subheader("Statistics by Gender")

        col1, col2, col3, col4 = st.columns(4)
        female_stats = gender_survival_stats[gender_survival_stats["Sex"] == "female"].iloc[0]
        male_stats = gender_survival_stats[gender_survival_stats["Sex"] == "male"].iloc[0]

        with col1:
            st.metric("Female Survivors", f"{int(female_stats['Survivors'])}/{int(female_stats['Total'])}")
        with col2:
            st.metric("Female Rate", f"{female_stats['Survival_Rate'] * 100:.1f}%")
        with col3:
            st.metric("Male Survivors", f"{int(male_stats['Survivors'])}/{int(male_stats['Total'])}")
        with col4:
            st.metric("Male Rate", f"{male_stats['Survival_Rate'] * 100:.1f}%")

    logger.info("Creating gender visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_gender_comparison = create_gender_survival_comparison(df)
        st.plotly_chart(fig_gender_comparison, use_container_width=True)

    with col2:
        fig_gender_class = create_gender_class_distribution(df)
        st.plotly_chart(fig_gender_class, use_container_width=True)

    st.subheader("Survival by Gender and Class")
    fig_gender_class_survival = create_gender_class_survival_analysis(df)
    st.plotly_chart(fig_gender_class_survival, use_container_width=True)

    if show_interpretations:
        st.markdown(
            """
            **Gender and Survival Interpretation:**

            - **"Women and children first" rule**: Clearly reflected by ~75% female survival versus ~20% male survival
            - **Sharp difference**: Gender was the strongest factor affecting survival
            - **Variation by class**: Even among women, passenger class changed the odds of survival

            The evacuation protocol strongly favored women.
            """
        )

elif analysis_focus == "Age and Survival":
    logger.info("Starting age and survival analysis")
    st.header("4. Age and Survival Analysis")

    logger.info("Creating age groups")
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 25, 40, 100],
        labels=["Children (0-12)", "Youth (13-25)", "Adults (26-40)", "Older Adults (41+)"],
    )

    age_survival_stats = df.groupby("Age_Group").agg({"Survived": ["sum", "count", "mean"]}).round(3)
    age_survival_stats.columns = ["Survivors", "Total", "Survival_Rate"]
    age_survival_stats = age_survival_stats.reset_index()
    logger.debug(f"Calculated age statistics for {len(age_survival_stats)} groups")

    if show_statistics:
        st.subheader("Statistics by Age Group")
        st.dataframe(age_survival_stats, use_container_width=True)

    logger.info("Creating age visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_age_dist = create_age_survival_distribution(df)
        st.plotly_chart(fig_age_dist, use_container_width=True)

    with col2:
        fig_age_rates = create_age_group_survival_rates(df)
        st.plotly_chart(fig_age_rates, use_container_width=True)

    st.subheader("Survival by Age and Gender")
    fig_age_gender = create_age_gender_survival_analysis(df)
    st.plotly_chart(fig_age_gender, use_container_width=True)

    if show_interpretations:
        st.markdown(
            """
            **Age and Survival Interpretation:**

            - **Children favored**: ~58% survival, reflecting the "children first" principle
            - **Young adults**: Lower survival rate (~36%), driven in part by many young men
            - **Gender differences**: These remain visible across all age groups
            - **Older adults**: Physical difficulties during evacuation (~36% survival)
            """
        )

elif analysis_focus == "Fare and Survival":
    logger.info("Starting fare and survival analysis")
    st.header("5. Ticket Fare and Survival Analysis")

    logger.info("Creating fare categories")
    df["Fare_Category"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop")

    fare_survival_stats = df.groupby("Fare_Category").agg(
        {"Survived": ["sum", "count", "mean"], "Fare": ["min", "max", "mean"]}
    ).round(2)
    fare_survival_stats.columns = ["Survivors", "Total", "Survival_Rate", "Fare_Min", "Fare_Max", "Average_Fare"]
    fare_survival_stats = fare_survival_stats.reset_index()
    logger.debug(f"Calculated fare statistics for {len(fare_survival_stats)} categories")

    if show_statistics:
        st.subheader("Statistics by Fare Category")
        st.dataframe(fare_survival_stats, use_container_width=True)

    logger.info("Creating fare visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_fare_survival = create_fare_category_survival(df)
        st.plotly_chart(fig_fare_survival, use_container_width=True)

    with col2:
        fig_fare_dist = create_fare_distribution_by_survival(df)
        st.plotly_chart(fig_fare_dist, use_container_width=True)

    st.subheader("Fare-Class-Survival Relationship")
    fig_fare_class = create_fare_class_survival_analysis(df)
    st.plotly_chart(fig_fare_class, use_container_width=True)

    if show_interpretations:
        st.markdown(
            """
            **Fare and Survival Interpretation:**

            - **Positive correlation**: Higher fares are associated with higher survival
            - **Critical threshold**: There is a meaningful gap between low and high fare groups
            - **Proxy for social status**: Fare reflects social position
            - **Access to resources**: More expensive tickets often meant cabins closer to lifeboats
            """
        )

elif analysis_focus == "Family and Survival":
    logger.info("Starting family and survival analysis")
    st.header("6. Family and Survival Analysis")

    logger.info("Calculating family statistics")
    family_survival_stats = df.groupby("Family_Size").agg({"Survived": ["sum", "count", "mean"]}).round(3)
    family_survival_stats.columns = ["Survivors", "Total", "Survival_Rate"]
    family_survival_stats = family_survival_stats.reset_index()
    logger.debug(f"Calculated family statistics for {len(family_survival_stats)} family sizes")

    if show_statistics:
        st.subheader("Statistics by Family Size")
        st.dataframe(family_survival_stats, use_container_width=True)

    logger.info("Creating family visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_family_survival = create_family_size_survival(df)
        st.plotly_chart(fig_family_survival, use_container_width=True)

    with col2:
        fig_alone_family = create_alone_vs_family_analysis(df)
        st.plotly_chart(fig_alone_family, use_container_width=True)

    st.subheader("Detailed Family Composition Analysis")
    fig_family_composition = create_family_composition_analysis(df)
    st.plotly_chart(fig_family_composition, use_container_width=True)

    if show_interpretations:
        st.markdown(
            """
            **Family and Survival Interpretation:**

            - **Small to medium families**: Best survival outcomes (2-4 members)
            - **Passengers traveling alone**: Lower survival rate (~32%)
            - **Large families**: Logistical difficulties during evacuation
            - **Mutual support**: Smaller families could help one another
            """
        )

elif analysis_focus == "Combined Analysis":
    logger.info("Starting combined multi-factor analysis")
    st.header("7. Combined Multi-Factor Analysis")

    st.subheader("Combined Factors Dashboard")
    fig_combined = create_multivariate_survival_analysis(df)
    st.plotly_chart(fig_combined, use_container_width=True)

    logger.info("Calculating influence factor ranking")
    st.subheader("Influence Factor Ranking")

    factors_ranking = calculate_survival_factors_ranking(df)
    if factors_ranking is not None:
        st.dataframe(factors_ranking, use_container_width=True)

    logger.info("Creating best and worst-case scenarios")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Survival Profile")
        optimal_profile = {
            "Gender": "Woman",
            "Class": "1st Class",
            "Age": "Girl or young woman",
            "Family": "Small family (2-4 members)",
            "Fare": "High",
        }
        for key, value in optimal_profile.items():
            st.write(f"**{key}**: {value}")

    with col2:
        st.subheader("Critical Survival Profile")
        critical_profile = {
            "Gender": "Man",
            "Class": "3rd Class",
            "Age": "Young adult",
            "Family": "Alone or very large family",
            "Fare": "Low",
        }
        for key, value in critical_profile.items():
            st.write(f"**{key}**: {value}")

logger.info(f"Page {__name__} completed successfully")
