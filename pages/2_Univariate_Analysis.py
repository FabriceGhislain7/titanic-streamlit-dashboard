"""
pages/2_Univariate_Analysis.py
Univariate analysis of individual Titanic dataset variables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

from src.components.charts import create_survival_overview_chart
from src.components.univariate_charts import *
from src.config import COLOR_PALETTES, COLUMN_LABELS, PAGE_CONFIG
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, detect_outliers_iqr, handle_outliers
from src.utils.log import logger

logger.info(f"Loading page {__name__}")


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
logger.info(f"Cleaned data shape: {df.shape}")

logger.info("Setting up page header")
st.title("Univariate Analysis - Single Variables")
st.markdown("Detailed exploration of the individual characteristics of each variable")

logger.info("Setting up sidebar controls")
with st.sidebar:
    st.header("Analysis Controls")

    numeric_variables = [col for col in df.select_dtypes(include=[np.number]).columns if col != "PassengerId"]
    categorical_variables = [col for col in df.select_dtypes(include=["object"]).columns if col not in ["Name", "Ticket"]]

    analysis_type = st.selectbox(
        "Analysis Type:",
        ["General Overview", "Numerical Variables", "Categorical Variables", "Focus on Age", "Focus on Survival"],
    )
    logger.debug(f"Selected analysis type: {analysis_type}")

    show_statistics = st.checkbox("Show detailed statistics", value=True)
    show_outliers = st.checkbox("Enable outlier analysis", value=True)
    logger.debug(f"Display options: stats={show_statistics}, outliers={show_outliers}")

if analysis_type == "General Overview":
    logger.info("Starting general overview analysis")
    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Passengers", f"{len(df):,}")
    with col2:
        st.metric("Numerical Variables", len(numeric_variables))
    with col3:
        st.metric("Categorical Variables", len(categorical_variables))
    with col4:
        survival_rate = df["Survived"].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
        logger.debug(f"Survival rate: {survival_rate:.1f}%")

    logger.info("Creating primary distributions")
    st.subheader("Main Distributions")

    col1, col2 = st.columns(2)

    with col1:
        logger.debug("Creating detailed age distribution chart")
        fig_age = create_age_distribution_detailed(df)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        logger.debug("Creating survival distribution chart")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)

    if show_statistics:
        logger.info("Calculating descriptive statistics")
        st.subheader("Descriptive Statistics for Numerical Variables")
        stats_df = df[numeric_variables].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

elif analysis_type == "Numerical Variables":
    logger.info("Starting numerical variable analysis")
    st.header("2. Numerical Variable Analysis")

    selected_var = st.selectbox(
        "Select a numerical variable:",
        numeric_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x),
    )
    logger.debug(f"Selected numerical variable: {selected_var}")

    if selected_var:
        st.subheader(f"Detailed analysis: {COLUMN_LABELS.get(selected_var, selected_var)}")

        col1, col2, col3, col4 = st.columns(4)
        var_data = df[selected_var].dropna()

        with col1:
            mean_val = var_data.mean()
            st.metric("Mean", f"{mean_val:.2f}")
        with col2:
            median_val = var_data.median()
            st.metric("Median", f"{median_val:.2f}")
        with col3:
            std_val = var_data.std()
            st.metric("Standard Deviation", f"{std_val:.2f}")
        with col4:
            unique_val = var_data.nunique()
            st.metric("Unique Values", f"{unique_val}")

        logger.debug(
            f"Statistics for {selected_var}: mean={mean_val:.2f}, median={median_val:.2f}, "
            f"std={std_val:.2f}, uniques={unique_val}"
        )

        logger.debug("Creating numerical analysis charts")
        fig = create_numerical_analysis_charts(df, selected_var)
        st.plotly_chart(fig, use_container_width=True)

        if show_outliers:
            logger.info("Analyzing outliers")
            st.subheader("Outlier Analysis")

            outliers, lower_bound, upper_bound = detect_outliers_iqr(var_data)
            logger.debug(f"Detected outliers: {len(outliers)}, bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Outliers", len(outliers))
            with col2:
                st.metric("Lower Bound", f"{lower_bound:.2f}")
            with col3:
                st.metric("Upper Bound", f"{upper_bound:.2f}")

            if len(outliers) > 0:
                st.write("**Outlier values:**")
                outliers_df = pd.DataFrame({"Index": outliers.index, "Value": outliers.values}).sort_values("Value")
                st.dataframe(outliers_df, use_container_width=True)

elif analysis_type == "Categorical Variables":
    logger.info("Starting categorical variable analysis")
    st.header("3. Categorical Variable Analysis")

    selected_cat_var = st.selectbox(
        "Select a categorical variable:",
        categorical_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x),
    )
    logger.debug(f"Selected categorical variable: {selected_cat_var}")

    if selected_cat_var:
        st.subheader(f"Detailed analysis: {COLUMN_LABELS.get(selected_cat_var, selected_cat_var)}")

        value_counts = df[selected_cat_var].value_counts()
        value_props = df[selected_cat_var].value_counts(normalize=True) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Categories", len(value_counts))
        with col2:
            st.metric("Most Frequent Category", value_counts.index[0])
        with col3:
            st.metric("Highest Frequency", f"{value_props.iloc[0]:.1f}%")

        logger.debug(
            f"Statistics for {selected_cat_var}: {len(value_counts)} categories, "
            f"top={value_counts.index[0]}, freq={value_props.iloc[0]:.1f}%"
        )

        freq_table = pd.DataFrame(
            {"Category": value_counts.index, "Count": value_counts.values, "Percentage": value_props.values.round(1)}
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Frequency Table:**")
            st.dataframe(freq_table, use_container_width=True)

        with col2:
            logger.debug("Creating categorical analysis chart")
            fig_cat = create_categorical_analysis_chart(df, selected_cat_var)
            st.plotly_chart(fig_cat, use_container_width=True)

elif analysis_type == "Focus on Age":
    logger.info("Starting age-focused analysis")
    st.header("4. In-Depth Age Analysis")

    age_data = df["Age"].dropna()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mean_age = age_data.mean()
        st.metric("Average Age", f"{mean_age:.1f} years")
    with col2:
        median_age = age_data.median()
        st.metric("Median Age", f"{median_age:.1f} years")
    with col3:
        min_age = age_data.min()
        st.metric("Minimum Age", f"{min_age:.1f} years")
    with col4:
        max_age = age_data.max()
        st.metric("Maximum Age", f"{max_age:.1f} years")
    with col5:
        std_age = age_data.std()
        st.metric("Standard Deviation", f"{std_age:.1f} years")

    logger.debug(
        f"Age statistics: mean={mean_age:.1f}, median={median_age:.1f}, min={min_age:.1f}, "
        f"max={max_age:.1f}, std={std_age:.1f}"
    )

    logger.debug("Creating age analysis charts")
    fig_age_complete = create_age_complete_analysis(df)
    st.plotly_chart(fig_age_complete, use_container_width=True)

    st.subheader("Analysis by Age Group")

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 25, 40, 100],
        labels=["Children (0-12)", "Youth (13-25)", "Adults (26-40)", "Older Adults (41+)"],
    )

    age_group_stats = df["Age_Group"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribution by Group:**")
        age_groups_df = pd.DataFrame(
            {
                "Group": age_group_stats.index,
                "Count": age_group_stats.values,
                "Percentage": (age_group_stats.values / len(df) * 100).round(1),
            }
        )
        st.dataframe(age_groups_df, use_container_width=True)

    with col2:
        logger.debug("Creating age group chart")
        fig_age_groups = px.bar(
            x=age_group_stats.index,
            y=age_group_stats.values,
            title="Distribution by Age Group",
            labels={"x": "Age Group", "y": "Number of Passengers"},
        )
        st.plotly_chart(fig_age_groups, use_container_width=True)

    if show_outliers:
        logger.info("Analyzing age outliers")
        st.subheader("Age Outlier Handling")

        outliers, lower_bound, upper_bound = detect_outliers_iqr(age_data)
        logger.debug(f"Detected age outliers: {len(outliers)}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Detected outliers:** {len(outliers)}")
            st.write(f"**Lower bound:** {lower_bound:.1f} years")
            st.write(f"**Upper bound:** {upper_bound:.1f} years")

            if len(outliers) > 0:
                outliers_list = sorted(outliers.values)
                st.write("**Age outliers:**", outliers_list)

        with col2:
            outlier_method = st.selectbox(
                "Outlier handling method:",
                ["None", "Removal", "Median Replacement", "Boundary Clipping"],
            )
            logger.debug(f"Selected outlier handling method: {outlier_method}")

            if outlier_method != "None":
                if outlier_method == "Removal":
                    df_processed = handle_outliers(df, method="remove", columns=["Age"])
                elif outlier_method == "Median Replacement":
                    df_processed = handle_outliers(df, method="replace_median", columns=["Age"])
                else:
                    df_processed = handle_outliers(df, method="clip", columns=["Age"])

                st.write(f"**Dataset after handling:** {len(df_processed)} rows")
                logger.debug(f"Rows after outlier handling: {len(df_processed)}")

                logger.debug("Creating outlier comparison chart")
                fig_comparison = create_outlier_comparison_chart(df, df_processed, "Age")
                st.plotly_chart(fig_comparison, use_container_width=True)

elif analysis_type == "Focus on Survival":
    logger.info("Starting survival-focused analysis")
    st.header("5. In-Depth Survival Analysis")

    survival_stats = df["Survived"].value_counts()
    survival_props = df["Survived"].value_counts(normalize=True) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        survived = int(survival_stats[1])
        st.metric("Survivors", survived)
    with col3:
        died = int(survival_stats[0])
        st.metric("Deaths", died)
    with col4:
        survival_rate = survival_props[1]
        st.metric("Survival Rate", f"{survival_rate:.1f}%")

    logger.debug(
        f"Survival statistics: total={len(df)}, survived={survived}, "
        f"died={died}, rate={survival_rate:.1f}%"
    )

    col1, col2 = st.columns(2)

    with col1:
        logger.debug("Creating survival pie chart")
        fig_pie = create_survival_overview_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        logger.debug("Creating survival bar chart")
        fig_bar = px.bar(
            x=["Non Survivors", "Survivors"],
            y=[died, survived],
            title="Survival Counts",
            color=["Non Survivors", "Survivors"],
            color_discrete_map={
                "Non Survivors": COLOR_PALETTES["survival"][0],
                "Survivors": COLOR_PALETTES["survival"][1],
            },
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Survival Distribution in the Dataset")

    survival_interpretation = f"""
    **Interpretation of the results:**

    - **Mortality rate**: {survival_props[0]:.1f}% of passengers did not survive
    - **Survival rate**: {survival_props[1]:.1f}% of passengers survived
    - **Difference**: The probability of death was {survival_props[0]/survival_props[1]:.1f} times higher than survival

    This confirms the severity of the Titanic disaster, where the majority of passengers lost their lives.
    """

    st.markdown(survival_interpretation)

logger.info(f"Page {__name__} completed successfully")
