"""
app.py - Main Streamlit application file
Titanic Survival Analysis Dashboard
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components.charts import create_class_distribution_chart, create_survival_overview_chart
from src.components.metrics import create_overview_metrics
from src.config import *
from src.utils.data_loader import get_data_summary, load_titanic_data
from src.utils.log import logger

logger.info(f"Loading file {__name__}")


def setup_page_config():
    """Configure the main Streamlit page."""
    logger.info("======================================== START APP ====================================")
    logger.info("Configuring the Streamlit page")
    st.set_page_config(**PAGE_CONFIG)


def main():
    """Main application function."""
    logger.info("Starting main()")

    setup_page_config()

    logger.info("Loading Titanic data")
    df = load_titanic_data()
    if df is None:
        logger.error("Failed to load Titanic data")
        st.error("Error loading data")
        return
    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    logger.info("Setting up the main header")
    st.title(APP_TEXTS["main_title"])
    st.markdown(APP_TEXTS["subtitle"])

    logger.info("Setting up the sidebar")
    with st.sidebar:
        st.header("Dataset Information")
        st.info(f"Total passengers: {len(df)}")
        st.info(f"Variables: {len(df.columns)}")
        st.info(APP_TEXTS["data_source"])

    logger.info("Creating overview metrics")
    st.subheader("General Overview")
    create_overview_metrics(df)

    logger.info("Setting up main chart columns")
    col1, col2 = st.columns(2)

    with col1:
        logger.debug("Creating survival chart")
        st.subheader("Survival Rate")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)

    with col2:
        logger.debug("Creating class distribution chart")
        st.subheader("Class Distribution")
        fig_class = create_class_distribution_chart(df)
        st.plotly_chart(fig_class, use_container_width=True)

    logger.info("Setting up the dataset details section")
    with st.expander("Dataset Details"):
        logger.debug("Generating summary data")
        summary = get_data_summary(df)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", summary["rows"])
            st.metric("Survivors", summary["survived"])

        with col2:
            st.metric("Columns", summary["columns"])
            st.metric("Deaths", summary["died"])

        with col3:
            st.metric("Missing Values", summary["missing_values"])
            st.metric("Survival Rate", f"{summary['survival_rate']:.1f}%")

    logger.info("Setting up the footer")
    st.markdown(APP_TEXTS["footer"])
    logger.info("Application started successfully")


if __name__ == "__main__":
    try:
        logger.info("Starting Streamlit application")
        main()
        logger.info("*************************** END APP *****************************")
    except Exception as e:
        logger.error(f"Error while running the application: {str(e)}")
        st.error("An error occurred in the application")
        raise
