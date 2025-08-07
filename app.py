# Main Streamlit application file
"""
app.py - Main Streamlit application file
Titanic Survival Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import *
from src.utils.data_loader import load_titanic_data, get_data_summary
from src.components.metrics import create_overview_metrics
from src.components.charts import create_survival_overview_chart, create_class_distribution_chart
from src.utils.log import logger

# Logger for file entry
logger.info(f"Loading file {__name__}")

# ----------------1. Main page configuration (from config.py)
def setup_page_config():
    """Configure Streamlit page"""
    logger.info("======================================== APP START ====================================")
    logger.info("Configuring Streamlit page")
    st.set_page_config(**PAGE_CONFIG)

def main():
    """Main application function"""
    logger.info("Starting main() function")
    
    setup_page_config()
    
    # ----------------2. Data loading (from notebook section 2.1 - Dataset Loading)
    logger.info("Loading Titanic data")
    df = load_titanic_data()
    if df is None:
        logger.error("Failed to load Titanic data")
        st.error("Error loading data")
        return
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    # ----------------3. Main header (from notebook Project Overview)
    logger.info("Setting up main header")
    st.title(APP_TEXTS['main_title'])
    st.markdown(APP_TEXTS['subtitle'])
    
    # ----------------4. Sidebar information
    logger.info("Setting up sidebar")
    with st.sidebar:
        st.header("Dataset Information")
        st.info(f"Total passengers: {len(df)}")
        st.info(f"Variables: {len(df.columns)}")
        st.info(APP_TEXTS['data_source'])
    
    # ----------------5. Overview Metrics (from notebook section 4.2.2 - Survival Analysis)
    logger.info("Creating overview metrics")
    st.subheader("General Overview")
    create_overview_metrics(df)
    
    # ----------------6. Main dashboard visualizations
    logger.info("Setting up main charts columns")
    col1, col2 = st.columns(2)
    
    with col1:
        # ----------------7. General survival chart (from notebook section 4.2.2)
        logger.debug("Creating survival chart")
        st.subheader("Survival Rate")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)
    
    with col2:
        # ----------------8. Class distribution (from notebook section 4.2.2.1)
        logger.debug("Creating class distribution chart")
        st.subheader("Distribution by Class")
        fig_class = create_class_distribution_chart(df)
        st.plotly_chart(fig_class, use_container_width=True)
    
    # ----------------9. Dataset information (from notebook section 2.1)
    logger.info("Setting up dataset details section")
    with st.expander("Dataset Details"):
        logger.debug("Generating data summary")
        summary = get_data_summary(df)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", summary['rows'])
            st.metric("Survived", summary['survived'])
        
        with col2:
            st.metric("Columns", summary['columns'])
            st.metric("Died", summary['died'])
        
        with col3:
            st.metric("Missing Values", summary['missing_values'])
            st.metric("Survival Rate", f"{summary['survival_rate']:.1f}%")
    
    # ----------------10. Footer (from config.py)
    logger.info("Setting up footer")
    st.markdown(APP_TEXTS['footer'])
    logger.info("Application started successfully")

if __name__ == "__main__":
    try:
        logger.info("Starting Streamlit application")
        main()
        logger.info("*************************** APP END *****************************")
    except Exception as e:
        logger.error(f"Error in application execution: {str(e)}")
        st.error("An error occurred in the application")
        raise