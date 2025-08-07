"""
pages/2_Univariate_Analysis.py
Univariate analysis of individual variables in the Titanic dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import PAGE_CONFIG, COLUMN_LABELS, COLOR_PALETTES
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, detect_outliers_iqr, handle_outliers
from src.components.charts import create_age_distribution_chart, create_survival_overview_chart
from src.components.univariate_charts import *
from src.utils.log import logger

# Logger for file entry
logger.info(f"Loading page {__name__}")

# ----------------1. Page configuration (from config.py)
def setup_page():
    """Configure Streamlit page"""
    logger.info("Configuring Streamlit page")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Data loading and cleaning (from notebook sections 2.1 and 3)
logger.info("Loading Titanic data")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load data")
    st.stop()

logger.info("Basic data cleaning")
df = clean_dataset_basic(df_original)
logger.info(f"Cleaned data. Shape: {df.shape}")

# ----------------3. Page header
logger.info("Setting up page header")
st.title("Univariate Analysis - Individual Variables")
st.markdown("Detailed exploration of individual characteristics of each variable")

# ----------------4. Sidebar controls
logger.info("Setting up sidebar controls")
with st.sidebar:
    st.header("Analysis Controls")
    
    # Variable selection for analysis
    numeric_variables = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col not in ['PassengerId']]
    categorical_variables = [col for col in df.select_dtypes(include=['object']).columns 
                           if col not in ['Name', 'Ticket']]
    
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["General Overview", "Numerical Variables", "Categorical Variables", "Age Focus", "Survival Focus"]
    )
    logger.debug(f"Selected analysis type: {analysis_type}")
    
    # Visualization options
    show_statistics = st.checkbox("Show detailed statistics", value=True)
    show_outliers = st.checkbox("Outlier analysis", value=True)
    logger.debug(f"Visualization options: stats={show_statistics}, outliers={show_outliers}")

# ----------------5. General Overview (from notebook section 4.1.1)
if analysis_type == "General Overview":
    logger.info("Starting general overview analysis")
    st.header("1. General Dataset Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Passengers", f"{len(df):,}")
    with col2:
        st.metric("Numerical Variables", len(numeric_variables))
    with col3:
        st.metric("Categorical Variables", len(categorical_variables))
    with col4:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
        logger.debug(f"Survival rate: {survival_rate:.1f}%")
    
    # ----------------6. Main distributions
    logger.info("Creating main distributions")
    st.subheader("Main Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating age distribution chart")
        fig_age = create_age_distribution_detailed(df)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        logger.debug("Creating survival distribution chart")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)
    
    # ----------------7. Summary statistics for all numerical variables
    if show_statistics:
        logger.info("Calculating descriptive statistics")
        st.subheader("Descriptive Statistics for Numerical Variables")
        stats_df = df[numeric_variables].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

# ----------------8. Numerical Variables Analysis (from notebook section 4.2.1)
elif analysis_type == "Numerical Variables":
    logger.info("Starting numerical variables analysis")
    st.header("2. Numerical Variables Analysis")
    
    # Specific variable selection
    selected_var = st.selectbox(
        "Select numerical variable:",
        numeric_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    logger.debug(f"Selected numerical variable: {selected_var}")
    
    if selected_var:
        st.subheader(f"Detailed analysis: {COLUMN_LABELS.get(selected_var, selected_var)}")
        
        # ----------------9. Basic statistics (from notebook section 4.2.1 - Age analysis)
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
        
        logger.debug(f"Statistics for {selected_var}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}, uniques={unique_val}")
        
        # ----------------10. Multiple visualizations (from notebook section 4.2.1)
        logger.debug("Creating numerical analysis charts")
        fig = create_numerical_analysis_charts(df, selected_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # ----------------11. Outlier analysis (from notebook section 4.2.1 - Outlier detection)
        if show_outliers:
            logger.info("Outlier analysis")
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
                outliers_df = pd.DataFrame({
                    'Index': outliers.index,
                    'Value': outliers.values
                }).sort_values('Value')
                st.dataframe(outliers_df, use_container_width=True)

# ----------------12. Categorical Variables Analysis (from notebook section 4.2.2)
elif analysis_type == "Categorical Variables":
    logger.info("Starting categorical variables analysis")
    st.header("3. Categorical Variables Analysis")
    
    # Categorical variable selection
    selected_cat_var = st.selectbox(
        "Select categorical variable:",
        categorical_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    logger.debug(f"Selected categorical variable: {selected_cat_var}")
    
    if selected_cat_var:
        st.subheader(f"Detailed analysis: {COLUMN_LABELS.get(selected_cat_var, selected_cat_var)}")
        
        # ----------------13. Categorical statistics (from notebook section 4.2.2)
        value_counts = df[selected_cat_var].value_counts()
        value_props = df[selected_cat_var].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Categories", len(value_counts))
        with col2:
            st.metric("Most Frequent Category", value_counts.index[0])
        with col3:
            st.metric("Maximum Frequency", f"{value_props.iloc[0]:.1f}%")
        
        logger.debug(f"Statistics for {selected_cat_var}: {len(value_counts)} categories, top={value_counts.index[0]}, freq={value_props.iloc[0]:.1f}%")
        
        # ----------------14. Frequency table
        freq_table = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': value_props.values.round(1)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Frequency Table:**")
            st.dataframe(freq_table, use_container_width=True)
        
        with col2:
            # ----------------15. Categorical visualization
            logger.debug("Creating categorical analysis chart")
            fig_cat = create_categorical_analysis_chart(df, selected_cat_var)
            st.plotly_chart(fig_cat, use_container_width=True)

# ----------------16. Age Focus (from notebook section 4.2.1 - Complete Age Analysis)
elif analysis_type == "Age Focus":
    logger.info("Starting age focus analysis")
    st.header("4. In-Depth Age Analysis")
    
    # ----------------17. Age statistics (from notebook section 4.2.1)
    age_data = df['Age'].dropna()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mean_age = age_data.mean()
        st.metric("Mean Age", f"{mean_age:.1f} years")
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
    
    logger.debug(f"Age statistics: mean={mean_age:.1f}, median={median_age:.1f}, min={min_age:.1f}, max={max_age:.1f}, std={std_age:.1f}")
    
    # ----------------18. Multiple age visualizations (from notebook section 4.2.1)
    logger.debug("Creating age analysis charts")
    fig_age_complete = create_age_complete_analysis(df)
    st.plotly_chart(fig_age_complete, use_container_width=True)
    
    # ----------------19. Age groups (from notebook section 4.2.2.4)
    st.subheader("Age Group Analysis")
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 25, 40, 100], 
                            labels=['Children (0-12)', 'Young (13-25)', 'Adults (26-40)', 'Seniors (41+)'])
    
    age_group_stats = df['Age_Group'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribution by Groups:**")
        age_groups_df = pd.DataFrame({
            'Group': age_group_stats.index,
            'Count': age_group_stats.values,
            'Percentage': (age_group_stats.values / len(df) * 100).round(1)
        })
        st.dataframe(age_groups_df, use_container_width=True)
    
    with col2:
        logger.debug("Creating age groups chart")
        fig_age_groups = px.bar(
            x=age_group_stats.index,
            y=age_group_stats.values,
            title="Distribution by Age Groups",
            labels={'x': 'Age Group', 'y': 'Number of Passengers'}
        )
        st.plotly_chart(fig_age_groups, use_container_width=True)
    
    # ----------------20. Age outlier treatment (from notebook section 4.2.1)
    if show_outliers:
        logger.info("Age outlier analysis")
        st.subheader("Age Outlier Management")
        
        outliers, lower_bound, upper_bound = detect_outliers_iqr(age_data)
        logger.debug(f"Age outliers detected: {len(outliers)}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Outliers detected:** {len(outliers)}")
            st.write(f"**Lower bound:** {lower_bound:.1f} years")
            st.write(f"**Upper bound:** {upper_bound:.1f} years")
            
            if len(outliers) > 0:
                outliers_list = sorted(outliers.values)
                st.write("**Outlier ages:**", outliers_list)
        
        with col2:
            # Outlier treatment options
            outlier_method = st.selectbox(
                "Outlier treatment method:",
                ["None", "Removal", "Replace with median", "Clip to bounds"]
            )
            logger.debug(f"Selected outlier treatment method: {outlier_method}")
            
            if outlier_method != "None":
                if outlier_method == "Removal":
                    df_processed = handle_outliers(df, method='remove', columns=['Age'])
                elif outlier_method == "Replace with median":
                    df_processed = handle_outliers(df, method='replace_median', columns=['Age'])
                else:  # Clipping
                    df_processed = handle_outliers(df, method='clip', columns=['Age'])
                
                st.write(f"**Dataset after treatment:** {len(df_processed)} rows")
                logger.debug(f"Data after outlier treatment: {len(df_processed)} rows")
                
                # Distribution comparison
                logger.debug("Creating outlier comparison chart")
                fig_comparison = create_outlier_comparison_chart(df, df_processed, 'Age')
                st.plotly_chart(fig_comparison, use_container_width=True)

# ----------------21. Survival Focus (from notebook section 4.2.2 - Survival Analysis)
elif analysis_type == "Survival Focus":
    logger.info("Starting survival focus analysis")
    st.header("5. In-Depth Survival Analysis")
    
    # ----------------22. Survival statistics (from notebook section 4.2.2)
    survival_stats = df['Survived'].value_counts()
    survival_props = df['Survived'].value_counts(normalize=True) * 100
    
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
    
    logger.debug(f"Survival statistics: total={len(df)}, survived={survived}, died={died}, rate={survival_rate:.1f}%")
    
    # ----------------23. Survival visualizations (from notebook section 4.2.2)
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating survival pie chart")
        fig_pie = create_survival_overview_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        logger.debug("Creating survival bar chart")
        fig_bar = px.bar(
            x=['Non-Survivors', 'Survivors'],
            y=[died, survived],
            title="Survival Count",
            color=['Non-Survivors', 'Survivors'],
            color_discrete_map={'Non-Survivors': COLOR_PALETTES['survival'][0], 
                               'Survivors': COLOR_PALETTES['survival'][1]}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ----------------24. Survival distribution analysis
    st.subheader("Survival Distribution in Dataset")
    
    survival_interpretation = f"""
    **Results interpretation:**
    
    - **Mortality rate**: {survival_props[0]:.1f}% of passengers did not survive
    - **Survival rate**: {survival_props[1]:.1f}% of passengers survived
    - **Difference**: The probability of death was {survival_props[0]/survival_props[1]:.1f} times higher than survival
    
    This data confirms the severity of the Titanic disaster, where the majority of passengers lost their lives.
    """
    
    st.markdown(survival_interpretation)

# ----------------25. Methodological notes
with st.expander("Methodological Notes"):
    st.markdown("""
    **Analysis methodology based on:**
    
    - **Notebook section 4.1.1**: General descriptive statistics
    - **Notebook section 4.2.1**: Univariate age analysis
    - **Notebook section 4.2.2**: Survival analysis
    - **Notebook section 4.1.1**: Outlier detection and management
    
    **Techniques used:**
    - Descriptive statistics (mean, median, standard deviation)
    - Multiple visualizations (histograms, boxplots, pie charts)
    - Outlier detection using IQR method
    - Frequency analysis for categorical variables
    - Age group creation for segmented analysis
    
    **Calculated metrics:**
    - Percentiles (25%, 50%, 75%)
    - Outlier bounds (Q1-1.5*IQR, Q3+1.5*IQR)
    - Frequency distributions
    - Survival rates
    """)

logger.info(f"Page {__name__} completed successfully")