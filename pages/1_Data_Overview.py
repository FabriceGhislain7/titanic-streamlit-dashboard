"""
pages/1_Data_Overview.py
Complete overview of the Titanic dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import PAGE_CONFIG, COLUMN_LABELS
from src.utils.data_loader import load_titanic_data, get_data_summary, get_missing_values_info, check_duplicates
from src.utils.data_processor import clean_dataset_basic, detect_outliers_summary
from src.components.charts import create_missing_values_heatmap, create_data_types_chart
from src.utils.log import logger

# Logger for file entry
logger.info(f"Loading page {__name__}")

# ----------------1. Page configuration (from config.py)
def setup_page():
    """Configure Streamlit page"""
    logger.info("Configuring Streamlit page")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Data loading (from notebook section 2.1 - Structure of dataset)
logger.info("Loading Titanic data")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load data")
    st.stop()
logger.info(f"Data loaded successfully. Shape: {df_original.shape}")

# ----------------3. Page header
logger.info("Setting up page header")
st.title("Titanic Dataset Overview")
st.markdown("Complete analysis of dataset structure, quality and characteristics")

# ----------------4. Sidebar controls
logger.info("Setting up sidebar controls")
with st.sidebar:
    st.header("Display Controls")
    
    # Display options
    show_raw_data = st.checkbox("Show raw data", value=False)
    show_cleaned_data = st.checkbox("Show cleaned data", value=True)
    show_statistics = st.checkbox("Show statistics", value=True)
    
    # Number of rows to display
    n_rows = st.slider("Rows to display", 5, 50, 10)
    logger.debug(f"Display parameters: raw={show_raw_data}, cleaned={show_cleaned_data}, stats={show_statistics}, rows={n_rows}")

# ----------------5. General dataset information (from notebook section 2.1)
logger.info("Displaying general dataset information")
st.header("1. General Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of Rows", f"{len(df_original):,}")
with col2:
    st.metric("Number of Columns", len(df_original.columns))
with col3:
    st.metric("Dataset Size", f"{df_original.memory_usage().sum() / 1024:.1f} KB")
with col4:
    duplicates_count = check_duplicates(df_original)
    st.metric("Duplicate Rows", duplicates_count)
    logger.debug(f"Found {duplicates_count} duplicate rows")

# ----------------6. Column structure (from notebook section 2.1)
logger.info("Displaying column structure")
st.subheader("Column Structure")

col1, col2 = st.columns(2)

with col1:
    st.write("**Column information:**")
    column_info = pd.DataFrame({
        'Column': df_original.columns,
        'Type': df_original.dtypes.astype(str),
        'Non-Null Values': df_original.count(),
        'Null Values': df_original.isnull().sum(),
        'Null Percentage': (df_original.isnull().sum() / len(df_original) * 100).round(2)
    })
    st.dataframe(column_info, use_container_width=True)

with col2:
    logger.debug("Creating data types chart")
    st.write("**Data types distribution:**")
    data_types = df_original.dtypes.value_counts()
    fig_types = px.pie(
        values=data_types.values,
        names=data_types.index.astype(str),
        title="Data Types Distribution"
    )
    st.plotly_chart(fig_types, use_container_width=True)

# ----------------8. Missing values analysis (from notebook section 2.2 - Missing values)
logger.info("Missing values analysis")
st.header("2. Missing Values Analysis")

missing_info = get_missing_values_info(df_original)

if missing_info is not None and not missing_info.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values Summary")
        st.dataframe(missing_info, use_container_width=True)
    
    with col2:
        logger.debug("Creating missing values chart")
        st.subheader("Missing Values Visualization")
        fig_missing = px.bar(
            missing_info,
            x='Column',
            y='Percentage',
            title="Missing Values Percentage by Column",
            color='Percentage',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    
    # Missing values heatmap
    logger.debug("Creating missing values heatmap")
    st.subheader("Missing Values Heatmap")
    fig_heatmap = create_missing_values_heatmap(df_original)
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    logger.info("No missing values detected")
    st.success("No missing values detected in the dataset!")

# ----------------10. Data Cleaning Preview (from notebook section 3 - Data Cleaning)
logger.info("Data cleaning preview")
st.header("3. Data Cleaning Preview")

# Apply basic cleaning
df_cleaned = clean_dataset_basic(df_original)
logger.info(f"Cleaned dataset. Shape: {df_cleaned.shape}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Dataset")
    st.write(f"Rows: {len(df_original)}")
    st.write(f"Columns: {len(df_original.columns)}")
    st.write(f"Missing values: {df_original.isnull().sum().sum()}")

with col2:
    st.subheader("Cleaned Dataset")
    st.write(f"Rows: {len(df_cleaned)}")
    st.write(f"Columns: {len(df_cleaned.columns)}")
    st.write(f"Missing values: {df_cleaned.isnull().sum().sum()}")

# ----------------11. Data visualization (from notebook section 2.1)
if show_raw_data:
    logger.debug("Displaying raw data")
    st.header("4. Raw Data")
    st.subheader("First rows of original dataset")
    st.dataframe(df_original.head(n_rows), use_container_width=True)
    
    st.subheader("Last rows of dataset")
    st.dataframe(df_original.tail(n_rows), use_container_width=True)

if show_cleaned_data:
    logger.debug("Displaying cleaned data")
    st.header("5. Cleaned Data")
    st.subheader("Dataset after basic cleaning")
    st.dataframe(df_cleaned.head(n_rows), use_container_width=True)

# ----------------12. Descriptive statistics (from notebook section 4.1.1 - Descriptive Statistics)
if show_statistics:
    logger.info("Calculating descriptive statistics")
    st.header("6. Descriptive Statistics")
    
    # Statistics for numerical variables
    st.subheader("Numerical Variables")
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df_cleaned[numeric_cols].describe(), use_container_width=True)
    
    # Statistics for categorical variables
    st.subheader("Categorical Variables")
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Name']  # Exclude Name
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            with st.expander(f"Analysis of {COLUMN_LABELS.get(col, col)}"):
                value_counts = df_cleaned[col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Counts:**")
                    st.dataframe(value_counts.reset_index())
                
                with col2:
                    logger.debug(f"Creating chart for {col}")
                    st.write("**Distribution:**")
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {COLUMN_LABELS.get(col, col)}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ----------------13. Outlier detection (from notebook section 4.1.1 - Outlier detection)
logger.info("Outlier detection")
st.header("7. Outlier Detection")

outliers_summary = detect_outliers_summary(df_cleaned)
if outliers_summary is not None:
    st.subheader("Outliers Summary by Variable")
    st.dataframe(outliers_summary, use_container_width=True)
    
    # Boxplot to visualize outliers
    st.subheader("Outliers Visualization")
    numeric_cols_for_outliers = [col for col in numeric_cols if col not in ['PassengerId']]
    
    if len(numeric_cols_for_outliers) > 0:
        selected_col = st.selectbox(
            "Select variable for outlier analysis:",
            numeric_cols_for_outliers,
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        
        logger.debug(f"Creating boxplot for {selected_col}")
        fig_box = px.box(
            df_cleaned,
            y=selected_col,
            title=f"Boxplot for {COLUMN_LABELS.get(selected_col, selected_col)}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ----------------14. Final summary
logger.info("Generating data quality summary")
st.header("8. Data Quality Summary")

quality_metrics = {
    "Completeness": f"{((df_cleaned.count().sum() / (len(df_cleaned) * len(df_cleaned.columns))) * 100):.1f}%",
    "Duplicates": f"{duplicates_count} rows",
    "Outliers detected": f"{outliers_summary['Outliers_Count'].sum() if outliers_summary is not None else 0} values",
    "Numerical variables": f"{len(numeric_cols)} columns",
    "Categorical variables": f"{len(categorical_cols)} columns"
}

col1, col2, col3 = st.columns(3)
metrics_items = list(quality_metrics.items())

for i, (metric, value) in enumerate(metrics_items):
    with [col1, col2, col3][i % 3]:
        st.metric(metric, value)

# ----------------15. Methodological notes (from notebook)
with st.expander("Methodological Notes"):
    st.markdown("""
    **Analysis methodology based on:**
    
    - **Notebook section 2.1**: Initial dataset structure
    - **Notebook section 2.2**: Missing values analysis
    - **Notebook section 2.3**: Duplicate checking
    - **Notebook section 3**: Data cleaning methods
    - **Notebook section 4.1.1**: Descriptive statistics
    
    **Applied transformations:**
    - Removal of 'Cabin' column (77% missing values)
    - Removal of duplicate rows
    - Outlier detection using IQR method
    """)

logger.info("Page 1_Data_Overview loaded successfully")