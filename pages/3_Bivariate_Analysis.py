"""
pages/3_Bivariate_Analysis.py
Bivariate analysis of factors that influence survival
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, create_basic_features
from src.components.bivariate_charts import *
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Page configuration (from config.py)
def setup_page():
    """Configure Streamlit page"""
    logger.info("Configuring Streamlit page")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Data loading and preparation (from notebook sections 2.1, 3, and feature engineering)
logger.info("Loading Titanic data")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load data")
    st.stop()

logger.info("Basic data cleaning")
df = clean_dataset_basic(df_original)
logger.info("Creating basic features")
df = create_basic_features(df)
logger.info(f"Data prepared. Shape: {df.shape}")

# ----------------3. Page header
logger.info("Setting up page header")
st.title("Bivariate Analysis - Survival Factors")
st.markdown("Exploration of relationships between variables and passenger survival")

# ----------------4. Sidebar controls
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
            "Combined Analysis"
        ]
    )
    logger.debug(f"Selected analysis focus: {analysis_focus}")
    
    show_statistics = st.checkbox("Show detailed statistics", value=True)
    show_interpretations = st.checkbox("Show interpretations", value=True)
    logger.debug(f"Display options: stats={show_statistics}, interpretations={show_interpretations}")

# ----------------5. General Overview (overview of main factors)
if analysis_focus == "General Overview":
    logger.info("Starting general overview analysis")
    st.header("1. Survival Factors Overview")
    
    # ----------------6. Main metrics (from multiple notebook sections)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_survival = df['Survived'].mean() * 100
        st.metric("Overall Survival", f"{overall_survival:.1f}%")
    
    with col2:
        female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
        st.metric("Female Survival", f"{female_survival:.1f}%")
    
    with col3:
        first_class_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100
        st.metric("1st Class Survival", f"{first_class_survival:.1f}%")
    
    with col4:
        child_survival = df[df['Age'] <= 12]['Survived'].mean() * 100
        st.metric("Children Survival", f"{child_survival:.1f}%")
    
    logger.debug(f"Calculated metrics: general={overall_survival:.1f}%, female={female_survival:.1f}%, first_class={first_class_survival:.1f}%, children={child_survival:.1f}%")
    
    # ----------------7. Overview charts (from notebook sections 4.2.2.x)
    logger.info("Creating overview charts")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating survival by class chart")
        # Survival by class
        fig_class = create_survival_by_class_detailed(df)
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col2:
        logger.debug("Creating survival by gender chart")
        # Survival by gender
        fig_gender = create_survival_by_gender_detailed(df)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # ----------------8. Correlation heatmap (from notebook section 4.1.2)
    logger.debug("Creating correlation heatmap")
    st.subheader("Correlation Matrix")
    fig_corr = create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------9. Class and Survival (from notebook sections 4.2.2.1 and 4.2.2.2)
elif analysis_focus == "Class and Survival":
    logger.info("Starting class and survival analysis")
    st.header("2. Passenger Class and Survival Analysis")
    
    # ----------------10. Class statistics (from notebook section 4.2.2.2)
    logger.info("Calculating class statistics")
    class_survival_stats = df.groupby('Pclass').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Age': 'mean',
        'Fare': 'mean'
    }).round(3)
    
    class_survival_stats.columns = ['Survivors', 'Total', 'Survival_Rate', 'Mean_Age', 'Mean_Fare']
    class_survival_stats = class_survival_stats.reset_index()
    logger.debug(f"Class statistics calculated for {len(class_survival_stats)} classes")
    
    if show_statistics:
        logger.debug("Showing class statistics")
        st.subheader("Statistics by Class")
        
        # Statistics table
        display_stats = class_survival_stats.copy()
        display_stats['Pclass'] = display_stats['Pclass'].map(VALUE_MAPPINGS['Pclass'])
        display_stats['Survival_Rate'] = (display_stats['Survival_Rate'] * 100).round(1)
        display_stats.columns = ['Class', 'Survivors', 'Total', 'Rate (%)', 'Mean Age', 'Mean Fare']
        
        st.dataframe(display_stats, use_container_width=True)
    
    # ----------------11. Class visualizations (from notebook section 4.2.2.2)
    logger.info("Creating class visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating class distribution chart")
        # Class counts
        fig_class_counts = create_class_distribution_analysis(df)
        st.plotly_chart(fig_class_counts, use_container_width=True)
    
    with col2:
        logger.debug("Creating survival rates by class chart")
        # Survival rates
        fig_class_survival = create_survival_rates_by_class(df)
        st.plotly_chart(fig_class_survival, use_container_width=True)
    
    # ----------------12. Detailed class analysis
    logger.debug("Creating detailed class analysis")
    st.subheader("Detailed Analysis by Class")
    fig_class_detailed = create_class_survival_detailed_analysis(df)
    st.plotly_chart(fig_class_detailed, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Showing class interpretations")
        st.markdown("""
        **Class and Survival Interpretation:**
        
        - **1st Class**: Highest survival rate (~65%), privileged access to lifeboats
        - **2nd Class**: Intermediate rate (~48%), intermediate position on ship
        - **3rd Class**: Lowest rate (~24%), difficulty accessing evacuation areas
        
        Social class had a significant impact on survival.
        """)

# ----------------13. Gender and Survival (from notebook section 4.2.2.3)
elif analysis_focus == "Gender and Survival":
    logger.info("Starting gender and survival analysis")
    st.header("3. Gender and Survival Analysis")
    
    # ----------------14. Gender statistics (from notebook section 4.2.2.3)
    logger.info("Calculating gender statistics")
    gender_survival_stats = df.groupby('Sex').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Age': 'mean',
        'Fare': 'mean'
    }).round(3)
    
    gender_survival_stats.columns = ['Survivors', 'Total', 'Survival_Rate', 'Mean_Age', 'Mean_Fare']
    gender_survival_stats = gender_survival_stats.reset_index()
    logger.debug(f"Gender statistics calculated for {len(gender_survival_stats)} genders")
    
    if show_statistics:
        logger.debug("Showing gender statistics")
        st.subheader("Statistics by Gender")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        female_stats = gender_survival_stats[gender_survival_stats['Sex'] == 'female'].iloc[0]
        male_stats = gender_survival_stats[gender_survival_stats['Sex'] == 'male'].iloc[0]
        
        with col1:
            st.metric("Women Survivors", f"{int(female_stats['Survivors'])}/{int(female_stats['Total'])}")
        with col2:
            st.metric("Women Rate", f"{female_stats['Survival_Rate']*100:.1f}%")
        with col3:
            st.metric("Men Survivors", f"{int(male_stats['Survivors'])}/{int(male_stats['Total'])}")
        with col4:
            st.metric("Men Rate", f"{male_stats['Survival_Rate']*100:.1f}%")
    
    # ----------------15. Gender visualizations (from notebook section 4.2.2.3)
    logger.info("Creating gender visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating gender survival comparison chart")
        # Gender survival comparison
        fig_gender_comparison = create_gender_survival_comparison(df)
        st.plotly_chart(fig_gender_comparison, use_container_width=True)
    
    with col2:
        logger.debug("Creating gender-class distribution chart")
        # Gender and class distribution
        fig_gender_class = create_gender_class_distribution(df)
        st.plotly_chart(fig_gender_class, use_container_width=True)
    
    # ----------------16. Gender by class analysis
    logger.debug("Creating gender by class analysis")
    st.subheader("Survival by Gender and Class")
    fig_gender_class_survival = create_gender_class_survival_analysis(df)
    st.plotly_chart(fig_gender_class_survival, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Showing gender interpretations")
        st.markdown("""
        **Gender and Survival Interpretation:**
        
        - **"Women and children first" rule**: Clearly applied with ~75% female survival vs ~20% male survival
        - **Dramatic difference**: Gender was the most determining factor for survival
        - **Class variations**: Even among women, class influenced probabilities
        
        The evacuation protocol clearly favored women.
        """)

# ----------------17. Age and Survival (from notebook section 4.2.2.4)
elif analysis_focus == "Age and Survival":
    logger.info("Starting age and survival analysis")
    st.header("4. Age and Survival Analysis")
    
    # ----------------18. Creating age groups (from notebook section 4.2.2.4)
    logger.info("Creating age groups")
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 25, 40, 100], 
                            labels=['Children (0-12)', 'Young (13-25)', 'Adults (26-40)', 'Seniors (41+)'])
    
    # Age group statistics
    age_survival_stats = df.groupby('Age_Group').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    age_survival_stats.columns = ['Survivors', 'Total', 'Survival_Rate']
    age_survival_stats = age_survival_stats.reset_index()
    logger.debug(f"Age statistics calculated for {len(age_survival_stats)} groups")
    
    if show_statistics:
        logger.debug("Showing age group statistics")
        st.subheader("Statistics by Age Group")
        st.dataframe(age_survival_stats, use_container_width=True)
    
    # ----------------19. Age visualizations (from notebook section 4.2.2.4)
    logger.info("Creating age visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating age distribution by survival chart")
        # Age distribution by survival
        fig_age_dist = create_age_survival_distribution(df)
        st.plotly_chart(fig_age_dist, use_container_width=True)
    
    with col2:
        logger.debug("Creating age group survival rates chart")
        # Rates by age group
        fig_age_rates = create_age_group_survival_rates(df)
        st.plotly_chart(fig_age_rates, use_container_width=True)
    
    # ----------------20. Age by gender analysis (from notebook section 4.2.2.4)
    logger.debug("Creating age by gender analysis")
    st.subheader("Survival by Age and Gender")
    fig_age_gender = create_age_gender_survival_analysis(df)
    st.plotly_chart(fig_age_gender, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Showing age interpretations")
        st.markdown("""
        **Age and Survival Interpretation:**
        
        - **Children favored**: ~58% survival, application of "children first"
        - **Young adults**: Lowest rate (~36%), many young men
        - **Gender differences**: Persistent across all age groups
        - **Seniors**: Physical difficulties in evacuation (~36% survival)
        """)

# ----------------21. Fare and Survival (from notebook section 4.2.2.5)
elif analysis_focus == "Fare and Survival":
    logger.info("Starting fare and survival analysis")
    st.header("5. Ticket Fare and Survival Analysis")
    
    # ----------------22. Fare categories (from notebook section 4.2.2.5)
    logger.info("Creating fare categories")
    df['Fare_Category'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    
    # Fare category statistics
    fare_survival_stats = df.groupby('Fare_Category').agg({
        'Survived': ['sum', 'count', 'mean'],
        'Fare': ['min', 'max', 'mean']
    }).round(2)
    
    fare_survival_stats.columns = ['Survivors', 'Total', 'Survival_Rate', 'Min_Fare', 'Max_Fare', 'Mean_Fare']
    fare_survival_stats = fare_survival_stats.reset_index()
    logger.debug(f"Fare statistics calculated for {len(fare_survival_stats)} categories")
    
    if show_statistics:
        logger.debug("Showing fare category statistics")
        st.subheader("Statistics by Fare Category")
        st.dataframe(fare_survival_stats, use_container_width=True)
    
    # ----------------23. Fare visualizations (from notebook section 4.2.2.5)
    logger.info("Creating fare visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating survival by fare category chart")
        # Survival by fare category
        fig_fare_survival = create_fare_category_survival(df)
        st.plotly_chart(fig_fare_survival, use_container_width=True)
    
    with col2:
        logger.debug("Creating fare distribution by survival chart")
        # Fare distribution by survival
        fig_fare_dist = create_fare_distribution_by_survival(df)
        st.plotly_chart(fig_fare_dist, use_container_width=True)
    
    # ----------------24. Fare-class correlation
    logger.debug("Creating fare-class-survival analysis")
    st.subheader("Fare-Class-Survival Relationship")
    fig_fare_class = create_fare_class_survival_analysis(df)
    st.plotly_chart(fig_fare_class, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Showing fare interpretations")
        st.markdown("""
        **Fare and Survival Interpretation:**
        
        - **Positive correlation**: Higher fare = greater survival
        - **Critical threshold**: Significant difference between low and high fares
        - **Social status proxy**: Fare reflects social position
        - **Resource access**: Expensive tickets = cabins near lifeboats
        """)

# ----------------25. Family and Survival (from notebook section 4.2.2.6)
elif analysis_focus == "Family and Survival":
    logger.info("Starting family and survival analysis")
    st.header("6. Family and Survival Analysis")
    
    # ----------------26. Family statistics (from notebook section 4.2.2.6)
    logger.info("Calculating family statistics")
    family_survival_stats = df.groupby('Family_Size').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    family_survival_stats.columns = ['Survivors', 'Total', 'Survival_Rate']
    family_survival_stats = family_survival_stats.reset_index()
    logger.debug(f"Family statistics calculated for {len(family_survival_stats)} sizes")
    
    if show_statistics:
        logger.debug("Showing family size statistics")
        st.subheader("Statistics by Family Size")
        st.dataframe(family_survival_stats, use_container_width=True)
    
    # ----------------27. Family visualizations (from notebook section 4.2.2.6)
    logger.info("Creating family visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating survival by family size chart")
        # Survival by family size
        fig_family_survival = create_family_size_survival(df)
        st.plotly_chart(fig_family_survival, use_container_width=True)
    
    with col2:
        logger.debug("Creating alone vs family chart")
        # Alone vs family
        fig_alone_family = create_alone_vs_family_analysis(df)
        st.plotly_chart(fig_alone_family, use_container_width=True)
    
    # ----------------28. Detailed family analysis
    logger.debug("Creating family composition analysis")
    st.subheader("Detailed Family Composition Analysis")
    fig_family_composition = create_family_composition_analysis(df)
    st.plotly_chart(fig_family_composition, use_container_width=True)
    
    if show_interpretations:
        logger.debug("Showing family interpretations")
        st.markdown("""
        **Family and Survival Interpretation:**
        
        - **Small-medium families**: Better survival (2-4 members)
        - **Solo travelers**: Reduced survival (~32%)
        - **Large families**: Logistical difficulties in evacuation
        - **Mutual support**: Small families help each other
        """)

# ----------------29. Combined Analysis (synthesis of all factors)
elif analysis_focus == "Combined Analysis":
    logger.info("Starting combined multi-factor analysis")
    st.header("7. Combined Multi-Factor Analysis")
    
    # ----------------30. Combined dashboard
    logger.info("Creating combined dashboard")
    st.subheader("Combined Factors Dashboard")
    
    # Multivariate analysis
    logger.debug("Creating multivariate analysis chart")
    fig_combined = create_multivariate_survival_analysis(df)
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # ----------------31. Top influence factors
    logger.info("Calculating influence factors ranking")
    st.subheader("Influence Factors Ranking")
    
    factors_ranking = calculate_survival_factors_ranking(df)
    if factors_ranking is not None:
        logger.debug("Showing factors ranking")
        st.dataframe(factors_ranking, use_container_width=True)
    else:
        logger.debug("Factors ranking not available")
    
    # ----------------32. Optimal/worst scenarios
    logger.info("Creating optimal/worst scenarios")
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Showing optimal profile")
        st.subheader("Optimal Survival Profile")
        optimal_profile = {
            "Gender": "Female",
            "Class": "1st Class", 
            "Age": "Child or young woman",
            "Family": "Small family (2-4 members)",
            "Fare": "High"
        }
        
        for key, value in optimal_profile.items():
            st.write(f"**{key}**: {value}")
    
    with col2:
        logger.debug("Showing critical profile")
        st.subheader("Critical Survival Profile")
        critical_profile = {
            "Gender": "Male",
            "Class": "3rd Class",
            "Age": "Young adult",
            "Family": "Alone or very large family",
            "Fare": "Low"
        }
        
        for key, value in critical_profile.items():
            st.write(f"**{key}**: {value}")

# ----------------33. Methodological notes
with st.expander("Methodological Notes"):
    st.markdown("""
    **Analysis methodology based on:**
    
    - **Notebook section 4.2.2.1**: Ticket analysis by class
    - **Notebook section 4.2.2.2**: Survival by class
    - **Notebook section 4.2.2.3**: Survival by gender
    - **Notebook section 4.2.2.4**: Survival by age groups
    - **Notebook section 4.2.2.5**: Survival by ticket fare
    - **Notebook section 4.2.2.6**: Survival by family size
    - **Notebook section 4.1.2**: Correlation matrix
    
    **Implemented analyses:**
    - Contingency tables
    - Chi-square tests (implicit in distributions)
    - Proportion analysis
    - Multiple correlations
    - Multivariate segmentation
    
    **Applied feature engineering:**
    - Standardized age groups
    - Fare categories by quartiles
    - Family size (SibSp + Parch + 1)
    - Binary indicators (Is_Alone)
    """)

logger.info(f"Page {__name__} completed successfully")