"""
pages/4_Advanced_Analytics.py
Advanced analytics, correlations, and feature engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, create_basic_features, handle_outliers
from src.components.advanced_charts import *
from src.utils.feature_engineering import *
from src.utils.statistical_analysis import *
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Page configuration (da config.py)
def setup_page():
    """Configure the Streamlit page"""
    logger.info("Configuring Streamlit page")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Load and prepare data (da notebook sezioni 2.1, 3, e feature engineering)
logger.info("Loading Titanic data")
df_original = load_titanic_data()
if df_original is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load the data")
    st.stop()

logger.info("Cleaning base data")
df = clean_dataset_basic(df_original)
logger.info("Creating base features")
df = create_basic_features(df)
logger.info(f"Data prepared. Shape: {df.shape}")

# ----------------3. Page header
logger.info("Setting up page header")
st.title("Advanced Analytics and Feature Engineering")
st.markdown("Advanced correlations, feature engineering, and deeper statistical analysis")

# ----------------4. Sidebar controls
logger.info("Setting up sidebar controls")
with st.sidebar:
    st.header("Advanced Analysis Controls")
    
    analysis_section = st.selectbox(
        "Analysis Section:",
        [
            "Advanced Correlation Matrix",
            "Feature Engineering",
            "Advanced Outlier Analysis", 
            "Statistical Analysis",
            "Pattern Mining",
            "Advanced Segmentation"
        ]
    )
    logger.debug(f"Selected analysis section: {analysis_section}")
    
    # Feature engineering options
    st.subheader("Feature Engineering")
    create_title_feature = st.checkbox("Extract title from name", value=True)
    create_deck_feature = st.checkbox("Extract deck from cabin", value=True)
    create_fare_per_person = st.checkbox("Calculate fare per person", value=True)
    logger.debug(f"Feature engineering options: title={create_title_feature}, deck={create_deck_feature}, fare_per_person={create_fare_per_person}")
    
    # Outlier options
    st.subheader("Outlier Handling")
    outlier_method = st.selectbox(
        "Outlier method:",
        ["None", "Removal", "Clipping", "Replacement"],
        index=0
    )
    logger.debug(f"Selected outlier method: {outlier_method}")

# ----------------5. Applying advanced feature engineering
logger.info("Applying advanced feature engineering")
df_engineered = df.copy()

if create_title_feature:
    logger.debug("Extracting title from name")
    df_engineered = extract_title_from_name(df_engineered)

if create_deck_feature:
    logger.debug("Extracting deck from cabin")
    df_engineered = extract_deck_from_cabin(df_engineered)

if create_fare_per_person:
    logger.debug("Calculating fare per person")
    df_engineered = calculate_fare_per_person(df_engineered)

# Apply outlier handling if requested
if outlier_method != "None":
    logger.info(f"Applying outlier handling: {outlier_method}")
    method_map = {
        "Removal": "remove",
        "Clipping": "clip", 
        "Replacement": "replace_median"
    }
    df_engineered = handle_outliers(df_engineered, method=method_map[outlier_method])
    logger.debug(f"Data after outlier handling. Shape: {df_engineered.shape}")
else:
    logger.debug("No outlier handling applied")

# ----------------6. Advanced Correlation Matrix (da notebook section 4.1.2)
if analysis_section == "Advanced Correlation Matrix":
    logger.info("Starting advanced correlation analysis")
    st.header("1. Advanced Correlation Analysis")
    
    # ----------------7. Correlations multiple (da notebook section 4.1.2 - Spearman correlation)
    logger.info("Creating multiple correlation matrices")
    st.subheader("Multiple Correlation Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logger.debug("Creating Pearson correlation")
        # Correlazione Pearson
        st.write("**Pearson correlation (linear):**")
        fig_pearson = create_correlation_matrix(df_engineered, method='pearson')
        st.plotly_chart(fig_pearson, use_container_width=True)
    
    with col2:
        logger.debug("Creating Spearman correlation")
        # Correlazione Spearman
        st.write("**Spearman correlation (monotonic):**")
        fig_spearman = create_correlation_matrix(df_engineered, method='spearman')
        st.plotly_chart(fig_spearman, use_container_width=True)
    
    # ----------------8. Top correlations con target (da notebook focus survival)
    logger.info("Calculating correlations with the target")
    st.subheader("Correlations with Survival")
    
    correlations_with_target = calculate_target_correlations(df_engineered, 'Survived')
    if correlations_with_target is not None:
        logger.debug("Displaying top correlations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Correlations Positive:**")
            st.dataframe(correlations_with_target.head(), use_container_width=True)
        
        with col2:
            logger.debug("Creating target correlation chart")
            # Grafico correlations
            fig_target_corr = create_target_correlation_chart(correlations_with_target)
            st.plotly_chart(fig_target_corr, use_container_width=True)
    else:
        logger.debug("Correlations with target unavailable")
    
    # ----------------9. Correlations per categorie (approfondimento)
    logger.debug("Creating correlations by category")
    st.subheader("Correlations by Category")
    
    # Correlations separate per gender
    fig_corr_by_gender = create_correlation_by_category(df_engineered, 'Sex')
    st.plotly_chart(fig_corr_by_gender, use_container_width=True)

# ----------------10. Feature Engineering (combinazione notebook + nuove features)
elif analysis_section == "Feature Engineering":
    logger.info("Starting feature engineering section")
    st.header("2. Advanced Feature Engineering")
    
    # ----------------11. Summary features create
    logger.info("Creating summary features")
    st.subheader("Existing vs New Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Features:**")
        original_features = [col for col in df.columns if col in df_original.columns]
        logger.debug(f"Original features: {len(original_features)}")
        for feature in original_features:
            st.write(f"- {COLUMN_LABELS.get(feature, feature)}")
    
    with col2:
        st.write("**Engineered Features:**")
        new_features = [col for col in df_engineered.columns if col not in df_original.columns]
        logger.debug(f"Features engineered: {len(new_features)}")
        for feature in new_features:
            st.write(f"- {feature}")
    
    # ----------------12. Analysis nuove features (da notebook + aggiunte)
    logger.info("Analyzing engineered features")
    st.subheader("Engineered Feature Analysis")
    
    # Title analysis (da nome)
    if 'Title' in df_engineered.columns:
        logger.debug("Analysis feature Title")
        st.write("**Title Analysis (extracted from Name):**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title_survival = df_engineered.groupby('Title')['Survived'].agg(['count', 'mean']).round(3)
            title_survival.columns = ['Count', 'Survival_Rate']
            st.dataframe(title_survival, use_container_width=True)
        
        with col2:
            fig_title = create_title_survival_analysis(df_engineered)
            st.plotly_chart(fig_title, use_container_width=True)
    else:
        logger.debug("Title feature unavailable")
    
    # Deck analysis (da cabin)
    if 'Deck' in df_engineered.columns:
        logger.debug("Analysis feature Deck")
        st.write("**Deck Analysis (extracted from Cabin):**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            deck_survival = df_engineered.groupby('Deck')['Survived'].agg(['count', 'mean']).round(3)
            deck_survival.columns = ['Count', 'Survival_Rate']
            st.dataframe(deck_survival, use_container_width=True)
        
        with col2:
            fig_deck = create_deck_survival_analysis(df_engineered)
            st.plotly_chart(fig_deck, use_container_width=True)
    else:
        logger.debug("Deck feature unavailable")
    
    # ----------------13. Feature importance approximation
    logger.info("Calculating feature importance")
    st.subheader("Approximate Feature Importance")
    
    feature_importance = calculate_feature_importance_proxy(df_engineered, 'Survived')
    if feature_importance is not None:
        logger.debug("Displaying feature importance")
        fig_importance = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        logger.debug("Feature importance unavailable")

# ----------------14. Advanced Outlier Analysis (da notebook section 4.1.1 estesa)
elif analysis_section == "Advanced Outlier Analysis":
    logger.info("Starting advanced outlier analysis")
    st.header("3. Advanced Outlier Analysis")
    
    # ----------------15. Detection outliers multivariato
    logger.info("Running multivariate outlier detection")
    st.subheader("Multivariate Outlier Detection")
    
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId']]
    logger.debug(f"Available numerical variables: {len(numeric_cols)}")
    
    if len(numeric_cols) > 1:
        selected_vars = st.multiselect(
            "Select variables for outlier analysis:",
            numeric_cols,
            default=numeric_cols[:3],
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        logger.debug(f"Variables selected for outlier analysis: {selected_vars}")
        
        if len(selected_vars) >= 2:
            logger.debug("Creating outlier visualizations")
            # ----------------16. Scatter plot outliers (da notebook esteso)
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter_outliers = create_outliers_scatter_plot(df_engineered, selected_vars[0], selected_vars[1])
                st.plotly_chart(fig_scatter_outliers, use_container_width=True)
            
            with col2:
                # Boxplot comparison
                fig_outliers_comparison = create_outliers_comparison_boxplot(df_engineered, selected_vars)
                st.plotly_chart(fig_outliers_comparison, use_container_width=True)
        else:
            logger.debug("Insufficient number of variables selected for outlier analysis")
    else:
        logger.debug("Insufficient number of numerical variables for outlier analysis")
    
    # ----------------17. Impatto outliers su correlations
    st.subheader("Impact of Outliers on Correlations")
    
    if outlier_method != "None":
        logger.info("Analyzing outlier impact on correlations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before handling:**")
            corr_before = df.select_dtypes(include=[np.number]).corr()['Survived'].abs().sort_values(ascending=False)
            st.dataframe(corr_before.head(10), use_container_width=True)
        
        with col2:
            st.write("**After handling:**")
            corr_after = df_engineered.select_dtypes(include=[np.number]).corr()['Survived'].abs().sort_values(ascending=False)
            st.dataframe(corr_after.head(10), use_container_width=True)
    else:
        logger.debug("No outlier handling available for correlation comparison")

# ----------------18. Statistical Analysis (test statistici e distribuzioni)
elif analysis_section == "Statistical Analysis":
    logger.info("Starting advanced statistical analysis")
    st.header("4. Advanced Statistical Analysis")
    
    # ----------------19. Test normalitÃ  (estensione analysis notebook)
    logger.info("Running normality tests")
    st.subheader("Normality Tests")
    
    numeric_vars = [col for col in df_engineered.select_dtypes(include=[np.number]).columns 
                   if col not in ['PassengerId', 'Survived']]
    logger.debug(f"Numerical variables available for normality tests: {len(numeric_vars)}")
    
    if numeric_vars:
        selected_var = st.selectbox(
            "Select a variable for normality testing:",
            numeric_vars,
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        logger.debug(f"Variable selected for normality testing: {selected_var}")
        
        # ----------------20. Normality visualizations
        logger.debug("Creating normality visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_normality = create_normality_test_plots(df_engineered, selected_var)
            st.plotly_chart(fig_normality, use_container_width=True)
        
        with col2:
            # Normality statistics
            logger.debug("Calculating normality statistics")
            normality_stats = calculate_normality_statistics(df_engineered, selected_var)
            st.write("**Normality Statistics:**")
            for stat_name, stat_value in normality_stats.items():
                st.metric(stat_name, f"{stat_value:.4f}")
    else:
        logger.debug("No numerical variable available for normality testing")
    
    # ----------------21. Group distribution comparison
    logger.info("Comparing distributions across groups")
    st.subheader("Distribution Comparison by Group")
    
    categorical_vars = ['Sex', 'Pclass']
    if 'Title' in df_engineered.columns:
        categorical_vars.append('Title')
    
    cat_var = st.selectbox("Grouping variable:", categorical_vars)
    num_var = st.selectbox("Numerical variable:", numeric_vars, index=1)
    logger.debug(f"Distribution comparison: {num_var} by {cat_var}")
    
    fig_dist_comparison = create_distribution_comparison_by_group(df_engineered, num_var, cat_var)
    st.plotly_chart(fig_dist_comparison, use_container_width=True)

# ----------------22. Pattern Mining (ricerca pattern interessanti)
elif analysis_section == "Pattern Mining":
    logger.info("Starting pattern mining")
    st.header("5. Pattern Mining and Insights")
    
    # ----------------23. Pattern survival (combinazioni interessanti)
    logger.info("Searching for survival patterns")
    st.subheader("Survival Patterns")
    
    survival_patterns = discover_survival_patterns(df_engineered)
    if survival_patterns is not None:
        logger.debug("Display pattern survival")
        st.dataframe(survival_patterns, use_container_width=True)
    else:
        logger.debug("Survival patterns unavailable")
    
    # ----------------24. Interesting anomalies
    logger.info("Searching for interesting anomalies")
    st.subheader("Interesting Anomalous Cases")
    
    # Find passengers with unusual characteristics
    anomalies = find_interesting_anomalies(df_engineered)
    if anomalies is not None:
        logger.debug("Displaying anomalies")
        st.write("**Passengers with unusual characteristics:**")
        st.dataframe(anomalies[['Name', 'Sex', 'Age', 'Pclass', 'Fare', 'Survived']], use_container_width=True)
    else:
        logger.debug("No interesting anomalies found")
    
    # ----------------25. Rare but significant combinations
    logger.info("Searching for rare but significant combinations")
    st.subheader("Rare but Significant Combinations")
    
    rare_combinations = find_rare_but_significant_combinations(df_engineered)
    if rare_combinations is not None:
        logger.debug("Displaying rare combinations")
        for combination, stats in rare_combinations.items():
            st.write(f"**{combination}:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Count", stats['count'])
            with col2:
                st.metric("Survival Rate", f"{stats['survival_rate']:.1f}%")
            with col3:
                st.metric("Significance", stats['significance'])
    else:
        logger.debug("No rare combinations found")

# ----------------26. Advanced Segmentation (cluster analysis)
elif analysis_section == "Advanced Segmentation":
    logger.info("Starting advanced segmentation")
    st.header("6. Advanced Segmentation")
    
    # ----------------27. Segmentation based on multiple features
    logger.info("Creating passenger segments")
    st.subheader("Passenger Segments")
    
    segments = create_passenger_segments(df_engineered)
    if segments is not None:
        logger.debug("Passenger segments created")
        # Add segments to the dataframe
        df_with_segments = df_engineered.copy()
        df_with_segments['Segment'] = segments
        
        # Segment analysis
        segment_analysis = analyze_segments(df_with_segments)
        st.dataframe(segment_analysis, use_container_width=True)
        
        # ----------------28. Display segments
        logger.debug("Creating segment visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_segments_survival = create_segments_survival_chart(df_with_segments)
            st.plotly_chart(fig_segments_survival, use_container_width=True)
        
        with col2:
            fig_segments_dist = create_segments_distribution_chart(df_with_segments)
            st.plotly_chart(fig_segments_dist, use_container_width=True)
    else:
        logger.debug("Passenger segments were not created")
    
    # ----------------29. Analysis RFM-style (Recency, Frequency, Monetary adapting to Titanic)
    logger.info("Creating profiles Age-Fare-Class")
    st.subheader("Passenger Profiles (Age-Fare-Class)")
    
    afc_profiles = create_age_fare_class_profiles(df_engineered)
    if afc_profiles is not None:
        logger.debug("Analyzing AFC profiles")
        df_with_profiles = df_engineered.copy()
        df_with_profiles['Profile'] = afc_profiles
        
        profile_survival = df_with_profiles.groupby('Profile')['Survived'].agg(['count', 'mean']).round(3)
        profile_survival.columns = ['Count', 'Survival_Rate']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(profile_survival, use_container_width=True)
        
        with col2:
            fig_profiles = create_profiles_chart(profile_survival)
            st.plotly_chart(fig_profiles, use_container_width=True)
    else:
        logger.debug("AFC profiles were not created")

# ----------------30. Summary e insights finali
logger.info("Creating advanced analysis summary")
st.header("Advanced Analysis Summary")

# ----------------31. Metriche chiave dall'analysis avanzata
col1, col2, col3, col4 = st.columns(4)

with col1:
    original_features = len([col for col in df.columns if col in df_original.columns])
    st.metric("Original Features", original_features)

with col2:
    engineered_features = len([col for col in df_engineered.columns if col not in df_original.columns])
    st.metric("Features Engineered", engineered_features)

with col3:
    if outlier_method != "None":
        outliers_handled = len(df) - len(df_engineered)
        st.metric("Handled Outliers", outliers_handled)
        logger.debug(f"Handled outliers: {outliers_handled}")
    else:
        st.metric("Handled Outliers", 0)
        logger.debug("No outliers handled")

with col4:
    data_quality_score = calculate_data_quality_score(df_engineered)
    st.metric("Data Quality Score", f"{data_quality_score:.1f}%")
    logger.debug(f"Data quality score: {data_quality_score:.1f}%")

logger.info(f"Page {__name__} completed successfully")



