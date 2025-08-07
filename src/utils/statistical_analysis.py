"""
src/utils/statistical_analysis.py
Functions for advanced statistical analysis
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Target Correlations (from extended notebook correlations)
def calculate_target_correlations(df, target_col):
    """
    Calculate correlations of all variables with target
    Extends notebook section 4.1.2
    """
    logger.info(f"Executing calculate_target_correlations for target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"Empty DataFrame or missing target column {target_col}")
        return None
    
    # Select only numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId', target_col]]
    
    if len(numeric_cols) == 0:
        logger.warning("No numerical columns found")
        return None
    
    correlations = {}
    
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # Only if we have enough data
            corr_pearson = df[col].corr(df[target_col])
            corr_spearman = df[col].corr(df[target_col], method='spearman')
            
            correlations[col] = {
                'Pearson': corr_pearson,
                'Spearman': corr_spearman,
                'Abs_Pearson': abs(corr_pearson)
            }
            logger.debug(f"Calculated correlations for {col}: Pearson={corr_pearson:.2f}, Spearman={corr_spearman:.2f}")
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
    
    logger.info(f"Calculated correlations for {len(correlations)} variables")
    return corr_df['Abs_Pearson']

# ----------------2. Normality Tests (distribution analysis extension)
def calculate_normality_statistics(df, variable):
    """
    Calculate normality statistics for a variable
    """
    logger.info(f"Executing calculate_normality_statistics for variable={variable}")
    if df is None or variable not in df.columns:
        logger.warning(f"Empty DataFrame or missing column {variable}")
        return None
    
    data = df[variable].dropna()
    
    if len(data) < 20:
        logger.warning(f"Insufficient data for {variable} (n={len(data)})")
        return {"Error": "Insufficient data"}
    
    # Basic statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    logger.debug(f"Basic statistics for {variable}: mean={mean_val:.2f}, skewness={skewness:.2f}")
    
    # Shapiro-Wilk test (for small samples)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        logger.debug(f"Shapiro-Wilk not performed for n={len(data)} > 5000")
    
    # Kolmogorov-Smirnov test
    # Normalize data for test
    normalized_data = (data - mean_val) / std_val
    ks_stat, ks_p = stats.kstest(normalized_data, 'norm')
    
    result = {
        "Mean": mean_val,
        "Median": median_val,
        "Standard Deviation": std_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Shapiro p-value": shapiro_p,
        "KS p-value": ks_p
    }
    
    logger.debug(f"Normality test results for {variable}: {result}")
    return result

# ----------------3. Feature Importance Proxy (without ML)
def calculate_feature_importance_proxy(df, target_col):
    """
    Calculate approximate feature importance without ML
    """
    logger.info(f"Executing calculate_feature_importance_proxy for target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"Empty DataFrame or missing target column {target_col}")
        return None
    
    importance_scores = []
    
    # For numerical variables: absolute correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId', target_col]]
    
    for col in numeric_cols:
        if df[col].notna().sum() > 10:
            corr = abs(df[col].corr(df[target_col]))
            importance_scores.append({
                'Feature': col,
                'Importance': corr,
                'Type': 'Numeric'
            })
            logger.debug(f"Numerical importance for {col}: {corr:.2f}")
    
    # For categorical variables: Cramér's V
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['Name', 'Ticket']]
    
    for col in categorical_cols:
        if df[col].notna().sum() > 10 and df[col].nunique() < 20:  # Not too many unique values
            try:
                cramers_v = calculate_cramers_v(df[col], df[target_col])
                importance_scores.append({
                    'Feature': col,
                    'Importance': cramers_v,
                    'Type': 'Categorical'
                })
                logger.debug(f"Categorical importance for {col}: {cramers_v:.2f}")
            except Exception as e:
                logger.warning(f"Error calculating Cramers V for {col}: {str(e)}")
                st.warning(f"Error calculating Cramers V for {col}: {str(e)}")
                continue
    
    if not importance_scores:
        logger.warning("No importance scores calculated")
        return None
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    logger.info(f"Calculated importance for {len(importance_df)} features")
    return importance_df

# ----------------4. Cramér's V for categorical variables
def calculate_cramers_v(x, y):
    """
    Calculate Cramér's V to measure association between categorical variables
    """
    logger.debug(f"Calculating Cramér's V between {x.name} and {y.name}")
    try:
        # Remove null values
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            logger.debug("Insufficient data for Cramér's V")
            return 0
        
        # Contingency table
        confusion_matrix = pd.crosstab(x_clean, y_clean)
        
        # Chi-square test
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        
        # Cramér's V
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0:
            logger.debug("Zero minimum dimension in contingency table")
            return 0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        logger.debug(f"Calculated Cramér's V: {cramers_v:.2f}")
        return cramers_v
        
    except Exception as e:
        logger.error(f"Error calculating Cramér's V: {str(e)}")
        return 0

# ----------------5. Pattern Mining - Survival Patterns
def discover_survival_patterns(df):
    """
    Discover interesting survival patterns
    """
    logger.info("Executing discover_survival_patterns")
    if df is None or 'Survived' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Survived' column")
        return None
    
    patterns = []
    
    # Patterns for variable combinations
    categorical_vars = ['Sex', 'Pclass']
    if 'Title' in df.columns:
        categorical_vars.append('Title')
    if 'Deck' in df.columns:
        categorical_vars.append('Deck')
    
    logger.debug(f"Pattern analysis for variables: {categorical_vars}")
    
    # Analyze pairwise combinations
    for i, var1 in enumerate(categorical_vars):
        for var2 in categorical_vars[i+1:]:
            if var1 in df.columns and var2 in df.columns:
                pattern_data = df.groupby([var1, var2]).agg({
                    'Survived': ['count', 'sum', 'mean']
                }).round(3)
                
                pattern_data.columns = ['Count', 'Survivors', 'Survival_Rate']
                pattern_data = pattern_data.reset_index()
                
                # Filter significant patterns
                significant_patterns = pattern_data[
                    (pattern_data['Count'] >= 10) & 
                    ((pattern_data['Survival_Rate'] >= 0.8) | (pattern_data['Survival_Rate'] <= 0.2))
                ]
                
                for _, row in significant_patterns.iterrows():
                    patterns.append({
                        'Pattern': f"{var1}={row[var1]}, {var2}={row[var2]}",
                        'Count': row['Count'],
                        'Survival_Rate': f"{row['Survival_Rate']*100:.1f}%",
                        'Type': 'High' if row['Survival_Rate'] >= 0.8 else 'Low'
                    })
                logger.debug(f"Found {len(significant_patterns)} significant patterns for {var1} and {var2}")
    
    if not patterns:
        logger.info("No significant patterns found")
        return None
    
    logger.info(f"Found {len(patterns)} significant patterns")
    return pd.DataFrame(patterns).sort_values('Count', ascending=False)

# ----------------6. Interesting Anomalies
def find_interesting_anomalies(df):
    """
    Find passengers with unusual but significant characteristics
    """
    logger.info("Executing find_interesting_anomalies")
    if df is None:
        logger.warning("Empty DataFrame")
        return None
    
    anomalies = []
    
    # Children in first class who didn't survive
    if 'Age' in df.columns:
        child_1st_died = df[
            (df['Age'] <= 12) & 
            (df['Pclass'] == 1) & 
            (df['Survived'] == 0)
        ]
        if len(child_1st_died) > 0:
            anomalies.append(child_1st_died)
            logger.debug(f"Found {len(child_1st_died)} children in 1st class who didn't survive")
    
    # Men in first class who survived (against trend)
    male_1st_survived = df[
        (df['Sex'] == 'male') & 
        (df['Pclass'] == 1) & 
        (df['Survived'] == 1)
    ]
    if len(male_1st_survived) > 0:
        anomalies.append(male_1st_survived)
        logger.debug(f"Found {len(male_1st_survived)} men in 1st class who survived")
    
    # Women in third class who didn't survive
    female_3rd_died = df[
        (df['Sex'] == 'female') & 
        (df['Pclass'] == 3) & 
        (df['Survived'] == 0)
    ]
    if len(female_3rd_died) > 0:
        anomalies.append(female_3rd_died)
        logger.debug(f"Found {len(female_3rd_died)} women in 3rd class who didn't survive")
    
    if not anomalies:
        logger.info("No interesting anomalies found")
        return None
    
    # Combine all anomalies
    combined_anomalies = pd.concat(anomalies, ignore_index=True).drop_duplicates()
    logger.info(f"Found {len(combined_anomalies)} interesting anomalies")
    return combined_anomalies

# ----------------7. Rare but Significant Combinations
def find_rare_but_significant_combinations(df):
    """
    Find rare combinations of characteristics with extreme survival rates
    """
    logger.info("Executing find_rare_but_significant_combinations")
    if df is None or 'Survived' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Survived' column")
        return None
    
    combinations = {}
    
    # Triple combinations: Gender, Class, Age Band
    if 'Age' in df.columns:
        df_temp = df.copy()
        df_temp['Age_Band'] = pd.cut(df_temp['Age'], bins=[0, 18, 60, 100], labels=['Young', 'Adult', 'Senior'])
        
        triple_analysis = df_temp.groupby(['Sex', 'Pclass', 'Age_Band']).agg({
            'Survived': ['count', 'mean']
        }).round(3)
        
        triple_analysis.columns = ['Count', 'Survival_Rate']
        triple_analysis = triple_analysis.reset_index()
        
        # Filter rare combinations (5-20 people) with extreme rates
        rare_significant = triple_analysis[
            (triple_analysis['Count'] >= 5) & 
            (triple_analysis['Count'] <= 20) &
            ((triple_analysis['Survival_Rate'] >= 0.8) | (triple_analysis['Survival_Rate'] <= 0.2))
        ]
        
        for _, row in rare_significant.iterrows():
            key = f"{row['Sex']}-Class{row['Pclass']}-{row['Age_Band']}"
            combinations[key] = {
                'count': int(row['Count']),
                'survival_rate': row['Survival_Rate'] * 100,
                'significance': 'High' if row['Survival_Rate'] >= 0.8 or row['Survival_Rate'] <= 0.2 else 'Medium'
            }
        logger.debug(f"Found {len(rare_significant)} rare significant combinations")
    
    if not combinations:
        logger.info("No rare significant combinations found")
        return None
    
    logger.info(f"Found {len(combinations)} rare significant combinations")
    return combinations

# ----------------8. Passenger Segmentation
def create_passenger_segments(df):
    """
    Create passenger segments based on multiple characteristics
    """
    logger.info("Executing create_passenger_segments")
    if df is None:
        logger.warning("Empty DataFrame")
        return None
    
    # Select variables for clustering
    cluster_vars = []
    
    if 'Age' in df.columns:
        cluster_vars.append('Age')
    if 'Fare' in df.columns:
        cluster_vars.append('Fare')
    if 'Family_Size' in df.columns:
        cluster_vars.append('Family_Size')
    
    # Add encoded variables
    if 'Sex' in df.columns:
        sex_encoded = (df['Sex'] == 'female').astype(int)
        cluster_vars.append('Sex_Encoded')
        df_cluster = df.copy()
        df_cluster['Sex_Encoded'] = sex_encoded
    
    if 'Pclass' in df.columns:
        cluster_vars.append('Pclass')
        df_cluster = df.copy() if 'df_cluster' not in locals() else df_cluster
    
    if len(cluster_vars) < 2:
        logger.warning(f"Insufficient variables for clustering: {cluster_vars}")
        return None
    
    logger.debug(f"Variables used for clustering: {cluster_vars}")
    
    # Prepare data for clustering
    cluster_data = df_cluster[cluster_vars].fillna(df_cluster[cluster_vars].median())
    
    # Normalize
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        segments = kmeans.fit_predict(cluster_data_scaled)
        logger.info(f"Created {len(np.unique(segments))} passenger segments")
        return segments
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        return None

# ----------------9. Segment Analysis
def analyze_segments(df):
    """
    Analyze characteristics of created segments
    """
    logger.info("Executing analyze_segments")
    if df is None or 'Segment' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Segment' column")
        return None
    
    # Variables to analyze
    analysis_vars = ['Age', 'Fare', 'Family_Size', 'Survived']
    available_vars = [var for var in analysis_vars if var in df.columns]
    
    if not available_vars:
        logger.warning("No variables available for analysis")
        return None
    
    logger.debug(f"Segment analysis for variables: {available_vars}")
    
    # Analysis by segment
    segment_analysis = df.groupby('Segment')[available_vars].agg(['count', 'mean']).round(3)
    
    # Flatten column names
    segment_analysis.columns = [f"{var}_{stat}" for var, stat in segment_analysis.columns]
    
    # Add female percentage if available
    if 'Sex' in df.columns:
        female_pct = df.groupby('Segment')['Sex'].apply(lambda x: (x == 'female').mean() * 100)
        segment_analysis['Female_Percentage'] = female_pct
        logger.debug("Added female percentage to analysis")
    
    if 'Pclass' in df.columns:
        avg_class = df.groupby('Segment')['Pclass'].mean()
        segment_analysis['Avg_Class'] = avg_class
        logger.debug("Added average class to analysis")
    
    logger.info(f"Segment analysis completed for {len(available_vars)} variables")
    return segment_analysis

# ----------------10. Age-Fare-Class Profiles
def create_age_fare_class_profiles(df):
    """
    Create profiles based on age, fare and class
    """
    logger.info("Executing create_age_fare_class_profiles")
    if df is None:
        logger.warning("Empty DataFrame")
        return None
    
    required_vars = ['Age', 'Fare', 'Pclass']
    if not all(var in df.columns for var in required_vars):
        logger.warning(f"Missing variables: {[var for var in required_vars if var not in df.columns]}")
        return None
    
    df_temp = df.copy()
    
    # Create bins for age and fare
    df_temp['Age_Bin'] = pd.qcut(df_temp['Age'], q=3, labels=['Young', 'Middle', 'Old'], duplicates='drop')
    df_temp['Fare_Bin'] = pd.qcut(df_temp['Fare'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    
    # Combine into profiles
    df_temp['Profile'] = (
        df_temp['Age_Bin'].astype(str) + '_' + 
        df_temp['Fare_Bin'].astype(str) + '_' + 
        'Class' + df_temp['Pclass'].astype(str)
    )
    
    # Filter profiles with sufficient data
    profile_counts = df_temp['Profile'].value_counts()
    valid_profiles = profile_counts[profile_counts >= 5].index
    
    profiles = df_temp['Profile'].where(df_temp['Profile'].isin(valid_profiles), 'Other')
    
    logger.info(f"Created {len(profile_counts)} profiles, {len(valid_profiles)} valid")
    return profiles

# ----------------11. Data Quality Score
def calculate_data_quality_score(df):
    """
    Calculate a data quality score
    """
    logger.info("Executing calculate_data_quality_score")
    if df is None:
        logger.warning("Empty DataFrame")
        return 0
    
    scores = []
    
    # Completeness (% non-null values)
    completeness = (df.count().sum() / (len(df) * len(df.columns))) * 100
    scores.append(completeness * 0.4)  # Weight 40%
    logger.debug(f"Completeness: {completeness:.1f}%")
    
    # Consistency (% duplicates)
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    scores.append(uniqueness * 0.2)  # Weight 20%
    logger.debug(f"Uniqueness: {uniqueness:.1f}%")
    
    # Validity (% values in expected ranges)
    validity_score = 100  # Assume valid by default
    
    # Specific checks
    if 'Age' in df.columns:
        invalid_age = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
        age_validity = (1 - invalid_age / len(df)) * 100
        validity_score = min(validity_score, age_validity)
        logger.debug(f"Age validity: {age_validity:.1f}%")
    
    if 'Fare' in df.columns:
        invalid_fare = (df['Fare'] < 0).sum()
        fare_validity = (1 - invalid_fare / len(df)) * 100
        validity_score = min(validity_score, fare_validity)
        logger.debug(f"Fare validity: {fare_validity:.1f}%")
    
    scores.append(validity_score * 0.2)  # Weight 20%
    
    # Richness (number of features vs baseline)
    baseline_features = 12  # Original dataset
    current_features = len(df.columns)
    richness = min(100, (current_features / baseline_features) * 100)
    scores.append(richness * 0.2)  # Weight 20%
    logger.debug(f"Richness: {richness:.1f}%")
    
    total_score = sum(scores)
    logger.info(f"Calculated data quality score: {total_score:.1f}/100")
    return total_score

logger.info(f"Loading completed {__name__}")