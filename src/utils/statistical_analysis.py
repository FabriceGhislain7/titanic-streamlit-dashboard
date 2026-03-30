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

# ----------------1. Correlations with Target (da notebook correlazioni estese)
def calculate_target_correlations(df, target_col):
    """
    Calculate correlations for all variables with the target
    Extends notebook section 4.1.2
    """
    logger.info(f"Running calculate_target_correlations per target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"Empty DataFrame or target column {target_col} missing")
        return None
    
    # Select only numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId', target_col]]
    
    if len(numeric_cols) == 0:
        logger.warning("No numerical column found")
        return None
    
    correlations = {}
    
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # Only if enough data is available
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

# ----------------2. Test NormalitÃ  (estensione analisi distribuzione)
def calculate_normality_statistics(df, variable):
    """
    Calcola statistics di normalitÃ  per una variable
    """
    logger.info(f"Running calculate_normality_statistics per variable={variable}")
    if df is None or variable not in df.columns:
        logger.warning(f"DataFrame vuoto o column {variable} missing")
        return None
    
    data = df[variable].dropna()
    
    if len(data) < 20:
        logger.warning(f"Insufficient data per {variable} (n={len(data)})")
        return {"Error": "Insufficient data"}
    
    # Base statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    logger.debug(f"Base statistics for {variable}: mean={mean_val:.2f}, skewness={skewness:.2f}")
    
    # Test Shapiro-Wilk (for small samples)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        logger.debug(f"Shapiro-Wilk not executed per n={len(data)} > 5000")
    
    # Test Kolmogorov-Smirnov
    # Normalize the data for the test
    normalized_data = (data - mean_val) / std_val
    ks_stat, ks_p = stats.kstest(normalized_data, 'norm')
    
    result = {
        "Mean": mean_val,
        "Meanna": median_val,
        "Standard Deviation": std_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Shapiro p-value": shapiro_p,
        "KS p-value": ks_p
    }
    
    logger.debug(f"Risultati test normalitÃ  per {variable}: {result}")
    return result

# ----------------3. Feature Importance Proxy (senza ML)
def calculate_feature_importance_proxy(df, target_col):
    """
    Calculate approximate feature importance without ML
    """
    logger.info(f"Running calculate_feature_importance_proxy per target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"Empty DataFrame or target column {target_col} missing")
        return None
    
    importance_scores = []
    
    # For numerical variables: correlazione assoluta
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
    
    # For categorical variables: CramÃ©r's V
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
                logger.warning(f"Error while calculating Cramers V for {col}: {str(e)}")
                st.warning(f"Error while calculating Cramers V for {col}: {str(e)}")
                continue
    
    if not importance_scores:
        logger.warning("No importance score calculated")
        return None
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    logger.info(f"Calculated importance for {len(importance_df)} features")
    return importance_df

# ----------------4. CramÃ©r's V per variables categoriche
def calculate_cramers_v(x, y):
    """
    Calcola CramÃ©r's V per misurare associazione tra variables categoriche
    """
    logger.debug(f"Calcolo CramÃ©r's V tra {x.name} e {y.name}")
    try:
        # Remove null values
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            logger.debug("Insufficient data per CramÃ©r's V")
            return 0
        
        # Contingency table
        confusion_matrix = pd.crosstab(x_clean, y_clean)
        
        # Chi-square test
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        
        # CramÃ©r's V
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0:
            logger.debug("Minimum dimension is zero in contingency table")
            return 0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        logger.debug(f"CramÃ©r's V calcolato: {cramers_v:.2f}")
        return cramers_v
        
    except Exception as e:
        logger.error(f"Error nel calcolo CramÃ©r's V: {str(e)}")
        return 0

# ----------------5. Pattern Mining - Pattern Survival
def discover_survival_patterns(df):
    """
    Discover interesting survival patterns
    """
    logger.info("Running discover_survival_patterns")
    if df is None or 'Survived' not in df.columns:
        logger.warning("DataFrame vuoto o column 'Survived' missing")
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
                
                # Filtra pattern significativi
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
                logger.debug(f"Trovati {len(significant_patterns)} pattern significativi per {var1} e {var2}")
    
    if not patterns:
        logger.info("Nessun pattern significativo trovato")
        return None
    
    logger.info(f"Trovati {len(patterns)} pattern significativi")
    return pd.DataFrame(patterns).sort_values('Count', ascending=False)

# ----------------6. Anomalie Interessanti
def find_interesting_anomalies(df):
    """
    Trova passeggeri con caratteristiche inusuali ma significative
    """
    logger.info("Running find_interesting_anomalies")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    anomalies = []
    
    # Bambini in prima class che non sono survivors
    if 'Age' in df.columns:
        child_1st_died = df[
            (df['Age'] <= 12) & 
            (df['Pclass'] == 1) & 
            (df['Survived'] == 0)
        ]
        if len(child_1st_died) > 0:
            anomalies.append(child_1st_died)
            logger.debug(f"Trovati {len(child_1st_died)} bambini in 1a class non survivors")
    
    # Uomini di prima class survivors (contro tendenza)
    male_1st_survived = df[
        (df['Sex'] == 'male') & 
        (df['Pclass'] == 1) & 
        (df['Survived'] == 1)
    ]
    if len(male_1st_survived) > 0:
        anomalies.append(male_1st_survived)
        logger.debug(f"Trovati {len(male_1st_survived)} uomini in 1a class survivors")
    
    # Donne di terza class non sopravvissute
    female_3rd_died = df[
        (df['Sex'] == 'female') & 
        (df['Pclass'] == 3) & 
        (df['Survived'] == 0)
    ]
    if len(female_3rd_died) > 0:
        anomalies.append(female_3rd_died)
        logger.debug(f"Trovati {len(female_3rd_died)} donne in 3a class non sopravvissute")
    
    if not anomalies:
        logger.info("Nessuna anomalia interessante trovata")
        return None
    
    # Combina tutte le anomalie
    combined_anomalies = pd.concat(anomalies, ignore_index=True).drop_duplicates()
    logger.info(f"Trovate {len(combined_anomalies)} anomalie interessanti")
    return combined_anomalies

# ----------------7. Combinazioni Rare ma Significative
def find_rare_but_significant_combinations(df):
    """
    Trova combinazioni rare di caratteristiche con tassi di survival estremi
    """
    logger.info("Running find_rare_but_significant_combinations")
    if df is None or 'Survived' not in df.columns:
        logger.warning("DataFrame vuoto o column 'Survived' missing")
        return None
    
    combinations = {}
    
    # Combinazioni tripla: Sesso, Class, Fascia EtÃ 
    if 'Age' in df.columns:
        df_temp = df.copy()
        df_temp['Age_Band'] = pd.cut(df_temp['Age'], bins=[0, 18, 60, 100], labels=['Young', 'Adult', 'Senior'])
        
        triple_analysis = df_temp.groupby(['Sex', 'Pclass', 'Age_Band']).agg({
            'Survived': ['count', 'mean']
        }).round(3)
        
        triple_analysis.columns = ['Count', 'Survival_Rate']
        triple_analysis = triple_analysis.reset_index()
        
        # Filtra combinazioni rare (5-20 persone) con tassi estremi
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
        logger.debug(f"Trovate {len(rare_significant)} combinazioni rare significative")
    
    if not combinations:
        logger.info("Nessuna combinazione rara significativa trovata")
        return None
    
    logger.info(f"Trovate {len(combinations)} combinazioni rare significative")
    return combinations

# ----------------8. Segmentazione Passeggeri
def create_passenger_segments(df):
    """
    Crea segmenti di passeggeri basati su caratteristiche multiple
    """
    logger.info("Running create_passenger_segments")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    # Seleziona variables per clustering
    cluster_vars = []
    
    if 'Age' in df.columns:
        cluster_vars.append('Age')
    if 'Fare' in df.columns:
        cluster_vars.append('Fare')
    if 'Family_Size' in df.columns:
        cluster_vars.append('Family_Size')
    
    # Aggiungi variables encoded
    if 'Sex' in df.columns:
        sex_encoded = (df['Sex'] == 'female').astype(int)
        cluster_vars.append('Sex_Encoded')
        df_cluster = df.copy()
        df_cluster['Sex_Encoded'] = sex_encoded
    
    if 'Pclass' in df.columns:
        cluster_vars.append('Pclass')
        df_cluster = df.copy() if 'df_cluster' not in locals() else df_cluster
    
    if len(cluster_vars) < 2:
        logger.warning(f"Variabili insufficienti per clustering: {cluster_vars}")
        return None
    
    logger.debug(f"Variabili usate per clustering: {cluster_vars}")
    
    # Prepara data per clustering
    cluster_data = df_cluster[cluster_vars].fillna(df_cluster[cluster_vars].median())
    
    # Normalizza
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        segments = kmeans.fit_predict(cluster_data_scaled)
        logger.info(f"Creati {len(np.unique(segments))} segmenti di passeggeri")
        return segments
    except Exception as e:
        logger.error(f"Error nel clustering: {str(e)}")
        return None

# ----------------9. Analisi Segmenti
def analyze_segments(df):
    """
    Analizza caratteristiche dei segmenti creati
    """
    logger.info("Running analyze_segments")
    if df is None or 'Segment' not in df.columns:
        logger.warning("DataFrame vuoto o column 'Segment' missing")
        return None
    
    # Variabili da analizzare
    analysis_vars = ['Age', 'Fare', 'Family_Size', 'Survived']
    available_vars = [var for var in analysis_vars if var in df.columns]
    
    if not available_vars:
        logger.warning("Nessuna variable disponibile per l'analisi")
        return None
    
    logger.debug(f"Analisi segmenti per variables: {available_vars}")
    
    # Analisi per segmento
    segment_analysis = df.groupby('Segment')[available_vars].agg(['count', 'mean']).round(3)
    
    # Flatten column names
    segment_analysis.columns = [f"{var}_{stat}" for var, stat in segment_analysis.columns]
    
    # Aggiungi percentuale femminile se disponibile
    if 'Sex' in df.columns:
        female_pct = df.groupby('Segment')['Sex'].apply(lambda x: (x == 'female').mean() * 100)
        segment_analysis['Female_Percentage'] = female_pct
        logger.debug("Aggiunta percentuale femminile all'analisi")
    
    if 'Pclass' in df.columns:
        avg_class = df.groupby('Segment')['Pclass'].mean()
        segment_analysis['Avg_Class'] = avg_class
        logger.debug("Aggiunta class media all'analisi")
    
    logger.info(f"Analisi segmenti completata per {len(available_vars)} variables")
    return segment_analysis

# ----------------10. Profili Age-Fare-Class
def create_age_fare_class_profiles(df):
    """
    Crea profili basati su etÃ , tariffa e class
    """
    logger.info("Running create_age_fare_class_profiles")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    required_vars = ['Age', 'Fare', 'Pclass']
    if not all(var in df.columns for var in required_vars):
        logger.warning(f"Variabili missing: {[var for var in required_vars if var not in df.columns]}")
        return None
    
    df_temp = df.copy()
    
    # Crea bins per etÃ  e tariffa
    df_temp['Age_Bin'] = pd.qcut(df_temp['Age'], q=3, labels=['Young', 'Middle', 'Old'], duplicates='drop')
    df_temp['Fare_Bin'] = pd.qcut(df_temp['Fare'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    
    # Combina in profili
    df_temp['Profile'] = (
        df_temp['Age_Bin'].astype(str) + '_' + 
        df_temp['Fare_Bin'].astype(str) + '_' + 
        'Class' + df_temp['Pclass'].astype(str)
    )
    
    # Filtra profili con data sufficienti
    profile_counts = df_temp['Profile'].value_counts()
    valid_profiles = profile_counts[profile_counts >= 5].index
    
    profiles = df_temp['Profile'].where(df_temp['Profile'].isin(valid_profiles), 'Other')
    
    logger.info(f"Creati {len(profile_counts)} profili, {len(valid_profiles)} validi")
    return profiles

# ----------------11. Data Quality Score
def calculate_data_quality_score(df):
    """
    Calcola un punteggio di qualitÃ  dei data
    """
    logger.info("Running calculate_data_quality_score")
    if df is None:
        logger.warning("DataFrame vuoto")
        return 0
    
    scores = []
    
    # Completezza (% valori non nulli)
    completeness = (df.count().sum() / (len(df) * len(df.columns))) * 100
    scores.append(completeness * 0.4)  # Peso 40%
    logger.debug(f"Completezza: {completeness:.1f}%")
    
    # Coerenza (% duplicates)
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    scores.append(uniqueness * 0.2)  # Peso 20%
    logger.debug(f"UnicitÃ : {uniqueness:.1f}%")
    
    # ValiditÃ  (% valori nei range attesi)
    validity_score = 100  # Assume valido di default
    
    # Controlli specifici
    if 'Age' in df.columns:
        invalid_age = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
        age_validity = (1 - invalid_age / len(df)) * 100
        validity_score = min(validity_score, age_validity)
        logger.debug(f"ValiditÃ  etÃ : {age_validity:.1f}%")
    
    if 'Fare' in df.columns:
        invalid_fare = (df['Fare'] < 0).sum()
        fare_validity = (1 - invalid_fare / len(df)) * 100
        validity_score = min(validity_score, fare_validity)
        logger.debug(f"ValiditÃ  tariffa: {fare_validity:.1f}%")
    
    scores.append(validity_score * 0.2)  # Peso 20%
    
    # Ricchezza (numero di feature vs baseline)
    baseline_features = 12  # Dataset originale
    current_features = len(df.columns)
    richness = min(100, (current_features / baseline_features) * 100)
    scores.append(richness * 0.2)  # Peso 20%
    logger.debug(f"Ricchezza: {richness:.1f}%")
    
    total_score = sum(scores)
    logger.info(f"Punteggio qualitÃ  data calcolato: {total_score:.1f}/100")
    return total_score

logger.info(f"Loading completato {__name__}")
