"""
src/utils/statistical_analysis.py
Funzioni per analisi statistiche avanzate
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

# ----------------1. Correlazioni con Target (da notebook correlazioni estese)
def calculate_target_correlations(df, target_col):
    """
    Calcola correlazioni di tutte le variabili con il target
    Estende notebook sezione 4.1.2
    """
    logger.info(f"Esecuzione calculate_target_correlations per target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"DataFrame vuoto o colonna target {target_col} mancante")
        return None
    
    # Seleziona solo variabili numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId', target_col]]
    
    if len(numeric_cols) == 0:
        logger.warning("Nessuna colonna numerica trovata")
        return None
    
    correlations = {}
    
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # Solo se abbiamo abbastanza dati
            corr_pearson = df[col].corr(df[target_col])
            corr_spearman = df[col].corr(df[target_col], method='spearman')
            
            correlations[col] = {
                'Pearson': corr_pearson,
                'Spearman': corr_spearman,
                'Abs_Pearson': abs(corr_pearson)
            }
            logger.debug(f"Calcolate correlazioni per {col}: Pearson={corr_pearson:.2f}, Spearman={corr_spearman:.2f}")
    
    # Converti in DataFrame e ordina per correlazione assoluta
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
    
    logger.info(f"Calcolate correlazioni per {len(correlations)} variabili")
    return corr_df['Abs_Pearson']

# ----------------2. Test Normalità (estensione analisi distribuzione)
def calculate_normality_statistics(df, variable):
    """
    Calcola statistiche di normalità per una variabile
    """
    logger.info(f"Esecuzione calculate_normality_statistics per variabile={variable}")
    if df is None or variable not in df.columns:
        logger.warning(f"DataFrame vuoto o colonna {variable} mancante")
        return None
    
    data = df[variable].dropna()
    
    if len(data) < 20:
        logger.warning(f"Dati insufficienti per {variable} (n={len(data)})")
        return {"Errore": "Dati insufficienti"}
    
    # Statistiche di base
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    logger.debug(f"Statistiche base per {variable}: mean={mean_val:.2f}, skewness={skewness:.2f}")
    
    # Test Shapiro-Wilk (per campioni piccoli)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        logger.debug(f"Shapiro-Wilk non eseguito per n={len(data)} > 5000")
    
    # Test Kolmogorov-Smirnov
    # Normalizza i dati per il test
    normalized_data = (data - mean_val) / std_val
    ks_stat, ks_p = stats.kstest(normalized_data, 'norm')
    
    result = {
        "Media": mean_val,
        "Mediana": median_val,
        "Deviazione Standard": std_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Shapiro p-value": shapiro_p,
        "KS p-value": ks_p
    }
    
    logger.debug(f"Risultati test normalità per {variable}: {result}")
    return result

# ----------------3. Feature Importance Proxy (senza ML)
def calculate_feature_importance_proxy(df, target_col):
    """
    Calcola importanza approssimata delle feature senza ML
    """
    logger.info(f"Esecuzione calculate_feature_importance_proxy per target={target_col}")
    if df is None or target_col not in df.columns:
        logger.warning(f"DataFrame vuoto o colonna target {target_col} mancante")
        return None
    
    importance_scores = []
    
    # Per variabili numeriche: correlazione assoluta
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
            logger.debug(f"Importanza numerica per {col}: {corr:.2f}")
    
    # Per variabili categoriche: Cramér's V
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['Name', 'Ticket']]
    
    for col in categorical_cols:
        if df[col].notna().sum() > 10 and df[col].nunique() < 20:  # Non troppi valori unici
            try:
                cramers_v = calculate_cramers_v(df[col], df[target_col])
                importance_scores.append({
                    'Feature': col,
                    'Importance': cramers_v,
                    'Type': 'Categorical'
                })
                logger.debug(f"Importanza categorica per {col}: {cramers_v:.2f}")
            except Exception as e:
                logger.warning(f"Errore nel calcolo Cramers V per {col}: {str(e)}")
                st.warning(f"Errore nel calcolo Cramers V per {col}: {str(e)}")
                continue
    
    if not importance_scores:
        logger.warning("Nessun punteggio di importanza calcolato")
        return None
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    logger.info(f"Calcolata importanza per {len(importance_df)} features")
    return importance_df

# ----------------4. Cramér's V per variabili categoriche
def calculate_cramers_v(x, y):
    """
    Calcola Cramér's V per misurare associazione tra variabili categoriche
    """
    logger.debug(f"Calcolo Cramér's V tra {x.name} e {y.name}")
    try:
        # Rimuovi valori nulli
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            logger.debug("Dati insufficienti per Cramér's V")
            return 0
        
        # Tabella di contingenza
        confusion_matrix = pd.crosstab(x_clean, y_clean)
        
        # Chi-square test
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        
        # Cramér's V
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0:
            logger.debug("Dimensione minima zero in tabella contingenza")
            return 0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        logger.debug(f"Cramér's V calcolato: {cramers_v:.2f}")
        return cramers_v
        
    except Exception as e:
        logger.error(f"Errore nel calcolo Cramér's V: {str(e)}")
        return 0

# ----------------5. Pattern Mining - Pattern Sopravvivenza
def discover_survival_patterns(df):
    """
    Scopre pattern interessanti di sopravvivenza
    """
    logger.info("Esecuzione discover_survival_patterns")
    if df is None or 'Survived' not in df.columns:
        logger.warning("DataFrame vuoto o colonna 'Survived' mancante")
        return None
    
    patterns = []
    
    # Pattern per combinazioni di variabili
    categorical_vars = ['Sex', 'Pclass']
    if 'Title' in df.columns:
        categorical_vars.append('Title')
    if 'Deck' in df.columns:
        categorical_vars.append('Deck')
    
    logger.debug(f"Analisi pattern per variabili: {categorical_vars}")
    
    # Analizza combinazioni a coppie
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
    logger.info("Esecuzione find_interesting_anomalies")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    anomalies = []
    
    # Bambini in prima classe che non sono sopravvissuti
    if 'Age' in df.columns:
        child_1st_died = df[
            (df['Age'] <= 12) & 
            (df['Pclass'] == 1) & 
            (df['Survived'] == 0)
        ]
        if len(child_1st_died) > 0:
            anomalies.append(child_1st_died)
            logger.debug(f"Trovati {len(child_1st_died)} bambini in 1a classe non sopravvissuti")
    
    # Uomini di prima classe sopravvissuti (contro tendenza)
    male_1st_survived = df[
        (df['Sex'] == 'male') & 
        (df['Pclass'] == 1) & 
        (df['Survived'] == 1)
    ]
    if len(male_1st_survived) > 0:
        anomalies.append(male_1st_survived)
        logger.debug(f"Trovati {len(male_1st_survived)} uomini in 1a classe sopravvissuti")
    
    # Donne di terza classe non sopravvissute
    female_3rd_died = df[
        (df['Sex'] == 'female') & 
        (df['Pclass'] == 3) & 
        (df['Survived'] == 0)
    ]
    if len(female_3rd_died) > 0:
        anomalies.append(female_3rd_died)
        logger.debug(f"Trovati {len(female_3rd_died)} donne in 3a classe non sopravvissute")
    
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
    Trova combinazioni rare di caratteristiche con tassi di sopravvivenza estremi
    """
    logger.info("Esecuzione find_rare_but_significant_combinations")
    if df is None or 'Survived' not in df.columns:
        logger.warning("DataFrame vuoto o colonna 'Survived' mancante")
        return None
    
    combinations = {}
    
    # Combinazioni tripla: Sesso, Classe, Fascia Età
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
                'significance': 'Alta' if row['Survival_Rate'] >= 0.8 or row['Survival_Rate'] <= 0.2 else 'Media'
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
    logger.info("Esecuzione create_passenger_segments")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    # Seleziona variabili per clustering
    cluster_vars = []
    
    if 'Age' in df.columns:
        cluster_vars.append('Age')
    if 'Fare' in df.columns:
        cluster_vars.append('Fare')
    if 'Family_Size' in df.columns:
        cluster_vars.append('Family_Size')
    
    # Aggiungi variabili encoded
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
    
    # Prepara dati per clustering
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
        logger.error(f"Errore nel clustering: {str(e)}")
        return None

# ----------------9. Analisi Segmenti
def analyze_segments(df):
    """
    Analizza caratteristiche dei segmenti creati
    """
    logger.info("Esecuzione analyze_segments")
    if df is None or 'Segment' not in df.columns:
        logger.warning("DataFrame vuoto o colonna 'Segment' mancante")
        return None
    
    # Variabili da analizzare
    analysis_vars = ['Age', 'Fare', 'Family_Size', 'Survived']
    available_vars = [var for var in analysis_vars if var in df.columns]
    
    if not available_vars:
        logger.warning("Nessuna variabile disponibile per l'analisi")
        return None
    
    logger.debug(f"Analisi segmenti per variabili: {available_vars}")
    
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
        logger.debug("Aggiunta classe media all'analisi")
    
    logger.info(f"Analisi segmenti completata per {len(available_vars)} variabili")
    return segment_analysis

# ----------------10. Profili Age-Fare-Class
def create_age_fare_class_profiles(df):
    """
    Crea profili basati su età, tariffa e classe
    """
    logger.info("Esecuzione create_age_fare_class_profiles")
    if df is None:
        logger.warning("DataFrame vuoto")
        return None
    
    required_vars = ['Age', 'Fare', 'Pclass']
    if not all(var in df.columns for var in required_vars):
        logger.warning(f"Variabili mancanti: {[var for var in required_vars if var not in df.columns]}")
        return None
    
    df_temp = df.copy()
    
    # Crea bins per età e tariffa
    df_temp['Age_Bin'] = pd.qcut(df_temp['Age'], q=3, labels=['Young', 'Middle', 'Old'], duplicates='drop')
    df_temp['Fare_Bin'] = pd.qcut(df_temp['Fare'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    
    # Combina in profili
    df_temp['Profile'] = (
        df_temp['Age_Bin'].astype(str) + '_' + 
        df_temp['Fare_Bin'].astype(str) + '_' + 
        'Class' + df_temp['Pclass'].astype(str)
    )
    
    # Filtra profili con dati sufficienti
    profile_counts = df_temp['Profile'].value_counts()
    valid_profiles = profile_counts[profile_counts >= 5].index
    
    profiles = df_temp['Profile'].where(df_temp['Profile'].isin(valid_profiles), 'Other')
    
    logger.info(f"Creati {len(profile_counts)} profili, {len(valid_profiles)} validi")
    return profiles

# ----------------11. Data Quality Score
def calculate_data_quality_score(df):
    """
    Calcola un punteggio di qualità dei dati
    """
    logger.info("Esecuzione calculate_data_quality_score")
    if df is None:
        logger.warning("DataFrame vuoto")
        return 0
    
    scores = []
    
    # Completezza (% valori non nulli)
    completeness = (df.count().sum() / (len(df) * len(df.columns))) * 100
    scores.append(completeness * 0.4)  # Peso 40%
    logger.debug(f"Completezza: {completeness:.1f}%")
    
    # Coerenza (% duplicati)
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    scores.append(uniqueness * 0.2)  # Peso 20%
    logger.debug(f"Unicità: {uniqueness:.1f}%")
    
    # Validità (% valori nei range attesi)
    validity_score = 100  # Assume valido di default
    
    # Controlli specifici
    if 'Age' in df.columns:
        invalid_age = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
        age_validity = (1 - invalid_age / len(df)) * 100
        validity_score = min(validity_score, age_validity)
        logger.debug(f"Validità età: {age_validity:.1f}%")
    
    if 'Fare' in df.columns:
        invalid_fare = (df['Fare'] < 0).sum()
        fare_validity = (1 - invalid_fare / len(df)) * 100
        validity_score = min(validity_score, fare_validity)
        logger.debug(f"Validità tariffa: {fare_validity:.1f}%")
    
    scores.append(validity_score * 0.2)  # Peso 20%
    
    # Ricchezza (numero di feature vs baseline)
    baseline_features = 12  # Dataset originale
    current_features = len(df.columns)
    richness = min(100, (current_features / baseline_features) * 100)
    scores.append(richness * 0.2)  # Peso 20%
    logger.debug(f"Ricchezza: {richness:.1f}%")
    
    total_score = sum(scores)
    logger.info(f"Punteggio qualità dati calcolato: {total_score:.1f}/100")
    return total_score

logger.info(f"Caricamento completato {__name__}")