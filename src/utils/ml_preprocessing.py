"""
src/utils/ml_preprocessing.py
Preprocessing avanzato per Machine Learning
"""

import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
# Import IterativeImputer correttamente
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats
import re

from src.config import FEATURE_ENGINEERING, PREPROCESSING_CONFIG, COLUMN_LABELS

logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

warnings.filterwarnings('ignore')

# ----------------1. Custom Transformers

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer personalizzato per feature engineering specifico del Titanic
    """
    
    def __init__(self, 
                 extract_title=True,
                 extract_deck=True,
                 create_family_features=True,
                 create_fare_features=True,
                 create_age_groups=True,
                 create_interaction_features=False):
        logger.debug(f"Inizializzazione TitanicFeatureEngineer: "
                    f"extract_title={extract_title}, "
                    f"extract_deck={extract_deck}, "
                    f"create_family_features={create_family_features}, "
                    f"create_fare_features={create_fare_features}, "
                    f"create_age_groups={create_age_groups}, "
                    f"create_interaction_features={create_interaction_features}")
        
        self.extract_title = extract_title
        self.extract_deck = extract_deck
        self.create_family_features = create_family_features
        self.create_fare_features = create_fare_features
        self.create_age_groups = create_age_groups
        self.create_interaction_features = create_interaction_features
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """Fit del transformer"""
        logger.info("Esecuzione fit TitanicFeatureEngineer")
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Applica feature engineering"""
        logger.info("Esecuzione transform TitanicFeatureEngineer")
        if not self.fitted_:
            logger.error("Transformer deve essere fittato prima del transform")
            raise ValueError("Transformer deve essere fittato prima del transform")
        
        X_transformed = X.copy()
        
        # Extract title from name
        if self.extract_title and 'Name' in X_transformed.columns:
            logger.debug("Estrazione titolo dal nome")
            X_transformed = self._extract_title(X_transformed)
        
        # Extract deck from cabin
        if self.extract_deck and 'Cabin' in X_transformed.columns:
            logger.debug("Estrazione deck dalla cabina")
            X_transformed = self._extract_deck(X_transformed)
        
        # Family features
        if self.create_family_features:
            logger.debug("Creazione feature famiglia")
            X_transformed = self._create_family_features(X_transformed)
        
        # Fare features
        if self.create_fare_features:
            logger.debug("Creazione feature tariffa")
            X_transformed = self._create_fare_features(X_transformed)
        
        # Age groups
        if self.create_age_groups and 'Age' in X_transformed.columns:
            logger.debug("Creazione gruppi età")
            X_transformed = self._create_age_groups(X_transformed)
        
        # Interaction features
        if self.create_interaction_features:
            logger.debug("Creazione feature interazione")
            X_transformed = self._create_interaction_features(X_transformed)
        
        logger.info(f"Feature engineering completato. Shape finale: {X_transformed.shape}")
        return X_transformed
    
    def _extract_title(self, X):
        """Estrae titolo dal nome"""
        logger.debug("Esecuzione _extract_title")
        def extract_title_from_name(name):
            if pd.isna(name):
                return 'Unknown'
            
            # Pattern per estrarre titolo
            title_pattern = r', ([A-Za-z]+)\.'
            match = re.search(title_pattern, str(name))
            
            if match:
                title = match.group(1)
                
                # Raggruppa titoli rari
                title_mapping = {
                    'Mr': 'Mr',
                    'Mrs': 'Mrs', 
                    'Miss': 'Miss',
                    'Master': 'Master',
                    'Dr': 'Officer',
                    'Rev': 'Officer',
                    'Col': 'Officer',
                    'Major': 'Officer',
                    'Mlle': 'Miss',
                    'Countess': 'Royal',
                    'Ms': 'Mrs',
                    'Lady': 'Royal',
                    'Jonkheer': 'Royal',
                    'Don': 'Royal',
                    'Dona': 'Royal',
                    'Mme': 'Mrs',
                    'Capt': 'Officer',
                    'Sir': 'Royal'
                }
                
                return title_mapping.get(title, 'Other')
            
            return 'Unknown'
        
        X['Title'] = X['Name'].apply(extract_title_from_name)
        logger.debug(f"Trovati {X['Title'].nunique()} titoli unici")
        return X
    
    def _extract_deck(self, X):
        """Estrae deck dalla cabina"""
        logger.debug("Esecuzione _extract_deck")
        def extract_deck_from_cabin(cabin):
            if pd.isna(cabin):
                return 'Unknown'
            
            # Primo carattere della cabina è il deck
            deck = str(cabin)[0]
            return deck if deck.isalpha() else 'Unknown'
        
        X['Deck'] = X['Cabin'].apply(extract_deck_from_cabin)
        logger.debug(f"Trovati {X['Deck'].nunique()} deck unici")
        return X
    
    def _create_family_features(self, X):
        """Crea features relative alla famiglia"""
        logger.debug("Esecuzione _create_family_features")
        if 'SibSp' in X.columns and 'Parch' in X.columns:
            X['Family_Size'] = X['SibSp'] + X['Parch'] + 1
            X['Is_Alone'] = (X['Family_Size'] == 1).astype(int)
            
            # Categorie famiglia
            def categorize_family_size(size):
                if size == 1:
                    return 'Alone'
                elif size <= 4:
                    return 'Small'
                else:
                    return 'Large'
            
            X['Family_Category'] = X['Family_Size'].apply(categorize_family_size)
            logger.debug("Create features: Family_Size, Is_Alone, Family_Category")
        
        return X
    
    def _create_fare_features(self, X):
        """Crea features relative al prezzo"""
        logger.debug("Esecuzione _create_fare_features")
        if 'Fare' in X.columns:
            # Fare per persona
            if 'Family_Size' in X.columns:
                X['Fare_Per_Person'] = X['Fare'] / X['Family_Size']
            
            # Binning del fare
            X['Fare_Binned'] = pd.qcut(X['Fare'].fillna(X['Fare'].median()), 
                                      q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
            
            # Log fare per ridurre skewness
            X['Fare_Log'] = np.log1p(X['Fare'].fillna(X['Fare'].median()))
            logger.debug("Create features: Fare_Per_Person, Fare_Binned, Fare_Log")
        
        return X
    
    def _create_age_groups(self, X):
        """Crea gruppi di età"""
        logger.debug("Esecuzione _create_age_groups")
        def categorize_age(age):
            if pd.isna(age):
                return 'Unknown'
            elif age < 13:
                return 'Child'
            elif age < 25:
                return 'Young_Adult'
            elif age < 40:
                return 'Adult'
            elif age < 60:
                return 'Middle_Aged'
            else:
                return 'Senior'
        
        X['Age_Group'] = X['Age'].apply(categorize_age)
        
        # Age binning
        X['Age_Binned'] = pd.cut(X['Age'].fillna(X['Age'].median()), 
                                bins=5, labels=['Very_Young', 'Young', 'Middle', 'Mature', 'Old'])
        logger.debug("Create features: Age_Group, Age_Binned")
        
        return X
    
    def _create_interaction_features(self, X):
        """Crea features di interazione"""
        logger.debug("Esecuzione _create_interaction_features")
        # Interazioni importanti per sopravvivenza
        if 'Sex' in X.columns and 'Pclass' in X.columns:
            X['Sex_Pclass'] = X['Sex'].astype(str) + '_' + X['Pclass'].astype(str)
        
        if 'Age_Group' in X.columns and 'Sex' in X.columns:
            X['Age_Sex'] = X['Age_Group'].astype(str) + '_' + X['Sex'].astype(str)
        
        if 'Title' in X.columns and 'Pclass' in X.columns:
            X['Title_Pclass'] = X['Title'].astype(str) + '_' + X['Pclass'].astype(str)
        
        logger.debug(f"Create {len(X.columns) - len([c for c in X.columns if not c.endswith('_Pclass') and not c.endswith('_Sex')])} feature di interazione")
        return X

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Imputer intelligente che sceglie strategia basata sul tipo di dato
    """
    
    def __init__(self, 
                 numerical_strategy='median',
                 categorical_strategy='most_frequent',
                 use_advanced_imputation=False):
        logger.debug(f"Inizializzazione SmartImputer: "
                    f"numerical_strategy={numerical_strategy}, "
                    f"categorical_strategy={categorical_strategy}, "
                    f"use_advanced_imputation={use_advanced_imputation}")
        
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.use_advanced_imputation = use_advanced_imputation
        self.imputers_ = {}
        self.feature_types_ = {}
        
    def fit(self, X, y=None):
        """Fit degli imputers"""
        logger.info("Esecuzione fit SmartImputer")
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                self.feature_types_[column] = 'numerical'
                
                if self.use_advanced_imputation and ITERATIVE_IMPUTER_AVAILABLE:
                    logger.debug(f"Usando IterativeImputer per {column}")
                    self.imputers_[column] = IterativeImputer(random_state=42)
                elif self.use_advanced_imputation:
                    logger.debug(f"Usando KNNImputer per {column}")
                    self.imputers_[column] = KNNImputer(n_neighbors=5)
                else:
                    logger.debug(f"Usando SimpleImputer ({self.numerical_strategy}) per {column}")
                    self.imputers_[column] = SimpleImputer(strategy=self.numerical_strategy)
            else:
                self.feature_types_[column] = 'categorical'
                logger.debug(f"Usando SimpleImputer ({self.categorical_strategy}) per {column}")
                self.imputers_[column] = SimpleImputer(strategy=self.categorical_strategy)
            
            # Fit dell'imputer
            self.imputers_[column].fit(X[[column]])
        
        return self
    
    def transform(self, X):
        """Transform con imputation"""
        logger.info("Esecuzione transform SmartImputer")
        X_imputed = X.copy()
        
        for column in X.columns:
            if column in self.imputers_:
                logger.debug(f"Imputazione colonna {column}")
                imputed_values = self.imputers_[column].transform(X[[column]])
                X_imputed[column] = imputed_values.flatten()
        
        logger.debug(f"Valori mancanti dopo imputazione: {X_imputed.isnull().sum().sum()}")
        return X_imputed

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Gestione outliers con multiple strategie
    """
    
    def __init__(self, 
                 method='iqr',
                 action='clip',
                 threshold=3,
                 columns=None):
        logger.debug(f"Inizializzazione OutlierHandler: "
                    f"method={method}, "
                    f"action={action}, "
                    f"threshold={threshold}, "
                    f"columns={columns}")
        
        self.method = method  # 'iqr', 'zscore', 'isolation'
        self.action = action  # 'clip', 'remove', 'transform'
        self.threshold = threshold
        self.columns = columns
        self.outlier_bounds_ = {}
        
    def fit(self, X, y=None):
        """Calcola bounds per outliers"""
        logger.info("Esecuzione fit OutlierHandler")
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if self.columns is not None:
            numeric_columns = [col for col in numeric_columns if col in self.columns]
        
        for column in numeric_columns:
            if self.method == 'iqr':
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.outlier_bounds_[column] = (lower_bound, upper_bound)
                logger.debug(f"Colonna {column}: bounds IQR ({lower_bound:.2f}, {upper_bound:.2f})")
            
            elif self.method == 'zscore':
                mean = X[column].mean()
                std = X[column].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                self.outlier_bounds_[column] = (lower_bound, upper_bound)
                logger.debug(f"Colonna {column}: bounds z-score ({lower_bound:.2f}, {upper_bound:.2f})")
        
        return self
    
    def transform(self, X):
        """Applica gestione outliers"""
        logger.info("Esecuzione transform OutlierHandler")
        X_cleaned = X.copy()
        
        for column, (lower_bound, upper_bound) in self.outlier_bounds_.items():
            if column in X_cleaned.columns:
                if self.action == 'clip':
                    outliers = ((X_cleaned[column] < lower_bound) | (X_cleaned[column] > upper_bound)).sum()
                    X_cleaned[column] = X_cleaned[column].clip(lower_bound, upper_bound)
                    logger.debug(f"Clippati {outliers} outliers in {column}")
                elif self.action == 'transform':
                    # Log transform per ridurre impact outliers
                    X_cleaned[column] = np.log1p(X_cleaned[column] - X_cleaned[column].min() + 1)
                    logger.debug(f"Applicata trasformazione log a {column}")
        
        return X_cleaned

# ----------------2. Advanced Encoding

class AdvancedEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder avanzato con multiple strategie
    """
    
    def __init__(self, 
                 encoding_strategy='auto',
                 handle_unknown='ignore',
                 target_encoding_smoothing=1.0):
        logger.debug(f"Inizializzazione AdvancedEncoder: "
                    f"encoding_strategy={encoding_strategy}, "
                    f"handle_unknown={handle_unknown}, "
                    f"target_encoding_smoothing={target_encoding_smoothing}")
        
        self.encoding_strategy = encoding_strategy
        self.handle_unknown = handle_unknown
        self.target_encoding_smoothing = target_encoding_smoothing
        self.encoders_ = {}
        self.encoding_methods_ = {}
        
    def fit(self, X, y=None):
        """Fit degli encoders"""
        logger.info("Esecuzione fit AdvancedEncoder")
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            unique_values = X[column].nunique()
            
            # Strategia automatica basata su cardinalità
            if self.encoding_strategy == 'auto':
                if unique_values <= 2:
                    method = 'label'
                elif unique_values <= 10:
                    method = 'onehot'
                else:
                    method = 'target' if y is not None else 'label'
            else:
                method = self.encoding_strategy
            
            self.encoding_methods_[column] = method
            logger.debug(f"Colonna {column}: strategia encoding={method} (unique_values={unique_values})")
            
            # Crea encoder
            if method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[column].astype(str))
            elif method == 'onehot':
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
                encoder.fit(X[[column]])
            elif method == 'target' and y is not None:
                # Target encoding manuale
                encoder = self._create_target_encoder(X[column], y)
            else:
                encoder = LabelEncoder()
                encoder.fit(X[column].astype(str))
            
            self.encoders_[column] = encoder
        
        return self
    
    def transform(self, X):
        """Transform con encoding"""
        logger.info("Esecuzione transform AdvancedEncoder")
        X_encoded = X.copy()
        
        for column, encoder in self.encoders_.items():
            if column in X_encoded.columns:
                method = self.encoding_methods_[column]
                
                if method == 'label':
                    # Gestisce valori non visti
                    try:
                        X_encoded[column] = encoder.transform(X_encoded[column].astype(str))
                    except ValueError:
                        # Sostituisce valori non visti con il più frequente
                        known_values = encoder.classes_
                        X_encoded[column] = X_encoded[column].apply(
                            lambda x: x if str(x) in known_values else known_values[0]
                        )
                        X_encoded[column] = encoder.transform(X_encoded[column].astype(str))
                        logger.warning(f"Valori non visti in {column}, sostituiti con {known_values[0]}")
                
                elif method == 'onehot':
                    encoded_array = encoder.transform(X_encoded[[column]])
                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                    
                    # Rimuovi colonna originale e aggiungi encoded
                    X_encoded = X_encoded.drop(column, axis=1)
                    for i, feature_name in enumerate(feature_names):
                        X_encoded[feature_name] = encoded_array[:, i]
                    logger.debug(f"OneHot encoding per {column}: creati {len(feature_names)} nuove colonne")
                
                elif method == 'target':
                    X_encoded[column] = X_encoded[column].map(encoder).fillna(encoder.get('__unknown__', 0))
                    logger.debug(f"Target encoding applicato a {column}")
        
        logger.info(f"Encoding completato. Nuove dimensioni: {X_encoded.shape}")
        return X_encoded
    
    def _create_target_encoder(self, categorical_series, target):
        """Crea target encoder manuale"""
        logger.debug(f"Creazione target encoder per {categorical_series.name}")
        # Calcola media target per categoria
        target_means = categorical_series.to_frame().assign(target=target).groupby(categorical_series.name)['target'].mean()
        
        # Smoothing
        overall_mean = target.mean()
        category_counts = categorical_series.value_counts()
        
        smoothed_means = {}
        for category, mean_val in target_means.items():
            count = category_counts[category]
            smoothed_mean = (count * mean_val + self.target_encoding_smoothing * overall_mean) / (count + self.target_encoding_smoothing)
            smoothed_means[category] = smoothed_mean
        
        # Aggiungi valore per categorie non viste
        smoothed_means['__unknown__'] = overall_mean
        
        return smoothed_means

# ----------------3. Feature Selection

class IntelligentFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selezione intelligente delle features
    """
    
    def __init__(self, 
                 selection_methods=['variance', 'univariate', 'correlation'],
                 k_best=20,
                 variance_threshold=0.01,
                 correlation_threshold=0.95):
        logger.debug(f"Inizializzazione IntelligentFeatureSelector: "
                    f"selection_methods={selection_methods}, "
                    f"k_best={k_best}, "
                    f"variance_threshold={variance_threshold}, "
                    f"correlation_threshold={correlation_threshold}")
        
        self.selection_methods = selection_methods
        self.k_best = k_best
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features_ = None
        self.feature_scores_ = {}
        
    def fit(self, X, y=None):
        """Seleziona features"""
        logger.info("Esecuzione fit IntelligentFeatureSelector")
        selected_features = set(X.columns)
        
        # 1. Variance threshold
        if 'variance' in self.selection_methods:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(X)
            variance_features = X.columns[selector.get_support()].tolist()
            selected_features &= set(variance_features)
            
            # Score: variance
            variances = X.var()
            self.feature_scores_['variance'] = variances.to_dict()
            logger.debug(f"Variance threshold: selezionate {len(variance_features)} features")
        
        # 2. Univariate selection
        if 'univariate' in self.selection_methods and y is not None:
            # Usa chi2 per features positive, f_classif altrimenti
            X_positive = X.copy()
            # Assicura valori positivi per chi2
            for col in X_positive.columns:
                if X_positive[col].min() < 0:
                    X_positive[col] = X_positive[col] - X_positive[col].min()
            
            try:
                selector = SelectKBest(score_func=chi2, k=min(self.k_best, len(selected_features)))
                selector.fit(X_positive[list(selected_features)], y)
                univariate_features = selector.get_feature_names_out()
                selected_features &= set(univariate_features)
                
                # Score: chi2
                scores = dict(zip(X_positive[list(selected_features)].columns, selector.scores_))
                self.feature_scores_['univariate'] = scores
                logger.debug(f"Univariate selection (chi2): selezionate {len(univariate_features)} features")
            except:
                # Fallback a f_classif
                selector = SelectKBest(score_func=f_classif, k=min(self.k_best, len(selected_features)))
                selector.fit(X[list(selected_features)], y)
                univariate_features = selector.get_feature_names_out()
                selected_features &= set(univariate_features)
                logger.debug(f"Univariate selection (f_classif): selezionate {len(univariate_features)} features")
        
        # 3. Correlation filtering
        if 'correlation' in self.selection_methods:
            corr_matrix = X[list(selected_features)].corr().abs()
            
            # Trova coppie altamente correlate
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Rimuovi una feature da ogni coppia altamente correlata
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                # Rimuovi quella con varianza minore
                if X[feat1].var() < X[feat2].var():
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            
            selected_features -= features_to_remove
            logger.debug(f"Correlation filtering: rimosse {len(features_to_remove)} features altamente correlate")
        
        self.selected_features_ = list(selected_features)
        logger.info(f"Selezionate {len(self.selected_features_)} features finali")
        return self
    
    def transform(self, X):
        """Transform con feature selection"""
        logger.info("Esecuzione transform IntelligentFeatureSelector")
        if self.selected_features_ is None:
            logger.error("Feature selector deve essere fittato prima del transform")
            raise ValueError("Feature selector deve essere fittato prima del transform")
        
        return X[self.selected_features_]
    
    def get_feature_importance_summary(self):
        """Restituisce summary importanza features"""
        logger.debug("Esecuzione get_feature_importance_summary")
        if not self.feature_scores_:
            logger.warning("Nessun feature score disponibile")
            return None
        
        summary = {}
        for feature in self.selected_features_:
            scores = {}
            for method, feature_scores in self.feature_scores_.items():
                if feature in feature_scores:
                    scores[method] = feature_scores[feature]
            summary[feature] = scores
        
        return summary

# ----------------4. Preprocessing Pipeline Builder

class PreprocessingPipelineBuilder:
    """
    Builder per creare pipeline di preprocessing personalizzate
    """
    
    def __init__(self):
        logger.debug("Inizializzazione PreprocessingPipelineBuilder")
        self.steps = []
        self.pipeline = None
        
    def add_feature_engineering(self, **kwargs):
        """Aggiunge feature engineering"""
        logger.debug(f"Aggiunto step feature_engineering con kwargs: {kwargs}")
        self.steps.append(('feature_engineering', TitanicFeatureEngineer(**kwargs)))
        return self
    
    def add_imputation(self, **kwargs):
        """Aggiunge imputazione"""
        logger.debug(f"Aggiunto step imputation con kwargs: {kwargs}")
        self.steps.append(('imputation', SmartImputer(**kwargs)))
        return self
    
    def add_outlier_handling(self, **kwargs):
        """Aggiunge gestione outliers"""
        logger.debug(f"Aggiunto step outlier_handling con kwargs: {kwargs}")
        self.steps.append(('outlier_handling', OutlierHandler(**kwargs)))
        return self
    
    def add_encoding(self, **kwargs):
        """Aggiunge encoding"""
        logger.debug(f"Aggiunto step encoding con kwargs: {kwargs}")
        self.steps.append(('encoding', AdvancedEncoder(**kwargs)))
        return self
    
    def add_feature_selection(self, **kwargs):
        """Aggiunge feature selection"""
        logger.debug(f"Aggiunto step feature_selection con kwargs: {kwargs}")
        self.steps.append(('feature_selection', IntelligentFeatureSelector(**kwargs)))
        return self
    
    def add_scaling(self, method='standard', **kwargs):
        """Aggiunge scaling"""
        logger.debug(f"Aggiunto step scaling (method={method}) con kwargs: {kwargs}")
        if method == 'standard':
            scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            scaler = MinMaxScaler(**kwargs)
        elif method == 'robust':
            scaler = RobustScaler(**kwargs)
        elif method == 'power':
            scaler = PowerTransformer(**kwargs)
        else:
            logger.error(f"Metodo scaling non supportato: {method}")
            raise ValueError(f"Metodo scaling non supportato: {method}")
        
        self.steps.append(('scaling', scaler))
        return self
    
    def add_dimensionality_reduction(self, method='pca', **kwargs):
        """Aggiunge riduzione dimensionalità"""
        logger.debug(f"Aggiunto step dimensionality_reduction (method={method}) con kwargs: {kwargs}")
        if method == 'pca':
            reducer = PCA(**kwargs)
        else:
            logger.error(f"Metodo riduzione dimensionalità non supportato: {method}")
            raise ValueError(f"Metodo riduzione dimensionalità non supportato: {method}")
        
        self.steps.append(('dimensionality_reduction', reducer))
        return self
    
    def build(self):
        """Costruisce la pipeline"""
        logger.info("Costruzione pipeline preprocessing")
        if not self.steps:
            logger.error("Almeno uno step deve essere aggiunto alla pipeline")
            raise ValueError("Almeno uno step deve essere aggiunto alla pipeline")
        
        self.pipeline = Pipeline(self.steps)
        logger.info(f"Pipeline creata con {len(self.steps)} steps")
        return self.pipeline
    
    def get_step_names(self):
        """Restituisce nomi degli step"""
        logger.debug("Esecuzione get_step_names")
        return [step[0] for step in self.steps]

# ----------------5. Data Quality Checks

class DataQualityChecker:
    """
    Checker per qualità dei dati
    """
    
    @staticmethod
    def check_data_quality(X, y=None):
        """
        Controlla qualità generale dei dati
        
        Returns:
            Report qualità dati
        """
        logger.info("Esecuzione check_data_quality")
        report = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'missing_values': {},
            'duplicates': X.duplicated().sum(),
            'data_types': X.dtypes.value_counts().to_dict(),
            'memory_usage': X.memory_usage(deep=True).sum(),
            'constant_features': [],
            'high_cardinality_features': [],
            'skewed_features': [],
            'outliers_summary': {}
        }
        
        # Missing values
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(X) * 100, 2)
                }
        
        # Constant features
        for col in X.columns:
            if X[col].nunique() <= 1:
                report['constant_features'].append(col)
        
        # High cardinality features
        for col in X.select_dtypes(include=['object']).columns:
            unique_ratio = X[col].nunique() / len(X)
            if unique_ratio > 0.9:
                report['high_cardinality_features'].append({
                    'feature': col,
                    'unique_count': X[col].nunique(),
                    'unique_ratio': round(unique_ratio, 3)
                })
        
        # Skewed features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            skewness = X[col].skew()
            if abs(skewness) > 2:
                report['skewed_features'].append({
                    'feature': col,
                    'skewness': round(skewness, 3)
                })
        
        # Outliers summary
        for col in numeric_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
            
            if len(outliers) > 0:
                report['outliers_summary'][col] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(X) * 100, 2)
                }
        
        # Target analysis (se disponibile)
        if y is not None:
            report['target_analysis'] = {
                'class_distribution': y.value_counts().to_dict(),
                'balance_ratio': min(y.value_counts()) / max(y.value_counts()),
                'is_balanced': (min(y.value_counts()) / max(y.value_counts())) > 0.5
            }
        
        logger.debug(f"Report qualità dati generato: {len(report['missing_values'])} colonne con missing, {len(report['outliers_summary'])} con outliers")
        return report
    
    @staticmethod
    def suggest_preprocessing_steps(quality_report):
        """
        Suggerisce step di preprocessing basati sul report qualità
        
        Args:
            quality_report: Report da check_data_quality
        
        Returns:
            Lista di suggerimenti
        """
        logger.debug("Esecuzione suggest_preprocessing_steps")
        suggestions = []
        
        # Missing values
        if quality_report['missing_values']:
            high_missing = [col for col, info in quality_report['missing_values'].items() 
                           if info['percentage'] > 50]
            if high_missing:
                suggestions.append(f"Considera rimozione colonne con >50% missing: {high_missing}")
            else:
                suggestions.append("Applica imputazione per missing values")
        
        # Duplicates
        if quality_report['duplicates'] > 0:
            suggestions.append(f"Rimuovi {quality_report['duplicates']} righe duplicate")
        
        # Constant features
        if quality_report['constant_features']:
            suggestions.append(f"Rimuovi features costanti: {quality_report['constant_features']}")
        
        # High cardinality
        if quality_report['high_cardinality_features']:
            high_card_features = [f['feature'] for f in quality_report['high_cardinality_features']]
            suggestions.append(f"Considera encoding speciale per features ad alta cardinalità: {high_card_features}")
        
        # Skewed features
        if quality_report['skewed_features']:
            skewed_features = [f['feature'] for f in quality_report['skewed_features']]
            suggestions.append(f"Applica trasformazione (log, Box-Cox) per features skewed: {skewed_features}")
        
        # Outliers
        if quality_report['outliers_summary']:
            outlier_features = list(quality_report['outliers_summary'].keys())
            suggestions.append(f"Gestisci outliers per: {outlier_features}")
        
        # Target imbalance
        if 'target_analysis' in quality_report:
            if not quality_report['target_analysis']['is_balanced']:
                suggestions.append("Dataset sbilanciato - considera tecniche di bilanciamento")
        
        # Memory optimization
        if quality_report['memory_usage'] > 100_000_000:  # >100MB
            suggestions.append("Considera ottimizzazione tipi di dato per ridurre memoria")
        
        logger.debug(f"Generati {len(suggestions)} suggerimenti di preprocessing")
        return suggestions

# ----------------6. Utility Functions

def create_titanic_preprocessing_pipeline(config='standard'):
    """
    Crea pipeline di preprocessing predefinita per Titanic
    
    Args:
        config: 'minimal', 'standard', 'advanced'
    
    Returns:
        Pipeline di preprocessing
    """
    logger.info(f"Creazione pipeline preprocessing (config={config})")
    builder = PreprocessingPipelineBuilder()
    
    if config == 'minimal':
        pipeline = (builder
                   .add_feature_engineering(extract_title=False, create_interaction_features=False)
                   .add_imputation()
                   .add_encoding(encoding_strategy='label')
                   .build())
    
    elif config == 'standard':
        pipeline = (builder
                   .add_feature_engineering()
                   .add_imputation()
                   .add_outlier_handling()
                   .add_encoding()
                   .add_scaling()
                   .build())
    
    elif config == 'advanced':
        pipeline = (builder
                   .add_feature_engineering(create_interaction_features=True)
                   .add_imputation(use_advanced_imputation=True)
                   .add_outlier_handling()
                   .add_encoding(encoding_strategy='auto')
                   .add_feature_selection()
                   .add_scaling(method='robust')
                   .build())
    
    else:
        logger.error(f"Configurazione non supportata: {config}")
        raise ValueError(f"Configurazione non supportata: {config}")
    
    logger.info(f"Pipeline creata con {len(pipeline.steps)} steps")
    return pipeline

def validate_preprocessing_pipeline(pipeline, X_train, X_test, y_train=None, y_test=None):
    """
    Valida pipeline di preprocessing
    
    Args:
        pipeline: Pipeline da validare
        X_train, X_test: Set di training e test
        y_train, y_test: Target (opzionali)
    
    Returns:
        Report di validazione
    """
    logger.info("Validazione pipeline preprocessing")
    report = {
        'pipeline_steps': [step[0] for step in pipeline.steps],
        'validation_passed': True,
        'warnings': [],
        'errors': [],
        'shape_changes': {},
        'feature_changes': {},
        'data_quality_before': {},
        'data_quality_after': {}
    }
    
    try:
        # Qualità dati prima
        report['data_quality_before'] = DataQualityChecker.check_data_quality(X_train, y_train)
        
        # Applica pipeline
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)
        
        # Qualità dati dopo
        if isinstance(X_train_transformed, np.ndarray):
            # Converti in DataFrame per analisi
            feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        else:
            X_train_df = X_train_transformed
            
        report['data_quality_after'] = DataQualityChecker.check_data_quality(X_train_df)
        
        # Cambiamenti di shape
        report['shape_changes'] = {
            'train_before': X_train.shape,
            'train_after': X_train_transformed.shape,
            'test_before': X_test.shape,
            'test_after': X_test_transformed.shape
        }
        
        # Cambiamenti features
        if hasattr(X_train_transformed, 'columns'):
            original_features = set(X_train.columns)
            new_features = set(X_train_transformed.columns)
            
            report['feature_changes'] = {
                'original_count': len(original_features),
                'new_count': len(new_features),
                'removed_features': list(original_features - new_features),
                'added_features': list(new_features - original_features),
                'kept_features': list(original_features & new_features)
            }
        
        # Warnings
        if X_train_transformed.shape[1] < X_train.shape[1] * 0.5:
            report['warnings'].append("Più del 50% delle features sono state rimosse")
        
        if X_train_transformed.shape[0] != X_train.shape[0]:
            report['warnings'].append("Il numero di samples è cambiato")
        
        # Check per valori infiniti o NaN
        if isinstance(X_train_transformed, np.ndarray):
            if np.any(np.isnan(X_train_transformed)) or np.any(np.isinf(X_train_transformed)):
                report['errors'].append("Presenza di valori NaN o infiniti dopo preprocessing")
                report['validation_passed'] = False
        else:
            if X_train_transformed.isnull().any().any():
                report['errors'].append("Presenza di valori NaN dopo preprocessing")
                report['validation_passed'] = False
        
    except Exception as e:
        report['errors'].append(f"Errore durante validazione: {str(e)}")
        report['validation_passed'] = False
    
    logger.info(f"Validazione completata: {'successo' if report['validation_passed'] else 'fallita'}")
    return report

def optimize_preprocessing_pipeline(X, y, base_pipeline, scoring='accuracy', cv=3):
    """
    Ottimizza iperparametri della pipeline di preprocessing
    
    Args:
        X, y: Dati di training
        base_pipeline: Pipeline base da ottimizzare
        scoring: Metrica per ottimizzazione
        cv: Numero fold cross-validation
    
    Returns:
        Pipeline ottimizzata
    """
    logger.info("Ottimizzazione pipeline preprocessing")
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Crea pipeline completa con classificatore
    full_pipeline = Pipeline([
        ('preprocessing', base_pipeline),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=50))
    ])
    
    # Griglia parametri per ottimizzazione
    param_grid = {}
    
    # Parametri per feature engineering (se presente)
    if any('feature_engineering' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__feature_engineering__extract_title': [True, False],
            'preprocessing__feature_engineering__create_interaction_features': [True, False]
        })
    
    # Parametri per imputation (se presente)
    if any('imputation' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__imputation__numerical_strategy': ['mean', 'median'],
            'preprocessing__imputation__use_advanced_imputation': [True, False]
        })
    
    # Parametri per outlier handling (se presente)
    if any('outlier' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__outlier_handling__method': ['iqr', 'zscore'],
            'preprocessing__outlier_handling__action': ['clip', 'transform']
        })
    
    # Parametri per feature selection (se presente)
    if any('feature_selection' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__feature_selection__k_best': [10, 15, 20],
            'preprocessing__feature_selection__selection_methods': [
                ['variance', 'univariate'],
                ['variance', 'univariate', 'correlation']
            ]
        })
    
    if not param_grid:
        logger.warning("Nessun parametro da ottimizzare - ritorno pipeline originale")
        return base_pipeline
    
    # Ottimizzazione
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Estrae pipeline preprocessing ottimizzata
    optimized_pipeline = grid_search.best_estimator_.named_steps['preprocessing']
    
    logger.info(f"Ottimizzazione completata. Miglior score: {grid_search.best_score_:.4f}")
    return optimized_pipeline, {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'optimization_results': grid_search.cv_results_
    }

def get_preprocessing_recommendations(X, y=None):
    """
    Fornisce raccomandazioni per preprocessing basate sui dati
    
    Args:
        X: Features
        y: Target (opzionale)
    
    Returns:
        Dizionario con raccomandazioni
    """
    logger.info("Generazione raccomandazioni preprocessing")
    # Analisi qualità dati
    quality_report = DataQualityChecker.check_data_quality(X, y)
    
    recommendations = {
        'suggested_pipeline_config': 'standard',
        'required_steps': [],
        'optional_steps': [],
        'warnings': [],
        'estimated_complexity': 'medium'
    }
    
    # Determina configurazione suggerita
    complexity_score = 0
    
    # Missing values
    if quality_report['missing_values']:
        max_missing_pct = max(info['percentage'] for info in quality_report['missing_values'].values())
        if max_missing_pct > 50:
            complexity_score += 2
            recommendations['required_steps'].append('advanced_imputation')
        elif max_missing_pct > 20:
            complexity_score += 1
            recommendations['required_steps'].append('smart_imputation')
    
    # Outliers
    if quality_report['outliers_summary']:
        total_outlier_pct = sum(info['percentage'] for info in quality_report['outliers_summary'].values())
        if total_outlier_pct > 20:
            complexity_score += 1
            recommendations['required_steps'].append('outlier_handling')
    
    # High cardinality features
    if quality_report['high_cardinality_features']:
        complexity_score += 1
        recommendations['required_steps'].append('advanced_encoding')
    
    # Skewed features
    if quality_report['skewed_features']:
        complexity_score += 1
        recommendations['optional_steps'].append('power_transformation')
    
    # Target imbalance
    if 'target_analysis' in quality_report and not quality_report['target_analysis']['is_balanced']:
        recommendations['warnings'].append('Dataset sbilanciato - considera tecniche di sampling')
    
    # Molte features
    if quality_report['n_features'] > 20:
        complexity_score += 1
        recommendations['optional_steps'].append('feature_selection')
    
    # Determina configurazione
    if complexity_score <= 2:
        recommendations['suggested_pipeline_config'] = 'minimal'
        recommendations['estimated_complexity'] = 'low'
    elif complexity_score <= 4:
        recommendations['suggested_pipeline_config'] = 'standard'
        recommendations['estimated_complexity'] = 'medium'
    else:
        recommendations['suggested_pipeline_config'] = 'advanced'
        recommendations['estimated_complexity'] = 'high'
    
    # Suggerimenti aggiuntivi
    suggestions = DataQualityChecker.suggest_preprocessing_steps(quality_report)
    recommendations['detailed_suggestions'] = suggestions
    
    logger.info(f"Raccomandazioni generate: config={recommendations['suggested_pipeline_config']}")
    return recommendations

def create_preprocessing_report(X_before, X_after, y=None, pipeline_steps=None):
    """
    Crea report dettagliato del preprocessing
    
    Args:
        X_before: Dati prima del preprocessing
        X_after: Dati dopo il preprocessing
        y: Target (opzionale)
        pipeline_steps: Lista degli step applicati
    
    Returns:
        Report completo
    """
    logger.info("Creazione report preprocessing")
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'pipeline_steps': pipeline_steps or [],
        'data_transformation_summary': {},
        'quality_improvement': {},
        'feature_analysis': {},
        'recommendations': []
    }
    
    # Summary trasformazione
    report['data_transformation_summary'] = {
        'original_shape': X_before.shape,
        'final_shape': X_after.shape,
        'features_removed': X_before.shape[1] - X_after.shape[1],
        'samples_kept': X_after.shape[0],
        'transformation_ratio': X_after.shape[1] / X_before.shape[1]
    }
    
    # Miglioramento qualità
    quality_before = DataQualityChecker.check_data_quality(X_before, y)
    
    if isinstance(X_after, np.ndarray):
        # Converti in DataFrame per analisi
        feature_names = [f'feature_{i}' for i in range(X_after.shape[1])]
        X_after_df = pd.DataFrame(X_after, columns=feature_names)
    else:
        X_after_df = X_after
    
    quality_after = DataQualityChecker.check_data_quality(X_after_df)
    
    report['quality_improvement'] = {
        'missing_values_before': len(quality_before['missing_values']),
        'missing_values_after': len(quality_after['missing_values']),
        'constant_features_before': len(quality_before['constant_features']),
        'constant_features_after': len(quality_after['constant_features']),
        'outliers_before': len(quality_before['outliers_summary']),
        'outliers_after': len(quality_after['outliers_summary'])
    }
    
    # Analisi features
    if hasattr(X_after, 'columns') and hasattr(X_before, 'columns'):
        original_features = set(X_before.columns)
        final_features = set(X_after.columns)
        
        report['feature_analysis'] = {
            'original_features': list(original_features),
            'final_features': list(final_features),
            'removed_features': list(original_features - final_features),
            'added_features': list(final_features - original_features),
            'feature_retention_rate': len(final_features & original_features) / len(original_features)
        }
    
    # Raccomandazioni post-processing
    if report['quality_improvement']['missing_values_after'] > 0:
        report['recommendations'].append("Ancora presenti missing values - verifica imputation")
    
    if report['data_transformation_summary']['features_removed'] > X_before.shape[1] * 0.7:
        report['recommendations'].append("Molte features rimosse - verifica soglie feature selection")
    
    if isinstance(X_after, np.ndarray) and (np.any(np.isnan(X_after)) or np.any(np.isinf(X_after))):
        report['recommendations'].append("Valori NaN/infiniti nel risultato finale")
    
    logger.debug(f"Report creato con {len(report['recommendations'])} raccomandazioni")
    return report

# ----------------7. Export Functions

def save_preprocessing_pipeline(pipeline, filepath):
    """Salva pipeline di preprocessing"""
    logger.info(f"Salvataggio pipeline in {filepath}")
    import joblib
    joblib.dump(pipeline, filepath)
    return filepath

def load_preprocessing_pipeline(filepath):
    """Carica pipeline di preprocessing"""
    logger.info(f"Caricamento pipeline da {filepath}")
    import joblib
    return joblib.load(filepath)

def export_feature_names(pipeline, original_features, output_path):
    """
    Esporta mapping nomi features dopo preprocessing
    
    Args:
        pipeline: Pipeline applicata
        original_features: Feature originali
        output_path: Path per salvare mapping
    """
    logger.info(f"Esportazione feature names in {output_path}")
    # Applica pipeline a dummy data per ottenere feature names
    dummy_data = pd.DataFrame(np.zeros((1, len(original_features))), columns=original_features)
    
    try:
        transformed_data = pipeline.transform(dummy_data)
        
        if hasattr(transformed_data, 'columns'):
            final_features = transformed_data.columns.tolist()
        else:
            final_features = [f'feature_{i}' for i in range(transformed_data.shape[1])]
        
        mapping = {
            'original_features': original_features,
            'final_features': final_features,
            'pipeline_steps': [step[0] for step in pipeline.steps],
            'transformation_date': pd.Timestamp.now().isoformat()
        }
        
        # Salva come JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.debug(f"Esportati {len(final_features)} feature names")
        return mapping
        
    except Exception as e:
        logger.error(f"Errore nell'export feature names: {str(e)}")
        return None

logger.info(f"Caricamento completato {__name__}")