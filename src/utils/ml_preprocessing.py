"""
src/utils/ml_preprocessing.py
Advanced Machine Learning Preprocessing
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
# Import IterativeImputer correctly
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
logger.info(f"Loading {__name__}")

warnings.filterwarnings('ignore')

# ----------------1. Custom Transformers

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Titanic-specific feature engineering
    """
    
    def __init__(self, 
                 extract_title=True,
                 extract_deck=True,
                 create_family_features=True,
                 create_fare_features=True,
                 create_age_groups=True,
                 create_interaction_features=False):
        logger.debug(f"Initializing TitanicFeatureEngineer: "
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
        """Fit the transformer"""
        logger.info("Executing fit TitanicFeatureEngineer")
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Apply feature engineering"""
        logger.info("Executing transform TitanicFeatureEngineer")
        if not self.fitted_:
            logger.error("Transformer must be fitted before transform")
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Extract title from name
        if self.extract_title and 'Name' in X_transformed.columns:
            logger.debug("Extracting title from name")
            X_transformed = self._extract_title(X_transformed)
        
        # Extract deck from cabin
        if self.extract_deck and 'Cabin' in X_transformed.columns:
            logger.debug("Extracting deck from cabin")
            X_transformed = self._extract_deck(X_transformed)
        
        # Family features
        if self.create_family_features:
            logger.debug("Creating family features")
            X_transformed = self._create_family_features(X_transformed)
        
        # Fare features
        if self.create_fare_features:
            logger.debug("Creating fare features")
            X_transformed = self._create_fare_features(X_transformed)
        
        # Age groups
        if self.create_age_groups and 'Age' in X_transformed.columns:
            logger.debug("Creating age groups")
            X_transformed = self._create_age_groups(X_transformed)
        
        # Interaction features
        if self.create_interaction_features:
            logger.debug("Creating interaction features")
            X_transformed = self._create_interaction_features(X_transformed)
        
        logger.info(f"Feature engineering completed. Final shape: {X_transformed.shape}")
        return X_transformed
    
    def _extract_title(self, X):
        """Extract title from name"""
        logger.debug("Executing _extract_title")
        def extract_title_from_name(name):
            if pd.isna(name):
                return 'Unknown'
            
            # Pattern to extract title
            title_pattern = r', ([A-Za-z]+)\.'
            match = re.search(title_pattern, str(name))
            
            if match:
                title = match.group(1)
                
                # Group rare titles
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
        logger.debug(f"Found {X['Title'].nunique()} unique titles")
        return X
    
    def _extract_deck(self, X):
        """Extract deck from cabin"""
        logger.debug("Executing _extract_deck")
        def extract_deck_from_cabin(cabin):
            if pd.isna(cabin):
                return 'Unknown'
            
            # First character of cabin is the deck
            deck = str(cabin)[0]
            return deck if deck.isalpha() else 'Unknown'
        
        X['Deck'] = X['Cabin'].apply(extract_deck_from_cabin)
        logger.debug(f"Found {X['Deck'].nunique()} unique decks")
        return X
    
    def _create_family_features(self, X):
        """Create family-related features"""
        logger.debug("Executing _create_family_features")
        if 'SibSp' in X.columns and 'Parch' in X.columns:
            X['Family_Size'] = X['SibSp'] + X['Parch'] + 1
            X['Is_Alone'] = (X['Family_Size'] == 1).astype(int)
            
            # Family categories
            def categorize_family_size(size):
                if size == 1:
                    return 'Alone'
                elif size <= 4:
                    return 'Small'
                else:
                    return 'Large'
            
            X['Family_Category'] = X['Family_Size'].apply(categorize_family_size)
            logger.debug("Created features: Family_Size, Is_Alone, Family_Category")
        
        return X
    
    def _create_fare_features(self, X):
        """Create fare-related features"""
        logger.debug("Executing _create_fare_features")
        if 'Fare' in X.columns:
            # Fare per person
            if 'Family_Size' in X.columns:
                X['Fare_Per_Person'] = X['Fare'] / X['Family_Size']
            
            # Fare binning
            X['Fare_Binned'] = pd.qcut(X['Fare'].fillna(X['Fare'].median()), 
                                      q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
            
            # Log fare to reduce skewness
            X['Fare_Log'] = np.log1p(X['Fare'].fillna(X['Fare'].median()))
            logger.debug("Created features: Fare_Per_Person, Fare_Binned, Fare_Log")
        
        return X
    
    def _create_age_groups(self, X):
        """Create age groups"""
        logger.debug("Executing _create_age_groups")
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
        logger.debug("Created features: Age_Group, Age_Binned")
        
        return X
    
    def _create_interaction_features(self, X):
        """Create interaction features"""
        logger.debug("Executing _create_interaction_features")
        # Important interactions for survival
        if 'Sex' in X.columns and 'Pclass' in X.columns:
            X['Sex_Pclass'] = X['Sex'].astype(str) + '_' + X['Pclass'].astype(str)
        
        if 'Age_Group' in X.columns and 'Sex' in X.columns:
            X['Age_Sex'] = X['Age_Group'].astype(str) + '_' + X['Sex'].astype(str)
        
        if 'Title' in X.columns and 'Pclass' in X.columns:
            X['Title_Pclass'] = X['Title'].astype(str) + '_' + X['Pclass'].astype(str)
        
        logger.debug("Created interaction features")
        return X

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Intelligent imputer that chooses strategy based on data type
    """
    
    def __init__(self, 
                 numerical_strategy='median',
                 categorical_strategy='most_frequent',
                 use_advanced_imputation=False):
        logger.debug(f"Initializing SmartImputer: "
                    f"numerical_strategy={numerical_strategy}, "
                    f"categorical_strategy={categorical_strategy}, "
                    f"use_advanced_imputation={use_advanced_imputation}")
        
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.use_advanced_imputation = use_advanced_imputation
        self.imputers_ = {}
        self.feature_types_ = {}
        
    def fit(self, X, y=None):
        """Fit the imputers"""
        logger.info("Executing fit SmartImputer")
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                self.feature_types_[column] = 'numerical'
                
                if self.use_advanced_imputation and ITERATIVE_IMPUTER_AVAILABLE:
                    logger.debug(f"Using IterativeImputer for {column}")
                    self.imputers_[column] = IterativeImputer(random_state=42)
                elif self.use_advanced_imputation:
                    logger.debug(f"Using KNNImputer for {column}")
                    self.imputers_[column] = KNNImputer(n_neighbors=5)
                else:
                    logger.debug(f"Using SimpleImputer ({self.numerical_strategy}) for {column}")
                    self.imputers_[column] = SimpleImputer(strategy=self.numerical_strategy)
            else:
                self.feature_types_[column] = 'categorical'
                logger.debug(f"Using SimpleImputer ({self.categorical_strategy}) for {column}")
                self.imputers_[column] = SimpleImputer(strategy=self.categorical_strategy)
            
            # Fit the imputer
            self.imputers_[column].fit(X[[column]])
        
        return self
    
    def transform(self, X):
        """Transform with imputation"""
        logger.info("Executing transform SmartImputer")
        X_imputed = X.copy()
        
        for column in X.columns:
            if column in self.imputers_:
                logger.debug(f"Imputing column {column}")
                imputed_values = self.imputers_[column].transform(X[[column]])
                X_imputed[column] = imputed_values.flatten()
        
        logger.debug(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        return X_imputed

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Outlier handling with multiple strategies
    """
    
    def __init__(self, 
                 method='iqr',
                 action='clip',
                 threshold=3,
                 columns=None):
        logger.debug(f"Initializing OutlierHandler: "
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
        """Calculate bounds for outliers"""
        logger.info("Executing fit OutlierHandler")
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
                logger.debug(f"Column {column}: IQR bounds ({lower_bound:.2f}, {upper_bound:.2f})")
            
            elif self.method == 'zscore':
                mean = X[column].mean()
                std = X[column].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                self.outlier_bounds_[column] = (lower_bound, upper_bound)
                logger.debug(f"Column {column}: z-score bounds ({lower_bound:.2f}, {upper_bound:.2f})")
        
        return self
    
    def transform(self, X):
        """Apply outlier handling"""
        logger.info("Executing transform OutlierHandler")
        X_cleaned = X.copy()
        
        for column, (lower_bound, upper_bound) in self.outlier_bounds_.items():
            if column in X_cleaned.columns:
                if self.action == 'clip':
                    outliers = ((X_cleaned[column] < lower_bound) | (X_cleaned[column] > upper_bound)).sum()
                    X_cleaned[column] = X_cleaned[column].clip(lower_bound, upper_bound)
                    logger.debug(f"Clipped {outliers} outliers in {column}")
                elif self.action == 'transform':
                    # Log transform to reduce outlier impact
                    X_cleaned[column] = np.log1p(X_cleaned[column] - X_cleaned[column].min() + 1)
                    logger.debug(f"Applied log transformation to {column}")
        
        return X_cleaned

# ----------------2. Advanced Encoding

class AdvancedEncoder(BaseEstimator, TransformerMixin):
    """
    Advanced encoder with multiple strategies
    """
    
    def __init__(self, 
                 encoding_strategy='auto',
                 handle_unknown='ignore',
                 target_encoding_smoothing=1.0):
        logger.debug(f"Initializing AdvancedEncoder: "
                    f"encoding_strategy={encoding_strategy}, "
                    f"handle_unknown={handle_unknown}, "
                    f"target_encoding_smoothing={target_encoding_smoothing}")
        
        self.encoding_strategy = encoding_strategy
        self.handle_unknown = handle_unknown
        self.target_encoding_smoothing = target_encoding_smoothing
        self.encoders_ = {}
        self.encoding_methods_ = {}
        
    def fit(self, X, y=None):
        """Fit the encoders"""
        logger.info("Executing fit AdvancedEncoder")
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            unique_values = X[column].nunique()
            
            # Automatic strategy based on cardinality
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
            logger.debug(f"Column {column}: encoding strategy={method} (unique_values={unique_values})")
            
            # Create encoder
            if method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[column].astype(str))
            elif method == 'onehot':
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
                encoder.fit(X[[column]])
            elif method == 'target' and y is not None:
                # Manual target encoding
                encoder = self._create_target_encoder(X[column], y)
            else:
                encoder = LabelEncoder()
                encoder.fit(X[column].astype(str))
            
            self.encoders_[column] = encoder
        
        return self
    
    def transform(self, X):
        """Transform with encoding"""
        logger.info("Executing transform AdvancedEncoder")
        X_encoded = X.copy()
        
        for column, encoder in self.encoders_.items():
            if column in X_encoded.columns:
                method = self.encoding_methods_[column]
                
                if method == 'label':
                    # Handle unseen values
                    try:
                        X_encoded[column] = encoder.transform(X_encoded[column].astype(str))
                    except ValueError:
                        # Replace unseen values with most frequent
                        known_values = encoder.classes_
                        X_encoded[column] = X_encoded[column].apply(
                            lambda x: x if str(x) in known_values else known_values[0]
                        )
                        X_encoded[column] = encoder.transform(X_encoded[column].astype(str))
                        logger.warning(f"Unseen values in {column}, replaced with {known_values[0]}")
                
                elif method == 'onehot':
                    encoded_array = encoder.transform(X_encoded[[column]])
                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                    
                    # Remove original column and add encoded
                    X_encoded = X_encoded.drop(column, axis=1)
                    for i, feature_name in enumerate(feature_names):
                        X_encoded[feature_name] = encoded_array[:, i]
                    logger.debug(f"OneHot encoding for {column}: created {len(feature_names)} new columns")
                
                elif method == 'target':
                    X_encoded[column] = X_encoded[column].map(encoder).fillna(encoder.get('__unknown__', 0))
                    logger.debug(f"Target encoding applied to {column}")
        
        logger.info(f"Encoding completed. New dimensions: {X_encoded.shape}")
        return X_encoded
    
    def _create_target_encoder(self, categorical_series, target):
        """Create manual target encoder"""
        logger.debug(f"Creating target encoder for {categorical_series.name}")
        # Calculate target mean per category
        target_means = categorical_series.to_frame().assign(target=target).groupby(categorical_series.name)['target'].mean()
        
        # Smoothing
        overall_mean = target.mean()
        category_counts = categorical_series.value_counts()
        
        smoothed_means = {}
        for category, mean_val in target_means.items():
            count = category_counts[category]
            smoothed_mean = (count * mean_val + self.target_encoding_smoothing * overall_mean) / (count + self.target_encoding_smoothing)
            smoothed_means[category] = smoothed_mean
        
        # Add value for unseen categories
        smoothed_means['__unknown__'] = overall_mean
        
        return smoothed_means

# ----------------3. Feature Selection

class IntelligentFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Intelligent feature selection
    """
    
    def __init__(self, 
                 selection_methods=['variance', 'univariate', 'correlation'],
                 k_best=20,
                 variance_threshold=0.01,
                 correlation_threshold=0.95):
        logger.debug(f"Initializing IntelligentFeatureSelector: "
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
        """Select features"""
        logger.info("Executing fit IntelligentFeatureSelector")
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
            logger.debug(f"Variance threshold: selected {len(variance_features)} features")
        
        # 2. Univariate selection
        if 'univariate' in self.selection_methods and y is not None:
            # Use chi2 for positive features, f_classif otherwise
            X_positive = X.copy()
            # Ensure positive values for chi2
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
                logger.debug(f"Univariate selection (chi2): selected {len(univariate_features)} features")
            except:
                # Fallback to f_classif
                selector = SelectKBest(score_func=f_classif, k=min(self.k_best, len(selected_features)))
                selector.fit(X[list(selected_features)], y)
                univariate_features = selector.get_feature_names_out()
                selected_features &= set(univariate_features)
                logger.debug(f"Univariate selection (f_classif): selected {len(univariate_features)} features")
        
        # 3. Correlation filtering
        if 'correlation' in self.selection_methods:
            corr_matrix = X[list(selected_features)].corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                # Remove the one with lower variance
                if X[feat1].var() < X[feat2].var():
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            
            selected_features -= features_to_remove
            logger.debug(f"Correlation filtering: removed {len(features_to_remove)} highly correlated features")
        
        self.selected_features_ = list(selected_features)
        logger.info(f"Selected {len(self.selected_features_)} final features")
        return self
    
    def transform(self, X):
        """Transform with feature selection"""
        logger.info("Executing transform IntelligentFeatureSelector")
        if self.selected_features_ is None:
            logger.error("Feature selector must be fitted before transform")
            raise ValueError("Feature selector must be fitted before transform")
        
        return X[self.selected_features_]
    
    def get_feature_importance_summary(self):
        """Return feature importance summary"""
        logger.debug("Executing get_feature_importance_summary")
        if not self.feature_scores_:
            logger.warning("No feature scores available")
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
    Builder for creating custom preprocessing pipelines
    """
    
    def __init__(self):
        logger.debug("Initializing PreprocessingPipelineBuilder")
        self.steps = []
        self.pipeline = None
        
    def add_feature_engineering(self, **kwargs):
        """Add feature engineering"""
        logger.debug(f"Added feature_engineering step with kwargs: {kwargs}")
        self.steps.append(('feature_engineering', TitanicFeatureEngineer(**kwargs)))
        return self
    
    def add_imputation(self, **kwargs):
        """Add imputation"""
        logger.debug(f"Added imputation step with kwargs: {kwargs}")
        self.steps.append(('imputation', SmartImputer(**kwargs)))
        return self
    
    def add_outlier_handling(self, **kwargs):
        """Add outlier handling"""
        logger.debug(f"Added outlier_handling step with kwargs: {kwargs}")
        self.steps.append(('outlier_handling', OutlierHandler(**kwargs)))
        return self
    
    def add_encoding(self, **kwargs):
        """Add encoding"""
        logger.debug(f"Added encoding step with kwargs: {kwargs}")
        self.steps.append(('encoding', AdvancedEncoder(**kwargs)))
        return self
    
    def add_feature_selection(self, **kwargs):
        """Add feature selection"""
        logger.debug(f"Added feature_selection step with kwargs: {kwargs}")
        self.steps.append(('feature_selection', IntelligentFeatureSelector(**kwargs)))
        return self
    
    def add_scaling(self, method='standard', **kwargs):
        """Add scaling"""
        logger.debug(f"Added scaling step (method={method}) with kwargs: {kwargs}")
        if method == 'standard':
            scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            scaler = MinMaxScaler(**kwargs)
        elif method == 'robust':
            scaler = RobustScaler(**kwargs)
        elif method == 'power':
            scaler = PowerTransformer(**kwargs)
        else:
            logger.error(f"Unsupported scaling method: {method}")
            raise ValueError(f"Unsupported scaling method: {method}")
        
        self.steps.append(('scaling', scaler))
        return self
    
    def add_dimensionality_reduction(self, method='pca', **kwargs):
        """Add dimensionality reduction"""
        logger.debug(f"Added dimensionality_reduction step (method={method}) with kwargs: {kwargs}")
        if method == 'pca':
            reducer = PCA(**kwargs)
        else:
            logger.error(f"Unsupported dimensionality reduction method: {method}")
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        self.steps.append(('dimensionality_reduction', reducer))
        return self
    
    def build(self):
        """Build the pipeline"""
        logger.info("Building preprocessing pipeline")
        if not self.steps:
            logger.error("At least one step must be added to the pipeline")
            raise ValueError("At least one step must be added to the pipeline")
        
        self.pipeline = Pipeline(self.steps)
        logger.info(f"Pipeline created with {len(self.steps)} steps")
        return self.pipeline
    
    def get_step_names(self):
        """Return step names"""
        logger.debug("Executing get_step_names")
        return [step[0] for step in self.steps]

# ----------------5. Data Quality Checks

class DataQualityChecker:
    """
    Data quality checker
    """
    
    @staticmethod
    def check_data_quality(X, y=None):
        """
        Check overall data quality
        
        Returns:
            Data quality report
        """
        logger.info("Executing check_data_quality")
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
        
        # Target analysis (if available)
        if y is not None:
            report['target_analysis'] = {
                'class_distribution': y.value_counts().to_dict(),
                'balance_ratio': min(y.value_counts()) / max(y.value_counts()),
                'is_balanced': (min(y.value_counts()) / max(y.value_counts())) > 0.5
            }
        
        logger.debug(f"Data quality report generated: {len(report['missing_values'])} columns with missing, {len(report['outliers_summary'])} with outliers")
        return report
    
    @staticmethod
    def suggest_preprocessing_steps(quality_report):
        """
        Suggest preprocessing steps based on quality report
        
        Args:
            quality_report: Report from check_data_quality
        
        Returns:
            List of suggestions
        """
        logger.debug("Executing suggest_preprocessing_steps")
        suggestions = []
        
        # Missing values
        if quality_report['missing_values']:
            high_missing = [col for col, info in quality_report['missing_values'].items() 
                           if info['percentage'] > 50]
            if high_missing:
                suggestions.append(f"Consider removing columns with >50% missing: {high_missing}")
            else:
                suggestions.append("Apply imputation for missing values")
        
        # Duplicates
        if quality_report['duplicates'] > 0:
            suggestions.append(f"Remove {quality_report['duplicates']} duplicate rows")
        
        # Constant features
        if quality_report['constant_features']:
            suggestions.append(f"Remove constant features: {quality_report['constant_features']}")
        
        # High cardinality
        if quality_report['high_cardinality_features']:
            high_card_features = [f['feature'] for f in quality_report['high_cardinality_features']]
            suggestions.append(f"Consider special encoding for high cardinality features: {high_card_features}")
        
        # Skewed features
        if quality_report['skewed_features']:
            skewed_features = [f['feature'] for f in quality_report['skewed_features']]
            suggestions.append(f"Apply transformation (log, Box-Cox) for skewed features: {skewed_features}")
        
        # Outliers
        if quality_report['outliers_summary']:
            outlier_features = list(quality_report['outliers_summary'].keys())
            suggestions.append(f"Handle outliers for: {outlier_features}")
        
        # Target imbalance
        if 'target_analysis' in quality_report:
            if not quality_report['target_analysis']['is_balanced']:
                suggestions.append("Imbalanced dataset - consider balancing techniques")
        
        # Memory optimization
        if quality_report['memory_usage'] > 100_000_000:  # >100MB
            suggestions.append("Consider data type optimization to reduce memory")
        
        logger.debug(f"Generated {len(suggestions)} preprocessing suggestions")
        return suggestions

# ----------------6. Utility Functions

def create_titanic_preprocessing_pipeline(config='standard'):
    """
    Create predefined preprocessing pipeline for Titanic
    
    Args:
        config: 'minimal', 'standard', 'advanced'
    
    Returns:
        Preprocessing pipeline
    """
    logger.info(f"Creating preprocessing pipeline (config={config})")
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
        logger.error(f"Unsupported configuration: {config}")
        raise ValueError(f"Unsupported configuration: {config}")
    
    logger.info(f"Pipeline created with {len(pipeline.steps)} steps")
    return pipeline

def validate_preprocessing_pipeline(pipeline, X_train, X_test, y_train=None, y_test=None):
    """
    Validate preprocessing pipeline
    
    Args:
        pipeline: Pipeline to validate
        X_train, X_test: Training and test sets
        y_train, y_test: Targets (optional)
    
    Returns:
        Validation report
    """
    logger.info("Validating preprocessing pipeline")
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
        # Data quality before
        report['data_quality_before'] = DataQualityChecker.check_data_quality(X_train, y_train)
        
        # Apply pipeline
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)
        
        # Data quality after
        if isinstance(X_train_transformed, np.ndarray):
            # Convert to DataFrame for analysis
            feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        else:
            X_train_df = X_train_transformed
            
        report['data_quality_after'] = DataQualityChecker.check_data_quality(X_train_df)
        
        # Shape changes
        report['shape_changes'] = {
            'train_before': X_train.shape,
            'train_after': X_train_transformed.shape,
            'test_before': X_test.shape,
            'test_after': X_test_transformed.shape
        }
        
        # Feature changes
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
            report['warnings'].append("More than 50% of features were removed")
        
        if X_train_transformed.shape[0] != X_train.shape[0]:
            report['warnings'].append("Number of samples changed")
        
        # Check for infinite or NaN values
        if isinstance(X_train_transformed, np.ndarray):
            if np.any(np.isnan(X_train_transformed)) or np.any(np.isinf(X_train_transformed)):
                report['errors'].append("Presence of NaN or infinite values after preprocessing")
                report['validation_passed'] = False
        else:
            if X_train_transformed.isnull().any().any():
                report['errors'].append("Presence of NaN values after preprocessing")
                report['validation_passed'] = False
        
    except Exception as e:
        report['errors'].append(f"Error during validation: {str(e)}")
        report['validation_passed'] = False
    
    logger.info(f"Validation completed: {'success' if report['validation_passed'] else 'failed'}")
    return report

def optimize_preprocessing_pipeline(X, y, base_pipeline, scoring='accuracy', cv=3):
    """
    Optimize preprocessing pipeline hyperparameters
    
    Args:
        X, y: Training data
        base_pipeline: Base pipeline to optimize
        scoring: Metric for optimization
        cv: Number of CV folds
    
    Returns:
        Optimized pipeline
    """
    logger.info("Optimizing preprocessing pipeline")
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Create full pipeline with classifier
    full_pipeline = Pipeline([
        ('preprocessing', base_pipeline),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=50))
    ])
    
    # Parameter grid for optimization
    param_grid = {}
    
    # Parameters for feature engineering (if present)
    if any('feature_engineering' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__feature_engineering__extract_title': [True, False],
            'preprocessing__feature_engineering__create_interaction_features': [True, False]
        })
    
    # Parameters for imputation (if present)
    if any('imputation' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__imputation__numerical_strategy': ['mean', 'median'],
            'preprocessing__imputation__use_advanced_imputation': [True, False]
        })
    
    # Parameters for outlier handling (if present)
    if any('outlier' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__outlier_handling__method': ['iqr', 'zscore'],
            'preprocessing__outlier_handling__action': ['clip', 'transform']
        })
    
    # Parameters for feature selection (if present)
    if any('feature_selection' in step[0] for step in base_pipeline.steps):
        param_grid.update({
            'preprocessing__feature_selection__k_best': [10, 15, 20],
            'preprocessing__feature_selection__selection_methods': [
                ['variance', 'univariate'],
                ['variance', 'univariate', 'correlation']
            ]
        })
    
    if not param_grid:
        logger.warning("No parameters to optimize - returning original pipeline")
        return base_pipeline
    
    # Optimization
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Extract optimized preprocessing pipeline
    optimized_pipeline = grid_search.best_estimator_.named_steps['preprocessing']
    
    logger.info(f"Optimization completed. Best score: {grid_search.best_score_:.4f}")
    return optimized_pipeline, {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'optimization_results': grid_search.cv_results_
    }

def get_preprocessing_recommendations(X, y=None):
    """
    Provide preprocessing recommendations based on data
    
    Args:
        X: Features
        y: Target (optional)
    
    Returns:
        Dictionary with recommendations
    """
    logger.info("Generating preprocessing recommendations")
    # Data quality analysis
    quality_report = DataQualityChecker.check_data_quality(X, y)
    
    recommendations = {
        'suggested_pipeline_config': 'standard',
        'required_steps': [],
        'optional_steps': [],
        'warnings': [],
        'estimated_complexity': 'medium'
    }
    
    # Determine suggested configuration
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
        recommendations['warnings'].append('Imbalanced dataset - consider sampling techniques')
    
    # Many features
    if quality_report['n_features'] > 20:
        complexity_score += 1
        recommendations['optional_steps'].append('feature_selection')
    
    # Determine configuration
    if complexity_score <= 2:
        recommendations['suggested_pipeline_config'] = 'minimal'
        recommendations['estimated_complexity'] = 'low'
    elif complexity_score <= 4:
        recommendations['suggested_pipeline_config'] = 'standard'
        recommendations['estimated_complexity'] = 'medium'
    else:
        recommendations['suggested_pipeline_config'] = 'advanced'
        recommendations['estimated_complexity'] = 'high'
    
    # Additional suggestions
    suggestions = DataQualityChecker.suggest_preprocessing_steps(quality_report)
    recommendations['detailed_suggestions'] = suggestions
    
    logger.info(f"Recommendations generated: config={recommendations['suggested_pipeline_config']}")
    return recommendations

def create_preprocessing_report(X_before, X_after, y=None, pipeline_steps=None):
    """
    Create detailed preprocessing report
    
    Args:
        X_before: Data before preprocessing
        X_after: Data after preprocessing
        y: Target (optional)
        pipeline_steps: List of applied steps
    
    Returns:
        Complete report
    """
    logger.info("Creating preprocessing report")
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'pipeline_steps': pipeline_steps or [],
        'data_transformation_summary': {},
        'quality_improvement': {},
        'feature_analysis': {},
        'recommendations': []
    }
    
    # Transformation summary
    report['data_transformation_summary'] = {
        'original_shape': X_before.shape,
        'final_shape': X_after.shape,
        'features_removed': X_before.shape[1] - X_after.shape[1],
        'samples_kept': X_after.shape[0],
        'transformation_ratio': X_after.shape[1] / X_before.shape[1]
    }
    
    # Quality improvement
    quality_before = DataQualityChecker.check_data_quality(X_before, y)
    
    if isinstance(X_after, np.ndarray):
        # Convert to DataFrame for analysis
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
    
    # Feature analysis
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
    
    # Post-processing recommendations
    if report['quality_improvement']['missing_values_after'] > 0:
        report['recommendations'].append("Still present missing values - verify imputation")
    
    if report['data_transformation_summary']['features_removed'] > X_before.shape[1] * 0.7:
        report['recommendations'].append("Many features removed - verify feature selection thresholds")
    
    if isinstance(X_after, np.ndarray) and (np.any(np.isnan(X_after)) or np.any(np.isinf(X_after))):
        report['recommendations'].append("NaN/infinite values in final result")
    
    logger.debug(f"Report created with {len(report['recommendations'])} recommendations")
    return report

# ----------------7. Export Functions

def save_preprocessing_pipeline(pipeline, filepath):
    """Save preprocessing pipeline"""
    logger.info(f"Saving pipeline to {filepath}")
    import joblib
    joblib.dump(pipeline, filepath)
    return filepath

def load_preprocessing_pipeline(filepath):
    """Load preprocessing pipeline"""
    logger.info(f"Loading pipeline from {filepath}")
    import joblib
    return joblib.load(filepath)

def export_feature_names(pipeline, original_features, output_path):
    """
    Export feature names mapping after preprocessing
    
    Args:
        pipeline: Applied pipeline
        original_features: Original features
        output_path: Path to save mapping
    """
    logger.info(f"Exporting feature names to {output_path}")
    # Apply pipeline to dummy data to get feature names
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
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.debug(f"Exported {len(final_features)} feature names")
        return mapping
        
    except Exception as e:
        logger.error(f"Error exporting feature names: {str(e)}")
        return None

logger.info(f"Loading completed {__name__}")