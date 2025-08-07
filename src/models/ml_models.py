"""
src/models/ml_models.py
Machine Learning Model Definitions and Configurations
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from src.config import ML_MODELS

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Base Model Class

class TitanicModel:
    """
    Base class for all Titanic models
    """
    def __init__(self, model_type, **kwargs):
        logger.info(f"Initializing TitanicModel for {model_type}")
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.requires_scaling = False
        self.hyperparameters = kwargs
        
    def get_model_info(self):
        """Return model information"""
        logger.debug("Requesting model information")
        return {
            'type': self.model_type,
            'name': ML_MODELS.get(self.model_type, {}).get('name', self.model_type),
            'requires_scaling': self.requires_scaling,
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters
        }

# ----------------2. Model Factory

class ModelFactory:
    """
    Factory for creating ML model instances
    """
    
    @staticmethod
    def create_model(model_type, custom_params=None):
        """
        Create instance of specified model
        
        Args:
            model_type (str): Type of model to create
            custom_params (dict): Custom parameters
        
        Returns:
            Instance of requested model
        """
        logger.info(f"Creating model {model_type}")
        
        # Combine default parameters with custom ones
        default_params = ML_MODELS.get(model_type, {}).get('params', {})
        params = {**default_params, **(custom_params or {})}
        
        if model_type == 'LogisticRegression':
            logger.debug("Creating LogisticRegressionModel")
            return LogisticRegressionModel(**params)
        elif model_type == 'RandomForestClassifier':
            logger.debug("Creating RandomForestModel")
            return RandomForestModel(**params)
        elif model_type == 'GradientBoostingClassifier':
            logger.debug("Creating GradientBoostingModel")
            return GradientBoostingModel(**params)
        elif model_type == 'SVC':
            logger.debug("Creating SVMModel")
            return SVMModel(**params)
        elif model_type == 'DecisionTreeClassifier':
            logger.debug("Creating DecisionTreeModel")
            return DecisionTreeModel(**params)
        elif model_type == 'GaussianNB':
            logger.debug("Creating NaiveBayesModel")
            return NaiveBayesModel(**params)
        elif model_type == 'KNeighborsClassifier':
            logger.debug("Creating KNNModel")
            return KNNModel(**params)
        elif model_type == 'ExtraTreesClassifier':
            logger.debug("Creating ExtraTreesModel")
            return ExtraTreesModel(**params)
        elif model_type == 'AdaBoostClassifier':
            logger.debug("Creating AdaBoostModel")
            return AdaBoostModel(**params)
        elif model_type == 'MLPClassifier':
            logger.debug("Creating MLPModel")
            return MLPModel(**params)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Return list of available models"""
        logger.debug("Requesting available models")
        return list(ML_MODELS.keys())
    
    @staticmethod
    def create_ensemble_models(model_types=None):
        """
        Create a set of models for ensemble
        
        Args:
            model_types (list): List of model types to create
        
        Returns:
            dict: Dictionary with model instances
        """
        logger.info("Creating ensemble models")
        
        if model_types is None:
            model_types = ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']
            logger.debug(f"Using default models for ensemble: {model_types}")
        
        models = {}
        for model_type in model_types:
            models[model_type] = ModelFactory.create_model(model_type)
        
        logger.info(f"Created {len(models)} models for ensemble")
        return models

# ----------------3. Concrete Model Classes

class LogisticRegressionModel(TitanicModel):
    """Logistic Regression Model"""
    
    def __init__(self, **kwargs):
        super().__init__('LogisticRegression', **kwargs)
        logger.debug("Initializing LogisticRegressionModel")
        self.requires_scaling = True
        self.model = LogisticRegression(**kwargs)
    
    def get_feature_importance(self):
        """Return coefficients as feature importance"""
        logger.debug("Requesting feature importance for LogisticRegression")
        
        if not self.is_trained:
            logger.warning("Model not trained - cannot get feature importance")
            return None
        
        # For logistic regression, use absolute value of coefficients
        importances = np.abs(self.model.coef_[0])
        return dict(zip(self.feature_names, importances))

# ... (continues for all other model classes with the same pattern)

# ----------------4. Model Configurations

class ModelConfigurations:
    """
    Predefined configurations for different scenarios
    """
    
    FAST_TRAINING = {
        'LogisticRegression': {'max_iter': 100},
        'RandomForestClassifier': {'n_estimators': 50},
        'GradientBoostingClassifier': {'n_estimators': 50},
        'DecisionTreeClassifier': {'max_depth': 5},
        'SVC': {'kernel': 'linear', 'max_iter': 100}
    }
    
    # ... (rest of configurations)

    @classmethod
    def get_config(cls, config_name):
        """Return specific configuration"""
        logger.info(f"Requesting configuration {config_name}")
        return getattr(cls, config_name.upper(), {})

# ----------------5. Hyperparameter Grids

class HyperparameterGrids:
    """
    Hyperparameter grids for GridSearch
    """
    
    # ... (hyperparameter grids)

    @classmethod
    def get_grid(cls, model_type):
        """Return grid for model type"""
        logger.info(f"Requesting grid for {model_type}")
        grid_name = model_type.upper().replace('CLASSIFIER', '')
        return getattr(cls, grid_name, {})

# ----------------6. Model Utilities

def get_model_complexity_score(model):
    """
    Calculate model complexity score
    """
    logger.debug("Calculating model complexity")
    
    if hasattr(model, 'n_estimators'):
        return model.n_estimators / 100
    elif hasattr(model, 'C'):
        return 1 / model.C
    elif hasattr(model, 'max_depth') and model.max_depth:
        return model.max_depth / 10
    else:
        logger.debug("Using default complexity value")
        return 0.5  # Default for models without obvious parameters

# ... (rest of utility functions with added logging)

# ----------------7. Model Ensemble

class ModelEnsemble:
    """
    Class for managing model ensembles
    """
    
    def __init__(self, models):
        logger.info(f"Initializing ModelEnsemble with {len(models)} models")
        self.models = models
        self.weights = None
        
    def set_weights(self, weights):
        """Set weights for weighted voting"""
        logger.debug(f"Setting ensemble weights: {weights}")
        self.weights = weights
    
    def predict_ensemble(self, X):
        """Ensemble prediction"""
        logger.info("Executing ensemble prediction")
        predictions = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                logger.debug(f"Executing prediction for {model_name}")
                pred = model.model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            logger.warning("No trained models available for ensemble")
            return None
        
        # Majority voting
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        logger.debug("Ensemble prediction completed")
        return ensemble_pred

logger.info(f"Loading completed {__name__}")