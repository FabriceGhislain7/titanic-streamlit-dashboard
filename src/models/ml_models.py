"""
src/models/ml_models.py
Definizioni e configurazioni dei modelli Machine Learning
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
logger.info(f"Caricamento {__name__}")

# ----------------1. Base Model Class

class TitanicModel:
    """
    Classe base per tutti i modelli Titanic
    """
    def __init__(self, model_type, **kwargs):
        logger.info(f"Inizializzazione TitanicModel per {model_type}")
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.requires_scaling = False
        self.hyperparameters = kwargs
        
    def get_model_info(self):
        """Restituisce informazioni sul modello"""
        logger.debug("Richiesta informazioni modello")
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
    Factory per creare istanze dei modelli ML
    """
    
    @staticmethod
    def create_model(model_type, custom_params=None):
        """
        Crea istanza del modello specificato
        
        Args:
            model_type (str): Tipo di modello da creare
            custom_params (dict): Parametri personalizzati
        
        Returns:
            Istanza del modello richiesto
        """
        logger.info(f"Creazione modello {model_type}")
        
        # Combina parametri di default con quelli personalizzati
        default_params = ML_MODELS.get(model_type, {}).get('params', {})
        params = {**default_params, **(custom_params or {})}
        
        if model_type == 'LogisticRegression':
            logger.debug("Creazione LogisticRegressionModel")
            return LogisticRegressionModel(**params)
        elif model_type == 'RandomForestClassifier':
            logger.debug("Creazione RandomForestModel")
            return RandomForestModel(**params)
        elif model_type == 'GradientBoostingClassifier':
            logger.debug("Creazione GradientBoostingModel")
            return GradientBoostingModel(**params)
        elif model_type == 'SVC':
            logger.debug("Creazione SVMModel")
            return SVMModel(**params)
        elif model_type == 'DecisionTreeClassifier':
            logger.debug("Creazione DecisionTreeModel")
            return DecisionTreeModel(**params)
        elif model_type == 'GaussianNB':
            logger.debug("Creazione NaiveBayesModel")
            return NaiveBayesModel(**params)
        elif model_type == 'KNeighborsClassifier':
            logger.debug("Creazione KNNModel")
            return KNNModel(**params)
        elif model_type == 'ExtraTreesClassifier':
            logger.debug("Creazione ExtraTreesModel")
            return ExtraTreesModel(**params)
        elif model_type == 'AdaBoostClassifier':
            logger.debug("Creazione AdaBoostModel")
            return AdaBoostModel(**params)
        elif model_type == 'MLPClassifier':
            logger.debug("Creazione MLPModel")
            return MLPModel(**params)
        else:
            logger.error(f"Tipo di modello non supportato: {model_type}")
            raise ValueError(f"Tipo di modello non supportato: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Restituisce lista dei modelli disponibili"""
        logger.debug("Richiesta modelli disponibili")
        return list(ML_MODELS.keys())
    
    @staticmethod
    def create_ensemble_models(model_types=None):
        """
        Crea un insieme di modelli per ensemble
        
        Args:
            model_types (list): Lista dei tipi di modello da creare
        
        Returns:
            dict: Dizionario con istanze dei modelli
        """
        logger.info("Creazione ensemble models")
        
        if model_types is None:
            model_types = ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']
            logger.debug(f"Usando modelli di default per ensemble: {model_types}")
        
        models = {}
        for model_type in model_types:
            models[model_type] = ModelFactory.create_model(model_type)
        
        logger.info(f"Creati {len(models)} modelli per ensemble")
        return models

# ----------------3. Concrete Model Classes

class LogisticRegressionModel(TitanicModel):
    """Modello Logistic Regression"""
    
    def __init__(self, **kwargs):
        super().__init__('LogisticRegression', **kwargs)
        logger.debug("Inizializzazione LogisticRegressionModel")
        self.requires_scaling = True
        self.model = LogisticRegression(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce coefficienti come importanza features"""
        logger.debug("Richiesta feature importance per LogisticRegression")
        
        if not self.is_trained:
            logger.warning("Modello non addestrato - impossibile ottenere feature importance")
            return None
        
        # Per logistic regression, usiamo valore assoluto dei coefficienti
        importances = np.abs(self.model.coef_[0])
        return dict(zip(self.feature_names, importances))

# ... (continua per tutte le altre classi di modello con lo stesso pattern)

# ----------------4. Model Configurations

class ModelConfigurations:
    """
    Configurazioni predefinite per diversi scenari
    """
    
    FAST_TRAINING = {
        'LogisticRegression': {'max_iter': 100},
        'RandomForestClassifier': {'n_estimators': 50},
        'GradientBoostingClassifier': {'n_estimators': 50},
        'DecisionTreeClassifier': {'max_depth': 5},
        'SVC': {'kernel': 'linear', 'max_iter': 100}
    }
    
    # ... (resto delle configurazioni)

    @classmethod
    def get_config(cls, config_name):
        """Restituisce configurazione specifica"""
        logger.info(f"Richiesta configurazione {config_name}")
        return getattr(cls, config_name.upper(), {})

# ----------------5. Hyperparameter Grids

class HyperparameterGrids:
    """
    Griglie di iperparametri per GridSearch
    """
    
    # ... (griglie di iperparametri)

    @classmethod
    def get_grid(cls, model_type):
        """Restituisce griglia per tipo di modello"""
        logger.info(f"Richiesta griglia per {model_type}")
        grid_name = model_type.upper().replace('CLASSIFIER', '')
        return getattr(cls, grid_name, {})

# ----------------6. Model Utilities

def get_model_complexity_score(model):
    """
    Calcola un punteggio di complessità del modello
    """
    logger.debug("Calcolo complessità modello")
    
    if hasattr(model, 'n_estimators'):
        return model.n_estimators / 100
    elif hasattr(model, 'C'):
        return 1 / model.C
    elif hasattr(model, 'max_depth') and model.max_depth:
        return model.max_depth / 10
    else:
        logger.debug("Usando valore di default per complessità")
        return 0.5  # Default per modelli senza parametri evidenti

# ... (resto delle utility functions con logging aggiunto)

# ----------------7. Model Ensemble

class ModelEnsemble:
    """
    Classe per gestire ensemble di modelli
    """
    
    def __init__(self, models):
        logger.info(f"Inizializzazione ModelEnsemble con {len(models)} modelli")
        self.models = models
        self.weights = None
        
    def set_weights(self, weights):
        """Imposta pesi per voting pesato"""
        logger.debug(f"Impostazione pesi ensemble: {weights}")
        self.weights = weights
    
    def predict_ensemble(self, X):
        """Predizione ensemble"""
        logger.info("Esecuzione predizione ensemble")
        predictions = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                logger.debug(f"Esecuzione predizione per {model_name}")
                pred = model.model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            logger.warning("Nessun modello addestrato disponibile per ensemble")
            return None
        
        # Voting majority
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        logger.debug("Predizione ensemble completata")
        return ensemble_pred

logger.info(f"Caricamento completato {__name__}")