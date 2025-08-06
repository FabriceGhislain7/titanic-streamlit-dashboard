"""
src/models/model_trainer.py
Training pipeline e gestione modelli Machine Learning
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
import time
import pickle
import os
from datetime import datetime

from src.models.ml_models import ModelFactory, ModelEnsemble, HyperparameterGrids
from src.config import PREPROCESSING_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)
logger.info(f"Caricamento {__name__}")

# ----------------1. Data Preparation

class DataPreprocessor:
    """
    Classe per preprocessing dati per ML
    """
    
    def __init__(self, scaling_method='standard'):
        logger.info(f"Inizializzazione DataPreprocessor con scaling={scaling_method}")
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        self.is_fitted = False
        
    def fit_transform(self, X, y=None):
        """Fit e transform dei dati"""
        logger.info("Avvio fit_transform")
        logger.debug(f"Input shape: {X.shape}")
        
        X_processed = self._handle_missing_values(X.copy())
        X_processed = self._encode_categorical_features(X_processed)
        
        # Salva nomi features
        self.feature_names = X_processed.columns.tolist()
        logger.debug(f"Feature names salvate: {len(self.feature_names)} features")
        
        # Scaling se richiesto
        if self.scaling_method and self.scaling_method != 'none':
            logger.debug(f"Applicazione scaling: {self.scaling_method}")
            X_scaled = self._apply_scaling(X_processed, fit=True)
        else:
            logger.debug("Nessuno scaling applicato")
            X_scaled = X_processed
            
        self.is_fitted = True
        logger.info("fit_transform completato")
        return X_scaled, y
    
    def transform(self, X):
        """Transform su nuovi dati"""
        logger.info("Avvio transform")
        if not self.is_fitted:
            logger.error("Preprocessor deve essere fittato prima del transform")
            raise ValueError("Preprocessor deve essere fittato prima del transform")
            
        X_processed = self._handle_missing_values(X.copy())
        X_processed = self._encode_categorical_features(X_processed, fit=False)
        
        # Assicurati che abbia tutte le features
        for col in self.feature_names:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[self.feature_names]
        
        # Scaling se richiesto
        if self.scaler:
            logger.debug("Applicazione scaling transform")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        else:
            logger.debug("Nessuno scaling applicato in transform")
            X_scaled = X_processed
            
        logger.info("Transform completato")
        return X_scaled
    
    def _handle_missing_values(self, X):
        """Gestisce valori mancanti"""
        logger.debug("Gestione valori mancanti")
        for col in X.columns:
            if X[col].isnull().any():
                logger.debug(f"Valori mancanti trovati in {col}")
                if X[col].dtype in ['float64', 'int64']:
                    # Numeriche: usa mediana
                    fill_value = X[col].median()
                    X[col].fillna(fill_value, inplace=True)
                    logger.debug(f"Colonna {col} (numerica) - riempita con mediana: {fill_value}")
                else:
                    # Categoriche: usa moda
                    fill_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                    X[col].fillna(fill_value, inplace=True)
                    logger.debug(f"Colonna {col} (categorica) - riempita con moda: {fill_value}")
        return X
    
    def _encode_categorical_features(self, X, fit=True):
        """Codifica features categoriche"""
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        logger.debug(f"Colonne categoriche trovate: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                logger.debug(f"Fit encoder per {col} - classi: {len(self.label_encoders[col].classes_)}")
            else:
                if col in self.label_encoders:
                    # Gestisce valori non visti durante training
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        # Se ci sono valori non visti, usa il valore più frequente
                        known_values = self.label_encoders[col].classes_
                        X[col] = X[col].apply(lambda x: x if x in known_values else known_values[0])
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                        logger.warning(f"Valori non visti in {col} - sostituiti con {known_values[0]}")
        
        return X
    
    def _apply_scaling(self, X, fit=False):
        """Applica scaling ai dati"""
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            
            logger.debug(f"Fitting scaler: {type(self.scaler).__name__}")
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ----------------2. Training Pipeline

class ModelTrainer:
    """
    Classe principale per training modelli
    """
    
    def __init__(self, random_state=42):
        logger.info(f"Inizializzazione ModelTrainer con random_state={random_state}")
        self.random_state = random_state
        self.trained_models = {}
        self.training_results = {}
        self.preprocessors = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X, y, test_size=0.2, stratify=True):
        """
        Prepara dati per training
        
        Args:
            X: Features
            y: Target
            test_size: Dimensione test set (se 0, non fa split)
            stratify: Se usare stratificazione
        """
        logger.info(f"Preparazione dati - test_size={test_size}, stratify={stratify}")
        logger.debug(f"Input shape: X={X.shape}, y={y.shape}")
        
        if test_size == 0 or test_size is None:
            logger.info("Nessuno split applicato - usando dati così come sono")
            self.X_train = X
            self.y_train = y
            self.X_test = None
            self.y_test = None
            return X, None, y, None
        
        stratify_param = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.debug(f"Train shape: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.debug(f"Test shape: X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_single_model(self, model_type, use_scaling=None, custom_params=None):
        """
        Addestra singolo modello
        
        Args:
            model_type: Tipo di modello da addestrare
            use_scaling: Se usare scaling (auto-detect se None)
            custom_params: Parametri personalizzati
        
        Returns:
            Risultati del training
        """
        logger.info(f"Avvio training per {model_type}")
        start_time = time.time()
        
        # Crea modello
        model = ModelFactory.create_model(model_type, custom_params)
        logger.debug(f"Modello creato - parametri: {model.hyperparameters}")
        
        # Determina se usare scaling
        if use_scaling is None:
            use_scaling = model.requires_scaling
        logger.debug(f"Uso scaling: {use_scaling}")
        
        # Preprocessing
        scaling_method = 'standard' if use_scaling else 'none'
        preprocessor = DataPreprocessor(scaling_method=scaling_method)
        
        X_train_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        X_test_processed = preprocessor.transform(self.X_test)
        logger.debug(f"Dati preprocessati - shape: {X_train_processed.shape}")
        
        # Training
        logger.info("Avvio fitting modello...")
        model.model.fit(X_train_processed, self.y_train)
        model.is_trained = True
        model.feature_names = X_train_processed.columns.tolist()
        
        training_time = time.time() - start_time
        logger.info(f"Training completato in {training_time:.2f} secondi")
        
        # Salva modello e preprocessor
        self.trained_models[model_type] = model
        self.preprocessors[model_type] = preprocessor
        
        # Risultati
        results = {
            'model': model,
            'preprocessor': preprocessor,
            'training_time': training_time,
            'X_train_processed': X_train_processed,
            'X_test_processed': X_test_processed
        }
        
        self.training_results[model_type] = results
        
        return results
    
    def train_multiple_models(self, model_types, progress_callback=None):
        """
        Addestra multiple modelli
        
        Args:
            model_types: Lista dei tipi di modello
            progress_callback: Callback per aggiornare progress bar
        
        Returns:
            Dizionario con risultati di tutti i modelli
        """
        logger.info(f"Avvio training multiplo per {len(model_types)} modelli")
        results = {}
        
        for i, model_type in enumerate(model_types):
            if progress_callback:
                progress_callback(i / len(model_types), f"Training {model_type}...")
            
            try:
                logger.info(f"Training modello {i+1}/{len(model_types)}: {model_type}")
                result = self.train_single_model(model_type)
                results[model_type] = result
                
                if progress_callback:
                    progress_callback((i + 1) / len(model_types), f"Completato {model_type}")
                    
            except Exception as e:
                logger.error(f"Errore nel training di {model_type}: {str(e)}")
                st.error(f"Errore nel training di {model_type}: {str(e)}")
                continue
        
        logger.info("Training multiplo completato")
        return results
    
    def cross_validate_model(self, model_type, cv_folds=5, scoring='accuracy'):
        """
        Cross validation per singolo modello
        
        Args:
            model_type: Tipo di modello
            cv_folds: Numero di fold
            scoring: Metrica di scoring
        
        Returns:
            Risultati cross validation
        """
        logger.info(f"Avvio cross validation per {model_type} (cv_folds={cv_folds})")
        
        if model_type not in self.trained_models:
            logger.error(f"Modello {model_type} non è stato addestrato")
            raise ValueError(f"Modello {model_type} non è stato addestrato")
        
        model = self.trained_models[model_type]
        preprocessor = self.preprocessors[model_type]
        
        # Prepara dati completi
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        
        X_processed, _ = preprocessor.fit_transform(X_full, y_full)
        logger.debug(f"Dati per CV - shape: {X_processed.shape}")
        
        # Cross validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model.model, X_processed, y_full, cv=cv, scoring=scoring)
        
        logger.info(f"CV completato - punteggi: {cv_scores}")
        
        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
    
    def hyperparameter_tuning(self, model_type, param_grid=None, cv_folds=3, scoring='accuracy'):
        """
        Hyperparameter tuning con GridSearch
        
        Args:
            model_type: Tipo di modello
            param_grid: Griglia parametri (usa default se None)
            cv_folds: Numero di fold per CV
            scoring: Metrica di scoring
        
        Returns:
            Risultati del tuning
        """
        logger.info(f"Avvio hyperparameter tuning per {model_type}")
        
        if param_grid is None:
            param_grid = HyperparameterGrids.get_grid(model_type)
            logger.debug(f"Usando griglia parametri di default: {param_grid}")
        
        if not param_grid:
            logger.error(f"Nessuna griglia parametri disponibile per {model_type}")
            raise ValueError(f"Nessuna griglia parametri disponibile per {model_type}")
        
        # Crea modello base
        base_model = ModelFactory.create_model(model_type)
        logger.debug(f"Modello base creato per tuning")
        
        # Preprocessing
        scaling_method = 'standard' if base_model.requires_scaling else 'none'
        preprocessor = DataPreprocessor(scaling_method=scaling_method)
        X_train_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        logger.debug(f"Dati preprocessati per tuning - shape: {X_train_processed.shape}")
        
        # GridSearch
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Avvio GridSearchCV...")
        grid_search.fit(X_train_processed, self.y_train)
        
        logger.info(f"Tuning completato - migliori parametri: {grid_search.best_params_}")
        logger.debug(f"Miglior score: {grid_search.best_score_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    def create_learning_curves(self, model_type, train_sizes=None):
        """
        Genera learning curves
        
        Args:
            model_type: Tipo di modello
            train_sizes: Dimensioni training set da testare
        
        Returns:
            Dati per learning curves
        """
        logger.info(f"Generazione learning curves per {model_type}")
        
        if model_type not in self.trained_models:
            logger.error(f"Modello {model_type} non è stato addestrato")
            raise ValueError(f"Modello {model_type} non è stato addestrato")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            logger.debug(f"Usando train_sizes di default: {train_sizes}")
        
        model = self.trained_models[model_type]
        preprocessor = self.preprocessors[model_type]
        
        # Prepara dati
        X_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        logger.debug(f"Dati per learning curves - shape: {X_processed.shape}")
        
        # Learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model.model,
            X_processed,
            self.y_train,
            train_sizes=train_sizes,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        logger.info("Learning curves generate con successo")
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': np.mean(train_scores, axis=1),
            'train_std': np.std(train_scores, axis=1),
            'val_mean': np.mean(val_scores, axis=1),
            'val_std': np.std(val_scores, axis=1)
        }
        
# ----------------3. Ensemble Training

class EnsembleTrainer:
    """
    Classe per training ensemble di modelli
    """
    
    def __init__(self, base_trainer):
        logger.info("Inizializzazione EnsembleTrainer")
        self.base_trainer = base_trainer
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, model_types, voting='soft', weights=None):
        """
        Crea ensemble con voting
        
        Args:
            model_types: Lista tipi di modello
            voting: 'soft' o 'hard'
            weights: Pesi per i modelli
        
        Returns:
            Ensemble model
        """
        logger.info(f"Creazione voting ensemble (voting={voting}) con modelli: {model_types}")
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for model_type in model_types:
            if model_type in self.base_trainer.trained_models:
                model = self.base_trainer.trained_models[model_type].model
                estimators.append((model_type, model))
                logger.debug(f"Aggiunto modello {model_type} all'ensemble")
        
        if not estimators:
            logger.error("Nessun modello addestrato trovato per l'ensemble")
            raise ValueError("Nessun modello addestrato trovato")
        
        # Crea voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        logger.debug(f"VotingClassifier creato con {len(estimators)} modelli")
        
        # Training su dati preprocessati
        first_model_type = model_types[0]
        preprocessor = self.base_trainer.preprocessors[first_model_type]
        X_train_processed = self.base_trainer.training_results[first_model_type]['X_train_processed']
        
        logger.info("Avvio training voting ensemble...")
        ensemble.fit(X_train_processed, self.base_trainer.y_train)
        logger.info("Training voting ensemble completato")
        
        self.ensemble_models['voting'] = {
            'model': ensemble,
            'preprocessor': preprocessor,
            'component_models': model_types
        }
        
        return ensemble
    
    def create_stacking_ensemble(self, base_models, meta_model='LogisticRegression'):
        """
        Crea ensemble con stacking
        
        Args:
            base_models: Lista modelli base
            meta_model: Tipo di meta-modello
        
        Returns:
            Stacking ensemble
        """
        logger.info(f"Creazione stacking ensemble con meta-model {meta_model} e base models: {base_models}")
        from sklearn.ensemble import StackingClassifier
        
        estimators = []
        for model_type in base_models:
            if model_type in self.base_trainer.trained_models:
                model = self.base_trainer.trained_models[model_type].model
                estimators.append((model_type, model))
                logger.debug(f"Aggiunto modello base {model_type}")
        
        # Meta learner
        meta_learner = ModelFactory.create_model(meta_model).model
        logger.debug(f"Meta-learner creato: {meta_model}")
        
        # Crea stacking classifier
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3
        )
        
        # Training
        first_model_type = base_models[0]
        preprocessor = self.base_trainer.preprocessors[first_model_type]
        X_train_processed = self.base_trainer.training_results[first_model_type]['X_train_processed']
        
        logger.info("Avvio training stacking ensemble...")
        stacking.fit(X_train_processed, self.base_trainer.y_train)
        logger.info("Training stacking ensemble completato")
        
        self.ensemble_models['stacking'] = {
            'model': stacking,
            'preprocessor': preprocessor,
            'base_models': base_models,
            'meta_model': meta_model
        }
        
        return stacking

# ----------------4. Model Persistence

class ModelPersistence:
    """
    Gestione salvataggio e caricamento modelli
    """
    
    @staticmethod
    def save_model(model, model_name, save_dir=None):
        """
        Salva modello su disco
        
        Args:
            model: Modello da salvare
            model_name: Nome del modello
            save_dir: Directory di salvataggio
        
        Returns:
            Path del file salvato
        """
        logger.info(f"Salvataggio modello {model_name}")
        if save_dir is None:
            save_dir = MODELS_DIR
            logger.debug(f"Usando directory di default: {save_dir}")
        
        if not os.path.exists(save_dir):
            logger.debug(f"Creazione directory: {save_dir}")
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Modello salvato con successo in: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Errore nel salvataggio del modello: {str(e)}")
            raise Exception(f"Errore nel salvataggio del modello: {str(e)}")
    
    @staticmethod
    def load_model(filepath):
        """
        Carica modello da disco
        
        Args:
            filepath: Path del file modello
        
        Returns:
            Modello caricato
        """
        logger.info(f"Caricamento modello da: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info("Modello caricato con successo")
            return model
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {str(e)}")
            raise Exception(f"Errore nel caricamento del modello: {str(e)}")
    
    @staticmethod
    def list_saved_models(save_dir=None):
        """
        Lista modelli salvati
        
        Args:
            save_dir: Directory da esplorare
        
        Returns:
            Lista dei file modello
        """
        logger.info("Lista modelli salvati")
        if save_dir is None:
            save_dir = MODELS_DIR
            logger.debug(f"Usando directory di default: {save_dir}")
        
        if not os.path.exists(save_dir):
            logger.warning(f"Directory non trovata: {save_dir}")
            return []
        
        model_files = []
        for file in os.listdir(save_dir):
            if file.endswith('.pkl'):
                filepath = os.path.join(save_dir, file)
                stat = os.stat(filepath)
                model_files.append({
                    'filename': file,
                    'filepath': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
                logger.debug(f"Trovato modello: {file}")
        
        logger.info(f"Trovati {len(model_files)} modelli salvati")
        return sorted(model_files, key=lambda x: x['modified'], reverse=True)

# ----------------5. Training Configuration

class TrainingConfig:
    """
    Configurazioni per training
    """
    
    QUICK_TRAINING = {
        'test_size': 0.3,
        'cv_folds': 3,
        'use_hyperparameter_tuning': False,
        'models': ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier']
    }
    
    COMPREHENSIVE_TRAINING = {
        'test_size': 0.2,
        'cv_folds': 5,
        'use_hyperparameter_tuning': True,
        'models': ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']
    }
    
    DEEP_TRAINING = {
        'test_size': 0.2,
        'cv_folds': 10,
        'use_hyperparameter_tuning': True,
        'use_ensemble': True,
        'models': ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'DecisionTreeClassifier']
    }
    
    @classmethod
    def get_config(cls, config_name):
        """Restituisce configurazione specifica"""
        logger.info(f"Richiesta configurazione: {config_name}")
        return getattr(cls, config_name.upper(), cls.QUICK_TRAINING)

# ----------------6. Training Pipeline Manager

class TrainingPipelineManager:
    """
    Manager completo per pipeline di training
    """
    
    def __init__(self, config_name='QUICK_TRAINING'):
        logger.info(f"Inizializzazione TrainingPipelineManager con config: {config_name}")
        self.config = TrainingConfig.get_config(config_name)
        self.trainer = ModelTrainer()
        self.ensemble_trainer = None
        self.results = {}
        
    def run_full_pipeline(self, X, y, progress_callback=None):
        """
        Esegue pipeline completa di training
        
        Args:
            X: Features
            y: Target
            progress_callback: Callback per progress updates
        
        Returns:
            Risultati completi del training
        """
        logger.info("Avvio pipeline di training completa")
        total_steps = len(self.config['models']) + 2  # +2 per data prep e final evaluation
        current_step = 0
        
        # Step 1: Preparazione dati
        if progress_callback:
            progress_callback(current_step / total_steps, "Preparazione dati...")
        logger.info("Step 1: Preparazione dati")
        
        self.trainer.prepare_data(X, y, test_size=self.config['test_size'])
        current_step += 1
        
        # Step 2: Training modelli
        logger.info(f"Step 2: Training {len(self.config['models'])} modelli")
        training_results = self.trainer.train_multiple_models(
            self.config['models'],
            lambda p, msg: progress_callback((current_step + p) / total_steps, msg) if progress_callback else None
        )
        current_step += len(self.config['models'])
        
        # Step 3: Cross validation
        if progress_callback:
            progress_callback(current_step / total_steps, "Cross validation...")
        logger.info("Step 3: Cross validation")
        
        cv_results = {}
        for model_type in self.config['models']:
            if model_type in training_results:
                logger.debug(f"CV per modello {model_type}")
                cv_results[model_type] = self.trainer.cross_validate_model(
                    model_type, cv_folds=self.config['cv_folds']
                )
        
        # Step 4: Hyperparameter tuning (opzionale)
        if self.config.get('use_hyperparameter_tuning', False):
            logger.info("Step 4: Hyperparameter tuning")
            tuning_results = {}
            for model_type in self.config['models']:
                if model_type in training_results:
                    try:
                        logger.debug(f"Tuning per {model_type}")
                        tuning_results[model_type] = self.trainer.hyperparameter_tuning(model_type)
                    except Exception as e:
                        logger.warning(f"Errore tuning {model_type}: {str(e)}")
                        continue
        else:
            tuning_results = {}
            logger.info("Step 4: Hyperparameter tuning saltato")
        
        # Step 5: Ensemble (opzionale)
        if self.config.get('use_ensemble', False):
            logger.info("Step 5: Creazione ensemble")
            self.ensemble_trainer = EnsembleTrainer(self.trainer)
            ensemble_results = {
                'voting': self.ensemble_trainer.create_voting_ensemble(self.config['models'])
            }
            logger.info("Ensemble creato con successo")
        else:
            ensemble_results = {}
            logger.info("Step 5: Creazione ensemble saltata")
        
        current_step += 1
        if progress_callback:
            progress_callback(1.0, "Training completato!")
        
        # Compila risultati
        self.results = {
            'training': training_results,
            'cross_validation': cv_results,
            'hyperparameter_tuning': tuning_results,
            'ensemble': ensemble_results,
            'config': self.config
        }
        
        logger.info("Pipeline di training completata con successo")
        return self.results
    
    def get_best_model(self, metric='accuracy'):
        """
        Restituisce il miglior modello basato su metrica
        
        Args:
            metric: Metrica per comparazione
        
        Returns:
            Nome e score del miglior modello
        """
        logger.info(f"Ricerca miglior modello per metrica: {metric}")
        if 'cross_validation' not in self.results:
            logger.warning("Nessun risultato CV disponibile")
            return None
        
        best_score = 0
        best_model = None
        
        for model_type, cv_result in self.results['cross_validation'].items():
            score = cv_result['mean']
            if score > best_score:
                best_score = score
                best_model = model_type
        
        if best_model:
            logger.info(f"Miglior modello trovato: {best_model} (score={best_score})")
        else:
            logger.warning("Nessun modello valido trovato")
        
        return {'model': best_model, 'score': best_score}
    
    def save_pipeline_results(self, filepath):
        """Salva risultati pipeline"""
        logger.info(f"Salvataggio risultati pipeline in: {filepath}")
        import json
        
        # Prepara dati serializzabili
        serializable_results = {
            'config': self.config,
            'cross_validation': self.results.get('cross_validation', {}),
            'best_model': self.get_best_model()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info("Risultati salvati con successo")
            return filepath
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei risultati: {str(e)}")
            raise

logger.info(f"Caricamento completato {__name__}")