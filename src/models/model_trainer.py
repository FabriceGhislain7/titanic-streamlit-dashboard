"""
src/models/model_trainer.py
Training pipeline and Machine Learning model management
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
logger.info(f"Loading {__name__}")

# ----------------1. Data Preparation

class DataPreprocessor:
    """
    Class for ML data preprocessing
    """
    
    def __init__(self, scaling_method='standard'):
        logger.info(f"Initializing DataPreprocessor with scaling={scaling_method}")
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        self.is_fitted = False
        
    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        logger.info("Starting fit_transform")
        logger.debug(f"Input shape: {X.shape}")
        
        X_processed = self._handle_missing_values(X.copy())
        X_processed = self._encode_categorical_features(X_processed)
        
        # Save feature names
        self.feature_names = X_processed.columns.tolist()
        logger.debug(f"Feature names saved: {len(self.feature_names)} features")
        
        # Scaling if required
        if self.scaling_method and self.scaling_method != 'none':
            logger.debug(f"Applying scaling: {self.scaling_method}")
            X_scaled = self._apply_scaling(X_processed, fit=True)
        else:
            logger.debug("No scaling applied")
            X_scaled = X_processed
            
        self.is_fitted = True
        logger.info("fit_transform completed")
        return X_scaled, y
    
    def transform(self, X):
        """Transform new data"""
        logger.info("Starting transform")
        if not self.is_fitted:
            logger.error("Preprocessor must be fitted before transform")
            raise ValueError("Preprocessor must be fitted before transform")
            
        X_processed = self._handle_missing_values(X.copy())
        X_processed = self._encode_categorical_features(X_processed, fit=False)
        
        # Ensure it has all features
        for col in self.feature_names:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[self.feature_names]
        
        # Scaling if required
        if self.scaler:
            logger.debug("Applying scaling transform")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        else:
            logger.debug("No scaling applied in transform")
            X_scaled = X_processed
            
        logger.info("Transform completed")
        return X_scaled
    
    def _handle_missing_values(self, X):
        """Handle missing values"""
        logger.debug("Handling missing values")
        for col in X.columns:
            if X[col].isnull().any():
                logger.debug(f"Missing values found in {col}")
                if X[col].dtype in ['float64', 'int64']:
                    # Numerical: use median
                    fill_value = X[col].median()
                    X[col].fillna(fill_value, inplace=True)
                    logger.debug(f"Column {col} (numerical) - filled with median: {fill_value}")
                else:
                    # Categorical: use mode
                    fill_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                    X[col].fillna(fill_value, inplace=True)
                    logger.debug(f"Column {col} (categorical) - filled with mode: {fill_value}")
        return X
    
    def _encode_categorical_features(self, X, fit=True):
        """Encode categorical features"""
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        logger.debug(f"Categorical columns found: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                logger.debug(f"Fit encoder for {col} - classes: {len(self.label_encoders[col].classes_)}")
            else:
                if col in self.label_encoders:
                    # Handle unseen values during training
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        # If there are unseen values, use most frequent value
                        known_values = self.label_encoders[col].classes_
                        X[col] = X[col].apply(lambda x: x if x in known_values else known_values[0])
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                        logger.warning(f"Unseen values in {col} - replaced with {known_values[0]}")
        
        return X
    
    def _apply_scaling(self, X, fit=False):
        """Apply scaling to data"""
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
    Main class for model training
    """
    
    def __init__(self, random_state=42):
        logger.info(f"Initializing ModelTrainer with random_state={random_state}")
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
        Prepare data for training
        
        Args:
            X: Features
            y: Target
            test_size: Test set size (if 0, no split)
            stratify: Whether to use stratification
        """
        logger.info(f"Data preparation - test_size={test_size}, stratify={stratify}")
        logger.debug(f"Input shape: X={X.shape}, y={y.shape}")
        
        if test_size == 0 or test_size is None:
            logger.info("No split applied - using data as is")
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
        Train single model
        
        Args:
            model_type: Type of model to train
            use_scaling: Whether to use scaling (auto-detect if None)
            custom_params: Custom parameters
        
        Returns:
            Training results
        """
        logger.info(f"Starting training for {model_type}")
        start_time = time.time()
        
        # Create model
        model = ModelFactory.create_model(model_type, custom_params)
        logger.debug(f"Model created - parameters: {model.hyperparameters}")
        
        # Determine whether to use scaling
        if use_scaling is None:
            use_scaling = model.requires_scaling
        logger.debug(f"Use scaling: {use_scaling}")
        
        # Preprocessing
        scaling_method = 'standard' if use_scaling else 'none'
        preprocessor = DataPreprocessor(scaling_method=scaling_method)
        
        X_train_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        X_test_processed = preprocessor.transform(self.X_test)
        logger.debug(f"Data preprocessed - shape: {X_train_processed.shape}")
        
        # Training
        logger.info("Starting model fitting...")
        model.model.fit(X_train_processed, self.y_train)
        model.is_trained = True
        model.feature_names = X_train_processed.columns.tolist()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save model and preprocessor
        self.trained_models[model_type] = model
        self.preprocessors[model_type] = preprocessor
        
        # Results
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
        Train multiple models
        
        Args:
            model_types: List of model types
            progress_callback: Callback to update progress bar
        
        Returns:
            Dictionary with results of all models
        """
        logger.info(f"Starting multiple training for {len(model_types)} models")
        results = {}
        
        for i, model_type in enumerate(model_types):
            if progress_callback:
                progress_callback(i / len(model_types), f"Training {model_type}...")
            
            try:
                logger.info(f"Training model {i+1}/{len(model_types)}: {model_type}")
                result = self.train_single_model(model_type)
                results[model_type] = result
                
                if progress_callback:
                    progress_callback((i + 1) / len(model_types), f"Completed {model_type}")
                    
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                st.error(f"Error training {model_type}: {str(e)}")
                continue
        
        logger.info("Multiple training completed")
        return results
    
    def cross_validate_model(self, model_type, cv_folds=5, scoring='accuracy'):
        """
        Cross validation for single model
        
        Args:
            model_type: Model type
            cv_folds: Number of folds
            scoring: Scoring metric
        
        Returns:
            Cross validation results
        """
        logger.info(f"Starting cross validation for {model_type} (cv_folds={cv_folds})")
        
        if model_type not in self.trained_models:
            logger.error(f"Model {model_type} has not been trained")
            raise ValueError(f"Model {model_type} has not been trained")
        
        model = self.trained_models[model_type]
        preprocessor = self.preprocessors[model_type]
        
        # Prepare complete data
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        
        X_processed, _ = preprocessor.fit_transform(X_full, y_full)
        logger.debug(f"Data for CV - shape: {X_processed.shape}")
        
        # Cross validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model.model, X_processed, y_full, cv=cv, scoring=scoring)
        
        logger.info(f"CV completed - scores: {cv_scores}")
        
        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
    
    def hyperparameter_tuning(self, model_type, param_grid=None, cv_folds=3, scoring='accuracy'):
        """
        Hyperparameter tuning with GridSearch
        
        Args:
            model_type: Model type
            param_grid: Parameter grid (use default if None)
            cv_folds: Number of folds for CV
            scoring: Scoring metric
        
        Returns:
            Tuning results
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        if param_grid is None:
            param_grid = HyperparameterGrids.get_grid(model_type)
            logger.debug(f"Using default parameter grid: {param_grid}")
        
        if not param_grid:
            logger.error(f"No parameter grid available for {model_type}")
            raise ValueError(f"No parameter grid available for {model_type}")
        
        # Create base model
        base_model = ModelFactory.create_model(model_type)
        logger.debug(f"Base model created for tuning")
        
        # Preprocessing
        scaling_method = 'standard' if base_model.requires_scaling else 'none'
        preprocessor = DataPreprocessor(scaling_method=scaling_method)
        X_train_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        logger.debug(f"Data preprocessed for tuning - shape: {X_train_processed.shape}")
        
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
        
        logger.info("Starting GridSearchCV...")
        grid_search.fit(X_train_processed, self.y_train)
        
        logger.info(f"Tuning completed - best parameters: {grid_search.best_params_}")
        logger.debug(f"Best score: {grid_search.best_score_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    def create_learning_curves(self, model_type, train_sizes=None):
        """
        Generate learning curves
        
        Args:
            model_type: Model type
            train_sizes: Training set sizes to test
        
        Returns:
            Data for learning curves
        """
        logger.info(f"Generating learning curves for {model_type}")
        
        if model_type not in self.trained_models:
            logger.error(f"Model {model_type} has not been trained")
            raise ValueError(f"Model {model_type} has not been trained")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            logger.debug(f"Using default train_sizes: {train_sizes}")
        
        model = self.trained_models[model_type]
        preprocessor = self.preprocessors[model_type]
        
        # Prepare data
        X_processed, _ = preprocessor.fit_transform(self.X_train, self.y_train)
        logger.debug(f"Data for learning curves - shape: {X_processed.shape}")
        
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
        
        logger.info("Learning curves generated successfully")
        
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
    Class for training model ensembles
    """
    
    def __init__(self, base_trainer):
        logger.info("Initializing EnsembleTrainer")
        self.base_trainer = base_trainer
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, model_types, voting='soft', weights=None):
        """
        Create ensemble with voting
        
        Args:
            model_types: List of model types
            voting: 'soft' or 'hard'
            weights: Weights for models
        
        Returns:
            Ensemble model
        """
        logger.info(f"Creating voting ensemble (voting={voting}) with models: {model_types}")
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for model_type in model_types:
            if model_type in self.base_trainer.trained_models:
                model = self.base_trainer.trained_models[model_type].model
                estimators.append((model_type, model))
                logger.debug(f"Added model {model_type} to ensemble")
        
        if not estimators:
            logger.error("No trained models found for ensemble")
            raise ValueError("No trained models found")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        logger.debug(f"VotingClassifier created with {len(estimators)} models")
        
        # Training on preprocessed data
        first_model_type = model_types[0]
        preprocessor = self.base_trainer.preprocessors[first_model_type]
        X_train_processed = self.base_trainer.training_results[first_model_type]['X_train_processed']
        
        logger.info("Starting voting ensemble training...")
        ensemble.fit(X_train_processed, self.base_trainer.y_train)
        logger.info("Voting ensemble training completed")
        
        self.ensemble_models['voting'] = {
            'model': ensemble,
            'preprocessor': preprocessor,
            'component_models': model_types
        }
        
        return ensemble
    
    def create_stacking_ensemble(self, base_models, meta_model='LogisticRegression'):
        """
        Create ensemble with stacking
        
        Args:
            base_models: List of base models
            meta_model: Type of meta-model
        
        Returns:
            Stacking ensemble
        """
        logger.info(f"Creating stacking ensemble with meta-model {meta_model} and base models: {base_models}")
        from sklearn.ensemble import StackingClassifier
        
        estimators = []
        for model_type in base_models:
            if model_type in self.base_trainer.trained_models:
                model = self.base_trainer.trained_models[model_type].model
                estimators.append((model_type, model))
                logger.debug(f"Added base model {model_type}")
        
        # Meta learner
        meta_learner = ModelFactory.create_model(meta_model).model
        logger.debug(f"Meta-learner created: {meta_model}")
        
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3
        )
        
        # Training
        first_model_type = base_models[0]
        preprocessor = self.base_trainer.preprocessors[first_model_type]
        X_train_processed = self.base_trainer.training_results[first_model_type]['X_train_processed']
        
        logger.info("Starting stacking ensemble training...")
        stacking.fit(X_train_processed, self.base_trainer.y_train)
        logger.info("Stacking ensemble training completed")
        
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
    Model saving and loading management
    """
    
    @staticmethod
    def save_model(model, model_name, save_dir=None):
        """
        Save model to disk
        
        Args:
            model: Model to save
            model_name: Model name
            save_dir: Save directory
        
        Returns:
            Path of saved file
        """
        logger.info(f"Saving model {model_name}")
        if save_dir is None:
            save_dir = MODELS_DIR
            logger.debug(f"Using default directory: {save_dir}")
        
        if not os.path.exists(save_dir):
            logger.debug(f"Creating directory: {save_dir}")
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved successfully to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise Exception(f"Error saving model: {str(e)}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load model from disk
        
        Args:
            filepath: Model file path
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Error loading model: {str(e)}")
    
    @staticmethod
    def list_saved_models(save_dir=None):
        """
        List saved models
        
        Args:
            save_dir: Directory to explore
        
        Returns:
            List of model files
        """
        logger.info("Listing saved models")
        if save_dir is None:
            save_dir = MODELS_DIR
            logger.debug(f"Using default directory: {save_dir}")
        
        if not os.path.exists(save_dir):
            logger.warning(f"Directory not found: {save_dir}")
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
                logger.debug(f"Found model: {file}")
        
        logger.info(f"Found {len(model_files)} saved models")
        return sorted(model_files, key=lambda x: x['modified'], reverse=True)

# ----------------5. Training Configuration

class TrainingConfig:
    """
    Training configurations
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
        """Return specific configuration"""
        logger.info(f"Requesting configuration: {config_name}")
        return getattr(cls, config_name.upper(), cls.QUICK_TRAINING)

# ----------------6. Training Pipeline Manager

class TrainingPipelineManager:
    """
    Complete manager for training pipeline
    """
    
    def __init__(self, config_name='QUICK_TRAINING'):
        logger.info(f"Initializing TrainingPipelineManager with config: {config_name}")
        self.config = TrainingConfig.get_config(config_name)
        self.trainer = ModelTrainer()
        self.ensemble_trainer = None
        self.results = {}
        
    def run_full_pipeline(self, X, y, progress_callback=None):
        """
        Run complete training pipeline
        
        Args:
            X: Features
            y: Target
            progress_callback: Callback for progress updates
        
        Returns:
            Complete training results
        """
        logger.info("Starting complete training pipeline")
        total_steps = len(self.config['models']) + 2  # +2 for data prep and final evaluation
        current_step = 0
        
        # Step 1: Data preparation
        if progress_callback:
            progress_callback(current_step / total_steps, "Preparing data...")
        logger.info("Step 1: Data preparation")
        
        self.trainer.prepare_data(X, y, test_size=self.config['test_size'])
        current_step += 1
        
        # Step 2: Model training
        logger.info(f"Step 2: Training {len(self.config['models'])} models")
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
                logger.debug(f"CV for model {model_type}")
                cv_results[model_type] = self.trainer.cross_validate_model(
                    model_type, cv_folds=self.config['cv_folds']
                )
        
        # Step 4: Hyperparameter tuning (optional)
        if self.config.get('use_hyperparameter_tuning', False):
            logger.info("Step 4: Hyperparameter tuning")
            tuning_results = {}
            for model_type in self.config['models']:
                if model_type in training_results:
                    try:
                        logger.debug(f"Tuning for {model_type}")
                        tuning_results[model_type] = self.trainer.hyperparameter_tuning(model_type)
                    except Exception as e:
                        logger.warning(f"Tuning error {model_type}: {str(e)}")
                        continue
        else:
            tuning_results = {}
            logger.info("Step 4: Hyperparameter tuning skipped")
        
        # Step 5: Ensemble (optional)
        if self.config.get('use_ensemble', False):
            logger.info("Step 5: Creating ensemble")
            self.ensemble_trainer = EnsembleTrainer(self.trainer)
            ensemble_results = {
                'voting': self.ensemble_trainer.create_voting_ensemble(self.config['models'])
            }
            logger.info("Ensemble created successfully")
        else:
            ensemble_results = {}
            logger.info("Step 5: Ensemble creation skipped")
        
        current_step += 1
        if progress_callback:
            progress_callback(1.0, "Training completed!")
        
        # Compile results
        self.results = {
            'training': training_results,
            'cross_validation': cv_results,
            'hyperparameter_tuning': tuning_results,
            'ensemble': ensemble_results,
            'config': self.config
        }
        
        logger.info("Training pipeline completed successfully")
        return self.results
    
    def get_best_model(self, metric='accuracy'):
        """
        Return best model based on metric
        
        Args:
            metric: Metric for comparison
        
        Returns:
            Name and score of best model
        """
        logger.info(f"Searching best model for metric: {metric}")
        if 'cross_validation' not in self.results:
            logger.warning("No CV results available")
            return None
        
        best_score = 0
        best_model = None
        
        for model_type, cv_result in self.results['cross_validation'].items():
            score = cv_result['mean']
            if score > best_score:
                best_score = score
                best_model = model_type
        
        if best_model:
            logger.info(f"Best model found: {best_model} (score={best_score})")
        else:
            logger.warning("No valid model found")
        
        return {'model': best_model, 'score': best_score}
    
    def save_pipeline_results(self, filepath):
        """Save pipeline results"""
        logger.info(f"Saving pipeline results to: {filepath}")
        import json
        
        # Prepare serializable data
        serializable_results = {
            'config': self.config,
            'cross_validation': self.results.get('cross_validation', {}),
            'best_model': self.get_best_model()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info("Results saved successfully")
            return filepath
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

logger.info(f"Loading completed {__name__}")