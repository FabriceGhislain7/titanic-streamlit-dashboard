"""
src/models/model_evaluator.py
Machine Learning Models Performance Evaluation and Analysis
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss,
    average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
# Safe import for brier_score_loss
try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    try:
        from sklearn.calibration import brier_score_loss
    except ImportError:
        # Fallback: custom implementation if not available
        def brier_score_loss(y_true, y_prob):
            return np.mean((y_prob - y_true) ** 2)

from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, permutation_test_score
import scipy.stats as stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import EVALUATION_METRICS, ML_MODELS

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Core Evaluation Class

class ModelEvaluator:
    """
    Main class for model evaluation
    """
    
    def __init__(self, models_dict, X_test, y_test):
        """
        Args:
            models_dict: Dictionary {model_name: trained_model}
            X_test: Test set features
            y_test: Test set target
        """
        logger.info("Initializing ModelEvaluator")
        logger.debug(f"Received models: {list(models_dict.keys())}")
        logger.debug(f"X_test dimensions: {X_test.shape}, y_test: {y_test.shape}")
        
        self.models = models_dict
        self.X_test = X_test
        self.y_test = y_test
        self.evaluation_results = {}
        self.predictions = {}
        self.probabilities = {}
        
    def evaluate_all_models(self, detailed=True):
        """
        Evaluate all available models
        
        Args:
            detailed: Whether to include detailed metrics
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {len(self.models)} models (detailed={detailed})")
        
        for model_name, model_data in self.models.items():
            logger.debug(f"Evaluating model: {model_name}")
            self.evaluation_results[model_name] = self.evaluate_single_model(
                model_name, model_data, detailed=detailed
            )
        
        logger.info("Evaluation completed for all models")
        return self.evaluation_results
    
    def evaluate_single_model(self, model_name, model_data, detailed=True):
        """
        Evaluate single model
        
        Args:
            model_name: Model name
            model_data: Model data (model + preprocessor)
            detailed: Whether to include detailed analysis
        
        Returns:
            Model evaluation results
        """
        logger.info(f"Evaluating model {model_name}")
        
        model = model_data['model']
        preprocessor = model_data.get('preprocessor')
        
        # Preprocessing if necessary
        if preprocessor:
            logger.debug("Applying preprocessor")
            X_test_processed = preprocessor.transform(self.X_test)
        else:
            X_test_processed = self.X_test
        
        # Predictions
        logger.debug("Generating predictions")
        y_pred = model.predict(X_test_processed)
        self.predictions[model_name] = y_pred
        
        # Probabilities (if available)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                logger.debug("Calculating prediction probabilities")
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                self.probabilities[model_name] = y_pred_proba
            except Exception as e:
                logger.warning(f"Error calculating probabilities for {model_name}: {str(e)}")
        
        # Basic metrics
        logger.debug("Calculating basic metrics")
        results = self._calculate_basic_metrics(y_pred, y_pred_proba)
        
        # Detailed metrics
        if detailed:
            logger.debug("Calculating detailed metrics")
            results.update(self._calculate_detailed_metrics(y_pred, y_pred_proba))
            results.update(self._calculate_confusion_matrix_metrics(y_pred))
            
            if y_pred_proba is not None:
                results.update(self._calculate_probability_metrics(y_pred_proba))
        
        # Model information
        results['model_info'] = self._get_model_info(model_name, model)
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
    
    def _calculate_basic_metrics(self, y_pred, y_pred_proba):
        """Calculate basic metrics"""
        logger.debug("Calculating basic metrics")
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(self.y_test, y_pred_proba)
        
        return metrics
    
    def _calculate_detailed_metrics(self, y_pred, y_pred_proba):
        """Calculate detailed metrics"""
        logger.debug("Calculating detailed metrics")
        metrics = {
            'matthews_corrcoef': matthews_corrcoef(self.y_test, y_pred),
            'cohen_kappa': cohen_kappa_score(self.y_test, y_pred),
            'specificity': self._calculate_specificity(y_pred),
            'sensitivity': recall_score(self.y_test, y_pred, zero_division=0),  # Alias for recall
            'npv': self._calculate_npv(y_pred),  # Negative Predictive Value
            'fallout': self._calculate_fallout(y_pred)  # False Positive Rate
        }
        
        if y_pred_proba is not None:
            metrics['log_loss'] = log_loss(self.y_test, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(self.y_test, y_pred_proba)
        
        return metrics
    
    def _calculate_confusion_matrix_metrics(self, y_pred):
        """Calculate confusion matrix metrics"""
        logger.debug("Calculating confusion matrix metrics")
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_errors': int(fp + fn),
            'error_rate': (fp + fn) / len(self.y_test)
        }
    
    def _calculate_probability_metrics(self, y_pred_proba):
        """Calculate probability-based metrics"""
        logger.debug("Calculating probability metrics")
        # Calibration
        fraction_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10
        )
        
        # ECE (Expected Calibration Error)
        ece = np.mean(np.abs(fraction_positives - mean_predicted_value))
        
        return {
            'calibration_curve': {
                'fraction_positives': fraction_positives,
                'mean_predicted_value': mean_predicted_value
            },
            'expected_calibration_error': ece
        }
    
    def _calculate_specificity(self, y_pred):
        """Calculate specificity (True Negative Rate)"""
        logger.debug("Calculating specificity")
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_npv(self, y_pred):
        """Calculate Negative Predictive Value"""
        logger.debug("Calculating NPV")
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0
    
    def _calculate_fallout(self, y_pred):
        """Calculate False Positive Rate"""
        logger.debug("Calculating fallout")
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def _get_model_info(self, model_name, model):
        """Get model information"""
        logger.debug(f"Retrieving info for model {model_name}")
        return {
            'name': ML_MODELS.get(model_name, {}).get('name', model_name),
            'type': model_name,
            'n_features': getattr(model, 'n_features_in_', 'Unknown'),
            'classes': getattr(model, 'classes_', ['Unknown'])
        }

# ----------------2. Model Comparison

class ModelComparison:
    """
    Class for model comparison
    """
    
    def __init__(self, evaluation_results):
        logger.info("Initializing ModelComparison")
        self.results = evaluation_results
        
    def create_comparison_table(self, metrics=None):
        """
        Create comparison table
        
        Args:
            metrics: List of metrics to include
        
        Returns:
            DataFrame with comparison
        """
        logger.info("Creating comparison table")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            logger.debug(f"Using default metrics: {metrics}")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': ML_MODELS.get(model_name, {}).get('name', model_name)}
            
            for metric in metrics:
                value = results.get(metric, np.nan)
                if not np.isnan(value):
                    row[metric.title()] = round(value, 4)
                else:
                    row[metric.title()] = 'N/A'
            
            comparison_data.append(row)
        
        logger.debug(f"Comparison table generated with {len(comparison_data)} models")
        return pd.DataFrame(comparison_data)
    
    def rank_models(self, metric='f1', ascending=False):
        """
        Rank models by metric
        
        Args:
            metric: Metric for ranking
            ascending: Ascending order
        
        Returns:
            Ordered list of models
        """
        logger.info(f"Ranking models by metric: {metric}")
        
        rankings = []
        
        for model_name, results in self.results.items():
            score = results.get(metric, 0)
            rankings.append({
                'model': model_name,
                'name': ML_MODELS.get(model_name, {}).get('name', model_name),
                'score': score
            })
        
        rankings = sorted(rankings, key=lambda x: x['score'], reverse=not ascending)
        logger.debug(f"Top 3 models: {rankings[:3]}")
        return rankings
    
    def find_best_model(self, metric='f1'):
        """
        Find best model for metric
        
        Args:
            metric: Reference metric
        
        Returns:
            Dictionary with best model info
        """
        logger.info(f"Searching best model for metric: {metric}")
        rankings = self.rank_models(metric)
        if rankings:
            best = rankings[0]
            logger.info(f"Best model found: {best['name']} (score={best['score']})")
            return {
                'model_type': best['model'],
                'model_name': best['name'],
                'score': best['score'],
                'metric': metric
            }
        logger.warning("No models available for search")
        return None
    
    def calculate_model_consensus(self):
        """
        Calculate consensus among models for each prediction
        
        Returns:
            Consensus analysis
        """
        logger.info("Calculating model consensus")
        if not hasattr(self, 'evaluator'):
            logger.warning("Evaluator not available for consensus")
            return None
        
        # Collect all predictions
        all_predictions = []
        model_names = []
        
        for model_name in self.results.keys():
            if model_name in self.evaluator.predictions:
                all_predictions.append(self.evaluator.predictions[model_name])
                model_names.append(model_name)
        
        if not all_predictions:
            logger.warning("No predictions available for consensus")
            return None
        
        predictions_matrix = np.array(all_predictions).T
        
        # Calculate consensus for each sample
        consensus_scores = []
        for i in range(len(predictions_matrix)):
            votes = predictions_matrix[i]
            consensus = np.mean(votes)
            consensus_scores.append(consensus)
        
        logger.debug(f"Consensus calculated for {len(consensus_scores)} samples")
        return {
            'consensus_scores': np.array(consensus_scores),
            'high_consensus': np.sum((np.array(consensus_scores) >= 0.8) | (np.array(consensus_scores) <= 0.2)),
            'low_consensus': np.sum((np.array(consensus_scores) > 0.2) & (np.array(consensus_scores) < 0.8)),
            'model_names': model_names
        }

# ----------------3. Statistical Tests

class StatisticalTests:
    """
    Statistical tests for model comparison
    """
    
    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2):
        """
        McNemar test for comparing two models
        
        Args:
            y_true: True values
            y_pred1: Model 1 predictions
            y_pred2: Model 2 predictions
        
        Returns:
            McNemar test results
        """
        logger.info("Executing McNemar test")
        from statsmodels.stats.contingency_tables import mcnemar
        
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # Count cases
        both_correct = np.sum(correct1 & correct2)
        model1_correct = np.sum(correct1 & ~correct2)
        model2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        logger.debug(f"Contingency table - Both correct: {both_correct}, Model1 correct: {model1_correct}, Model2 correct: {model2_correct}, Both wrong: {both_wrong}")
        
        # Contingency matrix
        contingency_table = np.array([[both_correct, model1_correct],
                                     [model2_correct, both_wrong]])
        
        # McNemar test
        try:
            result = mcnemar(contingency_table, exact=True)
            logger.info(f"McNemar test completed - p-value: {result.pvalue}")
            return {
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'contingency_table': contingency_table,
                'significant': result.pvalue < 0.05
            }
        except Exception as e:
            logger.error(f"Error in McNemar test: {str(e)}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'contingency_table': contingency_table,
                'significant': False
            }
    
    @staticmethod
    def paired_t_test(scores1, scores2):
        """
        Paired t-test for comparing cross-validation scores
        
        Args:
            scores1: Model 1 scores
            scores2: Model 2 scores
        
        Returns:
            T-test results
        """
        logger.info("Executing paired t-test")
        logger.debug(f"Model 1 scores: mean={np.mean(scores1)}, std={np.std(scores1)}")
        logger.debug(f"Model 2 scores: mean={np.mean(scores2)}, std={np.std(scores2)}")
        
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        logger.info(f"T-test completed - p-value: {p_value}, significant: {p_value < 0.05}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'mean_diff': np.mean(scores1 - scores2),
            'significant': p_value < 0.05,
            'better_model': 1 if np.mean(scores1) > np.mean(scores2) else 2
        }
    
    @staticmethod
    def wilcoxon_test(scores1, scores2):
        """
        Wilcoxon test for non-parametric comparison
        
        Args:
            scores1: Model 1 scores
            scores2: Model 2 scores
        
        Returns:
            Wilcoxon test results
        """
        logger.info("Executing Wilcoxon test")
        logger.debug(f"Model 1 scores: median={np.median(scores1)}")
        logger.debug(f"Model 2 scores: median={np.median(scores2)}")
        
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        
        logger.info(f"Wilcoxon test completed - p-value: {p_value}, significant: {p_value < 0.05}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_model': 1 if np.median(scores1) > np.median(scores2) else 2
        }

# ----------------4. Error Analysis

class ErrorAnalysis:
    """
    Detailed error analysis
    """
    
    def __init__(self, X_test, y_test, predictions, feature_names=None):
        logger.info("Initializing ErrorAnalysis")
        logger.debug(f"Number of models: {len(predictions)}, Number of features: {len(feature_names) if feature_names else X_test.shape[1]}")
        
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions
        self.feature_names = feature_names or X_test.columns.tolist()
        
    def analyze_prediction_errors(self, model_name):
        """
        Analyze prediction errors for single model
        
        Args:
            model_name: Model name
        
        Returns:
            Error analysis
        """
        logger.info(f"Analyzing errors for model {model_name}")
        
        if model_name not in self.predictions:
            logger.warning(f"Model {model_name} not found in predictions")
            return None
        
        y_pred = self.predictions[model_name]
        
        # Identify errors
        errors = y_pred != self.y_test
        false_positives = (y_pred == 1) & (self.y_test == 0)
        false_negatives = (y_pred == 0) & (self.y_test == 1)
        
        logger.debug(f"Total errors: {np.sum(errors)}, FP: {np.sum(false_positives)}, FN: {np.sum(false_negatives)}")
        
        # Feature analysis
        error_analysis = {}
        
        for feature in self.feature_names:
            if feature in self.X_test.columns:
                feature_values = self.X_test[feature]
                
                error_analysis[feature] = {
                    'total_errors': np.sum(errors),
                    'fp_mean': np.mean(feature_values[false_positives]) if np.sum(false_positives) > 0 else np.nan,
                    'fn_mean': np.mean(feature_values[false_negatives]) if np.sum(false_negatives) > 0 else np.nan,
                    'correct_mean': np.mean(feature_values[~errors]) if np.sum(~errors) > 0 else np.nan,
                    'error_correlation': np.corrcoef(feature_values, errors.astype(int))[0,1] if len(np.unique(feature_values)) > 1 else 0
                }
        
        logger.info(f"Error analysis completed for {model_name}")
        return {
            'total_errors': np.sum(errors),
            'false_positives': np.sum(false_positives),
            'false_negatives': np.sum(false_negatives),
            'error_rate': np.mean(errors),
            'feature_analysis': error_analysis,
            'error_indices': np.where(errors)[0].tolist()
        }
    
    def find_difficult_samples(self, threshold=0.5):
        """
        Find difficult samples to classify
        
        Args:
            threshold: Threshold for defining difficult samples
        
        Returns:
            Difficult samples analysis
        """
        logger.info(f"Searching difficult samples with threshold={threshold}")
        
        # Count errors per sample
        error_counts = np.zeros(len(self.y_test))
        
        for model_name, y_pred in self.predictions.items():
            errors = (y_pred != self.y_test).astype(int)
            error_counts += errors
        
        # Difficult samples
        n_models = len(self.predictions)
        difficult_mask = error_counts >= (n_models * threshold)
        
        difficult_indices = np.where(difficult_mask)[0]
        
        if len(difficult_indices) == 0:
            logger.info("No difficult samples found")
            return {'difficult_samples': 0, 'indices': []}
        
        logger.info(f"Found {len(difficult_indices)} difficult samples")
        
        # Difficult samples analysis
        difficult_samples = self.X_test.iloc[difficult_indices]
        difficult_targets = self.y_test.iloc[difficult_indices]
        
        return {
            'difficult_samples': len(difficult_indices),
            'indices': difficult_indices.tolist(),
            'percentage': len(difficult_indices) / len(self.y_test) * 100,
            'sample_data': difficult_samples,
            'target_distribution': difficult_targets.value_counts().to_dict(),
            'error_counts': error_counts[difficult_indices]
        }
    
    def analyze_misclassification_patterns(self):
        """
        Analyze misclassification patterns
        
        Returns:
            Common error patterns
        """
        logger.info("Analyzing misclassification patterns")
        patterns = {}
        
        for model_name, y_pred in self.predictions.items():
            logger.debug(f"Analyzing patterns for model {model_name}")
            
            # Patterns for this model
            fp_indices = np.where((y_pred == 1) & (self.y_test == 0))[0]
            fn_indices = np.where((y_pred == 0) & (self.y_test == 1))[0]
            
            patterns[model_name] = {
                'false_positive_patterns': self._analyze_feature_patterns(fp_indices),
                'false_negative_patterns': self._analyze_feature_patterns(fn_indices)
            }
        
        logger.info("Pattern analysis completed")
        return patterns
    
    def _analyze_feature_patterns(self, error_indices):
        """Analyze feature patterns for specific errors"""
        if len(error_indices) == 0:
            return {}
        
        error_samples = self.X_test.iloc[error_indices]
        patterns = {}
        
        for feature in self.feature_names:
            if feature in error_samples.columns:
                values = error_samples[feature]
                
                patterns[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'most_common': values.mode().iloc[0] if not values.mode().empty else np.nan
                }
        
        return patterns

# ----------------5. Performance Monitoring

class PerformanceMonitor:
    """
    Performance monitoring over time
    """
    
    def __init__(self):
        logger.info("Initializing PerformanceMonitor")
        self.performance_history = []
        
    def log_performance(self, model_name, metrics, timestamp=None):
        """
        Log performance for monitoring
        
        Args:
            model_name: Model name
            metrics: Metrics dictionary
            timestamp: Timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics.copy()
        }
        
        self.performance_history.append(entry)
        logger.debug(f"Performance logged for {model_name} at {timestamp}")
    
    def get_performance_trend(self, model_name, metric='accuracy', periods=10):
        """
        Get performance trend
        
        Args:
            model_name: Model name
            metric: Metric to analyze
            periods: Number of periods
        
        Returns:
            Trend analysis
        """
        logger.info(f"Analyzing trend for {model_name} on metric {metric}")
        
        model_history = [
            entry for entry in self.performance_history[-periods:]
            if entry['model_name'] == model_name
        ]
        
        if len(model_history) < 2:
            logger.warning(f"Insufficient data for trend analysis ({len(model_history)} points)")
            return None
        
        timestamps = [entry['timestamp'] for entry in model_history]
        values = [entry['metrics'].get(metric, np.nan) for entry in model_history]
        
        # Calculate trend
        if len(values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(values)), values
            )
        else:
            slope = 0
        
        logger.debug(f"Trend slope: {slope}, Latest value: {values[-1]}")
        
        return {
            'timestamps': timestamps,
            'values': values,
            'trend_slope': slope,
            'is_improving': slope > 0,
            'latest_value': values[-1] if values else np.nan,
            'change_from_first': values[-1] - values[0] if len(values) > 1 else 0
        }
    
    def detect_performance_degradation(self, model_name, metric='accuracy', threshold=0.05):
        """
        Detect performance degradation
        
        Args:
            model_name: Model name
            metric: Metric to monitor
            threshold: Degradation threshold
        
        Returns:
            Alert if degradation detected
        """
        logger.info(f"Checking degradation for {model_name} on {metric}")
        
        trend = self.get_performance_trend(model_name, metric)
        
        if trend is None:
            return None
        
        degradation_detected = (
            trend['trend_slope'] < -threshold or
            trend['change_from_first'] < -threshold
        )
        
        if degradation_detected:
            logger.warning(f"Degradation detected for {model_name} - slope: {trend['trend_slope']}, change: {trend['change_from_first']}")
        else:
            logger.info(f"No degradation detected for {model_name}")
        
        return {
            'degradation_detected': degradation_detected,
            'trend_slope': trend['trend_slope'],
            'change_from_baseline': trend['change_from_first'],
            'latest_value': trend['latest_value'],
            'recommendation': 'Retraining suggested' if degradation_detected else 'Performance stable'
        }

# ----------------6. Model Interpretability Metrics

class InterpretabilityMetrics:
    """
    Metrics for model interpretability
    """
    
    @staticmethod
    def calculate_feature_stability(feature_importances_history):
        """
        Calculate feature importance stability over time
        
        Args:
            feature_importances_history: List of feature importance dictionaries
        
        Returns:
            Stability score for features
        """
        logger.info("Calculating feature importance stability")
        
        if len(feature_importances_history) < 2:
            logger.warning("Insufficient data for stability calculation")
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_importances_history)
        logger.debug(f"Analyzing stability for {len(df.columns)} features over {len(df)} periods")
        
        stability_scores = {}
        for feature in df.columns:
            values = df[feature].values
            # Calculate coefficient of variation
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            stability_scores[feature] = 1 / (1 + cv)  # Score between 0 and 1
        
        logger.debug(f"Average stability: {np.mean(list(stability_scores.values()))}")
        return stability_scores
    
    @staticmethod
    def calculate_prediction_confidence_distribution(probabilities):
        """
        Analyze prediction confidence distribution
        
        Args:
            probabilities: Array of probabilities
        
        Returns:
            Confidence distribution analysis
        """
        logger.info("Analyzing prediction confidence distribution")
        
        # Convert probabilities to confidence (distance from 0.5)
        confidence = np.abs(probabilities - 0.5) * 2
        
        logger.debug(f"Average confidence: {np.mean(confidence)}, Low confidence ratio: {np.sum(confidence < 0.6) / len(confidence)}")
        
        return {
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'low_confidence_ratio': np.sum(confidence < 0.6) / len(confidence),
            'high_confidence_ratio': np.sum(confidence > 0.8) / len(confidence),
            'confidence_percentiles': {
                '25th': np.percentile(confidence, 25),
                '50th': np.percentile(confidence, 50),
                '75th': np.percentile(confidence, 75),
                '90th': np.percentile(confidence, 90)
            }
        }

# ----------------7. Evaluation Report Generator

class EvaluationReportGenerator:
    """
    Evaluation report generator
    """
    
    def __init__(self, evaluation_results):
        logger.info("Initializing EvaluationReportGenerator")
        self.results = evaluation_results
        
    def generate_summary_report(self):
        """
        Generate summary report
        
        Returns:
            Structured report
        """
        logger.info("Generating summary report")
        comparison = ModelComparison(self.results)
        best_model = comparison.find_best_model('f1')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(self.results),
            'best_model': best_model,
            'model_rankings': {
                'accuracy': comparison.rank_models('accuracy')[:3],
                'precision': comparison.rank_models('precision')[:3],
                'recall': comparison.rank_models('recall')[:3],
                'f1': comparison.rank_models('f1')[:3]
            },
            'performance_summary': self._create_performance_summary()
        }
        
        logger.info("Report generated successfully")
        return report
    
    def _create_performance_summary(self):
        """Create performance summary"""
        logger.debug("Creating performance summary")
        summary = {}
        
        for model_name, results in self.results.items():
            summary[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'f1': results.get('f1', 0),
                'roc_auc': results.get('roc_auc', 0),
                'key_strengths': self._identify_model_strengths(results),
                'key_weaknesses': self._identify_model_weaknesses(results)
            }
        
        return summary
    
    def _identify_model_strengths(self, results):
        """Identify model strengths"""
        strengths = []
        
        if results.get('precision', 0) > 0.8:
            strengths.append('High Precision')
        if results.get('recall', 0) > 0.8:
            strengths.append('High Recall')
        if results.get('roc_auc', 0) > 0.9:
            strengths.append('Excellent ROC-AUC')
        if results.get('balanced_accuracy', 0) > 0.8:
            strengths.append('Balanced Performance')
        
        return strengths
    
    def _identify_model_weaknesses(self, results):
        """Identify model weaknesses"""
        weaknesses = []
        
        if results.get('precision', 1) < 0.7:
            weaknesses.append('Low Precision')
        if results.get('recall', 1) < 0.7:
            weaknesses.append('Low Recall')
        if results.get('roc_auc', 1) < 0.8:
            weaknesses.append('Poor ROC-AUC')
        if results.get('false_positives', 0) > results.get('true_positives', 1):
            weaknesses.append('High False Positive Rate')
        
        return weaknesses

logger.info(f"Loading completed {__name__}")