"""
src/components/ml_charts.py
Advanced Machine Learning Visualization Components
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
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

import warnings
warnings.filterwarnings('ignore')

from src.config import COLOR_PALETTES, CHART_CONFIG, ML_MODELS
from src.models.model_evaluator import ModelComparison, StatisticalTests

# ----------------1. Training Visualization

class TrainingVisualizer:
    """
    Visualizations for training process
    """
    
    @staticmethod
    def create_training_progress_chart(training_results, metric='accuracy'):
        """
        Display training progress for multiple models
        
        Args:
            training_results: Training results from ModelTrainer
            metric: Metric to display
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = COLOR_PALETTES.get('seaborn_palettes', ['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        for i, (model_name, results) in enumerate(training_results.items()):
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            training_time = results.get('training_time', 0)
            
            fig.add_trace(go.Bar(
                name=model_display_name,
                x=[model_display_name],
                y=[training_time],
                marker_color=colors[i % len(colors)],
                text=f"{training_time:.2f}s",
                textposition='outside'
            ))
        
        fig.update_layout(
            title="Training Times by Model",
            xaxis_title="Models",
            yaxis_title="Time (seconds)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_cross_validation_chart(cv_results):
        """
        Display cross-validation results
        
        Args:
            cv_results: CV results from ModelTrainer
        
        Returns:
            Plotly figure
        """
        cv_data = []
        
        for model_name, cv_result in cv_results.items():
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            scores = cv_result['scores']
            
            for i, score in enumerate(scores):
                cv_data.append({
                    'Model': model_display_name,
                    'Fold': f'Fold {i+1}',
                    'Score': score,
                    'Mean': cv_result['mean'],
                    'Std': cv_result['std']
                })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig = px.box(
            cv_df,
            x='Model',
            y='Score',
            points='all',
            title="Cross-Validation Scores Distribution",
            color='Model'
        )
        
        # Add mean lines
        for model_name, cv_result in cv_results.items():
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            fig.add_hline(
                y=cv_result['mean'],
                line_dash="dash",
                annotation_text=f"Mean: {cv_result['mean']:.3f}",
                annotation_position="right"
            )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_hyperparameter_optimization_chart(tuning_results):
        """
        Display hyperparameter tuning results
        
        Args:
            tuning_results: Results from hyperparameter_tuning
        
        Returns:
            Plotly figure
        """
        if not tuning_results:
            return None
        
        # Take first model as example
        model_name = list(tuning_results.keys())[0]
        results = tuning_results[model_name]
        
        cv_results = results['cv_results']
        
        # Create DataFrame from results
        df_results = pd.DataFrame(cv_results)
        
        # Plot top 10 parameter configurations
        top_indices = df_results.nlargest(10, 'mean_test_score').index
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(top_indices))),
            y=df_results.loc[top_indices, 'mean_test_score'],
            mode='markers+lines',
            name='CV Score',
            error_y=dict(
                type='data',
                array=df_results.loc[top_indices, 'std_test_score'],
                visible=True
            )
        ))
        
        fig.update_layout(
            title=f"Top 10 Hyperparameter Configurations - {ML_MODELS.get(model_name, {}).get('name', model_name)}",
            xaxis_title="Configuration",
            yaxis_title="CV Score",
            height=400
        )
        
        return fig

# ----------------2. Performance Comparison

class PerformanceVisualizer:
    """
    Visualizations for model performance comparison
    """
    
    @staticmethod
    def create_metrics_comparison_radar(evaluation_results):
        """
        Radar chart for multiple metrics comparison
        
        Args:
            evaluation_results: Results from ModelEvaluator
        
        Returns:
            Plotly figure
        """
        main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            values = [results.get(metric, 0) for metric in main_metrics]
            values.append(values[0])  # Close the radar
            
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=main_metrics + [main_metrics[0]],
                fill='toself',
                name=model_display_name,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.1f'
                )
            ),
            title="Radar Chart - Performance Metrics Comparison",
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_performance_heatmap(evaluation_results):
        """
        Model performance heatmap
        
        Args:
            evaluation_results: Evaluation results
        
        Returns:
            Plotly figure
        """
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy']
        
        # Prepare matrix
        matrix_data = []
        model_names = []
        
        for model_name, results in evaluation_results.items():
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            model_names.append(model_display_name)
            
            row = []
            for metric in metrics_to_show:
                value = results.get(metric, 0)
                row.append(value)
            matrix_data.append(row)
        
        fig = px.imshow(
            matrix_data,
            x=[m.replace('_', ' ').title() for m in metrics_to_show],
            y=model_names,
            color_continuous_scale='RdYlBu_r',
            title="Performance Heatmap - All Models",
            text_auto='.3f'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    @staticmethod
    def create_model_ranking_chart(evaluation_results, metric='f1'):
        """
        Model ranking for specific metric
        
        Args:
            evaluation_results: Evaluation results
            metric: Metric for ranking
        
        Returns:
            Plotly figure
        """
        comparison = ModelComparison(evaluation_results)
        rankings = comparison.rank_models(metric)
        
        # Prepare data
        models = [r['name'] for r in rankings]
        scores = [r['score'] for r in rankings]
        
        # Colors from best to worst
        colors = px.colors.sequential.RdYlGn_r[:len(models)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=models,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{score:.3f}" for score in scores],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"Model Ranking by {metric.replace('_', ' ').title()}",
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Models",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig

# ----------------3. ROC and Precision-Recall Curves

class CurveVisualizer:
    """
    ROC and Precision-Recall curves visualizations
    """
    
    @staticmethod
    def create_roc_curves_comparison(evaluation_results, y_test, probabilities):
        """
        ROC curves comparison
        
        Args:
            evaluation_results: Evaluation results
            y_test: True targets
            probabilities: Probability dictionary by model
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Baseline (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            if model_name in probabilities and probabilities[model_name] is not None:
                y_proba = probabilities[model_name]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = results.get('roc_auc', 0)
                
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{model_display_name} (AUC={auc_score:.3f})",
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
        
        fig.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600,
            showlegend=True,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        
        return fig
    
    @staticmethod
    def create_precision_recall_curves(evaluation_results, y_test, probabilities):
        """
        Precision-Recall curves
        
        Args:
            evaluation_results: Evaluation results
            y_test: True targets
            probabilities: Model probabilities
        
        Returns:
            Plotly figure
        """
        from sklearn.metrics import average_precision_score
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            if model_name in probabilities and probabilities[model_name] is not None:
                y_proba = probabilities[model_name]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                ap_score = average_precision_score(y_test, y_proba)
                
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f"{model_display_name} (AP={ap_score:.3f})",
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
        
        # Baseline (positive proportion)
        baseline = np.mean(y_test)
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline: {baseline:.3f}"
        )
        
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=600,
            showlegend=True
        )
        
        return fig

# ----------------4. Confusion Matrix Visualization

class ConfusionMatrixVisualizer:
    """
    Confusion matrix visualizations
    """
    
    @staticmethod
    def create_confusion_matrices_grid(evaluation_results):
        """
        Grid of confusion matrices
        
        Args:
            evaluation_results: Results with confusion matrix
        
        Returns:
            Plotly figure with subplots
        """
        n_models = len(evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[ML_MODELS.get(name, {}).get('name', name) for name in evaluation_results.keys()],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            cm = results.get('confusion_matrix')
            if cm is not None:
                # Normalize for percentages
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                heatmap = go.Heatmap(
                    z=cm_norm,
                    text=cm,  # Show absolute values
                    texttemplate='%{text}',
                    textfont={"size": 14},
                    colorscale='Blues',
                    showscale=(i == 0),  # Show scale only for the first
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.1%}<extra></extra>'
                )
                
                fig.add_trace(heatmap, row=row, col=col)
        
        # Update axes labels
        class_names = ['Did Not Survive', 'Survived']
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(
                    tickvals=[0, 1],
                    ticktext=class_names,
                    title_text="Predicted" if i == rows else "",
                    row=i, col=j
                )
                fig.update_yaxes(
                    tickvals=[0, 1],
                    ticktext=class_names,
                    title_text="Actual" if j == 1 else "",
                    row=i, col=j
                )
        
        fig.update_layout(
            title="Confusion Matrices - All Models",
            height=250 * rows + 100
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_detailed(cm, model_name, class_names=['Did Not Survive', 'Survived']):
        """
        Detailed confusion matrix for single model
        
        Args:
            cm: Confusion matrix
            model_name: Model name
            class_names: Class names
        
        Returns:
            Plotly figure
        """
        model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations with count and percentage
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})"
                )
        
        annotations_matrix = np.array(annotations).reshape(cm.shape)
        
        fig = px.imshow(
            cm_norm,
            text_auto=False,
            title=f"Detailed Confusion Matrix - {model_display_name}",
            labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Rate'},
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues'
        )
        
        # Add custom annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=annotations_matrix[i, j],
                    showarrow=False,
                    font=dict(size=16, color="white" if cm_norm[i, j] > 0.5 else "black")
                )
        
        fig.update_layout(height=500)
        
        return fig

# ----------------5. Feature Importance Visualization

class FeatureImportanceVisualizer:
    """
    Feature importance visualizations
    """
    
    @staticmethod
    def create_feature_importance_chart(feature_importance_dict, top_n=15):
        """
        Feature importance chart for single model
        
        Args:
            feature_importance_dict: Dictionary {feature: importance}
            top_n: Number of top features to show
        
        Returns:
            Plotly figure
        """
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(
            list(feature_importance_dict.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top {top_n} Feature Importance",
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_comparison(importance_data_dict):
        """
        Feature importance comparison between models
        
        Args:
            importance_data_dict: {model_name: {feature: importance}}
        
        Returns:
            Plotly figure
        """
        # Find common features
        all_features = set()
        for importances in importance_data_dict.values():
            all_features.update(importances.keys())
        
        # Select top global features
        global_importance = {}
        for feature in all_features:
            total_importance = sum(
                importances.get(feature, 0) 
                for importances in importance_data_dict.values()
            )
            global_importance[feature] = total_importance
        
        top_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        top_feature_names = [f[0] for f in top_features]
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, importances) in enumerate(importance_data_dict.items()):
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            
            values = [importances.get(feature, 0) for feature in top_feature_names]
            
            fig.add_trace(go.Bar(
                name=model_display_name,
                x=top_feature_names,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Feature Importance Comparison Between Models",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            barmode='group',
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_heatmap(importance_matrix_df):
        """
        Feature importance heatmap
        
        Args:
            importance_matrix_df: DataFrame with models as rows, features as columns
        
        Returns:
            Plotly figure
        """
        fig = px.imshow(
            importance_matrix_df.values,
            x=importance_matrix_df.columns,
            y=importance_matrix_df.index,
            title="Feature Importance Heatmap - Models vs Features",
            labels={'x': 'Features', 'y': 'Models', 'color': 'Importance'},
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig

# ----------------6. Prediction Analysis

class PredictionVisualizer:
    """
    Visualizations for prediction analysis
    """
    
    @staticmethod
    def create_prediction_confidence_chart(predictions_data):
        """
        Prediction confidence chart
        
        Args:
            predictions_data: DataFrame with columns Model, Probability, Prediction
        
        Returns:
            Plotly figure
        """
        fig = px.bar(
            predictions_data,
            x='Model',
            y='Probability',
            color='Prediction',
            title="Prediction Confidence by Model",
            color_discrete_map={
                'Survived': COLOR_PALETTES['success'], 
                'Did Not Survive': COLOR_PALETTES['danger']
            }
        )
        
        # Decision threshold
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="black",
            annotation_text="Decision Threshold (50%)"
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    @staticmethod
    def create_consensus_visualization(consensus_data):
        """
        Visualize consensus between models
        
        Args:
            consensus_data: Consensus data from ModelComparison
        
        Returns:
            Plotly figure
        """
        if consensus_data is None:
            return None
        
        fig = go.Figure()
        
        # Consensus scores distribution
        fig.add_trace(go.Histogram(
            x=consensus_data['consensus_scores'],
            nbinsx=20,
            name='Consensus Distribution',
            marker_color=COLOR_PALETTES['primary'],
            opacity=0.7
        ))
        
        # Lines for high/low consensus
        fig.add_vline(
            x=0.2,
            line_dash="dash",
            line_color="red",
            annotation_text="Low Consensus"
        )
        
        fig.add_vline(
            x=0.8,
            line_dash="dash",
            line_color="green",
            annotation_text="High Consensus"
        )
        
        fig.update_layout(
            title="Consensus Distribution Among Models",
            xaxis_title="Consensus Score (0=all agree 'No', 1=all agree 'Yes')",
            yaxis_title="Number of Predictions",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_probability_distribution_chart(probabilities_dict, bins=30):
        """
        Probability distribution for all models
        
        Args:
            probabilities_dict: {model_name: probability array}
            bins: Number of histogram bins
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, probabilities) in enumerate(probabilities_dict.items()):
            if probabilities is not None:
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                
                fig.add_trace(go.Histogram(
                    x=probabilities,
                    nbinsx=bins,
                    name=model_display_name,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7
                ))
        
        # Decision threshold
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Decision Threshold"
        )
        
        fig.update_layout(
            title="Prediction Probability Distribution",
            xaxis_title="Survival Probability",
            yaxis_title="Frequency",
            barmode='overlay',
            height=500
        )
        
        return fig

# ----------------7. Error Analysis Visualization

class ErrorAnalysisVisualizer:
    """
    Visualizations for error analysis
    """
    
    @staticmethod
    def create_error_distribution_chart(error_data, feature_name):
        """
        Error distribution by feature
        
        Args:
            error_data: DataFrame with columns Feature_Value, Error, Actual, Predicted
            feature_name: Feature name
        
        Returns:
            Plotly figure
        """
        fig = px.histogram(
            error_data,
            x='Feature_Value',
            color='Error',
            title=f"Error Distribution by {feature_name}",
            color_discrete_map={
                True: COLOR_PALETTES['danger'], 
                False: COLOR_PALETTES['success']
            },
            barmode='group',
            marginal="rug"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    @staticmethod
    def create_error_types_chart(error_summary):
        """
        Error types chart (TP, TN, FP, FN)
        
        Args:
            error_summary: Dictionary with error counts
        
        Returns:
            Plotly figure
        """
        categories = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        values = [
            error_summary.get('true_negatives', 0),
            error_summary.get('false_positives', 0),
            error_summary.get('false_negatives', 0),
            error_summary.get('true_positives', 0)
        ]
        
        colors = [COLOR_PALETTES['success'], COLOR_PALETTES['warning'], 
                 COLOR_PALETTES['danger'], COLOR_PALETTES['success']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Prediction Types Distribution",
            yaxis_title="Number of Predictions",
            height=400,
            showlegend=False
        )
        
        return fig

# ----------------8. Calibration Visualization

class CalibrationVisualizer:
    """
    Model calibration visualizations
    """
    
    @staticmethod
    def create_calibration_plot(y_test, probabilities_dict, n_bins=10):
        """
        Calibration plot for multiple models
        
        Args:
            y_test: True targets
            probabilities_dict: Probabilities by model
            n_bins: Number of bins for calibration
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
        
        for i, (model_name, probabilities) in enumerate(probabilities_dict.items()):
            if probabilities is not None:
                fraction_positives, mean_predicted_value = calibration_curve(
                    y_test, probabilities, n_bins=n_bins
                )
                
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                
                fig.add_trace(go.Scatter(
                    x=mean_predicted_value,
                    y=fraction_positives,
                    mode='lines+markers',
                    name=f'{model_display_name}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='Calibration Plot - Probability Reliability',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_reliability_diagram(y_test, probabilities, model_name, n_bins=10):
        """
        Reliability diagram for single model
        
        Args:
            y_test: True targets
            probabilities: Model probabilities
            model_name: Model name
            n_bins: Number of bins
        
        Returns:
            Plotly figure
        """
        from sklearn.calibration import brier_score_loss
        
        model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
        
        # Calibration curve
        fraction_positives, mean_predicted_value = calibration_curve(
            y_test, probabilities, n_bins=n_bins
        )
        
        # Brier score
        brier_score = brier_score_loss(y_test, probabilities)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'Reliability Diagram - {model_display_name}',
                'Probability Histogram'
            ],
            column_widths=[0.7, 0.3]
        )
        
        # Reliability diagram
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_positives,
                mode='lines+markers',
                name=f'{model_display_name}<br>Brier Score: {brier_score:.3f}',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=probabilities,
                nbinsx=n_bins,
                name='Probability Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        return fig

# ----------------9. Learning Curves

class LearningCurveVisualizer:
    """
    Learning curve visualizations
    """
    
    @staticmethod
    def create_learning_curve_chart(learning_curve_data, model_name):
        """
        Learning curve for overfitting/underfitting analysis
        
        Args:
            learning_curve_data: Data from create_learning_curves
            model_name: Model name
        
        Returns:
            Plotly figure
        """
        model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_mean']
        train_std = learning_curve_data['train_std']
        val_mean = learning_curve_data['val_mean']
        val_std = learning_curve_data['val_std']
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=COLOR_PALETTES['primary'], width=2),
            error_y=dict(
                type='data',
                array=train_std,
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=COLOR_PALETTES['secondary'], width=2),
            error_y=dict(
                type='data',
                array=val_std,
                visible=True
            )
        ))
        
        fig.update_layout(
            title=f'Learning Curve - {model_display_name}',
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_validation_curve_chart(param_range, train_scores, val_scores, param_name, model_name):
        """
        Validation curve for hyperparameter analysis
        
        Args:
            param_range: Parameter value range
            train_scores: Training scores
            val_scores: Validation scores
            param_name: Parameter name
            model_name: Model name
        
        Returns:
            Plotly figure
        """
        model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=param_range,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=COLOR_PALETTES['primary']),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=param_range,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=COLOR_PALETTES['secondary']),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title=f'Validation Curve - {model_display_name} ({param_name})',
            xaxis_title=param_name,
            yaxis_title='Score',
            height=500
        )
        
        return fig

# ----------------10. Advanced Analysis Visualization

class AdvancedAnalysisVisualizer:
    """
    Visualizations for advanced analysis
    """
    
    @staticmethod
    def create_bias_variance_analysis(models_performance):
        """
        Bias-variance trade-off analysis
        
        Args:
            models_performance: Model performance with variance
        
        Returns:
            Plotly figure
        """
        model_names = []
        bias_scores = []
        variance_scores = []
        
        for model_name, performance in models_performance.items():
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            model_names.append(model_display_name)
            
            # Approximation: bias = 1 - accuracy, variance = std CV scores
            bias_scores.append(1 - performance.get('accuracy', 0))
            variance_scores.append(performance.get('cv_std', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=bias_scores,
            y=variance_scores,
            mode='markers+text',
            text=model_names,
            textposition='top center',
            marker=dict(
                size=15,
                color=list(range(len(model_names))),
                colorscale='viridis',
                showscale=True
            ),
            name='Models'
        ))
        
        fig.update_layout(
            title='Bias-Variance Trade-off',
            xaxis_title='Bias (1 - Accuracy)',
            yaxis_title='Variance (CV Std Dev)',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_model_complexity_chart(models_info):
        """
        Model complexity vs performance comparison
        
        Args:
            models_info: Model info with complexity and performance
        
        Returns:
            Plotly figure
        """
        complexity_scores = []
        performance_scores = []
        model_names = []
        
        for model_name, info in models_info.items():
            model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
            model_names.append(model_display_name)
            
            # Complexity score based on model type
            complexity_map = {
                'LogisticRegression': 1,
                'DecisionTreeClassifier': 2,
                'RandomForestClassifier': 3,
                'GradientBoostingClassifier': 4,
                'SVC': 3,
                'MLPClassifier': 5
            }
            
            complexity_scores.append(complexity_map.get(model_name, 2))
            performance_scores.append(info.get('f1', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=complexity_scores,
            y=performance_scores,
            mode='markers+text',
            text=model_names,
            textposition='top center',
            marker=dict(
                size=15,
                color=performance_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="F1 Score")
            )
        ))
        
        fig.update_layout(
            title='Model Complexity vs Performance',
            xaxis_title='Model Complexity',
            yaxis_title='F1 Score',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_feature_selection_impact_chart(feature_selection_results):
        """
        Feature selection impact on performance
        
        Args:
            feature_selection_results: Results with different numbers of features
        
        Returns:
            Plotly figure
        """
        n_features = feature_selection_results['n_features']
        scores = feature_selection_results['scores']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=n_features,
            y=scores,
            mode='lines+markers',
            name='Performance vs N Features',
            line=dict(color=COLOR_PALETTES['primary'], width=3),
            marker=dict(size=8)
        ))
        
        # Find optimal point
        best_idx = np.argmax(scores)
        fig.add_trace(go.Scatter(
            x=[n_features[best_idx]],
            y=[scores[best_idx]],
            mode='markers',
            name=f'Optimum: {n_features[best_idx]} features',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title='Feature Selection Impact on Performance',
            xaxis_title='Number of Features',
            yaxis_title='Cross-Validation Score',
            height=500
        )
        
        return fig

# ----------------11. Ensemble Visualization

class EnsembleVisualizer:
    """
    Visualizations for ensemble methods
    """
    
    @staticmethod
    def create_ensemble_weight_chart(ensemble_weights, model_names):
        """
        Display ensemble weights
        
        Args:
            ensemble_weights: Array of model weights
            model_names: Model names
        
        Returns:
            Plotly figure
        """
        display_names = [ML_MODELS.get(name, {}).get('name', name) for name in model_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=display_names,
                y=ensemble_weights,
                marker_color=COLOR_PALETTES['seaborn_palettes'][:len(model_names)],
                text=[f"{w:.3f}" for w in ensemble_weights],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Ensemble Model Weights',
            xaxis_title='Models',
            yaxis_title='Weight',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_ensemble_diversity_chart(predictions_matrix):
        """
        Ensemble prediction diversity analysis
        
        Args:
            predictions_matrix: Prediction matrix (samples x models)
        
        Returns:
            Plotly figure
        """
        # Calculate agreement between models
        n_models = predictions_matrix.shape[1]
        agreement_scores = []
        
        for i in range(predictions_matrix.shape[0]):
            predictions = predictions_matrix[i, :]
            agreement = np.mean(predictions) if np.var(predictions) == 0 else 1 - np.var(predictions)
            agreement_scores.append(agreement)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=agreement_scores,
            nbinsx=20,
            name='Agreement Distribution',
            marker_color=COLOR_PALETTES['primary'],
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Ensemble Prediction Diversity',
            xaxis_title='Agreement Score (0=max diversity, 1=complete agreement)',
            yaxis_title='Number of Samples',
            height=400
        )
        
        return fig

# ----------------12. Utility Functions

def create_comprehensive_model_report_visualization(evaluation_results, training_results=None):
    """
    Create comprehensive model report visualization
    
    Args:
        evaluation_results: Evaluation results
        training_results: Training results (optional)
    
    Returns:
        List of figures for complete report
    """
    visualizations = []
    
    # Performance comparison
    perf_viz = PerformanceVisualizer()
    visualizations.append(perf_viz.create_metrics_comparison_radar(evaluation_results))
    visualizations.append(perf_viz.create_performance_heatmap(evaluation_results))
    
    # Confusion matrices
    cm_viz = ConfusionMatrixVisualizer()
    visualizations.append(cm_viz.create_confusion_matrices_grid(evaluation_results))
    
    # Feature importance (if available)
    models_with_importance = ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier']
    importance_data = {}
    
    for model_name in evaluation_results.keys():
        if model_name in models_with_importance:
            # Placeholder - in practice would come from trained model
            importance_data[model_name] = {f'feature_{i}': np.random.random() for i in range(10)}
    
    if importance_data:
        fi_viz = FeatureImportanceVisualizer()
        visualizations.append(fi_viz.create_feature_importance_comparison(importance_data))
    
    # Training times (if available)
    if training_results:
        train_viz = TrainingVisualizer()
        visualizations.append(train_viz.create_training_progress_chart(training_results))
    
    return visualizations

def save_visualization(fig, filepath, format='html'):
    """
    Save visualization
    
    Args:
        fig: Plotly figure
        filepath: Save path
        format: Format ('html', 'png', 'pdf')
    """
    if format == 'html':
        fig.write_html(filepath)
    elif format == 'png':
        fig.write_image(filepath)
    elif format == 'pdf':
        fig.write_image(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

def customize_chart_theme(fig, theme='default'):
    """
    Apply custom theme
    
    Args:
        fig: Plotly figure
        theme: Theme name
    
    Returns:
        Figure with applied theme
    """
    if theme == 'dark':
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    elif theme == 'minimal':
        fig.update_layout(
            template='simple_white',
            showlegend=False
        )
    elif theme == 'presentation':
        fig.update_layout(
            font=dict(size=16),
            title_font_size=20,
            height=600
        )
    
    return fig

def create_interactive_dashboard_components(evaluation_results):
    """
    Create components for interactive dashboard
    
    Args:
        evaluation_results: Evaluation results
    
    Returns:
        Dictionary with dashboard components
    """
    components = {}
    
    # Summary metrics
    comparison = ModelComparison(evaluation_results)
    components['summary_table'] = comparison.create_comparison_table()
    components['best_models'] = {
        'accuracy': comparison.find_best_model('accuracy'),
        'precision': comparison.find_best_model('precision'),
        'recall': comparison.find_best_model('recall'),
        'f1': comparison.find_best_model('f1')
    }
    
    # Interactive visualizations
    perf_viz = PerformanceVisualizer()
    components['radar_chart'] = perf_viz.create_metrics_comparison_radar(evaluation_results)
    components['heatmap'] = perf_viz.create_performance_heatmap(evaluation_results)
    
    return components