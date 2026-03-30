"""
pages/5_ML_Predictions.py
Machine learning models for Titanic survival prediction
Modular version with complete architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import ML modules
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic
from src.utils.ml_preprocessing import (
    create_titanic_preprocessing_pipeline, 
    DataQualityChecker,
    get_preprocessing_recommendataons,
    validate_preprocessing_pipeline
)
from src.models.ml_models import ModelFactory, ModelConfigurations, HyperparameterGrids
from src.models.model_trainer import ModelTrainer, TrainingPipelineManager, ModelPersistence
from src.models.model_evaluator import ModelEvaluator, ModelComparison, StatisticalTests, ErrorAnalysis
from src.components.ml_charts import (
    TrainingVisualizer, PerformanceVisualizer, CurveVisualizer,
    ConfusionMatrixVisualizer, FeatureImportanceVisualizer, PredictionVisualizer,
    ErrorAnalysisVisualizer, CalibrationVisualizer, create_comprehensive_model_report_visualization
)
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__}")

# ----------------1. Page configuration (da config.py)
def setup_page():
    """Configure the Streamlit page"""
    logger.info("Page configuration Streamlit")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Loading and preparing base data
@st.cache_data(ttl=3600)
def load_and_prepare_base_data():
    """Load and prepare base data"""
    logger.info("Loading and preparing base data")
    df_original = load_titanic_data()
    if df_original is None:
        logger.error("Unable to load original data")
        return None, None
    
    df_cleaned = clean_dataset_basic(df_original)
    logger.debug(f"Data loaded and cleaned. Original shape: {df_original.shape}, puliti: {df_cleaned.shape}")
    return df_original, df_cleaned

logger.info("Loading data Titanic")
df_original, df = load_and_prepare_base_data()
if df is None:
    logger.error("Unable to load Titanic data")
    st.error("Unable to load the data")
    st.stop()

# ----------------3. Page header
logger.info("Setting up page header")
st.title("Machine Learning Predictions")
st.markdown("Complete machine learning pipeline for predicting Titanic passenger survival")

# ----------------4. Sidebar controlli avanzati
logger.info("Setting up advanced sidebar controls")
with st.sidebar:
    st.header("Advanced ML Controls")
    
    # Main section
    ml_section = st.selectbox(
        "ML Section:",
        [
            "Data Quality & Preprocessing",
            "Model Training",
            "Model Evaluation", 
            "Model Comparison",
            "Feature Analysis",
            "Predictions & Deployment",
            "Model Reports"
        ]
    )
    logger.debug(f"Selected ML section: {ml_section}")
    
    st.markdown("---")
    
    # Preprocessing settings
    st.subheader("Preprocessing")
    preprocessing_config = st.selectbox(
        "Preprocessing configuration:",
        ["minimal", "standard", "advanced"],
        index=1,
        help="Minimal: basic, Standard: complete, Advanced: with feature selection"
    )
    logger.debug(f"Preprocessing configuration: {preprocessing_config}")
    
    # Training settings
    st.subheader("Training Configuration")
    training_mode = st.selectbox(
        "Training mode:",
        ["QUICK_TRAINING", "COMPREHENSIVE_TRAINING", "DEEP_TRAINING"],
        index=1,
        help="Quick: fast, Comprehensive: complete, Deep: with ensemble methods"
    )
    logger.debug(f"ModalitÃ  training: {training_mode}")
    
    # Models to use
    available_models = ModelFactory.get_available_models()
    selected_models = st.multiselect(
        "Select models:",
        available_models,
        default=available_models[:4],
        format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
    )
    logger.debug(f"Selected models: {selected_models}")
    
    # Advanced options
    st.subheader("Advanced Options")
    use_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
    use_cross_validataon = st.checkbox("Cross Validataon", value=True)
    save_models = st.checkbox("Save trained models", value=False)
    logger.debug(f"Advanced options: hp_tuning={use_hyperparameter_tuning}, cv={use_cross_validataon}, save={save_models}")

# ----------------5. Data Quality & Preprocessing
if ml_section == "Data Quality & Preprocessing":
    logger.info("Starting section Data Quality & Preprocessing")
    st.header("1. Data Quality Analysis and Preprocessing")
    
    # ----------------6. Data Quality Analysis
    logger.info("Analyzing data quality")
    st.subheader("Complete Quality Report")
    
    with st.expander("Complete Quality Report", expanded=True):
        logger.debug("Running data quality check")
        quality_report = DataQualityChecker.check_data_quality(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", quality_report['n_samples'])
            st.metric("Features", quality_report['n_features'])
        
        with col2:
            missing_count = len(quality_report['missing_values'])
            st.metric("Missing Values", missing_count)
            st.metric("Duplicates", quality_report['duplicates'])
        
        with col3:
            constant_count = len(quality_report['constant_features'])
            st.metric("Constant Features", constant_count)
            outliers_count = len(quality_report['outliers_summary'])
            st.metric("Features with Outliers", outliers_count)
        
        with col4:
            memory_mb = quality_report['memory_usage'] / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            skewed_count = len(quality_report['skewed_features'])
            st.metric("Skewed Features", skewed_count)
        
        logger.debug(f"Quality report: samples={quality_report['n_samples']}, features={quality_report['n_features']}, missing={missing_count}")
        
        # Problem details
        if quality_report['missing_values']:
            logger.debug("Displaying missing value details")
            st.write("**Missing Values Details:**")
            missing_df = pd.DataFrame([
                {
                    'Feature': col,
                    'Count': info['count'],
                    'Percentage': f"{info['percentage']:.1f}%"
                }
                for col, info in quality_report['missing_values'].items()
            ])
            st.dataframe(missing_df, use_container_width=True)
    
    # ----------------7. Preprocessing Recommendataons
    logger.info("Generating preprocessing recommendations")
    st.subheader("Preprocessing Recommendations")
    
    recommendataons = get_preprocessing_recommendataons(df)
    logger.debug(f"Recommendations: config={recommendataons['suggested_pipeline_config']}, complexity={recommendataons['estimated_complexity']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Suggested Configuration:** {recommendataons['suggested_pipeline_config'].title()}")
        st.info(f"**Estimated Complexity:** {recommendataons['estimated_complexity'].title()}")
        
        if recommendataons['required_steps']:
            logger.debug(f"Required steps: {len(recommendataons['required_steps'])}")
            st.write("**Required Steps:**")
            for step in recommendataons['required_steps']:
                st.write(f"- {step}")
    
    with col2:
        if recommendataons['optional_steps']:
            logger.debug(f"Optional steps: {len(recommendataons['optional_steps'])}")
            st.write("**Optional Steps:**")
            for step in recommendataons['optional_steps']:
                st.write(f"- {step}")
        
        if recommendataons['warnings']:
            logger.debug(f"Warnings: {len(recommendataons['warnings'])}")
            st.warning("**Warnings:**")
            for warning in recommendataons['warnings']:
                st.write(f"- {warning}")
    
    # ----------------8. Pipeline Creation & Validataon
    logger.info("Pipeline creation and validation section")
    st.subheader("Pipeline Creation and Validation")
    
    if st.button("Create and Validate Pipeline", type="primary"):
        logger.info("Starting pipeline creation and validation")
        with st.spinner("Creating pipeline..."):
            # Create pipeline
            logger.debug("Creating pipeline preprocessing")
            pipeline = create_titanic_preprocessing_pipeline(preprocessing_config)
            
            # Prepare data for validation
            target_col = 'Survived'
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            from sklearn.model_selection import train_test_split
            logger.debug("Train/test split for validation")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Valida pipeline
            logger.debug("Pipeline validation")
            validataon_report = validate_preprocessing_pipeline(
                pipeline, X_train, X_test, y_train, y_test
            )
            
            # Salva in session state
            st.session_state['preprocessing_pipeline'] = pipeline
            st.session_state['validataon_report'] = validataon_report
            st.session_state['prepared_data'] = (X_train, X_test, y_train, y_test)
            logger.debug("Pipeline and data saved in session state")
            
        # Show results validazione
        if validataon_report['validataon_passed']:
            logger.info("Pipeline validated successfully")
            st.success("Pipeline validated successfully!")
        else:
            logger.error("Problems during pipeline validation")
            st.error("Pipeline validation issues detected")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Shape Changes:**")
            shape_changes = validataon_report['shape_changes']
            st.write(f"Train: {shape_changes['train_before']} -> {shape_changes['train_after']}")
            st.write(f"Test: {shape_changes['test_before']} -> {shape_changes['test_after']}")
        
        with col2:
            if validataon_report['warnings']:
                logger.debug(f"Warnings validazione: {len(validataon_report['warnings'])}")
                st.warning("**Warnings:**")
                for warning in validataon_report['warnings']:
                    st.write(f"- {warning}")
            
            if validataon_report['errors']:
                logger.error(f"Errori validazione: {len(validataon_report['errors'])}")
                st.error("**Errors:**")
                for error in validataon_report['errors']:
                    st.write(f"- {error}")

# ----------------9. Model Training
elif ml_section == "Model Training":
    logger.info("Starting section Model Training")
    st.header("2. Machine Learning Model Training")
    
    # Verifica prerequisiti
    if 'preprocessing_pipeline' not in st.session_state:
        logger.warning("Pipeline preprocessing non trovata")
        st.warning("Run the Data Quality & Preprocessing section first to create the pipeline")
        st.stop()
    
    pipeline = st.session_state['preprocessing_pipeline']
    X_train, X_test, y_train, y_test = st.session_state['prepared_data']
    logger.debug(f"Data preparati uploadti. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # ----------------10. Training Configuration Display
    logger.info("Displaying training configuration")
    st.subheader("Training Configuration")
    
    training_config = TrainingPipelineManager(training_mode).config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Test Size:** {training_config['test_size']}")
        st.info(f"**CV Folds:** {training_config['cv_folds']}")
    
    with col2:
        st.info(f"**Hyperparameter Tuning:** {'Yes' if use_hyperparameter_tuning else 'No'}")
        st.info(f"**Cross Validation:** {'Yes' if use_cross_validataon else 'No'}")
    
    with col3:
        st.info(f"**Selected Models:** {len(selected_models)}")
        st.info(f"**Model Saving:** {'Yes' if save_models else 'No'}")
    
    # ----------------11. Training Execution
    logger.info("Sezione esecuzione training")
    st.subheader("Training Execution")
    
    if st.button("Start Full Training", type="primary"):
        if not selected_models:
            logger.error("No model selectto")
            st.error("Select at least one model from the sidebar")
            st.stop()
        
        logger.info(f"Avvio training con {len(selected_models)} models")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            logger.debug(f"Progress: {progress:.2f} - {message}")
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Inizializza trainer
        logger.debug("Inizializzazione trainer")
        trainer = ModelTrainer(random_state=42)
        
        # Imposta data giÃ  splittati direttamente
        trainer.X_train = X_train
        trainer.y_train = y_train
        trainer.X_test = X_test
        trainer.y_test = y_test
        
        # Applica preprocessing
        with st.spinner("Applicazione preprocessing..."):
            logger.debug("Applicazione preprocessing ai data")
            X_train_processed = pipeline.fit_transform(trainer.X_train)
            X_test_processed = pipeline.transform(trainer.X_test)
            
            # Aggiorna trainer con data processati
            trainer.X_train = pd.DataFrame(X_train_processed) if hasattr(X_train_processed, 'shape') else X_train_processed
            trainer.X_test = pd.DataFrame(X_test_processed) if hasattr(X_test_processed, 'shape') else X_test_processed
            logger.debug(f"Preprocessing applicato. Train shape: {trainer.X_train.shape}, Test shape: {trainer.X_test.shape}")
        
        training_results = {}
        evaluation_results = {}
        cv_results = {}
        
        # Training loop
        for i, model_type in enumerate(selected_models):
            update_progress(i / len(selected_models), f"Training {ML_MODELS[model_type]['name']}...")
            logger.info(f"Training model: {model_type}")
            
            try:
                # Training
                result = trainer.train_single_model(model_type)
                training_results[model_type] = result
                logger.debug(f"Training {model_type} completato")
                
                # Cross validataon se richiesto
                if use_cross_validataon:
                    logger.debug(f"Cross validataon {model_type}")
                    cv_result = trainer.cross_validate_model(model_type)
                    cv_results[model_type] = cv_result
                
                # Evaluation
                logger.debug(f"Evaluation {model_type}")
                model_dict = {model_type: {'model': result['model'], 'preprocessor': None}}
                evaluator = ModelEvaluator(model_dict, trainer.X_test, trainer.y_test)
                eval_result = evaluator.evaluate_single_model(model_type, result)
                evaluation_results[model_type] = eval_result
                
            except Exception as e:
                logger.error(f"Error nel training di {model_type}: {str(e)}")
                st.error(f"Error while training {model_type}: {str(e)}")
                continue
        
        update_progress(1.0, "Training completato!")
        logger.info("Training completato per tutti i models")
        
        # Salva results in session state
        st.session_state['training_results'] = training_results
        st.session_state['evaluation_results'] = evaluation_results
        st.session_state['cv_results'] = cv_results
        st.session_state['trained_models'] = trainer.trained_models
        logger.debug("Training results saved in session state")
        
        # ----------------12. Training Results Display
        st.success("Training completed successfully!")
        
        # Summary veloce
        comparison = ModelComparison(evaluation_results)
        best_model = comparison.find_best_model('f1')
        
        if best_model:
            logger.info(f"Best model: {best_model['model_name']} (F1: {best_model['score']:.3f})")
            st.success(f"**Best Model:** {best_model['model_name']} (F1: {best_model['score']:.3f})")
        
        # Tabella results
        logger.debug("Creating tabella results")
        st.subheader("Training Results")
        results_table = comparison.create_comparison_table()
        st.dataframe(results_table, use_container_width=True)
        
        # Display training times
        if training_results:
            logger.debug("Creating visualizzazioni training")
            train_viz = TrainingVisualizer()
            fig_times = train_viz.create_training_progress_chart(training_results)
            st.plotly_chart(fig_times, use_container_width=True)
        
        # Cross validataon results
        if cv_results:
            logger.debug("Creating visualizzazioni cross validataon")
            fig_cv = train_viz.create_cross_validataon_chart(cv_results)
            st.plotly_chart(fig_cv, use_container_width=True)

# ----------------13. Model Evaluation
elif ml_section == "Model Evaluation":
    logger.info("Starting section Model Evaluation")
    st.header("3. Detailed Model Evaluation")
    
    # Verifica prerequisiti
    if 'evaluation_results' not in st.session_state:
        logger.warning("Evaluation results not found")
        st.warning("Prima esegui il training dei models")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Results evaluation uploadti per {len(evaluation_results)} models")
    
    # ----------------14. Performance Overview
    logger.info("Creating performance overview")
    st.subheader("Performance Overview")
    
    # Metrics radar chart
    logger.debug("Creating radar chart metriche")
    perf_viz = PerformanceVisualizer()
    fig_radar = perf_viz.create_metrics_comparison_radar(evaluation_results)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance heatmap
    logger.debug("Creating heatmap performance")
    fig_heatmap = perf_viz.create_performance_heatmap(evaluation_results)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ----------------15. Detailed Metrics Analysis
    logger.info("Analysis metriche detailsate")
    st.subheader("Detailed Metric Analysis")
    
    # Selettore model per analysis detailsata
    selected_model_detail = st.selectbox(
        "Select model for detailed analysis:",
        list(evaluation_results.keys()),
        format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
    )
    logger.debug(f"Modello selectto per details: {selected_model_detail}")
    
    if selected_model_detail:
        model_results = evaluation_results[selected_model_detail]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_results.get('accuracy', 0):.3f}")
            st.metric("Precision", f"{model_results.get('precision', 0):.3f}")
        
        with col2:
            st.metric("Recall", f"{model_results.get('recall', 0):.3f}")
            st.metric("F1-Score", f"{model_results.get('f1', 0):.3f}")
        
        with col3:
            st.metric("ROC-AUC", f"{model_results.get('roc_auc', 0):.3f}")
            st.metric("Balanced Accuracy", f"{model_results.get('balanced_accuracy', 0):.3f}")
        
        with col4:
            st.metric("Matthews Corr", f"{model_results.get('matthews_corrcoef', 0):.3f}")
            st.metric("Cohen Kappa", f"{model_results.get('cohen_kappa', 0):.3f}")
        
        # Confusion Matrix detailsata
        if 'confusion_matrix' in model_results:
            logger.debug("Creating confusion matrix detailsata")
            cm_viz = ConfusionMatrixVisualizer()
            fig_cm = cm_viz.create_confusion_matrix_detailed(
                model_results['confusion_matrix'], 
                selected_model_detail
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ----------------16. Model Comparison
elif ml_section == "Model Comparison":
    logger.info("Starting section Model Comparison")
    st.header("4. In-Depth Model Comparison")
    
    if 'evaluation_results' not in st.session_state:
        logger.warning("Evaluation results not found per comparison")
        st.warning("Prima esegui il training dei models")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Comparison tra {len(evaluation_results)} models")
    
    # ----------------17. ROC Curves Comparison
    logger.info("Creating curve ROC")
    st.subheader("Curve ROC")
    
    if 'trained_models' in st.session_state:
        logger.debug("Calcolo probabilitÃ  per curve ROC")
        # Calcola probabilitÃ  per ROC
        probabilities = {}
        X_test = st.session_state['prepared_data'][1]
        y_test = st.session_state['prepared_data'][3]
        pipeline = st.session_state['preprocessing_pipeline']
        
        X_test_processed = pipeline.transform(X_test)
        
        for model_name, model_data in st.session_state['trained_models'].items():
            if hasattr(model_data['model'], 'predict_proba'):
                try:
                    proba = model_data['model'].predict_proba(X_test_processed)[:, 1]
                    probabilities[model_name] = proba
                    logger.debug(f"Calculated probabilities for {model_name}")
                except:
                    probabilities[model_name] = None
                    logger.debug(f"Error calculating probabilities for {model_name}")
        
        curve_viz = CurveVisualizer()
        fig_roc = curve_viz.create_roc_curves_comparison(evaluation_results, y_test, probabilities)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Precision-Recall curves
        logger.debug("Creating curve Precision-Recall")
        fig_pr = curve_viz.create_precision_recall_curves(evaluation_results, y_test, probabilities)
        st.plotly_chart(fig_pr, use_container_width=True)
    
    # ----------------18. Statistical Significance Tests
    logger.info("Statistical significance tests")
    st.subheader("Statistical Significance Tests")
    
    if len(evaluation_results) >= 2:
        logger.debug("Setting up selectors for statistical comparison")
        # Selettori per comparison
        col1, col2 = st.columns(2)
        
        with col1:
            model1 = st.selectbox(
                "Model 1:",
                list(evaluation_results.keys()),
                format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
            )
        
        with col2:
            model2 = st.selectbox(
                "Model 2:",
                [m for m in evaluation_results.keys() if m != model1],
                format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
            )
        
        logger.debug(f"Models for statistical comparison: {model1} vs {model2}")
        
        if st.button("Run Statistical Tests"):
            logger.info(f"Running statistical tests: {model1} vs {model2}")
            # McNemar test (richiede predictions)
            if 'trained_models' in st.session_state:
                y_test = st.session_state['prepared_data'][3]
                X_test_processed = pipeline.transform(st.session_state['prepared_data'][1])
                
                pred1 = st.session_state['trained_models'][model1]['model'].predict(X_test_processed)
                pred2 = st.session_state['trained_models'][model2]['model'].predict(X_test_processed)
                
                mcnemar_result = StatisticalTests.mcnemar_test(y_test, pred1, pred2)
                logger.debug(f"McNemar test result: statistic={mcnemar_result['statistic']:.3f}, p-value={mcnemar_result['p_value']:.4f}")
                
                st.write("**McNemar Test:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Statistic", f"{mcnemar_result['statistic']:.3f}")
                
                with col2:
                    st.metric("P-value", f"{mcnemar_result['p_value']:.4f}")
                
                with col3:
                    significance = "Significant" if mcnemar_result['significant'] else "Not Significant"
                    st.metric("Result", significance)
    else:
        logger.debug("Numero insufficiente di models per test statistici")
    
    # ----------------19. Model Rankings
    logger.info("Creating ranking models")
    st.subheader("Model Rankings")
    
    metrics_for_ranking = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for metric in metrics_for_ranking:
        with st.expander(f"Ranking by {metric.title()}"):
            logger.debug(f"Creating ranking by metric: {metric}")
            fig_ranking = perf_viz.create_model_ranking_chart(evaluation_results, metric)
            st.plotly_chart(fig_ranking, use_container_width=True)

# ----------------20. Feature Analysis
elif ml_section == "Feature Analysis":
    logger.info("Starting Feature Analysis section")
    st.header("5. Feature Importance Analysis")
    
    if 'trained_models' not in st.session_state:
        logger.warning("Trained models not found for feature analysis")
        st.warning("Run model training first")
        st.stop()
    
    # ----------------21. Feature Importance per model
    logger.info("Analysis feature importance per model")
    st.subheader("Feature Importance by Model")
    
    models_with_importance = ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier']
    available_models = [m for m in st.session_state['trained_models'].keys() if m in models_with_importance]
    logger.debug(f"Modelli con feature importance disponibili: {available_models}")
    
    if not available_models:
        logger.warning("No model con feature importance disponibile")
        st.warning("No model with feature importance available")
    else:
        selected_model_fi = st.selectbox(
            "Select model per feature importance:",
            available_models,
            format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
        )
        logger.debug(f"Selected model for feature importance: {selected_model_fi}")
        
        model_obj = st.session_state['trained_models'][selected_model_fi]['model']
        
        if hasattr(model_obj, 'feature_importances_'):
            logger.debug("Estrazione feature importance")
            # Ottieni nomi features (potrebbero essere numerici dopo preprocessing)
            if hasattr(model_obj, 'feature_names_in_'):
                feature_names = model_obj.feature_names_in_
            else:
                # Genera nomi features generici
                n_features = len(model_obj.feature_importances_)
                feature_names = [f'feature_{i}' for i in range(n_features)]
            
            importance_dict = dict(zip(feature_names, model_obj.feature_importances_))
            logger.debug(f"Feature importance estratte: {len(importance_dict)} features")
            
            fi_viz = FeatureImportanceVisualizer()
            fig_fi = fi_viz.create_feature_importance_chart(importance_dict)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            # Tabella feature importance
            fi_df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            ])
            
            st.dataframe(fi_df, use_container_width=True, height=300)
        else:
            logger.debug(f"Modello {selected_model_fi} non ha feature_importances_")
    
    # ----------------22. Comparison Feature Importance
    if len(available_models) > 1:
        logger.info("Comparison feature importance tra models")
        st.subheader("Comparison Feature Importance")
        
        importance_data = {}
        for model_name in available_models:
            model_obj = st.session_state['trained_models'][model_name]['model']
            if hasattr(model_obj, 'feature_importances_'):
                if hasattr(model_obj, 'feature_names_in_'):
                    feature_names = model_obj.feature_names_in_
                else:
                    n_features = len(model_obj.feature_importances_)
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                
                importance_data[model_name] = dict(zip(feature_names, model_obj.feature_importances_))
        
        logger.debug(f"Data importanza per comparison: {len(importance_data)} models")
        
        if importance_data:
            logger.debug("Creating chart comparison feature importance")
            fig_fi_comp = fi_viz.create_feature_importance_comparison(importance_data)
            st.plotly_chart(fig_fi_comp, use_container_width=True)

# ----------------23. Predictions & Deployment
elif ml_section == "Predictions & Deployment":
    logger.info("Starting Predictions & Deployment section")
    st.header("6. Predictions and Deployment")
    
    if 'trained_models' not in st.session_state:
        logger.warning("Trained models not found for predictions")
        st.warning("Run model training first")
        st.stop()
    
    # ----------------24. Single Prediction Interface
    logger.info("Setting up single prediction interface")
    st.subheader("Single Prediction")
    
    st.write("Enter passenger data to predict survival:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_pclass = st.selectbox("Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
        new_sex = st.selectbox("Sex", ["male", "female"], format_func=lambda x: "Male" if x == "male" else "Female")
        new_age = st.number_input("Age", min_value=0, max_value=100, value=30)
    
    with col2:
        new_sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
        new_parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
        new_fare = st.number_input("Ticket Fare", min_value=0.0, max_value=500.0, value=50.0)
    
    with col3:
        new_embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], 
                                   format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
        
        # Derived family information
        family_size = new_sibsp + new_parch + 1
        is_alone = "Yes" if family_size == 1 else "No"
        st.info(f"**Family Size:** {family_size} members")
        st.info(f"**Traveling Alone:** {is_alone}")
    
    logger.debug(f"Input data: class={new_pclass}, sex={new_sex}, age={new_age}, family={family_size}")
    
    # ----------------25. Esegui Prediction
    if st.button("Predict Survival", type="primary"):
        logger.info("Running single prediction")
        # Crea DataFrame input
        input_data = pd.DataFrame({
            'Pclass': [new_pclass],
            'Sex': [new_sex],
            'Age': [new_age],
            'SibSp': [new_sibsp],
            'Parch': [new_parch],
            'Fare': [new_fare],
            'Embarked': [new_embarked]
        })
        
        # Applica preprocessing
        logger.debug("Applying preprocessing to single input")
        pipeline = st.session_state['preprocessing_pipeline']
        input_processed = pipeline.transform(input_data)
        
        # Predictions da tutti i models
        logger.debug("Running predictions from all models")
        predictions = {}
        probabilities = {}
        
        for model_name, model_data in st.session_state['trained_models'].items():
            model = model_data['model']
            
            pred = model.predict(input_processed)[0]
            predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_processed)[0][1]
                probabilities[model_name] = proba
                logger.debug(f"Prediction {model_name}: {pred}, probability: {proba:.3f}")
            else:
                probabilities[model_name] = None
                logger.debug(f"Prediction {model_name}: {pred}")
        
        # ----------------26. Visualizza Results
        logger.info("Displaying prediction results")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predictions by Model:**")
            for model_name, pred in predictions.items():
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                prob = probabilities[model_name]
                
                if pred == 1:
                    st.success(f"**{model_display_name}:** SURVIVES")
                else:
                    st.error(f"**{model_display_name}:** DOES NOT SURVIVE")
                
                if prob is not None:
                    st.write(f"Probability: {prob:.2%}")
        
        with col2:
            # Consensus prediction
            survival_votes = sum(predictions.values())
            total_votes = len(predictions)
            consensus_prob = survival_votes / total_votes
            logger.debug(f"Consensus: {survival_votes}/{total_votes} = {consensus_prob:.3f}")
            
            st.write("**Consensus Prediction:**")
            if consensus_prob > 0.5:
                st.success(f"**CONSENSUS: SURVIVES**")
                st.write(f"Agreement: {survival_votes}/{total_votes} models")
            else:
                st.error(f"**CONSENSUS: DOES NOT SURVIVE**")
                st.write(f"Agreement: {total_votes - survival_votes}/{total_votes} models")
            
            confidence = max(consensus_prob, 1-consensus_prob)
            st.metric("Confidence Level", f"{confidence:.1%}")
        
        # Probability chart
        if any(prob is not None for prob in probabilities.values()):
            logger.debug("Creating prediction probability chart")
            prob_data = []
            for model_name, prob in probabilities.items():
                if prob is not None:
                    model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                    prob_data.append({
                        'Model': model_display_name,
                        'Probability': prob,
                        'Prediction': 'Survives' if prob > 0.5 else 'Does Not Survive'
                    })
            
            if prob_data:
                prob_df = pd.DataFrame(prob_data)
                pred_viz = PredictionVisualizer()
                fig_pred = pred_viz.create_prediction_confidence_chart(prob_df)
                st.plotly_chart(fig_pred, use_container_width=True)
    
    # ----------------27. Batch Predictions
    logger.info("Setting up batch predictions")
    st.subheader("Batch Predictions")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch predictions:",
        type=['csv'],
        help="The file must contain the same columns as the training dataset"
    )
    
    if uploaded_file is not None:
        logger.debug("CSV file uploaded for batch predictions")
        batch_data = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**")
        st.dataframe(batch_data.head(), use_container_width=True)
        logger.debug(f"Uploaded batch data. Shape: {batch_data.shape}")
        
        if st.button("Run Batch Predictions"):
            logger.info("Running batch predictions")
            try:
                # Preprocessing
                logger.debug("Preprocessing data batch")
                pipeline = st.session_state['preprocessing_pipeline']
                batch_processed = pipeline.transform(batch_data)
                
                # Predictions
                batch_results = batch_data.copy()
                
                # Usa il best model per batch predictions
                evaluation_results = st.session_state['evaluation_results']
                comparison = ModelComparison(evaluation_results)
                best_model_info = comparison.find_best_model('f1')
                
                if best_model_info:
                    logger.info(f"Usando best model per batch: {best_model_info['model_name']}")
                    best_model_name = best_model_info['model_type']
                    best_model = st.session_state['trained_models'][best_model_name]['model']
                    
                    predictions = best_model.predict(batch_processed)
                    batch_results['Predicted_Survival'] = predictions
                    batch_results['Predicted_Survival_Text'] = batch_results['Predicted_Survival'].map({0: 'Does Not Survive', 1: 'Survives'})
                    
                    if hasattr(best_model, 'predict_proba'):
                        probabilities = best_model.predict_proba(batch_processed)[:, 1]
                        batch_results['Survival_Probability'] = probabilities
                    
                    logger.debug(f"Predictions batch completate: {len(predictions)} predictions")
                    st.success(f"Predictions completed using {best_model_info['model_name']}")
                    st.dataframe(batch_results, use_container_width=True)
                    
                    # Download results
                    csv = batch_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name=f"titanic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                
            except Exception as e:
                logger.error(f"Error nelle predictions batch: {str(e)}")
                st.error(f"Error nelle predictions batch: {str(e)}")
    
    # ----------------28. Model Deployment Info
    logger.info("Sezione informazioni deployment")
    st.subheader("Deployment Information")
    
    with st.expander("Deployment Guide", expanded=False):
        st.markdown("""
        **Model deployment options:**
        
        1. **REST API**: Use FastAPI or Flask to create endpoints
        2. **Streamlit Cloud**: Direct deployment of this app
        3. **Docker Container**: Containerize the application
        4. **Cloud Services**: AWS SageMaker, Azure ML, Google AI Platform
        
        **Files required for deployment:**
        - Trained model (`pickle`/`joblib`)
        - Preprocessing pipeline
        - `requirements.txt`
        - `Dockerfile` (optional)
        
        **Production considerations:**
        - Model performance monitoring
        - A/B testing between models
        - Automated retraining
        - Data drift detection
        """)
    
    # Model saving
    if save_models:
        logger.info("Model saving section")
        st.subheader("Model Saving")
        
        if st.button("Save All Models"):
            logger.info("Starting save for all models")
            saved_models = []
            
            for model_name, model_data in st.session_state['trained_models'].items():
                try:
                    logger.debug(f"Salvataggio model: {model_name}")
                    # Salva model
                    model_path = ModelPersistence.save_model(
                        model_data['model'], 
                        model_name
                    )
                    saved_models.append(f"{model_name}: {model_path}")
                    
                    # Salva anche preprocessing pipeline
                    pipeline_path = ModelPersistence.save_model(
                        st.session_state['preprocessing_pipeline'],
                        f"preprocessing_pipeline_{model_name}"
                    )
                    saved_models.append(f"Pipeline {model_name}: {pipeline_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving {model_name}: {str(e)}")
                    st.error(f"Error saving {model_name}: {str(e)}")
            
            if saved_models:
                logger.info(f"Models saved successfully: {len(saved_models)}")
                st.success("Models saved successfully!")
                for saved in saved_models:
                    st.write(f"- {saved}")

# ----------------29. Model Reports
elif ml_section == "Model Reports":
    logger.info("Starting section Model Reports")
    st.header("7. Complete Model Reports")
    
    if 'evaluation_results' not in st.session_state:
        logger.warning("Evaluation results not found per reports")
        st.warning("Run training and evaluation first")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Report per {len(evaluation_results)} models")
    
    # ----------------30. Executive Summary
    logger.info("Creating executive summary")
    st.subheader("Executive Summary")
    
    comparison = ModelComparison(evaluation_results)
    
    # Best models per metrica
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_acc = comparison.find_best_model('accuracy')
        if best_acc:
            st.metric(
                "Best Accuracy",
                f"{best_acc['score']:.3f}",
                delta=best_acc['model_name']
            )
    
    with col2:
        best_prec = comparison.find_best_model('precision')
        if best_prec:
            st.metric(
                "Best Precision", 
                f"{best_prec['score']:.3f}",
                delta=best_prec['model_name']
            )
    
    with col3:
        best_rec = comparison.find_best_model('recall')
        if best_rec:
            st.metric(
                "Best Recall",
                f"{best_rec['score']:.3f}",
                delta=best_rec['model_name']
            )
    
    with col4:
        best_f1 = comparison.find_best_model('f1')
        if best_f1:
            st.metric(
                "Best F1",
                f"{best_f1['score']:.3f}",
                delta=best_f1['model_name']
            )
            logger.debug(f"Best F1: {best_f1['model_name']} ({best_f1['score']:.3f})")
    
    # ----------------31. Comprehensive Visualizations
    logger.info("Creating visualizzazioni complete")
    st.subheader("Visualizzazioni Complete")
    
    # Crea report visualizzazioni complete
    training_results = st.session_state.get('training_results')
    logger.debug("Creating comprehensive visualization report")
    comprehensive_viz = create_comprehensive_model_report_visualization(
        evaluation_results, training_results
    )
    
    for i, fig in enumerate(comprehensive_viz):
        if fig is not None:
            logger.debug(f"Display figura {i}")
            st.plotly_chart(fig, use_container_width=True)
    
    # ----------------32. Error Analysis
    logger.info("Analysis errori avanzata")
    st.subheader("Advanced Error Analysis")
    
    if 'trained_models' in st.session_state:
        # Selettore model per error analysis
        selected_model_error = st.selectbox(
            "Select model per analysis errori:",
            list(st.session_state['trained_models'].keys()),
            format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
        )
        logger.debug(f"Modello selectto per error analysis: {selected_model_error}")
        
        # Calcola predictions per error analysis
        X_test = st.session_state['prepared_data'][1]
        y_test = st.session_state['prepared_data'][3]
        pipeline = st.session_state['preprocessing_pipeline']
        
        logger.debug("Calcolo predictions per error analysis")
        X_test_processed = pipeline.transform(X_test)
        model = st.session_state['trained_models'][selected_model_error]['model']
        predictions = model.predict(X_test_processed)
        
        # Error analysis
        if hasattr(X_test_processed, 'columns'):
            feature_names = X_test_processed.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X_test_processed.shape[1])]
        
        predictions_dict = {selected_model_error: predictions}
        error_analyzer = ErrorAnalysis(X_test_processed, y_test, predictions_dict, feature_names)
        
        error_analysis = error_analyzer.analyze_prediction_errors(selected_model_error)
        logger.debug(f"Error analysis completata: {error_analysis['total_errors'] if error_analysis else 'N/A'} errori")
        
        if error_analysis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Errors", error_analysis['total_errors'])
            
            with col2:
                st.metric("False Positives", error_analysis['false_positives'])
            
            with col3:
                st.metric("False Negatives", error_analysis['false_negatives'])
            
            with col4:
                st.metric("Error Rate", f"{error_analysis['error_rate']:.1%}")
            
            # Difficult samples analysis
            difficult_samples = error_analyzer.find_difficult_samples()
            
            if difficult_samples['difficult_samples'] > 0:
                logger.debug(f"Difficult samples found: {difficult_samples['difficult_samples']} ({difficult_samples['percentage']:.1f}%)")
                st.write(f"**Difficult Samples to Classify:** {difficult_samples['difficult_samples']} ({difficult_samples['percentage']:.1f}%)")
    
    # ----------------33. Model Comparison Table
    logger.info("Creating tabella comparison completa")
    st.subheader("Complete Comparison Table")
    
    detailed_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', 'matthews_corrcoef']
    comparison_table = comparison.create_comparison_table(detailed_metrics)
    st.dataframe(comparison_table, use_container_width=True)
    
    # ----------------34. Export Reports
    logger.info("Sezione export reports")
    st.subheader("Export Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Genera Report JSON"):
            logger.info("Generating report JSON")
            # Crea report strutturato
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'models_evaluated': len(evaluation_results),
                'best_models': {
                    'accuracy': comparison.find_best_model('accuracy'),
                    'precision': comparison.find_best_model('precision'),
                    'recall': comparison.find_best_model('recall'),
                    'f1': comparison.find_best_model('f1')
                },
                'detailed_results': comparison_table.to_dict('records'),
                'preprocessing_config': preprocessing_config,
                'training_config': training_mode
            }
            
            import json
            json_str = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                label="Download Report JSON",
                data=json_str,
                file_name=f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )
    
    with col2:
        if st.button("Genera Report CSV"):
            logger.info("Generating report CSV")
            csv_report = comparison_table.to_csv(index=False)
            
            st.download_button(
                label="Download Report CSV",
                data=csv_report,
                file_name=f"ml_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

# ----------------35. Footer e Summary Generale
logger.info("Creating footer e summary generale")
st.markdown("---")

# Summary stato corrente
if 'trained_models' in st.session_state:
    n_models = len(st.session_state['trained_models'])
    logger.debug(f"Pipeline attiva con {n_models} models")
    st.success(f"**Active ML Pipeline:** {n_models} models trained with configuration {preprocessing_config}")
    
    if 'evaluation_results' in st.session_state:
        comparison = ModelComparison(st.session_state['evaluation_results'])
        best_model = comparison.find_best_model('f1')
        if best_model:
            logger.info(f"Best model globale: {best_model['model_name']} (F1: {best_model['score']:.3f})")
            st.info(f"**Best Overall Model:** {best_model['model_name']} (F1: {best_model['score']:.3f})")

st.markdown("""
**Machine Learning Pipeline Completed**

This implementation represents a complete, production-ready ML pipeline for Titanic analysis,
with a modular architecture, intelligent preprocessing, automated training, and in-depth evaluation.

**Next Steps:** Consider deployment, production monitoring, and automated retraining for a fully operational ML system.
""")

logger.info(f"Pagina {__name__} completata con successo")



