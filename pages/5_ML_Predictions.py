"""
pages/5_ML_Predictions.py
Machine Learning Models per Predizione Sopravvivenza Titanic
Versione modulare con architettura completa
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import dei moduli ML sviluppati
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic
from src.utils.ml_preprocessing import (
    create_titanic_preprocessing_pipeline, 
    DataQualityChecker,
    get_preprocessing_recommendations,
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
logger.info(f"Caricamento {__name__}")

# ----------------1. Configurazione pagina (da config.py)
def setup_page():
    """Configura la pagina Streamlit"""
    logger.info("Configurazione pagina Streamlit")
    st.set_page_config(**PAGE_CONFIG)

setup_page()

# ----------------2. Caricamento e preparazione dati base
@st.cache_data(ttl=3600)
def load_and_prepare_base_data():
    """Carica e prepara dati base"""
    logger.info("Funzione caricamento e preparazione dati base")
    df_original = load_titanic_data()
    if df_original is None:
        logger.error("Impossibile caricare dati originali")
        return None, None
    
    df_cleaned = clean_dataset_basic(df_original)
    logger.debug(f"Dati caricati e puliti. Shape originale: {df_original.shape}, puliti: {df_cleaned.shape}")
    return df_original, df_cleaned

logger.info("Caricamento dati Titanic")
df_original, df = load_and_prepare_base_data()
if df is None:
    logger.error("Impossibile caricare i dati Titanic")
    st.error("Impossibile caricare i dati")
    st.stop()

# ----------------3. Header pagina
logger.info("Setup header pagina")
st.title("Machine Learning Predictions")
st.markdown("Pipeline completa di Machine Learning per predire la sopravvivenza dei passeggeri del Titanic")

# ----------------4. Sidebar controlli avanzati
logger.info("Setup sidebar controlli avanzati")
with st.sidebar:
    st.header("Controlli ML Avanzati")
    
    # Sezione principale
    ml_section = st.selectbox(
        "Sezione ML:",
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
    logger.debug(f"Sezione ML selezionata: {ml_section}")
    
    st.markdown("---")
    
    # Configurazioni preprocessing
    st.subheader("Preprocessing")
    preprocessing_config = st.selectbox(
        "Configurazione preprocessing:",
        ["minimal", "standard", "advanced"],
        index=1,
        help="Minimal: basic, Standard: completo, Advanced: con feature selection"
    )
    logger.debug(f"Configurazione preprocessing: {preprocessing_config}")
    
    # Configurazioni training
    st.subheader("Training Configuration")
    training_mode = st.selectbox(
        "Modalità training:",
        ["QUICK_TRAINING", "COMPREHENSIVE_TRAINING", "DEEP_TRAINING"],
        index=1,
        help="Quick: veloce, Comprehensive: completo, Deep: con ensemble"
    )
    logger.debug(f"Modalità training: {training_mode}")
    
    # Modelli da utilizzare
    available_models = ModelFactory.get_available_models()
    selected_models = st.multiselect(
        "Seleziona modelli:",
        available_models,
        default=available_models[:4],
        format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
    )
    logger.debug(f"Modelli selezionati: {selected_models}")
    
    # Opzioni avanzate
    st.subheader("Opzioni Avanzate")
    use_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
    use_cross_validation = st.checkbox("Cross Validation", value=True)
    save_models = st.checkbox("Salva modelli addestrati", value=False)
    logger.debug(f"Opzioni avanzate: hp_tuning={use_hyperparameter_tuning}, cv={use_cross_validation}, save={save_models}")

# ----------------5. Data Quality & Preprocessing
if ml_section == "Data Quality & Preprocessing":
    logger.info("Avvio sezione Data Quality & Preprocessing")
    st.header("1. Analisi Qualità Dati e Preprocessing")
    
    # ----------------6. Data Quality Analysis
    logger.info("Analisi qualità dati")
    st.subheader("Report Qualità Completo")
    
    with st.expander("Report Qualità Completo", expanded=True):
        logger.debug("Esecuzione check qualità dati")
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
            st.metric("Features con Outliers", outliers_count)
        
        with col4:
            memory_mb = quality_report['memory_usage'] / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            skewed_count = len(quality_report['skewed_features'])
            st.metric("Skewed Features", skewed_count)
        
        logger.debug(f"Quality report: samples={quality_report['n_samples']}, features={quality_report['n_features']}, missing={missing_count}")
        
        # Dettagli problemi
        if quality_report['missing_values']:
            logger.debug("Visualizzazione dettagli missing values")
            st.write("**Missing Values Dettaglio:**")
            missing_df = pd.DataFrame([
                {
                    'Feature': col,
                    'Count': info['count'],
                    'Percentage': f"{info['percentage']:.1f}%"
                }
                for col, info in quality_report['missing_values'].items()
            ])
            st.dataframe(missing_df, use_container_width=True)
    
    # ----------------7. Preprocessing Recommendations
    logger.info("Generazione raccomandazioni preprocessing")
    st.subheader("Raccomandazioni Preprocessing")
    
    recommendations = get_preprocessing_recommendations(df)
    logger.debug(f"Raccomandazioni: config={recommendations['suggested_pipeline_config']}, complexity={recommendations['estimated_complexity']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Configurazione Suggerita:** {recommendations['suggested_pipeline_config'].title()}")
        st.info(f"**Complessità Stimata:** {recommendations['estimated_complexity'].title()}")
        
        if recommendations['required_steps']:
            logger.debug(f"Step richiesti: {len(recommendations['required_steps'])}")
            st.write("**Step Richiesti:**")
            for step in recommendations['required_steps']:
                st.write(f"- {step}")
    
    with col2:
        if recommendations['optional_steps']:
            logger.debug(f"Step opzionali: {len(recommendations['optional_steps'])}")
            st.write("**Step Opzionali:**")
            for step in recommendations['optional_steps']:
                st.write(f"- {step}")
        
        if recommendations['warnings']:
            logger.debug(f"Warnings: {len(recommendations['warnings'])}")
            st.warning("**Avvertimenti:**")
            for warning in recommendations['warnings']:
                st.write(f"- {warning}")
    
    # ----------------8. Pipeline Creation & Validation
    logger.info("Sezione creazione e validazione pipeline")
    st.subheader("Creazione e Validazione Pipeline")
    
    if st.button("Crea e Valida Pipeline", type="primary"):
        logger.info("Avvio creazione e validazione pipeline")
        with st.spinner("Creazione pipeline in corso..."):
            # Crea pipeline
            logger.debug("Creazione pipeline preprocessing")
            pipeline = create_titanic_preprocessing_pipeline(preprocessing_config)
            
            # Prepara dati per validazione
            target_col = 'Survived'
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            from sklearn.model_selection import train_test_split
            logger.debug("Split train/test per validazione")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Valida pipeline
            logger.debug("Validazione pipeline")
            validation_report = validate_preprocessing_pipeline(
                pipeline, X_train, X_test, y_train, y_test
            )
            
            # Salva in session state
            st.session_state['preprocessing_pipeline'] = pipeline
            st.session_state['validation_report'] = validation_report
            st.session_state['prepared_data'] = (X_train, X_test, y_train, y_test)
            logger.debug("Pipeline e dati salvati in session state")
            
        # Mostra risultati validazione
        if validation_report['validation_passed']:
            logger.info("Pipeline validata con successo")
            st.success("Pipeline validata con successo!")
        else:
            logger.error("Problemi nella validazione pipeline")
            st.error("Problemi nella validazione pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cambiamenti Shape:**")
            shape_changes = validation_report['shape_changes']
            st.write(f"Train: {shape_changes['train_before']} → {shape_changes['train_after']}")
            st.write(f"Test: {shape_changes['test_before']} → {shape_changes['test_after']}")
        
        with col2:
            if validation_report['warnings']:
                logger.debug(f"Warnings validazione: {len(validation_report['warnings'])}")
                st.warning("**Warnings:**")
                for warning in validation_report['warnings']:
                    st.write(f"- {warning}")
            
            if validation_report['errors']:
                logger.error(f"Errori validazione: {len(validation_report['errors'])}")
                st.error("**Errori:**")
                for error in validation_report['errors']:
                    st.write(f"- {error}")

# ----------------9. Model Training
elif ml_section == "Model Training":
    logger.info("Avvio sezione Model Training")
    st.header("2. Training Modelli Machine Learning")
    
    # Verifica prerequisiti
    if 'preprocessing_pipeline' not in st.session_state:
        logger.warning("Pipeline preprocessing non trovata")
        st.warning("Prima esegui la sezione 'Data Quality & Preprocessing' per creare la pipeline")
        st.stop()
    
    pipeline = st.session_state['preprocessing_pipeline']
    X_train, X_test, y_train, y_test = st.session_state['prepared_data']
    logger.debug(f"Dati preparati caricati. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # ----------------10. Training Configuration Display
    logger.info("Visualizzazione configurazione training")
    st.subheader("Configurazione Training")
    
    training_config = TrainingPipelineManager(training_mode).config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Test Size:** {training_config['test_size']}")
        st.info(f"**CV Folds:** {training_config['cv_folds']}")
    
    with col2:
        st.info(f"**Hyperparameter Tuning:** {'✅' if use_hyperparameter_tuning else '❌'}")
        st.info(f"**Cross Validation:** {'✅' if use_cross_validation else '❌'}")
    
    with col3:
        st.info(f"**Modelli Selezionati:** {len(selected_models)}")
        st.info(f"**Salvataggio:** {'✅' if save_models else '❌'}")
    
    # ----------------11. Training Execution
    logger.info("Sezione esecuzione training")
    st.subheader("Esecuzione Training")
    
    if st.button("Avvia Training Completo", type="primary"):
        if not selected_models:
            logger.error("Nessun modello selezionato")
            st.error("Seleziona almeno un modello dalla sidebar")
            st.stop()
        
        logger.info(f"Avvio training con {len(selected_models)} modelli")
        
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
        
        # Imposta dati già splittati direttamente
        trainer.X_train = X_train
        trainer.y_train = y_train
        trainer.X_test = X_test
        trainer.y_test = y_test
        
        # Applica preprocessing
        with st.spinner("Applicazione preprocessing..."):
            logger.debug("Applicazione preprocessing ai dati")
            X_train_processed = pipeline.fit_transform(trainer.X_train)
            X_test_processed = pipeline.transform(trainer.X_test)
            
            # Aggiorna trainer con dati processati
            trainer.X_train = pd.DataFrame(X_train_processed) if hasattr(X_train_processed, 'shape') else X_train_processed
            trainer.X_test = pd.DataFrame(X_test_processed) if hasattr(X_test_processed, 'shape') else X_test_processed
            logger.debug(f"Preprocessing applicato. Train shape: {trainer.X_train.shape}, Test shape: {trainer.X_test.shape}")
        
        training_results = {}
        evaluation_results = {}
        cv_results = {}
        
        # Training loop
        for i, model_type in enumerate(selected_models):
            update_progress(i / len(selected_models), f"Training {ML_MODELS[model_type]['name']}...")
            logger.info(f"Training modello: {model_type}")
            
            try:
                # Training
                result = trainer.train_single_model(model_type)
                training_results[model_type] = result
                logger.debug(f"Training {model_type} completato")
                
                # Cross validation se richiesto
                if use_cross_validation:
                    logger.debug(f"Cross validation {model_type}")
                    cv_result = trainer.cross_validate_model(model_type)
                    cv_results[model_type] = cv_result
                
                # Evaluation
                logger.debug(f"Evaluation {model_type}")
                model_dict = {model_type: {'model': result['model'], 'preprocessor': None}}
                evaluator = ModelEvaluator(model_dict, trainer.X_test, trainer.y_test)
                eval_result = evaluator.evaluate_single_model(model_type, result)
                evaluation_results[model_type] = eval_result
                
            except Exception as e:
                logger.error(f"Errore nel training di {model_type}: {str(e)}")
                st.error(f"Errore nel training di {model_type}: {str(e)}")
                continue
        
        update_progress(1.0, "Training completato!")
        logger.info("Training completato per tutti i modelli")
        
        # Salva risultati in session state
        st.session_state['training_results'] = training_results
        st.session_state['evaluation_results'] = evaluation_results
        st.session_state['cv_results'] = cv_results
        st.session_state['trained_models'] = trainer.trained_models
        logger.debug("Risultati training salvati in session state")
        
        # ----------------12. Training Results Display
        st.success("Training completato con successo!")
        
        # Summary veloce
        comparison = ModelComparison(evaluation_results)
        best_model = comparison.find_best_model('f1')
        
        if best_model:
            logger.info(f"Miglior modello: {best_model['model_name']} (F1: {best_model['score']:.3f})")
            st.success(f"**Miglior Modello:** {best_model['model_name']} (F1: {best_model['score']:.3f})")
        
        # Tabella risultati
        logger.debug("Creazione tabella risultati")
        st.subheader("Risultati Training")
        results_table = comparison.create_comparison_table()
        st.dataframe(results_table, use_container_width=True)
        
        # Visualizzazione training times
        if training_results:
            logger.debug("Creazione visualizzazioni training")
            train_viz = TrainingVisualizer()
            fig_times = train_viz.create_training_progress_chart(training_results)
            st.plotly_chart(fig_times, use_container_width=True)
        
        # Cross validation results
        if cv_results:
            logger.debug("Creazione visualizzazioni cross validation")
            fig_cv = train_viz.create_cross_validation_chart(cv_results)
            st.plotly_chart(fig_cv, use_container_width=True)

# ----------------13. Model Evaluation
elif ml_section == "Model Evaluation":
    logger.info("Avvio sezione Model Evaluation")
    st.header("3. Valutazione Dettagliata Modelli")
    
    # Verifica prerequisiti
    if 'evaluation_results' not in st.session_state:
        logger.warning("Risultati evaluation non trovati")
        st.warning("Prima esegui il training dei modelli")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Risultati evaluation caricati per {len(evaluation_results)} modelli")
    
    # ----------------14. Performance Overview
    logger.info("Creazione performance overview")
    st.subheader("Performance Overview")
    
    # Metrics radar chart
    logger.debug("Creazione radar chart metriche")
    perf_viz = PerformanceVisualizer()
    fig_radar = perf_viz.create_metrics_comparison_radar(evaluation_results)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance heatmap
    logger.debug("Creazione heatmap performance")
    fig_heatmap = perf_viz.create_performance_heatmap(evaluation_results)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ----------------15. Detailed Metrics Analysis
    logger.info("Analisi metriche dettagliate")
    st.subheader("Analisi Metriche Dettagliate")
    
    # Selettore modello per analisi dettagliata
    selected_model_detail = st.selectbox(
        "Seleziona modello per analisi dettagliata:",
        list(evaluation_results.keys()),
        format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
    )
    logger.debug(f"Modello selezionato per dettagli: {selected_model_detail}")
    
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
        
        # Confusion Matrix dettagliata
        if 'confusion_matrix' in model_results:
            logger.debug("Creazione confusion matrix dettagliata")
            cm_viz = ConfusionMatrixVisualizer()
            fig_cm = cm_viz.create_confusion_matrix_detailed(
                model_results['confusion_matrix'], 
                selected_model_detail
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ----------------16. Model Comparison
elif ml_section == "Model Comparison":
    logger.info("Avvio sezione Model Comparison")
    st.header("4. Confronto Approfondito Modelli")
    
    if 'evaluation_results' not in st.session_state:
        logger.warning("Risultati evaluation non trovati per comparison")
        st.warning("Prima esegui il training dei modelli")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Confronto tra {len(evaluation_results)} modelli")
    
    # ----------------17. ROC Curves Comparison
    logger.info("Creazione curve ROC")
    st.subheader("Curve ROC")
    
    if 'trained_models' in st.session_state:
        logger.debug("Calcolo probabilità per curve ROC")
        # Calcola probabilità per ROC
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
                    logger.debug(f"Probabilità calcolate per {model_name}")
                except:
                    probabilities[model_name] = None
                    logger.debug(f"Errore calcolo probabilità per {model_name}")
        
        curve_viz = CurveVisualizer()
        fig_roc = curve_viz.create_roc_curves_comparison(evaluation_results, y_test, probabilities)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Precision-Recall curves
        logger.debug("Creazione curve Precision-Recall")
        fig_pr = curve_viz.create_precision_recall_curves(evaluation_results, y_test, probabilities)
        st.plotly_chart(fig_pr, use_container_width=True)
    
    # ----------------18. Statistical Significance Tests
    logger.info("Test significatività statistica")
    st.subheader("Test Significatività Statistica")
    
    if len(evaluation_results) >= 2:
        logger.debug("Setup selettori per confronto statistico")
        # Selettori per confronto
        col1, col2 = st.columns(2)
        
        with col1:
            model1 = st.selectbox(
                "Modello 1:",
                list(evaluation_results.keys()),
                format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
            )
        
        with col2:
            model2 = st.selectbox(
                "Modello 2:",
                [m for m in evaluation_results.keys() if m != model1],
                format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
            )
        
        logger.debug(f"Modelli per confronto statistico: {model1} vs {model2}")
        
        if st.button("Esegui Test Statistici"):
            logger.info(f"Esecuzione test statistici: {model1} vs {model2}")
            # McNemar test (richiede predizioni)
            if 'trained_models' in st.session_state:
                y_test = st.session_state['prepared_data'][3]
                X_test_processed = pipeline.transform(st.session_state['prepared_data'][1])
                
                pred1 = st.session_state['trained_models'][model1]['model'].predict(X_test_processed)
                pred2 = st.session_state['trained_models'][model2]['model'].predict(X_test_processed)
                
                mcnemar_result = StatisticalTests.mcnemar_test(y_test, pred1, pred2)
                logger.debug(f"Risultato McNemar test: statistic={mcnemar_result['statistic']:.3f}, p-value={mcnemar_result['p_value']:.4f}")
                
                st.write("**Test di McNemar:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Statistica", f"{mcnemar_result['statistic']:.3f}")
                
                with col2:
                    st.metric("P-value", f"{mcnemar_result['p_value']:.4f}")
                
                with col3:
                    significance = "Significativo" if mcnemar_result['significant'] else "Non Significativo"
                    st.metric("Risultato", significance)
    else:
        logger.debug("Numero insufficiente di modelli per test statistici")
    
    # ----------------19. Model Rankings
    logger.info("Creazione ranking modelli")
    st.subheader("Classifiche Modelli")
    
    metrics_for_ranking = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for metric in metrics_for_ranking:
        with st.expander(f"Ranking per {metric.title()}"):
            logger.debug(f"Creazione ranking per metrica: {metric}")
            fig_ranking = perf_viz.create_model_ranking_chart(evaluation_results, metric)
            st.plotly_chart(fig_ranking, use_container_width=True)

# ----------------20. Feature Analysis
elif ml_section == "Feature Analysis":
    logger.info("Avvio sezione Feature Analysis")
    st.header("5. Analisi Feature Importance")
    
    if 'trained_models' not in st.session_state:
        logger.warning("Modelli addestrati non trovati per feature analysis")
        st.warning("Prima esegui il training dei modelli")
        st.stop()
    
    # ----------------21. Feature Importance per modello
    logger.info("Analisi feature importance per modello")
    st.subheader("Feature Importance per Modello")
    
    models_with_importance = ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier']
    available_models = [m for m in st.session_state['trained_models'].keys() if m in models_with_importance]
    logger.debug(f"Modelli con feature importance disponibili: {available_models}")
    
    if not available_models:
        logger.warning("Nessun modello con feature importance disponibile")
        st.warning("Nessun modello con feature importance disponibile")
    else:
        selected_model_fi = st.selectbox(
            "Seleziona modello per feature importance:",
            available_models,
            format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
        )
        logger.debug(f"Modello selezionato per feature importance: {selected_model_fi}")
        
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
    
    # ----------------22. Confronto Feature Importance
    if len(available_models) > 1:
        logger.info("Confronto feature importance tra modelli")
        st.subheader("Confronto Feature Importance")
        
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
        
        logger.debug(f"Dati importanza per confronto: {len(importance_data)} modelli")
        
        if importance_data:
            logger.debug("Creazione grafico confronto feature importance")
            fig_fi_comp = fi_viz.create_feature_importance_comparison(importance_data)
            st.plotly_chart(fig_fi_comp, use_container_width=True)

# ----------------23. Predictions & Deployment
elif ml_section == "Predictions & Deployment":
    logger.info("Avvio sezione Predictions & Deployment")
    st.header("6. Predizioni e Deploy")
    
    if 'trained_models' not in st.session_state:
        logger.warning("Modelli addestrati non trovati per predizioni")
        st.warning("Prima esegui il training dei modelli")
        st.stop()
    
    # ----------------24. Single Prediction Interface
    logger.info("Setup interfaccia predizione singola")
    st.subheader("Predizione Singola")
    
    st.write("Inserisci i dati di un passeggero per predire la sopravvivenza:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_pclass = st.selectbox("Classe", [1, 2, 3], format_func=lambda x: f"{x}ª Classe")
        new_sex = st.selectbox("Sesso", ["male", "female"], format_func=lambda x: "Uomo" if x == "male" else "Donna")
        new_age = st.number_input("Età", min_value=0, max_value=100, value=30)
    
    with col2:
        new_sibsp = st.number_input("Fratelli/Coniugi", min_value=0, max_value=10, value=0)
        new_parch = st.number_input("Genitori/Figli", min_value=0, max_value=10, value=0)
        new_fare = st.number_input("Prezzo Biglietto", min_value=0.0, max_value=500.0, value=50.0)
    
    with col3:
        new_embarked = st.selectbox("Porto Imbarco", ["S", "C", "Q"], 
                                   format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
        
        # Info famiglia calcolate
        family_size = new_sibsp + new_parch + 1
        is_alone = "Sì" if family_size == 1 else "No"
        st.info(f"**Famiglia:** {family_size} membri")
        st.info(f"**Solo:** {is_alone}")
    
    logger.debug(f"Dati input: classe={new_pclass}, sesso={new_sex}, età={new_age}, famiglia={family_size}")
    
    # ----------------25. Esegui Predizione
    if st.button("Predici Sopravvivenza", type="primary"):
        logger.info("Esecuzione predizione singola")
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
        logger.debug("Applicazione preprocessing a input singolo")
        pipeline = st.session_state['preprocessing_pipeline']
        input_processed = pipeline.transform(input_data)
        
        # Predizioni da tutti i modelli
        logger.debug("Esecuzione predizioni da tutti i modelli")
        predictions = {}
        probabilities = {}
        
        for model_name, model_data in st.session_state['trained_models'].items():
            model = model_data['model']
            
            pred = model.predict(input_processed)[0]
            predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_processed)[0][1]
                probabilities[model_name] = proba
                logger.debug(f"Predizione {model_name}: {pred}, prob: {proba:.3f}")
            else:
                probabilities[model_name] = None
                logger.debug(f"Predizione {model_name}: {pred}")
        
        # ----------------26. Visualizza Risultati
        logger.info("Visualizzazione risultati predizione")
        st.subheader("Risultati Predizione")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predizioni per Modello:**")
            for model_name, pred in predictions.items():
                model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                prob = probabilities[model_name]
                
                if pred == 1:
                    st.success(f"**{model_display_name}:** SOPRAVVIVE")
                else:
                    st.error(f"**{model_display_name}:** NON SOPRAVVIVE")
                
                if prob is not None:
                    st.write(f"Probabilità: {prob:.2%}")
        
        with col2:
            # Consensus predizione
            survival_votes = sum(predictions.values())
            total_votes = len(predictions)
            consensus_prob = survival_votes / total_votes
            logger.debug(f"Consensus: {survival_votes}/{total_votes} = {consensus_prob:.3f}")
            
            st.write("**Consensus Predizione:**")
            if consensus_prob > 0.5:
                st.success(f"**CONSENSUS: SOPRAVVIVE**")
                st.write(f"Accordo: {survival_votes}/{total_votes} modelli")
            else:
                st.error(f"**CONSENSUS: NON SOPRAVVIVE**")
                st.write(f"Accordo: {total_votes - survival_votes}/{total_votes} modelli")
            
            confidence = max(consensus_prob, 1-consensus_prob)
            st.metric("Confidence Level", f"{confidence:.1%}")
        
        # Grafico probabilità
        if any(prob is not None for prob in probabilities.values()):
            logger.debug("Creazione grafico probabilità predizioni")
            prob_data = []
            for model_name, prob in probabilities.items():
                if prob is not None:
                    model_display_name = ML_MODELS.get(model_name, {}).get('name', model_name)
                    prob_data.append({
                        'Model': model_display_name,
                        'Probability': prob,
                        'Prediction': 'Sopravvive' if prob > 0.5 else 'Non Sopravvive'
                    })
            
            if prob_data:
                prob_df = pd.DataFrame(prob_data)
                pred_viz = PredictionVisualizer()
                fig_pred = pred_viz.create_prediction_confidence_chart(prob_df)
                st.plotly_chart(fig_pred, use_container_width=True)
    
    # ----------------27. Batch Predictions
    logger.info("Setup predizioni batch")
    st.subheader("Predizioni Batch")
    
    uploaded_file = st.file_uploader(
        "Carica file CSV per predizioni batch:",
        type=['csv'],
        help="Il file deve contenere le stesse colonne del dataset di training"
    )
    
    if uploaded_file is not None:
        logger.debug("File CSV caricato per predizioni batch")
        batch_data = pd.read_csv(uploaded_file)
        st.write("**Preview dati caricati:**")
        st.dataframe(batch_data.head(), use_container_width=True)
        logger.debug(f"Dati batch caricati. Shape: {batch_data.shape}")
        
        if st.button("Esegui Predizioni Batch"):
            logger.info("Esecuzione predizioni batch")
            try:
                # Preprocessing
                logger.debug("Preprocessing dati batch")
                pipeline = st.session_state['preprocessing_pipeline']
                batch_processed = pipeline.transform(batch_data)
                
                # Predizioni
                batch_results = batch_data.copy()
                
                # Usa il miglior modello per batch predictions
                evaluation_results = st.session_state['evaluation_results']
                comparison = ModelComparison(evaluation_results)
                best_model_info = comparison.find_best_model('f1')
                
                if best_model_info:
                    logger.info(f"Usando miglior modello per batch: {best_model_info['model_name']}")
                    best_model_name = best_model_info['model_type']
                    best_model = st.session_state['trained_models'][best_model_name]['model']
                    
                    predictions = best_model.predict(batch_processed)
                    batch_results['Predicted_Survival'] = predictions
                    batch_results['Predicted_Survival_Text'] = batch_results['Predicted_Survival'].map({0: 'Non Sopravvive', 1: 'Sopravvive'})
                    
                    if hasattr(best_model, 'predict_proba'):
                        probabilities = best_model.predict_proba(batch_processed)[:, 1]
                        batch_results['Survival_Probability'] = probabilities
                    
                    logger.debug(f"Predizioni batch completate: {len(predictions)} predizioni")
                    st.success(f"Predizioni completate usando {best_model_info['model_name']}")
                    st.dataframe(batch_results, use_container_width=True)
                    
                    # Download risultati
                    csv = batch_results.to_csv(index=False)
                    st.download_button(
                        label="Scarica Risultati CSV",
                        data=csv,
                        file_name=f"titanic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                
            except Exception as e:
                logger.error(f"Errore nelle predizioni batch: {str(e)}")
                st.error(f"Errore nelle predizioni batch: {str(e)}")
    
    # ----------------28. Model Deployment Info
    logger.info("Sezione informazioni deployment")
    st.subheader("Informazioni Deployment")
    
    with st.expander("Guida Deployment", expanded=False):
        st.markdown("""
        **Opzioni per deployment del modello:**
        
        1. **API REST**: Usa FastAPI o Flask per creare endpoint
        2. **Streamlit Cloud**: Deploy diretto di questa app
        3. **Docker Container**: Containerizza l'applicazione
        4. **Cloud Services**: AWS SageMaker, Azure ML, Google AI Platform
        
        **File necessari per deployment:**
        - Modello addestrato (pickle/joblib)
        - Pipeline preprocessing
        - Requirements.txt
        - Dockerfile (opzionale)
        
        **Considerazioni produzione:**
        - Monitoring performance modelli
        - A/B testing tra modelli
        - Retraining automatico
        - Data drift detection
        """)
    
    # Model saving
    if save_models:
        logger.info("Sezione salvataggio modelli")
        st.subheader("Salvataggio Modelli")
        
        if st.button("Salva Tutti i Modelli"):
            logger.info("Avvio salvataggio tutti i modelli")
            saved_models = []
            
            for model_name, model_data in st.session_state['trained_models'].items():
                try:
                    logger.debug(f"Salvataggio modello: {model_name}")
                    # Salva modello
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
                    logger.error(f"Errore salvando {model_name}: {str(e)}")
                    st.error(f"Errore salvando {model_name}: {str(e)}")
            
            if saved_models:
                logger.info(f"Modelli salvati con successo: {len(saved_models)}")
                st.success("Modelli salvati con successo!")
                for saved in saved_models:
                    st.write(f"✅ {saved}")

# ----------------29. Model Reports
elif ml_section == "Model Reports":
    logger.info("Avvio sezione Model Reports")
    st.header("7. Report Completi Modelli")
    
    if 'evaluation_results' not in st.session_state:
        logger.warning("Risultati evaluation non trovati per reports")
        st.warning("Prima esegui il training e valutazione dei modelli")
        st.stop()
    
    evaluation_results = st.session_state['evaluation_results']
    logger.debug(f"Report per {len(evaluation_results)} modelli")
    
    # ----------------30. Executive Summary
    logger.info("Creazione executive summary")
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
            logger.debug(f"Miglior F1: {best_f1['model_name']} ({best_f1['score']:.3f})")
    
    # ----------------31. Comprehensive Visualizations
    logger.info("Creazione visualizzazioni complete")
    st.subheader("Visualizzazioni Complete")
    
    # Crea report visualizzazioni complete
    training_results = st.session_state.get('training_results')
    logger.debug("Creazione comprehensive visualization report")
    comprehensive_viz = create_comprehensive_model_report_visualization(
        evaluation_results, training_results
    )
    
    for i, fig in enumerate(comprehensive_viz):
        if fig is not None:
            logger.debug(f"Visualizzazione figura {i}")
            st.plotly_chart(fig, use_container_width=True)
    
    # ----------------32. Error Analysis
    logger.info("Analisi errori avanzata")
    st.subheader("Analisi Errori Avanzata")
    
    if 'trained_models' in st.session_state:
        # Selettore modello per error analysis
        selected_model_error = st.selectbox(
            "Seleziona modello per analisi errori:",
            list(st.session_state['trained_models'].keys()),
            format_func=lambda x: ML_MODELS.get(x, {}).get('name', x)
        )
        logger.debug(f"Modello selezionato per error analysis: {selected_model_error}")
        
        # Calcola predizioni per error analysis
        X_test = st.session_state['prepared_data'][1]
        y_test = st.session_state['prepared_data'][3]
        pipeline = st.session_state['preprocessing_pipeline']
        
        logger.debug("Calcolo predizioni per error analysis")
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
                logger.debug(f"Campioni difficili trovati: {difficult_samples['difficult_samples']} ({difficult_samples['percentage']:.1f}%)")
                st.write(f"**Campioni Difficili da Classificare:** {difficult_samples['difficult_samples']} ({difficult_samples['percentage']:.1f}%)")
    
    # ----------------33. Model Comparison Table
    logger.info("Creazione tabella confronto completa")
    st.subheader("Tabella Confronto Completa")
    
    detailed_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', 'matthews_corrcoef']
    comparison_table = comparison.create_comparison_table(detailed_metrics)
    st.dataframe(comparison_table, use_container_width=True)
    
    # ----------------34. Export Reports
    logger.info("Sezione export reports")
    st.subheader("Export Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Genera Report JSON"):
            logger.info("Generazione report JSON")
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
                label="Scarica Report JSON",
                data=json_str,
                file_name=f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )
    
    with col2:
        if st.button("Genera Report CSV"):
            logger.info("Generazione report CSV")
            csv_report = comparison_table.to_csv(index=False)
            
            st.download_button(
                label="Scarica Report CSV",
                data=csv_report,
                file_name=f"ml_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

# ----------------35. Footer e Summary Generale
logger.info("Creazione footer e summary generale")
st.markdown("---")

# Summary stato corrente
if 'trained_models' in st.session_state:
    n_models = len(st.session_state['trained_models'])
    logger.debug(f"Pipeline attiva con {n_models} modelli")
    st.success(f"**Pipeline ML Attiva:** {n_models} modelli addestrati con configurazione {preprocessing_config}")
    
    if 'evaluation_results' in st.session_state:
        comparison = ModelComparison(st.session_state['evaluation_results'])
        best_model = comparison.find_best_model('f1')
        if best_model:
            logger.info(f"Miglior modello globale: {best_model['model_name']} (F1: {best_model['score']:.3f})")
            st.info(f"**Miglior Modello Globale:** {best_model['model_name']} (F1: {best_model['score']:.3f})")

# Note metodologiche
with st.expander("Note Metodologiche e Architettura", expanded=False):
    st.markdown("""
    **Architettura Modulare Implementata:**
    
    **Data & Preprocessing:**
    - `ml_preprocessing.py`: Pipeline intelligente con feature engineering automatico
    - `DataQualityChecker`: Analisi qualità dati e raccomandazioni
    - Configurazioni: Minimal, Standard, Advanced
    
    **Models & Training:**
    - `ml_models.py`: Factory pattern per modelli con configurazioni ottimizzate
    - `model_trainer.py`: Pipeline training completa con CV e hyperparameter tuning
    - Supporto ensemble e persistence
    
    **Evaluation & Analysis:**
    - `model_evaluator.py`: Metriche avanzate, test statistici, error analysis
    - `ModelComparison`: Confronti automatici e ranking
    - Monitoring performance e drift detection
    
    **Visualization:**
    - `ml_charts.py`: Visualizzazioni professionali per ogni aspetto ML
    - Dashboard interattive e report automatici
    - Export multi-formato
    
    **Features Avanzate:**
    - **Preprocessing intelligente** con raccomandazioni automatiche
    - **Training configurabile** (Quick/Comprehensive/Deep)
    - **Evaluation completa** con 15+ metriche
    - **Statistical significance** testing
    - **Feature importance** analysis
    - **Batch predictions** e deployment ready
    - **Error analysis** avanzata
    - **Report automatici** JSON/CSV
    
    **Limitazioni:**
    - Dataset storicamente limitato (1912)
    - Bias nei dati originali
    - Generalizzazione a contesti moderni limitata
    
    **Best Practices Implementate:**
    - Modular architecture con separation of concerns
    - Error handling robusto
    - Caching intelligente per performance
    - Validation pipeline completa
    - Reproducibility con random states
    - Scalabilità per nuovi modelli/metriche
    """)

st.markdown("""
**Machine Learning Pipeline completata**

Questa implementazione rappresenta una pipeline ML completa e production-ready per l'analisi del Titanic,
con architettura modulare, preprocessing intelligente, training automatizzato e evaluation approfondita.

**Prossimi Step:** Considera deployment, monitoring in produzione e retraining automatico per un sistema ML completo.
""")

logger.info(f"Pagina {__name__} completata con successo")