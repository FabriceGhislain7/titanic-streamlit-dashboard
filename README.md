# Titanic Survival Prediction Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)

Una moderna applicazione web per l'analisi completa e la predizione della sopravvivenza dei passeggeri del Titanic, costruita con **architettura modulare** e **pipeline di Machine Learning production-ready**.

## Links del Progetto

- **🌐 Live Demo**: [https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
- **💻 GitHub Repository**: [https://github.com/FabriceGhislain7/titanic-streamlit-dashboard](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)
- **📖 Documentation**: [In Progress]

## Overview del Progetto

Questa applicazione rappresenta un'implementazione completa di data science end-to-end, dalla **Data Quality Analysis** al **Model Deployment**, passando per **Feature Engineering** avanzato, **Statistical Testing** e **Error Analysis** dettagliata.

### Obiettivi

- **Analisi Esplorativa**: Insight approfonditi sui fattori di sopravvivenza
- **Feature Engineering**: Creazione intelligente di variabili predittive
- **ML Pipeline**: Training, validation e comparison automatizzati
- **Production Ready**: Deployment, monitoring e batch predictions
- **User Experience**: Dashboard interattiva e intuitiva

## Quick Start

### Prerequisiti

- **Python 3.8+**
- **Git**
- **Virtual Environment** (raccomandato)

### Installazione

1. **Clona il repository:**
   ### Analisi Strutturale del Progetto

Per osservare la struttura del progetto senza l'ambiente virtuale, esegui il comando PowerShell:

```powershell
Get-ChildItem -Recurse | Where-Object { $_.FullName -notmatch "venv" } | Select-Object FullName
```

## Tecnologie e Stack

### Core Framework
- **Streamlit**: Modern web framework for data applications
- **Python 3.8+**: Primary programming language

### Data Science Stack
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing foundation
- **Scikit-learn**: Machine learning library
- **SciPy**: Scientific computing and statistical tests

### Visualization Libraries
- **Plotly**: Interactive web-based plotting
- **Matplotlib**: Static plotting and publication-quality figures
- **Seaborn**: Statistical data visualization

### Machine Learning Pipeline
- **Feature Engineering**: Automated feature creation and selection
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and statistical testing
- **Cross-Validation**: Robust model validation strategies

### Development Tools
- **Logging**: Comprehensive application logging
- **Testing**: Unit test infrastructure
- **Configuration**: Centralized config management
- **Documentation**: Inline docs and notebooks

## Funzionalità Principali

### 1. Data Overview
- **Dataset Summary**: Comprehensive statistical overview
- **Data Quality Assessment**: Missing values, duplicates, outliers
- **Interactive Filtering**: Dynamic data exploration
- **Export Capabilities**: Cleaned data download

### 2. Univariate Analysis
- **Distribution Analysis**: Histograms, box plots, density plots
- **Statistical Testing**: Normality tests, descriptive statistics
- **Outlier Detection**: IQR method with visualization
- **Categorical Analysis**: Frequency tables and bar charts

### 3. Bivariate Analysis
- **Correlation Analysis**: Pearson and Spearman correlation matrices
- **Survival Factor Analysis**: Impact of individual variables
- **Cross-tabulation**: Detailed contingency tables
- **Statistical Significance**: Chi-square and other tests

### 4. Advanced Analytics
- **Feature Engineering**: Title extraction, family size, fare per person
- **Pattern Mining**: Rare but significant combinations
- **Segmentation Analysis**: Passenger clustering and profiling
- **Correlation Deep-dive**: Multi-level relationship exploration

### 5. Machine Learning Pipeline
- **Preprocessing Pipeline**: Automated data preparation
- **Model Training**: 8+ algorithms with cross-validation
- **Hyperparameter Tuning**: Grid search and random search
- **Model Evaluation**: 15+ performance metrics
- **Model Comparison**: Statistical significance testing
- **Feature Importance**: Interpretability analysis
- **Prediction Interface**: Single and batch predictions
- **Model Deployment**: Export and persistence capabilities

## Metodologia di Sviluppo

### Data Science Workflow
1. **Exploratory Data Analysis (EDA)**
2. **Data Quality Assessment**
3. **Feature Engineering & Selection**
4. **Model Development & Training**
5. **Model Evaluation & Comparison**
6. **Model Interpretation & Deployment**

### Software Engineering Practices
- **Modular Architecture**: Separation of concerns
- **Configuration Management**: Centralized settings
- **Comprehensive Logging**: Full application tracing
- **Error Handling**: Robust exception management
- **Code Documentation**: Inline and external docs
- **Testing Infrastructure**: Unit test foundation

## Metriche e Performance

### Machine Learning Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Probabilistic Metrics**: ROC-AUC, Precision-Recall AUC
- **Advanced Metrics**: Matthews Correlation, Cohen's Kappa
- **Cross-Validation**: K-fold validation with confidence intervals

### Statistical Analysis
- **Correlation Analysis**: Pearson, Spearman, Kendall
- **Significance Testing**: T-tests, Chi-square, McNemar
- **Distribution Analysis**: Kolmogorov-Smirnov, Shapiro-Wilk
- **Effect Size**: Cohen's d, Cramér's V

## Configurazione e Personalizzazione

### File di Configurazione

- **src/config.py**: Configurazioni principali, palette colori, mappature
- **.streamlit/config.toml**: Configurazioni Streamlit UI/UX
- **requirements.txt**: Dipendenze production
- **logs/**: Directory logging automatico

### Personalizzazione

1. **Modifiche Dataset**: Sostituisci `src/data/data_titanic.csv`
2. **Nuovi Modelli**: Estendi `src/models/ml_models.py`
3. **Custom Charts**: Aggiungi in `src/components/`
4. **Styling**: Modifica `assets/styles/main.css`

## Testing e Quality Assurance

### Test Suite
```bash
# Esegui tutti i test
python -m pytest tests/

# Test specifici
python -m pytest tests/test_data_processing.py
python -m pytest tests/test_models.py
```

### Code Quality
- **Logging**: Comprehensive logging in tutti i moduli
- **Error Handling**: Try-catch robusto
- **Type Hints**: Documentazione tipi (in sviluppo)
- **Docstrings**: Documentazione inline

## Performance e Scalabilità

### Ottimizzazioni Implementate
- **Caching**: `@st.cache_data` per operazioni costose
- **Lazy Loading**: Caricamento dati on-demand
- **Memory Management**: Gestione efficiente memoria
- **Preprocessing Pipeline**: Ottimizzazione scikit-learn

### Limitazioni Attuali
- **Dataset Size**: Ottimizzato per dataset ~1000 righe
- **Memory Usage**: ~100MB per sessione completa
- **Concurrent Users**: Testato per uso single-user

## Deployment e Produzione

### Deployment Options

1. **Streamlit Cloud**:
   ```bash
   # Push to GitHub e collega Streamlit Cloud
   git push origin main
   ```

2. **Docker Container**:
   ```dockerfile
   FROM python:3.8-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

3. **Local Production**:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

### Production Considerations
- **Environment Variables**: Configurazioni sensibili
- **Monitoring**: Log analysis e performance tracking
- **Backup Strategy**: Dataset e model versioning
- **Security**: Input validation e sanitization

## Sviluppo e Contributi

### Development Setup
```bash
# Clone del repository
git clone https://github.com/FabriceGhislain7/titanic-streamlit-dashboard.git
cd titanic-streamlit-dashboard

# Setup development environment
pip install -r requirements_complesso.txt

# Pre-commit hooks (opzionale)
pre-commit install
```

### Project Evolution
- **Origine**: Migrazione da Jupyter Notebook (`docs/data_analysis.ipynb`)
- **Architettura**: Refactoring modulare per scalabilità
- **ML Pipeline**: Implementazione production-ready
- **UI/UX**: Dashboard professionale con Streamlit

### Roadmap Futuro
- **Database Integration**: PostgreSQL/MongoDB support
- **Real-time Predictions**: API REST endpoints
- **A/B Testing**: Model comparison framework
- **Advanced ML**: Deep Learning models integration
- **Monitoring**: MLOps e model drift detection

## Licenza e Credits

### Dataset
- **Source**: Kaggle Titanic Competition
- **License**: Public Domain
- **Purpose**: Educational e demonstration

### Codebase
- **GitHub**: [https://github.com/FabriceGhislain7/titanic-streamlit-dashboard](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)
- **Live Demo**: [https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
- **License**: MIT License
- **Author**: Fabrice Ghislain
- **Version**: 1.0.0

### Acknowledgments
- **Streamlit Community**: Framework e documentation
- **Scikit-learn**: ML library foundation
- **Plotly Team**: Interactive visualization tools

---

**Nota**: Questo progetto rappresenta un esempio completo di applicazione data science moderna, con particolare attenzione all'architettura scalabile, best practices di ML engineering e user experience professionale.bash
   git clone <repository-url>
   cd titanic-streamlit-dashboard
   ```

2. **Crea e attiva virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installa le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Tip**: Per generare requirements.txt da zero: `pip freeze > requirements.txt`

4. **Avvia l'applicazione:**
   ```bash
   streamlit run app.py
   ```

5. **Apri nel browser:** `http://localhost:8501`

## Architettura del Progetto

```
titanic-streamlit-dashboard/
├── app.py                       # Entry point principale
├── src/                         # Core business logic
│   ├── components/              # UI Components & Visualizations  
│   │   ├── advanced_charts.py   # Advanced analytics charts
│   │   ├── bivariate_charts.py  # Correlation & relationship plots
│   │   ├── charts.py            # Base charting functions
│   │   ├── metrics.py           # KPI and metrics components
│   │   ├── ml_charts.py         # ML-specific visualizations
│   │   ├── univariate_charts.py # Single variable analysis
│   │   └── __init__.py          
│   ├── data/                    # Dataset storage
│   │   └── data_titanic.csv     # Original Titanic dataset
│   ├── models/                  # ML Models & Evaluation
│   │   ├── ml_models.py         # Model factory & configurations
│   │   ├── model_evaluator.py   # Comprehensive evaluation suite
│   │   ├── model_trainer.py     # Training pipeline manager
│   │   └── __init__.py
│   ├── utils/                   # Utility functions
│   │   ├── data_loader.py       # Data loading utilities
│   │   ├── data_processor.py    # Data cleaning & preprocessing
│   │   ├── feature_engineering.py # Advanced feature creation
│   │   ├── helpers.py           # General helper functions
│   │   ├── log.py              # Logging configuration
│   │   ├── ml_preprocessing.py  # ML-specific preprocessing
│   │   ├── statistical_analysis.py # Statistical testing suite
│   │   └── __init__.py
│   └── config.py               # Global configuration & constants
├── pages/                      # Multi-page application
│   ├── 1_Data_Overview.py      # Dataset exploration & summary
│   ├── 2_Univariate_Analysis.py # Single variable deep-dive
│   ├── 3_Bivariate_Analysis.py # Correlation & relationship analysis
│   ├── 4_Advanced_Analytics.py # Feature engineering & patterns
│   └── 5_ML_Predictions.py     # Complete ML pipeline
├── assets/                     # Static resources
│   ├── styles/                 
│   │   └── main.css           # Custom styling
│   └── images/                # Image assets
├── tests/                     # Unit testing suite
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── __init__.py
├── docs/                      # Documentation & research
│   ├── data_analysis.ipynb    # Original Jupyter analysis
│   ├── machine_learning.ipynb # ML development notebook
│   └── README.md
├── logs/                      # Application logs
│   └── titanic_app.log       # Runtime logging
├── .streamlit/               # Streamlit configuration
│   ├── config.toml           # App configuration
│   └── secrets.toml          # Environment secrets
├── requirements.txt          # Production dependencies
├── requirements_complesso.txt # Development dependencies
└── README.md                # This documentation
```