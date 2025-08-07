# Titanic Survival Prediction Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)

A modern web application for comprehensive analysis and prediction of Titanic passenger survival, built with **modular architecture** and **production-ready Machine Learning pipeline**.

## Project Links

- **🌐 Live Demo**: [https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
- **💻 GitHub Repository**: [https://github.com/FabriceGhislain7/titanic-streamlit-dashboard](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)
- **📖 Documentation**: [In Progress]

## Project Overview

This application represents a complete end-to-end data science implementation, from **Data Quality Analysis** to **Model Deployment**, through advanced **Feature Engineering**, **Statistical Testing**, and detailed **Error Analysis**.

### Objectives

- **Exploratory Analysis**: Deep insights into survival factors
- **Feature Engineering**: Intelligent creation of predictive variables
- **ML Pipeline**: Automated training, validation, and comparison
- **Production Ready**: Deployment, monitoring, and batch predictions
- **User Experience**: Interactive and intuitive dashboard

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Git**
- **Virtual Environment** (recommended)

### Installation

1. **Clone the repository:**
   ### Project Structural Analysis

To observe the project structure without the virtual environment, run the PowerShell command:

```powershell
Get-ChildItem -Recurse | Where-Object { $_.FullName -notmatch "venv" } | Select-Object FullName
```

```bash
   git clone <repository-url>
   cd titanic-streamlit-dashboard
```

2. **Create and activate virtual environment:**
   ```bash
      # Windows
      python -m venv venv
      .\venv\Scripts\activate
      
      # Linux/Mac
      python3 -m venv venv
      source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Tip**: To generate requirements.txt from scratch: `pip freeze > requirements.txt`

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:** `http://localhost:8501`

## Technologies and Stack

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

## Key Features

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

## Development Methodology

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

## Metrics and Performance

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

## Configuration and Customization

### Configuration Files

- **src/config.py**: Main configurations, color palettes, mappings
- **.streamlit/config.toml**: Streamlit UI/UX configurations
- **requirements.txt**: Production dependencies
- **logs/**: Automatic logging directory

### Customization

1. **Dataset Modifications**: Replace `src/data/data_titanic.csv`
2. **New Models**: Extend `src/models/ml_models.py`
3. **Custom Charts**: Add to `src/components/`
4. **Styling**: Modify `assets/styles/main.css`

## Testing and Quality Assurance

### Test Suite
```bash
# Run all tests
python -m pytest tests/

# Specific tests
python -m pytest tests/test_data_processing.py
python -m pytest tests/test_models.py
```

### Code Quality
- **Logging**: Comprehensive logging in all modules
- **Error Handling**: Robust try-catch
- **Type Hints**: Type documentation (in development)
- **Docstrings**: Inline documentation

## Performance and Scalability

### Implemented Optimizations
- **Caching**: `@st.cache_data` for expensive operations
- **Lazy Loading**: On-demand data loading
- **Memory Management**: Efficient memory handling
- **Preprocessing Pipeline**: Scikit-learn optimization

### Current Limitations
- **Dataset Size**: Optimized for ~1000 rows datasets
- **Memory Usage**: ~100MB per complete session
- **Concurrent Users**: Tested for single-user usage

## Deployment and Production

### Deployment Options

1. **Streamlit Cloud**:
   ```bash
      # Push to GitHub and connect Streamlit Cloud
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
- **Environment Variables**: Sensitive configurations
- **Monitoring**: Log analysis and performance tracking
- **Backup Strategy**: Dataset and model versioning
- **Security**: Input validation and sanitization

## Development and Contributions

### Development Setup
```bash
# Repository clone
git clone https://github.com/FabriceGhislain7/titanic-streamlit-dashboard.git
cd titanic-streamlit-dashboard

# Setup development environment
pip install -r requirements_complesso.txt

# Pre-commit hooks (optional)
pre-commit install
```

### Project Evolution
- **Origin**: Migration from Jupyter Notebook (`docs/data_analysis.ipynb`)
- **Architecture**: Modular refactoring for scalability
- **ML Pipeline**: Production-ready implementation
- **UI/UX**: Professional dashboard with Streamlit

### Future Roadmap
- **Database Integration**: PostgreSQL/MongoDB support
- **Real-time Predictions**: REST API endpoints
- **A/B Testing**: Model comparison framework
- **Advanced ML**: Deep Learning models integration
- **Monitoring**: MLOps and model drift detection

## Project Architecture

```
titanic-streamlit-dashboard/
├── app.py                       # Main entry point
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

## License and Credits

### Dataset
- **Source**: Kaggle Titanic Competition
- **License**: Public Domain
- **Purpose**: Educational and demonstration

### Codebase
- **GitHub**: [https://github.com/FabriceGhislain7/titanic-streamlit-dashboard](https://github.com/FabriceGhislain7/titanic-streamlit-dashboard)
- **Live Demo**: [https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/](https://titanic-app-dashboard-ogxxezhe82g8tggobo5l2n.streamlit.app/)
- **License**: MIT License
- **Author**: Fabrice Ghislain
- **Version**: 1.0.0

### Acknowledgments
- **Streamlit Community**: Framework and documentation
- **Scikit-learn**: ML library foundation
- **Plotly Team**: Interactive visualization tools

---

**Note**: This project represents a complete example of modern data science application, with particular attention to scalable architecture, ML engineering best practices, and professional user experience.