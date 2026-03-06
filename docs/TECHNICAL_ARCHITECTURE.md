# Titanic Survival Prediction Dashboard

## Technical Documentation

This document provides detailed technical documentation for the **Titanic Survival Prediction Dashboard**, including architecture, data processing pipelines, statistical analysis, and machine learning implementation details.

The goal of this document is to preserve the **technical knowledge and development decisions** behind the project.

---

# System Architecture

The application follows a **modular architecture** separating UI, data processing, machine learning, and utilities.

```text
titanic-streamlit-dashboard
│
├── app.py
├── src
│   ├── components
│   ├── data
│   ├── models
│   ├── utils
│   └── config.py
│
├── pages
├── assets
├── tests
├── docs
```

## Architectural Principles

The project architecture is designed around:

* modular code organization
* separation of concerns
* reusable components
* scalable machine learning pipeline

Key design goals:

* maintainability
* reproducibility
* extensibility

---

# Data Processing Pipeline

The data pipeline includes several stages.

## Data Loading

Dataset is loaded from:

```
src/data/data_titanic.csv
```

The loader ensures:

* correct column types
* missing value detection
* dataset validation

Main module:

```
src/utils/data_loader.py
```

---

## Data Cleaning

Data cleaning operations include:

* missing value handling
* categorical encoding
* outlier identification
* data normalization

Main module:

```
src/utils/data_processor.py
```

---

# Feature Engineering

Advanced feature engineering techniques are applied to improve predictive power.

### Engineered Features

Examples include:

**Passenger Title**

Extracted from the passenger name to capture social status.

Example:

```
Mr, Mrs, Miss, Master
```

---

**Family Size**

Calculated as:

```
FamilySize = SibSp + Parch + 1
```

---

**Fare Per Passenger**

Normalized fare value.

```
FarePerPerson = Fare / FamilySize
```

---

These features improve model performance and help capture hidden patterns in the dataset.

Implementation:

```
src/utils/feature_engineering.py
```

---

# Statistical Analysis

The project includes a dedicated statistical analysis module.

Location:

```
src/utils/statistical_analysis.py
```

Implemented techniques include:

### Correlation Analysis

* Pearson correlation
* Spearman correlation
* Kendall correlation

### Hypothesis Testing

* Chi-square tests
* T-tests
* McNemar test

### Distribution Testing

* Shapiro-Wilk test
* Kolmogorov-Smirnov test

These analyses help validate relationships between variables and survival outcomes.

---

# Machine Learning Pipeline

The machine learning pipeline is implemented inside:

```
src/models
```

Key modules:

```
model_trainer.py
model_evaluator.py
ml_models.py
```

---

## Model Training

Multiple machine learning algorithms are trained and evaluated automatically.

Typical models include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine
* K-Nearest Neighbors

Cross-validation is used to ensure robust performance evaluation.

---

## Model Evaluation

Evaluation metrics include:

Classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score

Probabilistic metrics:

* ROC-AUC
* Precision-Recall AUC

Advanced metrics:

* Matthews Correlation Coefficient
* Cohen's Kappa

The evaluation module also supports:

* confusion matrix visualization
* model comparison
* statistical performance analysis

---

# Visualization System

The visualization system is implemented using modular components.

Location:

```
src/components
```

Visualization libraries used:

* Plotly
* Matplotlib
* Seaborn

Types of visualizations implemented:

* distribution plots
* correlation heatmaps
* feature importance charts
* survival analysis charts

---

# Application Interface

The application uses a **multi-page interface** implemented with Streamlit.

Pages include:

```
1_Data_Overview
2_Univariate_Analysis
3_Bivariate_Analysis
4_Advanced_Analytics
5_ML_Predictions
```

Each page focuses on a specific part of the data science workflow.

---

# Logging System

A structured logging system is implemented.

Location:

```
src/utils/log.py
```

The logging system supports:

* runtime diagnostics
* debugging support
* monitoring of ML pipeline execution

Logs are stored in:

```
logs/titanic_app.log
```

---

# Testing Infrastructure

Unit tests are implemented in:

```
tests/
```

Current tests include:

* data processing tests
* machine learning pipeline tests

Tests are executed using:

```bash
pytest
```

---

# Performance Optimization

Several optimizations are implemented to ensure efficient execution.

### Streamlit Caching

Expensive operations use:

```
@st.cache_data
```

This reduces recomputation and improves application responsiveness.

---

### Lazy Loading

Data and models are loaded only when required.

This minimizes memory usage and startup time.

---

# Known Limitations

Current limitations include:

* optimized for relatively small datasets (~1000 rows)
* single-user deployment model
* no external database integration

---

# Future Improvements

Planned improvements include:

* PostgreSQL integration
* REST API for model predictions
* model monitoring and MLOps pipeline
* model drift detection
* advanced machine learning models

---

# Purpose of this Document

This document exists to:

* preserve technical insights
* document architectural decisions
* support future development

It complements the main repository README, which provides a **high-level project overview**.

