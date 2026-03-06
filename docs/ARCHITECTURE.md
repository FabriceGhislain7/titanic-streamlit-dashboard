# Titanic Survival Prediction Dashboard

## System Architecture Documentation

This document describes the internal architecture of the **Titanic Survival Prediction Dashboard**, including module responsibilities, system structure, and interactions between components.

The goal is to provide a **clear technical understanding of the system design** for developers and contributors.

---

# Architectural Overview

The application follows a **modular layered architecture** designed for maintainability and scalability.

Main layers:

```
User Interface Layer
        │
        ▼
Application Layer
        │
        ▼
Data Processing Layer
        │
        ▼
Machine Learning Layer
        │
        ▼
Data Layer
```

Each layer is isolated and communicates through well-defined interfaces.

---

# High-Level System Structure

```
titanic-streamlit-dashboard
│
├── app.py
│
├── src
│   ├── components
│   ├── data
│   ├── models
│   ├── utils
│   └── config.py
│
├── pages
│
├── assets
│
├── tests
│
├── docs
│
└── requirements.txt
```

The architecture separates:

* user interface
* data processing
* machine learning logic
* visualization
* configuration

This structure improves **code organization and maintainability**.

---

# Entry Point

```
app.py
```

This file acts as the **main application entry point**.

Responsibilities include:

* initializing the Streamlit application
* configuring the global environment
* loading shared resources
* routing between pages

The application runs through:

```bash
streamlit run app.py
```

---

# Source Code Architecture

All core application logic is contained inside:

```
src/
```

The `src` directory implements the **business logic of the application**.

---

# Components Layer

Location:

```
src/components/
```

This layer contains all **visualization and UI components** used by the Streamlit pages.

Modules include:

```
charts.py
advanced_charts.py
bivariate_charts.py
univariate_charts.py
metrics.py
ml_charts.py
```

Responsibilities:

* generate charts and visualizations
* display key metrics
* encapsulate plotting logic
* separate visualization from analysis logic

Visualization libraries used:

* Plotly
* Matplotlib
* Seaborn

---

# Data Layer

Location:

```
src/data/
```

This directory stores the dataset used by the application.

```
data_titanic.csv
```

The dataset represents passenger information from the Titanic disaster and is used for:

* exploratory analysis
* feature engineering
* model training

---

# Utilities Layer

Location:

```
src/utils/
```

This layer contains **data processing and utility modules** used throughout the system.

Modules include:

```
data_loader.py
data_processor.py
feature_engineering.py
helpers.py
log.py
ml_preprocessing.py
statistical_analysis.py
```

Responsibilities include:

* loading datasets
* data cleaning and preprocessing
* feature engineering
* statistical analysis
* logging configuration
* helper functions

---

# Data Loading

Module:

```
data_loader.py
```

Responsibilities:

* loading CSV datasets
* validating schema
* detecting missing values
* preparing the dataset for analysis

---

# Data Processing

Module:

```
data_processor.py
```

Responsibilities:

* handling missing values
* cleaning invalid data
* transforming categorical variables
* preparing structured datasets

---

# Feature Engineering

Module:

```
feature_engineering.py
```

This module generates additional predictive features.

Examples include:

Passenger Title extraction

```
Mr
Mrs
Miss
Master
```

Family Size

```
FamilySize = SibSp + Parch + 1
```

Fare normalization

```
FarePerPerson = Fare / FamilySize
```

These engineered features significantly improve model performance.

---

# Statistical Analysis

Module:

```
statistical_analysis.py
```

Implements statistical testing and exploratory analysis methods.

Supported methods include:

Correlation analysis

* Pearson
* Spearman
* Kendall

Hypothesis testing

* Chi-square test
* T-tests
* McNemar test

Distribution testing

* Shapiro-Wilk
* Kolmogorov-Smirnov

These techniques help identify relationships between variables and survival outcomes.

---

# Machine Learning Layer

Location:

```
src/models/
```

Modules:

```
ml_models.py
model_trainer.py
model_evaluator.py
```

This layer implements the **machine learning pipeline**.

---

# Model Factory

Module:

```
ml_models.py
```

This module defines and configures the machine learning models available in the system.

Typical models include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machines
* K-Nearest Neighbors

The factory approach allows easy addition of new models.

---

# Model Training Pipeline

Module:

```
model_trainer.py
```

Responsibilities:

* model training
* hyperparameter tuning
* cross-validation
* training pipeline orchestration

---

# Model Evaluation

Module:

```
model_evaluator.py
```

Responsibilities include:

* computing evaluation metrics
* generating confusion matrices
* model comparison
* statistical validation of model performance

Metrics implemented include:

Classification metrics

* Accuracy
* Precision
* Recall
* F1 Score

Advanced metrics

* ROC-AUC
* Matthews Correlation
* Cohen's Kappa

---

# Multi-Page Application Layer

Location:

```
pages/
```

The application is implemented as a **multi-page Streamlit dashboard**.

Pages include:

```
1_Data_Overview.py
2_Univariate_Analysis.py
3_Bivariate_Analysis.py
4_Advanced_Analytics.py
5_ML_Predictions.py
```

Each page represents a stage of the data science workflow.

---

# Asset Layer

Location:

```
assets/
```

Contains static resources such as:

```
styles/main.css
images/
```

Used for:

* UI customization
* application styling
* branding elements

---

# Logging System

Logging configuration is implemented in:

```
src/utils/log.py
```

Logs are stored in:

```
logs/titanic_app.log
```

The logging system supports:

* debugging
* monitoring
* error tracking

---

# Testing Infrastructure

Location:

```
tests/
```

Current test modules include:

```
test_data_processing.py
test_models.py
```

Tests validate:

* data processing logic
* machine learning pipeline

Tests are executed using:

```bash
pytest
```

---

# Configuration System

Global configuration is managed through:

```
src/config.py
```

This file centralizes:

* constants
* configuration parameters
* visualization settings
* feature mappings

This improves maintainability and consistency across modules.

---

# Performance Optimization

The application implements several optimizations.

### Streamlit caching

Expensive computations use:

```
@st.cache_data
```

This reduces repeated computations and improves responsiveness.

---

### Lazy loading

Models and datasets are loaded only when required.

This reduces startup time and memory usage.

---

# Architectural Principles

The project architecture follows several key principles:

**Separation of concerns**

Each module has a single responsibility.

**Modularity**

Code is organized into reusable modules.

**Scalability**

The structure allows integration of:

* new machine learning models
* additional datasets
* API services

**Maintainability**

Clear module boundaries simplify debugging and development.

---

# Future Architectural Improvements

Planned architectural improvements include:

* external database integration
* REST API service for predictions
* containerized deployment with Docker
* MLOps pipeline integration
* model monitoring and drift detection

