# System Architecture Diagram

```mermaid
flowchart TD

User[User Browser]

UI[Streamlit UI Layer]
Pages[Streamlit Pages]

Components[Visualization Components]

DataProcessing[Data Processing Layer]

FeatureEngineering[Feature Engineering]

StatisticalAnalysis[Statistical Analysis]

MLPipeline[Machine Learning Pipeline]

Models[ML Models]

Evaluation[Model Evaluation]

Dataset[(Titanic Dataset)]

User --> UI
UI --> Pages

Pages --> Components

Pages --> DataProcessing
DataProcessing --> FeatureEngineering
FeatureEngineering --> StatisticalAnalysis

StatisticalAnalysis --> MLPipeline
MLPipeline --> Models
Models --> Evaluation

Dataset --> DataProcessing
```

---

# Layered Architecture Diagram

This diagram shows the **architectural layers**.

```mermaid
flowchart TB

subgraph Presentation Layer
UI[Streamlit UI]
Pages[Dashboard Pages]
Charts[Visualization Components]
end

subgraph Application Layer
Controller[Application Logic]
Config[Configuration]
end

subgraph Data Processing Layer
Loader[Data Loader]
Processor[Data Processor]
Features[Feature Engineering]
Stats[Statistical Analysis]
end

subgraph Machine Learning Layer
Trainer[Model Trainer]
Factory[Model Factory]
Evaluator[Model Evaluator]
end

subgraph Data Layer
Dataset[(Titanic Dataset)]
end

UI --> Pages
Pages --> Charts
Pages --> Controller

Controller --> Loader
Loader --> Dataset

Loader --> Processor
Processor --> Features
Features --> Stats

Stats --> Trainer
Trainer --> Factory
Factory --> Evaluator
```

---

# Internal Module Architecture

This represents **the actual structure of the project**.

```mermaid
flowchart LR

app[app.py]

subgraph src
components[components]
data[data]
models[models]
utils[utils]
config[config.py]
end

subgraph pages
overview[Data Overview]
univariate[Univariate Analysis]
bivariate[Bivariate Analysis]
advanced[Advanced Analytics]
mlpred[ML Predictions]
end

subgraph tests
testsuite[Test Suite]
end

app --> overview
app --> univariate
app --> bivariate
app --> advanced
app --> mlpred

overview --> components
univariate --> components
bivariate --> components
advanced --> components
mlpred --> components

overview --> utils
univariate --> utils
bivariate --> utils
advanced --> utils
mlpred --> models

models --> utils
utils --> data
```

---

# Data Science Workflow Diagram

Questo è molto utile per **recruiter e data scientist**.

```mermaid
flowchart LR

RawData[Raw Dataset]

EDA[Exploratory Data Analysis]

Cleaning[Data Cleaning]

FeatureEngineering[Feature Engineering]

Training[Model Training]

Evaluation[Model Evaluation]

Prediction[Prediction Interface]

RawData --> EDA
EDA --> Cleaning
Cleaning --> FeatureEngineering
FeatureEngineering --> Training
Training --> Evaluation
Evaluation --> Prediction
```

