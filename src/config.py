"""
Titanic Survival Analysis - Configuration File
==============================================

Centralized configuration for the Titanic analysis Streamlit application.
All parameters, paths and settings are defined here to facilitate
maintenance and customization.

Author: Data Analyst
Date: 2025
"""

import os

def find_project_root():
    """
    Find the project root folder by searching for characteristic marker files.
    This approach works wherever the project is moved.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # Files/folders that indicate the project root
    project_markers = [
        'requirements.txt',
        'README.md', 
        '.git',
        'app.py',
        'Home.py',
        'src'
    ]
    
    # Go up in the hierarchy until finding the root
    max_levels = 5  # Limit to avoid infinite loops
    for _ in range(max_levels):
        # Check if markers exist in current folder
        for marker in project_markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path
        
        # If already at filesystem root, stop
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            break
            
        current_path = parent_path
    
    # If root not found, use config file folder
    return os.path.dirname(os.path.abspath(__file__))

# Automatically find project root
PROJECT_ROOT = find_project_root()

# Project structure directories (paths relative to root)
BASE_DIR = PROJECT_ROOT
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLES_DIR = os.path.join(ASSETS_DIR, "styles")

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, MODELS_DIR, ASSETS_DIR, STYLES_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Data files
DATA_FILE = os.path.join(DATA_DIR, "data_titanic.csv")
DATA_URL = "https://raw.githubusercontent.com/FabriceGhislain7/data_analyst_scientist/main/titanic_project/data_titanic.csv"

# Dataset columns
DATASET_COLUMNS = {
    'NUMERICAL': ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare'],
    'CATEGORICAL': ['Survived', 'Pclass', 'Sex', 'Embarked'],
    'TEXT': ['Name', 'Ticket', 'Cabin'],
    'TARGET': 'Survived'
}

# Mapping for readable labels
COLUMN_LABELS = {
    'PassengerId': 'Passenger ID',
    'Survived': 'Survived',
    'Pclass': 'Class',
    'Name': 'Name',
    'Sex': 'Gender',
    'Age': 'Age',
    'SibSp': 'Siblings/Spouses',
    'Parch': 'Parents/Children',
    'Ticket': 'Ticket',
    'Fare': 'Ticket Fare',
    'Cabin': 'Cabin',
    'Embarked': 'Port of Embarkation'
}

# Mapping for categorical values
VALUE_MAPPINGS = {
    'Survived': {0: 'No', 1: 'Yes'},
    'Pclass': {1: '1st Class', 2: '2nd Class', 3: '3rd Class'},
    'Sex': {'male': 'Male', 'female': 'Female'},
    'Embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
}

# Parameters for outlier handling
OUTLIER_CONFIG = {
    'method': 'IQR',
    'lower_quantile': 0.25,
    'upper_quantile': 0.75,
    'iqr_multiplier': 1.5,
    'replacement_methods': ['remove', 'mean', 'median', 'mode']
}

# Age groups configuration
AGE_GROUPS = {
    'Children': {'min': 0, 'max': 12, 'label': 'Children (0-12 years)'},
    'Young_Adults': {'min': 13, 'max': 25, 'label': 'Young Adults (13-25 years)'},
    'Middle_Adults': {'min': 26, 'max': 40, 'label': 'Adults (26-40 years)'},
    'Older_Adults': {'min': 41, 'max': 100, 'label': 'Seniors (41+ years)'}
}

# Fare categories configuration
FARE_CATEGORIES = {
    'bins': 4,
    'labels': ['Low', 'Medium', 'High', 'Very High'],
    'method': 'qcut'
}

# Thresholds for missing values
MISSING_VALUE_THRESHOLDS = {
    'drop_column_threshold': 0.5,
    'drop_row_threshold': 0.8
}

# Color palettes
COLOR_PALETTES = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'success': '#45B7D1',
    'warning': '#FFA07A',
    'danger': '#FF6B6B',
    'survival': ['#FF6B6B', '#4ECDC4'],
    'gender': ['#87CEEB', '#FFB6C1'],
    'class': ['#FFD700', '#87CEEB', '#DDA0DD'],
    'age_groups': ['#FFE4B5', '#98FB98', '#87CEEB', '#DDA0DD'],
    'seaborn_palettes': ['viridis', 'plasma', 'coolwarm', 'Set2', 'Dark2']
}

# Chart configurations
CHART_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'whitegrid',
    'font_scale': 1.1,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 10
}

# Histogram configuration
HISTOGRAM_CONFIG = {
    'bins': 20,
    'kde': True,
    'alpha': 0.7,
    'edgecolor': 'black',
    'linewidth': 0.5
}

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Titanic Survival Analysis',
    'page_icon': '🚢',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Texts and descriptions
APP_TEXTS = {
    'main_title': 'Titanic Survival Analysis',
    'subtitle': 'Exploratory and predictive analysis of factors that influenced survival',
    'description': '''
    This application provides a comprehensive analysis of the famous Titanic dataset,
    exploring the factors that influenced passenger survival during
    the tragic sinking of 1912.
    ''',
    'data_source': 'Data source: Kaggle Titanic Competition',
    'footer': '''
    ---
    Developed with: Python, Streamlit, Pandas, Scikit-learn
    
    Note: This analysis is based on available historical data and serves educational purposes.
    '''
}

# ML models to use
ML_MODELS = {
    'LogisticRegression': {
        'name': 'Logistic Regression',
        'params': {'random_state': 42, 'max_iter': 1000}
    },
    'RandomForestClassifier': {
        'name': 'Random Forest',
        'params': {'n_estimators': 100, 'random_state': 42}
    },
    'GradientBoostingClassifier': {
        'name': 'Gradient Boosting',
        'params': {'random_state': 42}
    },
    'SVC': {
        'name': 'Support Vector Machine',
        'params': {'random_state': 42, 'probability': True}
    },
    'DecisionTreeClassifier': {
        'name': 'Decision Tree',
        'params': {'random_state': 42}
    }
}

# Feature engineering
FEATURE_ENGINEERING = {
    'create_family_size': True,
    'create_is_alone': True,
    'create_title_from_name': True,
    'create_fare_per_person': True,
    'create_age_groups': True,
    'create_deck_from_cabin': True
}

# Preprocessing
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'scale_features': True,
    'handle_missing': 'median'
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1',
    'roc_auc', 'confusion_matrix', 'classification_report'
]

# Debug settings
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
SHOW_RAW_DATA = DEBUG_MODE
SHOW_PROCESSING_STEPS = DEBUG_MODE

# Cache settings
CACHE_CONFIG = {
    'ttl': 3600,
    'max_entries': 100,
    'allow_output_mutation': True
}

# Mapping of notebook sections to Streamlit pages
NOTEBOOK_SECTIONS = {
    'data_loading': {
        'page': '1_Data_Overview.py',
        'section': 'Dataset Loading and Initial Exploration'
    },
    'missing_values': {
        'page': '1_Data_Overview.py', 
        'section': 'Missing Values Analysis'
    },
    'duplicates': {
        'page': '1_Data_Overview.py',
        'section': 'Duplicate Detection'
    },
    'data_cleaning': {
        'page': '1_Data_Overview.py',
        'section': 'Data Cleaning Methods'
    },
    'descriptive_stats': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Descriptive Statistics Analysis'
    },
    'numerical_viz': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Numerical Variables Visualization'
    },
    'age_analysis': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Single Variable Analysis - Age'
    },
    'categorical_analysis': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Categorical Variables - Survival Analysis'
    },
    'survival_by_class': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Class-based Survival Analysis'
    },
    'survival_by_gender': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Gender-based Survival Analysis'
    },
    'survival_by_age_groups': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Age Group Survival Analysis'
    },
    'survival_by_fare': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Fare-based Survival Analysis'
    },
    'survival_by_family': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Family Size Impact on Survival'
    },
    'correlation_analysis': {
        'page': '4_Advanced_Analytics.py',
        'section': 'Correlation Matrix and Heatmaps'
    },
    'outlier_detection': {
        'page': '4_Advanced_Analytics.py',
        'section': 'Advanced Outlier Detection'
    },
    'machine_learning': {
        'page': '5_ML_Predictions.py',
        'section': 'Machine Learning Models'
    }
}

# Updated utility functions
def get_data_path():
    """Returns the data file path as string"""
    return DATA_FILE

def get_model_path(model_name):
    """Returns the path to save a model"""
    ensure_directories()  # Ensure directory exists
    return os.path.join(MODELS_DIR, f"{model_name}.pkl")

def load_custom_css():
    """Load custom CSS if available"""
    css_file = os.path.join(STYLES_DIR, "main.css")
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def get_color_palette(palette_name='survival'):
    """Returns a specific color palette"""
    return COLOR_PALETTES.get(palette_name, COLOR_PALETTES['survival'])

def format_percentage(value, decimals=1):
    """Format a value as percentage"""
    return f"{value:.{decimals}f}%"

def format_number(value, decimals=2):
    """Format a number with specified decimals"""
    return f"{value:.{decimals}f}"

def get_project_info():
    """Returns project information and paths"""
    return {
        'project_root': PROJECT_ROOT,
        'config_location': os.path.abspath(__file__),
        'data_directory': DATA_DIR,
        'models_directory': MODELS_DIR,
        'data_file_exists': os.path.exists(DATA_FILE),
        'directories_created': all(os.path.exists(d) for d in [DATA_DIR, MODELS_DIR, ASSETS_DIR])
    }

# Automatic initialization
if __name__ == "__main__":
    # When config file is run directly, 
    # show path information
    ensure_directories()
    info = get_project_info()
    
    print("=== TITANIC PROJECT CONFIGURATION ===")
    print(f"Project Root: {info['project_root']}")
    print(f"Config File: {info['config_location']}")
    print(f"Data Directory: {info['data_directory']}")
    print(f"Models Directory: {info['models_directory']}")
    print(f"Data file exists: {info['data_file_exists']}")
    print(f"All directories created: {info['directories_created']}")
    
    if not info['data_file_exists']:
        print(f"\n⚠️  Data file not found at: {DATA_FILE}")
        print("Download it from:", DATA_URL)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': os.path.join(BASE_DIR, 'logs', 'titanic_app.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
        'streamlit': {
            'level': 'WARNING',
            'propagate': False
        }
    }
}

# Create logs folder if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

def setup_logging():
    """Configure global logging system"""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured correctly")
    return logger

# Automatic configuration if DEBUG_MODE is True
if DEBUG_MODE:
    LOGGING_CONFIG['handlers']['console']['level'] = 'DEBUG'
    LOGGING_CONFIG['loggers']['']['level'] = 'DEBUG'