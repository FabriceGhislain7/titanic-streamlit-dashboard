"""
Initialization file for models package

This module initializes the models package and makes all model classes
available at the package level for easier imports.
"""

import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.info(f"Initializing {__name__} package")

# Import all model classes to make them available at package level
try:
    from .ml_models import *
    from .model_trainer import *
    from .model_evaluator import *
    
    logger.info("Successfully imported all model modules")
except ImportError as e:
    logger.error(f"Error importing model modules: {str(e)}")
    raise

# Package version
__version__ = '1.0.0'
logger.info(f"Package version: {__version__}")

# Package documentation
__doc__ = """
Titanic Survival Prediction Models

This package contains all machine learning models and utilities
for training and evaluating survival prediction models.
"""

logger.info(f"Completed initialization of {__name__}")