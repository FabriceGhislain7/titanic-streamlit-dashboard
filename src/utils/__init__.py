"""
Initialization file for utils package

This module initializes the utilities package and makes all helper functions
and classes available at the package level for easier imports.
"""

import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.info(f"Initializing {__name__} package")

# Import all utility modules to make them available at package level
try:
    from .data_loader import *
    from .data_processor import *
    from .feature_engineering import *
    from .helpers import *
    from .ml_preprocessing import *
    from .statistical_analysis import *
    
    logger.info("Successfully imported all utility modules")
except ImportError as e:
    logger.error(f"Error importing utility modules: {str(e)}")
    raise

# Package version
__version__ = '1.0.0'
logger.info(f"Package version: {__version__}")

# Package documentation
__doc__ = """
Titanic Data Utilities

This package contains all utility functions for data loading, processing,
feature engineering, and statistical analysis.
"""

logger.info(f"Completed initialization of {__name__}")