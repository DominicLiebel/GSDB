"""
Error Handling Utilities

This module provides standardized error handling and exception classes
for the Gastric Stained Dataset project.
"""

import logging
import sys
import traceback
from typing import Optional, Any, Dict, Union, Callable
from functools import wraps


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""
    pass


class DataLoadingError(DatasetError):
    """Exception raised for errors during data loading."""
    pass


class ConfigurationError(Exception):
    """Exception raised for errors in configuration parameters."""
    pass


class ModelError(Exception):
    """Base exception for all model-related errors."""
    pass


class TrainingError(ModelError):
    """Exception raised for errors during model training."""
    pass


class EvaluationError(ModelError):
    """Exception raised for errors during model evaluation."""
    pass


def handle_exception(e: Exception, log_level: int = logging.ERROR,
                     exit_code: Optional[int] = None) -> None:
    """Standard exception handler for consistent error logging.
    
    Args:
        e: The exception to handle
        log_level: Logging level (default: ERROR)
        exit_code: If provided, exit with this code after logging
    """
    # Get detailed traceback info
    tb_str = traceback.format_exception(type(e), e, e.__traceback__)
    
    # Log the exception
    logging.log(log_level, f"Exception occurred: {str(e)}")
    logging.log(log_level, f"Traceback:\n{''.join(tb_str)}")
    
    # Exit if requested
    if exit_code is not None:
        sys.exit(exit_code)


def error_handler(func: Callable) -> Callable:
    """Decorator for consistent error handling across functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (DatasetError, ConfigurationError, ModelError) as e:
            # Log custom exceptions with details
            handle_exception(e, logging.ERROR)
            raise
        except Exception as e:
            # Log unexpected exceptions
            handle_exception(e, logging.CRITICAL)
            raise
    return wrapper


def validate_dataset_config(config: Dict[str, Any]) -> None:
    """Validate dataset configuration parameters.
    
    Args:
        config: Dataset configuration dictionary
        
    Raises:
        ConfigurationError: If any configuration parameters are invalid
    """
    required_keys = ['task', 'stains', 'scanners']
    
    # Validate required keys
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")
    
    # Validate specific values
    if config['task'] not in ['inflammation', 'tissue']:
        raise ConfigurationError(f"Invalid task: {config['task']}. Must be 'inflammation' or 'tissue'")
    
    if not isinstance(config['stains'], list) or not config['stains']:
        raise ConfigurationError("'stains' must be a non-empty list")
    
    if not isinstance(config['scanners'], list) or not config['scanners']:
        raise ConfigurationError("'scanners' must be a non-empty list")


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Raises:
        ConfigurationError: If any configuration parameters are invalid
    """
    required_keys = ['architecture', 'batch_size', 'optimizer']
    
    # Validate required keys
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")
    
    # Validate specific values
    if 'learning_rate' in config['optimizer'] and config['optimizer']['learning_rate'] <= 0:
        raise ConfigurationError("Learning rate must be positive")
    
    if config['batch_size'] <= 0:
        raise ConfigurationError("Batch size must be positive")
    
    if 'epochs' in config and config['epochs'] <= 0:
        raise ConfigurationError("Number of epochs must be positive")