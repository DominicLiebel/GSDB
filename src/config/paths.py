"""
Path Configuration Module

This module provides centralized path configuration for the project.
It allows setting paths via environment variables, config files, or command-line arguments.
"""

import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import yaml
import argparse

def get_base_dir() -> Path:
    """
    Get the base directory from environment variable or default location.
    
    The base directory can be set using the environment variable GASTRIC_BASE_DIR.
    If not set, will use the default relative path from the project root.
    
    Returns:
        Path: Base directory path
    """
    # Try environment variable first
    env_base_dir = os.environ.get("GASTRIC_BASE_DIR")
    if env_base_dir:
        base_dir = Path(env_base_dir)
        if base_dir.exists():
            return base_dir
        else:
            logging.warning(f"Environment variable GASTRIC_BASE_DIR points to non-existent path: {base_dir}")
            logging.warning("Falling back to default path")
    
    # Try to detect base directory based on project structure
    current_dir = Path(__file__).resolve().parent  # config directory
    project_root = current_dir.parent.parent  # Move up to project root
    
    # Fallback to a reasonable default
    default_dir = project_root
    if default_dir.exists():
        return default_dir
    else:
        # Create it if it doesn't exist
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

def get_project_paths(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get all project paths based on base directory.
    
    Args:
        base_dir: Base directory. If None, will be determined using get_base_dir()
    
    Returns:
        Dict[str, Path]: Dictionary of project paths
    """
    if base_dir is None:
        base_dir = get_base_dir()
    
    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": base_dir / "data",
        "RAW_DIR": base_dir / "data/raw",
        "PROCESSED_DIR": base_dir / "data/processed",
        "SPLITS_DIR": base_dir / "data/splits",
        "RESULTS_DIR": base_dir / "results",
        "LOGS_DIR": base_dir / "results/logs",
        "FIGURES_DIR": base_dir / "results/figures",
        "TABLES_DIR": base_dir / "results/tables",
        "MODELS_DIR": base_dir / "results/models",
        "METRICS_DIR": base_dir / "results/metrics",
        "CONFIG_DIR": base_dir / "configs",
        "TUNING_DIR": base_dir / "results/tuning"
    }
    
    # Don't automatically create directories
    # Only create them when actually needed
    
    return paths

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return {}

def add_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add path-related arguments to an argument parser.
    
    Args:
        parser: Argument parser to add arguments to
    
    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for all data and outputs"
    )
    parser.add_argument(
        "--data-dir", 
        type=Path,
        default=None,
        help="Directory containing data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path, 
        default=None,
        help="Directory for outputs"
    )
    return parser

# Initialize project paths
DEFAULT_PATHS = get_project_paths()