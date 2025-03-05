# Project Organization Guide

This document outlines the organization of the GSDB (Gastric Stained Dataset) project codebase
to help new contributors understand the project structure and architecture.

## Directory Structure

```
GSDB/
├── configs/               # Configuration files
│   └── model_config.yaml  # Model and training configuration
├── data/                  # Data storage
│   ├── processed/         # Processed data (tiles)
│   ├── raw/               # Raw data (whole slide images) - not included in repo
│   └── splits/            # Dataset splits (train/val/test)
├── docs/                  # Documentation
├── results/               # Experimental results
│   ├── figures/           # Generated figures
│   ├── logs/              # Run logs
│   ├── metrics/           # Performance metrics
│   ├── models/            # Saved models
│   ├── tables/            # Result tables
│   └── tuning/            # Hyperparameter tuning results
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── config/            # Configuration utilities
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementation
│   │   ├── architectures/ # Model architecture definitions
│   └── utils/             # Helper utilities
├── environment.yml        # Conda environment specification
└── README.md              # Project overview
```

## Key Code Modules

### Configuration (`src/config/`)

- `paths.py`: Centralized path management for consistent file locations

### Data Processing (`src/data/`)

- `create_splits.py`: Creates train/validation/test splits based on patient IDs
- `dataset_analysis.py`: Analyzes dataset statistics and distributions
- `extract_tiles.py`: Extracts fixed-size tiles from whole slide images
- `process_dataset.py`: Main data processing workflow

### Models (`src/models/`)

- `architectures/`: Modular model definitions
  - `base.py`: Base class for histology classifiers
  - `gigapath.py`: GigaPath-based classifier implementation
- `dataset.py`: PyTorch dataset implementation for histology data
- `evaluate.py`: Model evaluation pipeline
- `metrics_utils.py`: Metrics calculation and visualization
- `train.py`: Model training pipeline
- `training_utils.py`: Common training utilities and functions
- `tune.py`: Hyperparameter optimization

### Utilities (`src/utils/`)

- `error_handling.py`: Standardized error handling and exception classes

## Architecture Overview

### Data Flow

1. **Data Preparation**:
   - Raw WSIs → Particle extraction → Tile generation → Split creation

2. **Training Pipeline**:
   - Dataset → DataLoader → Model → Training loop → Evaluation → Metrics

3. **Evaluation Pipeline**:
   - Model → Test data → Threshold optimization → Hierarchical metrics

### Model Architecture

The project uses a modular architecture approach:

1. **Base Model Class**: `HistologyClassifier` in `src/models/architectures/base.py`
   - Supports multiple backbones (ResNet, DenseNet, ConvNeXt, Swin Transformer)
   - Standardized interface for training and evaluation

2. **Specialized Models**: 
   - `GigaPathClassifier` in `src/models/architectures/gigapath.py`
   - Feature extraction using pre-trained GigaPath model

3. **Model Training**:
   - Mixed precision training with reproducibility controls
   - Support for multi-GPU parallelization
   - Early stopping and model checkpointing

### Evaluation Framework

The evaluation framework is designed for scientific reproducibility:

1. **Hierarchical Evaluation**:
   - Metrics at tile, particle (tissue), and slide (inflammation) levels
   - Validation-optimized thresholds to avoid data leakage

2. **Threshold Optimization**:
   - Strict separation of validation and test data
   - Multiple aggregation strategies for hierarchical levels

3. **Metrics Reporting**:
   - Accuracy, sensitivity, specificity, precision, F1 score, and AUC

## Configuration System

The project uses a hierarchical configuration system:

1. **Base Configuration**: Default parameters in configuration files
2. **Model-Specific Configuration**: Specialized parameters for specific models
3. **Command-Line Overrides**: Runtime parameter adjustments

Key configuration file:
- `configs/model_config.yaml`: Model architectures and training parameters

## Development Guidelines

1. **Code Organization**:
   - Follow the existing module structure
   - Place reusable utilities in appropriate utility modules
   - Keep model architecture definitions separate from training code

2. **Error Handling**:
   - Use custom exception classes from `src/utils/error_handling.py`
   - Log all errors with appropriate context
   - Catch specific exceptions rather than using broad exception handlers

3. **Documentation**:
   - Add docstrings to all functions and classes
   - Include type hints for parameters and return values
   - Update documentation when adding new features

4. **Reproducibility**:
   - Set random seeds for all randomized operations
   - Document all hyperparameters and configuration settings
   - Save configuration with model checkpoints

5. **Testing**:
   - Run tests before committing changes
   - Include tests for new functionality
   - Verify results against previous benchmarks