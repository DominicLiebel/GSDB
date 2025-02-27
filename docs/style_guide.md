# Coding Style Guide

This document outlines the coding style conventions for the GSDB project to ensure code
consistency, readability, and maintainability across the codebase.

## Python Code Style

### General Guidelines

1. Follow [PEP 8](https://pep8.org/) style guidelines for Python code
2. Use 4 spaces for indentation (no tabs)
3. Keep line length to a maximum of 100 characters
4. Use meaningful variable and function names
5. Break long lines before operators

### Naming Conventions

1. **Variables and Functions**: Use `snake_case` for variable and function names
   ```python
   def calculate_metrics(predictions, labels):
       mean_accuracy = sum(predictions == labels) / len(labels)
   ```

2. **Classes**: Use `PascalCase` for class names
   ```python
   class HistologyDataset(Dataset):
       def __init__(self, split):
           self.split = split
   ```

3. **Constants**: Use `UPPER_CASE` for constants
   ```python
   DEFAULT_BATCH_SIZE = 32
   MAX_EPOCHS = 100
   ```

4. **Private Members**: Prefix with underscore for private methods and variables
   ```python
   def _preprocess_image(self, image):
       # Internal implementation
       return processed_image
   ```

### Documentation

1. **Module Docstrings**: Every module should have a descriptive docstring
   ```python
   """
   Metrics Calculation Module
   
   This module implements various evaluation metrics for histology classification.
   """
   ```

2. **Function Docstrings**: Use Google style docstrings for functions
   ```python
   def calculate_f1(precision: float, recall: float) -> float:
       """Calculate F1 score from precision and recall.
       
       Args:
           precision: Precision value between 0 and 1
           recall: Recall value between 0 and 1
           
       Returns:
           F1 score (harmonic mean of precision and recall)
           
       Raises:
           ValueError: If precision or recall are outside [0,1] range
       """
   ```

3. **Class Docstrings**: Include class-level documentation
   ```python
   class ModelTracker:
       """Tracks best models during hyperparameter optimization.
       
       This class maintains the best model for each architecture type
       based on validation metrics.
       
       Attributes:
           best_models: Dictionary mapping architecture names to best models
           best_metrics: Dictionary mapping architecture names to best metrics
       """
   ```

4. **Type Hints**: Include type hints for function parameters and return values
   ```python
   def load_config(config_path: Path) -> Dict[str, Any]:
       """Load configuration from YAML file."""
   ```

### Imports

1. **Import Order**: Group imports in the following order:
   - Standard library imports
   - Third-party library imports
   - Local application imports
   
   ```python
   # Standard library
   import os
   import sys
   from pathlib import Path
   from typing import Dict, List, Optional
   
   # Third-party
   import numpy as np
   import torch
   import pandas as pd
   
   # Local application
   from src.utils.error_handling import ConfigurationError
   from src.models.architectures import HistologyClassifier
   ```

2. **Import Style**: Prefer explicit imports over wildcard imports
   ```python
   # Good
   from torch.utils.data import DataLoader, Dataset
   
   # Avoid
   from torch.utils.data import *
   ```

### Error Handling

1. **Specific Exceptions**: Catch specific exceptions rather than using bare `except:`
   ```python
   try:
       config = load_config(config_path)
   except FileNotFoundError:
       logging.error(f"Config file not found: {config_path}")
       raise
   except yaml.YAMLError as e:
       logging.error(f"Error parsing config file: {e}")
       raise
   ```

2. **Custom Exceptions**: Use custom exception classes for specific error cases
   ```python
   if task not in ['inflammation', 'tissue']:
       raise ConfigurationError(f"Unknown task: {task}")
   ```

3. **Error Messages**: Write clear, informative error messages
   ```python
   if threshold < 0 or threshold > 1:
       raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
   ```

### Code Organization

1. **Function Length**: Keep functions focused and reasonably sized (< 50 lines)
2. **Class Organization**: Organize class methods in logical groups:
   - Constructor and initialization methods
   - Public API methods
   - Internal helper methods
   - Utility methods
3. **Module Structure**: Order components in modules as follows:
   - Module docstring
   - Imports
   - Constants
   - Exception classes
   - Functions
   - Classes

## PyTorch-Specific Guidelines

### Model Implementation

1. **Model Classes**: Implement models as subclasses of `nn.Module`
   ```python
   class HistologyClassifier(nn.Module):
       def __init__(self, model_name, num_classes=1):
           super().__init__()
           # Initialize model
   ```

2. **Forward Method**: Implement clear forward methods
   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Forward pass through the model."""
       return self.backbone(x)
   ```

3. **Device Management**: Handle device placement consistently
   ```python
   def to(self, device):
       super().to(device)
       self.device = device
       return self
   ```

### Training Code

1. **DataLoader Parameters**: Use consistent DataLoader parameters
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=is_training,
       num_workers=num_workers,
       pin_memory=True
   )
   ```

2. **Training Loop**: Structure training loops consistently
   ```python
   # Zero gradients
   optimizer.zero_grad()
   
   # Forward pass
   outputs = model(inputs)
   loss = criterion(outputs, targets)
   
   # Backward pass
   loss.backward()
   
   # Optimizer step
   optimizer.step()
   ```

## Configuration Files

### YAML Style

1. **Structure**: Use hierarchical structure for related parameters
   ```yaml
   optimizer:
     name: AdamW
     learning_rate: 0.001
     weight_decay: 0.0001
   ```

2. **Naming**: Use consistent parameter naming across files
   ```yaml
   # Consistent naming
   batch_size: 32
   learning_rate: 0.001
   ```

3. **Comments**: Include explanatory comments for non-obvious settings
   ```yaml
   # Number of tiles to sample per particle
   # Higher values give better representation but slower training
   tiles_per_particle: 20
   ```

## Version Control

### Git Practices

1. **Commit Messages**: Write meaningful commit messages
   ```
   Add threshold optimization for hierarchical metrics
   
   - Implement validation-based threshold optimization
   - Ensure no test data leakage during optimization
   - Add visualization of ROC curves for different thresholds
   ```

2. **Branch Naming**: Use descriptive branch names
   ```
   feature/add-swin-transformer
   bugfix/fix-data-leakage
   refactor/reorganize-model-architecture
   ```

3. **Pull Requests**: Provide detailed descriptions for pull requests
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Any relevant issues addressed