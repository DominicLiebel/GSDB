"""
Common Training Utilities for Histology Classification

This module provides standardized training functions that can be used by both
train.py and tune.py to ensure consistency in the training approach.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import os
from datetime import datetime

import sys
from pathlib import Path

# Try to get project root dynamically to avoid import errors
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.models.metrics_utils as metrics_utils

def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds to ensure reproducibility.
    
    This function sets seeds for Python's random module, NumPy, PyTorch,
    and configures CUDA for deterministic behavior.
    
    Args:
        seed: Integer seed value
        
    Note:
        For fully deterministic behavior, this may slow down training.
        CUDA operations will be deterministic but potentially slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # These settings ensure deterministic behavior but may reduce performance
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"All random seeds set to {seed} for reproducibility")
    logging.info("Note: Deterministic mode is enabled, which may slightly reduce performance")

def train_epoch(model: nn.Module, 
               train_loader: DataLoader, 
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               scaler: torch.amp.GradScaler,
               device: torch.device,
               use_amp: bool = True,
               deterministic: bool = True) -> Dict[str, Any]:
    """Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on
        use_amp: Whether to use automatic mixed precision (AMP)
        deterministic: Whether to enforce deterministic behavior
        
    Returns:
        Dict containing training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_raw_preds = []
    metadata = []
    
    # Enforce deterministic behavior if requested
    # This is particularly important for mixed precision training
    if deterministic and torch.cuda.is_available():
        if not torch.backends.cudnn.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.debug("Set CUDNN to deterministic mode for training")
    
    for inputs, labels, batch_metadata in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use AMP only if requested, with deterministic settings for float16 ops
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard full-precision training (more deterministic)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Store predictions and labels for additional metrics
        with torch.no_grad():
            preds = (outputs > 0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(outputs.detach().cpu().numpy())
            
            # Store metadata for hierarchical metrics calculation
            for i in range(len(labels)):
                meta_dict = {k: v[i] if isinstance(v, list) else v for k, v in batch_metadata.items()}
                metadata.append(meta_dict)
        
        total_loss += loss.item()
    
    # Calculate basic metrics
    metrics = metrics_utils.calculate_basic_metrics(
        all_labels, all_preds, all_raw_preds
    )
    
    # Add additional metrics
    metrics['loss'] = total_loss / len(train_loader)
    
    # Calculate confusion matrix metrics directly
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total = len(all_labels)
    
    # Ensure all required metrics are present
    metrics['accuracy'] = 100 * (tp + tn) / total if total > 0 else 0
    metrics['sensitivity'] = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Add a log message showing the primary metrics
    logging.debug(f"Train metrics - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%, " + 
                 f"Sensitivity: {metrics['sensitivity']:.2f}%, Specificity: {metrics['specificity']:.2f}%, " +
                 f"F1: {metrics['f1']:.2f}%, AUC: {metrics.get('auc', 0):.2f}%")
    
    return metrics

def validate(model: nn.Module, 
            val_loader: DataLoader, 
            criterion: nn.Module,
            device: torch.device) -> Dict[str, Any]:
    """Validate the model.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
        
    Returns:
        Dict containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_raw_preds = []
    metadata = []
    
    # Handle empty validation set
    if len(val_loader) == 0:
        logging.warning("Validation dataloader is empty!")
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc': 50.0
        }
        return metrics
    
    with torch.no_grad():
        for inputs, labels, batch_metadata in tqdm(val_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Calculate predictions
            preds = (outputs > 0).float()
            
            # Store predictions and labels for additional metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(outputs.cpu().numpy())
            
            # Store metadata for hierarchical metrics calculation
            for i in range(len(labels)):
                meta_dict = {k: v[i] if isinstance(v, list) else v for k, v in batch_metadata.items()}
                metadata.append(meta_dict)
            
            total_loss += loss.item()
    
    # Calculate basic metrics
    metrics = metrics_utils.calculate_basic_metrics(
        all_labels, all_preds, all_raw_preds
    )
    
    # Add additional metrics
    metrics['loss'] = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    # Add a log message showing the primary metrics
    logging.debug(f"Validation metrics - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%, " + 
                 f"Sensitivity: {metrics['sensitivity']:.2f}%, Specificity: {metrics['specificity']:.2f}%, " +
                 f"F1: {metrics['f1']:.2f}%, AUC: {metrics.get('auc', 0):.2f}%")
    
    return metrics

def handle_multi_gpu_model(model: nn.Module) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs are available.
    
    Args:
        model: The neural network model
        
    Returns:
        Model, possibly wrapped in DataParallel
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model

def configure_training_components(model: nn.Module, 
                                 learning_rate: float, 
                                 weight_decay: float,
                                 pos_weight: float = 1.0,
                                 optimizer_name: str = "AdamW", 
                                 momentum: float = 0.9,
                                 device: torch.device = None) -> Dict[str, Any]:
    """Configure optimizer, criterion, and scaler with support for SGD.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        pos_weight: Positive class weight for BCEWithLogitsLoss
        optimizer_name: Name of optimizer to use ("AdamW" or "SGD")
        momentum: Momentum parameter for SGD optimizer
        device: Device to place tensors on
        
    Returns:
        Dict containing optimizer, criterion, and scaler
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert string parameters to float if needed
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    if isinstance(momentum, str):
        momentum = float(momentum)
    if isinstance(pos_weight, str):
        pos_weight = float(pos_weight)
    
    # Create optimizer based on name
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        logging.info(f"Using SGD optimizer with learning_rate={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
    else:  # Default to AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info(f"Using AdamW optimizer with learning_rate={learning_rate}, weight_decay={weight_decay}")
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    
    scaler = torch.amp.GradScaler()
    
    return {
        'optimizer': optimizer,
        'criterion': criterion,
        'scaler': scaler
    }

def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, Any],
                   config: Dict[str, Any],
                   filepath: str,
                   seed: int = None,
                   deterministic: bool = False) -> None:
    """Save model checkpoint with reproducibility information.
    
    Args:
        model: The neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Validation metrics
        config: Model configuration
        filepath: Path to save checkpoint
        seed: Random seed used for training
        deterministic: Whether deterministic mode was enabled
    """
    # Ensure all required metrics are present
    required_metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    for metric in required_metrics:
        if metric not in metrics:
            logging.warning(f"Metric '{metric}' missing from checkpoint metrics")
    
    # Include reproducibility information in checkpoint
    reproducibility_info = {
        'random_seed': seed,
        'deterministic_mode': deterministic,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'reproducibility': reproducibility_info
    }
    
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")
    
    # Log mtraetrics to console
    logging.info(f"Checkpoint metrics - Acc: {metrics.get('accuracy', 0):.2f}%, " +
               f"Sens: {metrics.get('sensitivity', 0):.2f}%, " +
               f"Spec: {metrics.get('specificity', 0):.2f}%, " +
               f"F1: {metrics.get('f1', 0):.2f}%, " +
               f"AUC: {metrics.get('auc', 0):.2f}%")
    
class EarlyStopping:
    """Early stopping handler to prevent overfitting.
    
    Monitors validation metrics and stops training if no improvement
    is seen for a specified number of epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """Initialize early stopping handler.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' if lower is better (e.g., loss), 'max' if higher is better (e.g., accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.min_validation_loss = float('inf')
        
        logging.info(f"Early stopping initialized: patience={patience}, min_delta={min_delta}, mode={mode}")
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should be stopped.
        
        Args:
            current_score: Current validation metric value
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.mode == 'min':
            score = -1.0 * current_score
        else:
            score = current_score
            
        if self.best_score == float('inf') or self.best_score == float('-inf'):
            # First epoch
            self.best_score = score
            return False
            
        delta = score - self.best_score
        
        if delta > self.min_delta:
            # Improvement found
            self.best_score = score
            self.counter = 0
            
            # Update min validation loss for reference
            if self.mode == 'min':
                self.min_validation_loss = current_score
                
            return False
        
        # No improvement
        self.counter += 1
        if self.counter >= self.patience:
            logging.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            self.early_stop = True
            return True
            
        return False
        
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.min_validation_loss = float('inf')