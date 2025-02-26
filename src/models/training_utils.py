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

import metrics_utils

def train_epoch(model: nn.Module, 
               train_loader: DataLoader, 
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               scaler: torch.amp.GradScaler,
               device: torch.device) -> Dict[str, Any]:
    """Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on
        
    Returns:
        Dict containing training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_raw_preds = []
    metadata = []
    
    for inputs, labels, batch_metadata in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
    metrics['loss'] = total_loss / len(val_loader)
    
    # Calculate confusion matrix metrics directly
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total = len(all_labels)
    
    # Ensure all required metrics are present
    metrics['accuracy'] = 100 * (tp + tn) / total if total > 0 else 0
    metrics['sensitivity'] = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
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
                                 device: torch.device = None) -> Dict[str, Any]:
    """Configure optimizer, criterion, and scaler.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        pos_weight: Positive class weight for BCEWithLogitsLoss
        device: Device to place tensors on
        
    Returns:
        Dict containing optimizer, criterion, and scaler
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
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
                   filepath: str) -> None:
    """Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Validation metrics
        config: Model configuration
        filepath: Path to save checkpoint
    """
    # Ensure all required metrics are present
    required_metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    for metric in required_metrics:
        if metric not in metrics:
            logging.warning(f"Metric '{metric}' missing from checkpoint metrics")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")
    
    # Log mtraetrics to console
    logging.info(f"Checkpoint metrics - Acc: {metrics.get('accuracy', 0):.2f}%, " +
               f"Sens: {metrics.get('sensitivity', 0):.2f}%, " +
               f"Spec: {metrics.get('specificity', 0):.2f}%, " +
               f"F1: {metrics.get('f1', 0):.2f}%, " +
               f"AUC: {metrics.get('auc', 0):.2f}%")