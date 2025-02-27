"""
Histology Image Classification Training

This script trains deep learning models for histological image classification.
Supports both inflammation detection and tissue type classification.

Key Features:
- Multiple model architectures (ResNet18, GigaPath, ConvNeXt, Swin Transformer)
- State-of-the-art transfer learning capabilities
- Multi-GPU training support
- Mixed precision training
- Advanced data augmentation
- Comprehensive logging

Example Usage:
    # List available models for inflammation task:
    python train.py --task inflammation --list-models

    # Train inflammation classifier using ResNet18:
    python train.py --task inflammation --model resnet18 --batch-size 64

    # Train tissue classifier with specific settings:
    python train.py --task tissue --model swin_v2_b --batch-size 32 --epochs 50
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Union, Any
import os
from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import json
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Local imports
from src.models.dataset import HistologyDataset
import src.models.metrics_utils as metrics_utils
import src.models.training_utils as training_utils
from src.config.paths import get_project_paths, add_path_args

# Import models and weights
from torchvision.models import (
    convnext_large, ConvNeXt_Large_Weights,
    swin_v2_b, Swin_V2_B_Weights,
    resnet18, ResNet18_Weights
)

def load_config(config_dir: Path, task: str, model_name: str = None) -> dict:
    """Load task-specific and model-specific configuration from unified YAML file.
    
    Args:
        config_dir (Path): Directory containing configuration files
        task (str): Task identifier ('inflammation' or 'tissue')
        model_name (str, optional): Model identifier to load specific configuration
            
    Returns:
        dict: Configuration dictionary containing model and training parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If specified task or model not found in config file
    """
    config_path = config_dir / 'model_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract common configuration
        common_config = config.get('common', {})
        
        # Extract task-specific data
        if task not in config:
            raise KeyError(f"Task '{task}' not found in config file")
            
        task_config = config[task]
        
        # If no model specified, use first model in the task
        if model_name is None:
            model_name = next(iter(task_config.keys()))
            logging.info(f"No model specified. Using default model: {model_name}")
        
        # Extract model-specific configuration
        if model_name not in task_config:
            available_models = list(task_config.keys())
            raise KeyError(f"Model '{model_name}' not found for task '{task}'. Available models: {available_models}")
            
        model_config = task_config[model_name]
        
        # Combine configurations with priority order
        # 1. Model-specific config (highest priority)
        # 2. Common config (lowest priority)
        result_config = common_config.copy()
        result_config.update(model_config)
        
        # Add architecture data separately from architectures
        result_config['architectures'] = config.get('architectures', {})
        
        # Add task and selected model for logging purposes
        result_config['task'] = task
        result_config['selected_model'] = model_name
        
        return result_config
        
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise
    except KeyError as e:
        logging.error(f"Configuration error: {e}")
        raise

def get_transforms(task: str, is_training: bool = False, config: dict = None) -> T.Compose:
    """Get data transformations for training or evaluation based on configuration.
    
    Args:
        task: 'inflammation' or 'tissue'
        is_training: Whether to include training augmentations
        config: Configuration dictionary containing transform settings
        
    Returns:
        Composed transforms
    """
    if config is None:
        # Fallback defaults if no config is provided
        return _get_default_transforms(task, is_training)
    
    transforms_list = []
    
    # Get common normalization params
    common_config = config.get('common', {})
    norm_config = common_config.get('transforms', {}).get('normalization', {})
    norm_mean = norm_config.get('mean', [0.485, 0.456, 0.406])
    norm_std = norm_config.get('std', [0.229, 0.224, 0.225])
    
    if is_training:
        # Get task and model specific transform settings
        transform_config = config.get('transforms', {}).get('training', {})
        
        # Random resized crop
        if 'random_resized_crop' in transform_config:
            rrc_config = transform_config['random_resized_crop']
            size = rrc_config.get('size', 224)
            scale = tuple(rrc_config.get('scale', [0.8, 1.0]))
            transforms_list.append(T.RandomResizedCrop(size, scale=scale))
        
        # Random flips
        if transform_config.get('random_horizontal_flip', False):
            transforms_list.append(T.RandomHorizontalFlip())
            
        if transform_config.get('random_vertical_flip', False):
            transforms_list.append(T.RandomVerticalFlip())
        
        # Color jitter
        if 'color_jitter' in transform_config:
            jitter_config = transform_config['color_jitter']
            brightness = jitter_config.get('brightness', 0)
            contrast = jitter_config.get('contrast', 0)
            saturation = jitter_config.get('saturation', 0)
            hue = jitter_config.get('hue', 0)
            transforms_list.append(T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
        
        # Random rotation
        if transform_config.get('random_rotation', {}).get('enabled', False):
            degrees = transform_config.get('random_rotation', {}).get('degrees', 180)
            transforms_list.append(T.RandomRotation(degrees))
        
        # Random blur
        if transform_config.get('random_blur', {}).get('enabled', False):
            kernel_size = transform_config.get('random_blur', {}).get('kernel_size', 3)
            transforms_list.append(T.GaussianBlur(kernel_size=kernel_size))
        
        # Random affine transform
        if transform_config.get('random_affine', {}).get('enabled', False):
            degrees = transform_config.get('random_affine', {}).get('degrees', 15)
            translate = tuple(transform_config.get('random_affine', {}).get('translate', [0.1, 0.1]))
            transforms_list.append(T.RandomAffine(degrees=degrees, translate=translate))
    else:
        # Validation transforms from common config
        val_config = common_config.get('transforms', {}).get('validation', {})
        resize = val_config.get('resize', 256)
        center_crop = val_config.get('center_crop', 224)
        
        transforms_list.extend([
            T.Resize(resize),
            T.CenterCrop(center_crop)
        ])
    
    # Common transforms
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return T.Compose(transforms_list)

def _get_default_transforms(task: str, is_training: bool = False) -> T.Compose:
    """Fallback function for default transforms if no config is provided.
    
    Args:
        task: 'inflammation' or 'tissue'
        is_training: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    transforms_list = []
    
    if is_training:
        # Basic augmentations for all tasks
        transforms_list.extend([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        # Default task-specific augmentations
        if task == 'inflammation':
            transforms_list.extend([
                T.RandomRotation(180),
                T.GaussianBlur(kernel_size=3),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1))
            ])
        else:  # tissue
            transforms_list.append(T.RandomRotation(180))
    else:
        transforms_list.extend([
            T.Resize(256),
            T.CenterCrop(224)
        ])
    
    # Common transforms
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return T.Compose(transforms_list)

def setup_logging(output_dir: Path):
    """Configure logging to both file and console.
    
    Args:
        output_dir (Path): Directory for saving log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def configure_device(config: dict = None) -> torch.device:
    """Configure device for training/evaluation.
    
    Args:
        config (dict, optional): Configuration dictionary that may contain
            device-specific settings
            
    Returns:
        torch.device: Configured device (CPU or CUDA)
    """
    if torch.cuda.is_available():
        # Set CUDA device if specified in config
        if config and 'gpu_id' in config:
            torch.cuda.set_device(config['gpu_id'])
            return torch.device(f'cuda:{config["gpu_id"]}')
        return torch.device('cuda')
    return torch.device('cpu')

class HistologyClassifier(nn.Module):
    """Advanced histology image classifier supporting multiple architectures."""
    
    def __init__(self, model_name: str, num_classes: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create backbone based on model name
        if model_name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Modify classifier head
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )
            
        elif model_name == 'convnext_large':
            from torchvision.models import convnext_large, ConvNeXt_Large_Weights
            self.backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
            # Modify classifier head
            self.backbone.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(1536, self.num_classes)
            )
            
        elif model_name == 'swin_v2_b':
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            self.backbone = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            # Modify classifier head
            in_features = self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )
            
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'resnet18', 'convnext_large', or 'swin_v2_b'. For GigaPath, use GigaPathClassifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)


    def configure_optimizers(self, lr: float, weight_decay: float):
        """Configure optimizer and related training components."""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler()
        return self.optimizer

    def train(self, mode: bool = True) -> 'HistologyClassifier':
        """Set training mode."""
        super().train(mode)
        self.backbone.train(mode)
        return self
        
    def eval(self) -> 'HistologyClassifier':
        """Set evaluation mode."""
        super().eval()
        self.backbone.eval()
        return self
        
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = device
        self.backbone = self.backbone.to(device)
        return self
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch with optimized performance."""
        self.train()
        metrics = {
            'train_loss': 0.0,
            'correct': 0,
            'total': 0,
            'preds': [],
            'labels': [],
            'raw_preds': []
        }
        
        torch.backends.cudnn.benchmark = True
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels, _ = batch
            inputs = inputs.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = self.backbone(inputs).squeeze()
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            
            if hasattr(self, 'gradient_clipping') and self.gradient_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.gradient_clipping)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics using raw logits
            metrics['train_loss'] += loss.item()
            with torch.no_grad():
                preds = (outputs > 0).float()  # No sigmoid needed
            metrics['correct'] += (preds == labels).sum().item()
            metrics['total'] += labels.size(0)
            metrics['preds'].extend(preds.cpu().numpy())
            metrics['labels'].extend(labels.cpu().numpy())
            metrics['raw_preds'].extend(outputs.detach().cpu().numpy())
            
        return self._calculate_metrics(metrics, len(train_loader), 'train')
    
    def evaluate(self, data_loader: DataLoader, prefix: str = 'val') -> Dict:
        """Evaluate the model on the provided data."""
        self.eval()
        metrics = {
            f'{prefix}_loss': 0.0,
            'correct': 0,
            'total': 0,
            'preds': [],
            'labels': [],
            'raw_preds': []
        }
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {prefix}"):
                inputs, labels, _ = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.backbone(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                
                metrics[f'{prefix}_loss'] += loss.item()
                preds = (outputs > 0).float()
                metrics['correct'] += (preds == labels).sum().item()
                metrics['total'] += labels.size(0)
                metrics['preds'].extend(preds.cpu().numpy())
                metrics['labels'].extend(labels.cpu().numpy())
                metrics['raw_preds'].extend(outputs.cpu().numpy())
                
        return self._calculate_metrics(metrics, len(data_loader), prefix)

    def _calculate_metrics(self, metrics: Dict, num_batches: int, prefix: str) -> Dict:
        """Calculate performance metrics.
        
        Args:
            metrics: Dictionary containing raw metrics data
            num_batches: Number of batches processed
            prefix: Prefix for metric names (e.g., 'train' or 'val')
            
        Returns:
            Dict: Dictionary containing calculated metrics
        """
        # Calculate basic metrics using imported function
        basic_metrics = calculate_basic_metrics(
            y_true=metrics['labels'],
            y_pred=metrics['preds'],
            y_scores=metrics['raw_preds']
        )
        
        # Format metrics with prefix
        formatted_metrics = {
            f'{prefix}_loss': metrics[f'{prefix}_loss'] / num_batches,
        }
        
        # Add basic metrics with prefix
        for key, value in basic_metrics.items():
            formatted_metrics[f'{prefix}_{key}'] = value
            
        return formatted_metrics

# GigaPath model implementation (moved from tune.py)
class GigaPathClassifier(nn.Module):
    """Classifier using Prov-GigaPath features for histology classification."""
    
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "gigapath"
        
        try:
            import timm
            # Load pretrained GigaPath tile encoder
            self.tile_encoder = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True,
                num_classes=0  # Disable classification head
            )
            
            # Freeze the encoder weights
            for param in self.tile_encoder.parameters():
                param.requires_grad = False
                
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(1536, 512),  # GigaPath outputs 1536-dim features
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            logging.error("timm library not found. Please install with: pip install timm")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # Ensure encoder remains in inference mode
            features = self.tile_encoder(x)
        return self.classifier(features)
        
    def train(self, mode: bool = True) -> 'GigaPathClassifier':
        """Set training mode for classifier only."""
        super().train(mode)
        # Keep encoder in eval mode
        self.tile_encoder.eval()
        return self

def train_model(args, paths):
    """Main training function with standardized approach and config-based transforms."""
    
    # Load configuration with model selection
    config = load_config(paths["CONFIG_DIR"], args.task, args.model)
    
    # Extract architecture configuration
    arch_config = config['architecture']
    model_name = arch_config['name']
    
    # Log the selected model
    logging.info(f"Training {args.task} classifier using {model_name} architecture")
    
    # Override configuration with command line arguments if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        logging.info(f"Overriding batch size: {args.batch_size}")
        
    if args.learning_rate is not None:
        config['optimizer']['learning_rate'] = args.learning_rate
        logging.info(f"Overriding learning rate: {args.learning_rate}")
        
    if args.weight_decay is not None:
        config['optimizer']['weight_decay'] = args.weight_decay
        logging.info(f"Overriding weight decay: {args.weight_decay}")
        
    if args.epochs is not None:
        config['epochs'] = args.epochs
        logging.info(f"Overriding epochs: {args.epochs}")
    
    # Create model with proper configuration
    if model_name == 'gigapath':
        # Special case for GigaPath model
        model = GigaPathClassifier(
            num_classes=1,
            dropout_rate=config['dropout_rate']
        )
    else:
        # Standard models from HistologyClassifier
        model = HistologyClassifier(
            model_name=model_name,
            num_classes=1,
            dropout_rate=config['dropout_rate']
        )
    
    # Setup device and multi-GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)

    # Configure training components with optimizer name and momentum for SGD
    optimizer_name = config['optimizer'].get('name', 'AdamW')
    momentum = config['optimizer'].get('momentum', 0.9)
    
    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=config.get('pos_weight', 1.0),
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device
    )
    
    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']
    
    # Setup early stopping if enabled
    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")
    
    # Create datasets and dataloaders with transforms from config
    train_transform = get_transforms(args.task, is_training=True, config=config)
    val_transform = get_transforms(args.task, is_training=False, config=config)
    
    train_dataset = HistologyDataset(
        split='train',
        transform=train_transform,
        task=args.task,
        paths=paths
    )
    
    val_dataset = HistologyDataset(
        split='val',
        transform=val_transform,
        task=args.task,
        paths=paths
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Configure scheduler
    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        # Training phase
        train_metrics = training_utils.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device
        )
        
        # Validation phase
        val_metrics = training_utils.validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Log metrics
        logging.info(
            f"Epoch {epoch+1}/{config['epochs']}\n"
            f"Train Metrics:\n"
            f"  Loss: {train_metrics['loss']:.4f}\n"
            f"  Accuracy: {train_metrics['accuracy']:.2f}%\n"
            f"  Sensitivity: {train_metrics['sensitivity']:.2f}%\n"
            f"  Specificity: {train_metrics['specificity']:.2f}%\n"
            f"  F1: {train_metrics['f1']:.2f}%\n"
            f"  AUC: {train_metrics['auc']:.2f}%\n"
            f"Validation Metrics:\n"
            f"  Loss: {val_metrics['loss']:.4f}\n"
            f"  Accuracy: {val_metrics['accuracy']:.2f}%\n"
            f"  Sensitivity: {val_metrics['sensitivity']:.2f}%\n"
            f"  Specificity: {val_metrics['specificity']:.2f}%\n"
            f"  F1: {val_metrics['f1']:.2f}%\n"
            f"  AUC: {val_metrics['auc']:.2f}%"
        )
        
        # Check early stopping
        if early_stopping and early_stopping(val_metrics['loss']):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = paths["MODELS_DIR"] / f"best_model_{args.task}_{model_name}.pt"
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path)
            )
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f} to {model_path}")
            
    # Save final training history
    history_path = paths["RESULTS_DIR"] / f"training_history_{args.task}_{model_name}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    logging.info(f"Training complete. Training history saved to {history_path}")

def calculate_basic_metrics(y_true, y_pred, y_scores=None):
    """Calculate basic classification metrics."""
    metrics = {}
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    total = len(y_true)
    metrics['accuracy'] = 100 * (tp + tn) / total if total > 0 else 0
    metrics['sensitivity'] = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    metrics['f1'] = 100 * f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC if scores are provided
    if y_scores is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['auc'] = 100 * roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc'] = 50  # Default AUC for random classifier
    
    return metrics

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with improved model selection."""
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument(
        "--task",
        required=True,
        choices=['inflammation', 'tissue'],
        help="Classification task"
    )
    parser.add_argument(
        "--model",
        choices=['resnet18', 'gigapath', 'convnext_large', 'swin_v2_b'],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Override weight decay from config"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models for the specified task and exit"
    )
    
    # Add path arguments using the utility function
    parser = add_path_args(parser)
    
    args = parser.parse_args()
    
    return args

def main():
    """Main training function with improved model selection."""
    args = parse_args()

    # Get project paths with any overrides from command line
    paths = get_project_paths(base_dir=args.base_dir)
    
    # Override specific directories if provided
    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed" 
        paths["SPLITS_DIR"] = args.data_dir / "splits"
    
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"
        paths["MODELS_DIR"] = args.output_dir / "models"
        paths["FIGURES_DIR"] = args.output_dir / "figures"
        paths["TABLES_DIR"] = args.output_dir / "tables"
    
    # Create required directories
    for dir_path in [paths["LOGS_DIR"], paths["MODELS_DIR"]]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # If list-models flag is set, show available models and exit
    if args.list_models:
        if not args.task:
            print("Error: --task must be specified with --list-models")
            sys.exit(1)
            
        try:
            # Load configuration to list available models
            config_path = paths["CONFIG_DIR"] / 'model_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if args.task in config:
                available_models = list(config[args.task].keys())
                print(f"\nAvailable models for {args.task} task:")
                for model in available_models:
                    print(f"  - {model}")
                print("\nUse --model [name] to select a specific model.")
            else:
                print(f"Error: Task '{args.task}' not found in configuration.")
            sys.exit(0)
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            sys.exit(1)
    
    # Setup output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = paths["MODELS_DIR"] / f"{args.task}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(model_dir)
    
    logging.info(f"Starting training for {args.task} task")
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # Log paths
    logging.info("Project paths:")
    for path_name, path_value in paths.items():
        logging.info(f"  {path_name}: {path_value}")
    
    # Train model
    train_model(args, paths)
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()