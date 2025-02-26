"""
Histology Image Classification Training

This script trains deep learning models for histological image classification.
Supports both inflammation detection and tissue type classification.

Key Features:
- State-of-the-art model architectures (ConvNeXt, EfficientNetV2, Swin Transformer)
- Multi-GPU training support
- Mixed precision training
- Advanced data augmentation
- Comprehensive logging

Example Usage:
    # Train inflammation classifier:
    python train.py --task inflammation --model convnext_large --batch-size 64

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

# Add project root to path
import sys
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

def load_config(config_dir: Path, task: str) -> dict:
    """Load task-specific configuration from unified YAML file.
    
    Args:
        config_dir (Path): Directory containing configuration files
        task (str): Task identifier ('inflammation' or 'tissue')
        
    Returns:
        dict: Configuration dictionary containing model and training parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = config_dir / 'model_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract task-specific configuration
        if task not in config:
            raise KeyError(f"Task '{task}' not found in config file")
            
        # Combine common and task-specific configs
        task_config = config[task].copy()
        if 'common' in config:
            task_config.update({
                k: v for k, v in config['common'].items() 
                if k not in task_config
            })
            
        return task_config
        
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise
    except KeyError as e:
        logging.error(f"Configuration error: {e}")
        raise

def get_transforms(task: str, is_training: bool = False) -> T.Compose:
    """Get data transformations for training or evaluation.
    
    Args:
        task: 'inflammation' or 'tissue'
        is_training: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    if is_training:
        # Basic augmentations for all tasks
        transforms.extend([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Keep 224 for standard size
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        # Task-specific augmentations
        if task == 'inflammation':
            transforms.append(T.RandomAffine(degrees=15, translate=(0.1, 0.1)))
        else:  # tissue
            transforms.append(T.RandomRotation(180))
    else:
        transforms.extend([
            T.Resize(256),
            T.CenterCrop(224)  # Keep 224 for standard size
        ])
    
    # Common transforms
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return T.Compose(transforms)

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
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
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
        self.scaler = torch.amp.GradScaler('cuda')
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
    
def train_model(args, paths):
    """Main training function with standardized approach."""
    
    # Load configuration
    config = load_config(paths["CONFIG_DIR"], args.task)
    
    # Extract architecture configuration
    arch_config = config['architecture']
    model_name = arch_config['name']
    
    # Create model with proper configuration
    model = HistologyClassifier(
        model_name=model_name,
        num_classes=1,
        dropout_rate=config['dropout_rate']
    )
    
    # Setup device and multi-GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)

    # Configure training components
    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay'],
        pos_weight=config['pos_weight'],
        device=device
    )
    
    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']
    
    # Create datasets and dataloaders
    train_dataset = HistologyDataset(
        split='train',
        transform=get_transforms(args.task, is_training=True),
        task=args.task,
        paths=paths
    )
    
    val_dataset = HistologyDataset(
        split='val',
        transform=get_transforms(args.task, is_training=False),
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
    if config['scheduler']['enabled']:
        if config['scheduler']['name'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler']['T_max']
            )
        elif config['scheduler']['name'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['scheduler']['factor'],
                patience=config['scheduler']['patience']
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument(
        "--task",
        required=True,
        choices=['inflammation', 'tissue'],
        help="Classification task"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to hyperparameter config JSON from Optuna"
    )
    parser.add_argument(
        "--model",
        help="Model architecture (if not using config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (if not using config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (if not using config)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay (if not using config)"
    )
    
    # Add path arguments using the utility function
    parser = add_path_args(parser)
    
    return parser.parse_args()

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

def main():
    """Main training function using either Optuna results or manual settings."""
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
    
    # Load configuration
    config = load_config(paths["CONFIG_DIR"], args.task)
    logging.info(f"Loaded configuration from {paths['CONFIG_DIR'] / 'model_config.yaml'}")
    logging.info(f"Configuration:\n{yaml.dump(config, indent=2)}")
    
    # Train model
    train_model(args, paths)
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()