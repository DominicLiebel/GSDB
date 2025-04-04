"""
Hyperparameter Optimization for Histology Classification

This script performs hyperparameter optimization using Optuna for histology image classification.
It supports both inflammation and tissue classification tasks.

Key Features:
- Efficient hyperparameter search using TPE sampler
- Multi-GPU support
- Automatic mixed precision
- Trial pruning for efficiency
- Comprehensive logging

Example Usage:
    # Tune inflammation classifier:
    python tune.py --task inflammation --n-trials 200 --storage sqlite:///inflammation_study.db

    # Tune tissue classifier with specific settings:
    python tune.py --task tissue --n-trials 200 --timeout 48 --gpus 0,1
"""

# Standard library imports
import os
import sys
import argparse
import logging
import json
import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Third-party imports
import optuna
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_parallel_coordinate, plot_slice
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import timm

# Local imports
from dataset import HistologyDataset
import src.models.metrics_utils as metrics_utils
from src.models.model_utils import load_model, get_transforms
import training_utils
from src.models.architectures import GigaPathClassifier, HistologyClassifier

# Import project paths from config module
from src.config.paths import get_project_paths

# Get project paths
paths = get_project_paths()
BASE_DIR = paths["BASE_DIR"]
CONFIG_DIR = BASE_DIR / 'configs'
RESULTS_DIR = paths["RESULTS_DIR"]
LOG_DIR = RESULTS_DIR / 'logs'
MODEL_DIR = RESULTS_DIR / 'models'
TUNING_DIR = RESULTS_DIR / 'tuning'

def setup_logging(args):
    """Configure logging to both file and console."""
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{args.task}_{args.study_name}_{timestamp}.log"
    
    # Configure logging to both file and console
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Remove any existing handlers (important for proper setup)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure new handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Set format for both handlers
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Log file created at: {log_file}")
    return log_file


class ModelTracker:
    """Tracks best models for each architecture type."""
    
    def __init__(self, save_dir: str, task: str):
        """Initialize model tracker.
        
        Args:
            save_dir: Directory to save models
            task: Task type ('inflammation' or 'tissue')
        """
        self.save_dir = Path(save_dir)
        self.task = task
        self.best_models = {}
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking dictionary for only the supported models
        for model_name in [
            'resnet18', 'convnext_large', 'swin_v2_b', 'gigapath', 'densenet121', 'densenet169'
        ]:
            self.best_models[model_name] = {
                'val_loss': float('inf'),
                'metrics': None,
                'params': None,
                'trial_number': None
            }

    
    def update(self, trial_number: int, model_name: str, val_loss: float, 
            model: torch.nn.Module, params: Dict, metrics: Dict) -> bool:
        """Update the best model tracker if a better model is found.
        
        Args:
            trial_number: Current trial number
            model_name: Model architecture name
            val_loss: Validation loss
            model: Model to save if better than current best
            params: Hyperparameters used
            metrics: Validation metrics
            
        Returns:
            bool: True if model was updated, False otherwise
        """
        if val_loss < self.best_models[model_name]['val_loss']:
            # Calculate additional metrics using metrics_utils
            all_metrics = metrics_utils.calculate_basic_metrics(
                metrics['labels'],
                metrics['preds'],
                metrics['raw_preds']
            )
            
            model_filename = f"{self.task}_{model_name}.pt"
            
            # Save model state with standardized format matching train.py
            save_dict = {
                'epoch': -1,  # Placeholder since we don't track epochs in tuning
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,  # Placeholder for consistent format
                'metrics': all_metrics,
                'config': params,  # Match config key in train.py
                # Additional tuning-specific information
                'trial_number': trial_number,
                'val_loss': val_loss
            }
            
            torch.save(save_dict, self.save_dir / model_filename)
            
            # Update tracking
            self.best_models[model_name] = {
                'val_loss': val_loss,
                'metrics': all_metrics,
                'params': params,
                'trial_number': trial_number,
                'filename': model_filename
            }
            
            self._save_tracking_info()
            return True
        return False
    
    def _save_tracking_info(self):
        """Save tracking information to JSON file."""
        tracking_info = {
            'task': self.task,
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'models': self.best_models
        }
        
        tracking_file = self.save_dir / f"{self.task}_model_tracking.json"
        with open(tracking_file, 'w') as f:
            json.dump(tracking_info, f, indent=2)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                       choices=['inflammation', 'tissue'])
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--final-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--study-name', type=str, required=True)
    parser.add_argument('--base-dir', type=Path,
                       default=BASE_DIR,
                       help='Base directory for all data and outputs')
    parser.add_argument('--output-dir', type=Path, 
                       default=TUNING_DIR,
                       help='Directory to save results')
    parser.add_argument('--timeout', type=int, help='Optimization timeout in hours')
    parser.add_argument('--gpus', type=str, help='Comma-separated GPU indices (e.g., "0,1")')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel trials (1 for better stability with multi-GPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloaders(args, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        args: Command line arguments
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = HistologyDataset(
        split='train',
        transform=get_transforms(is_training=True),
        task=args.task
    )
    
    val_dataset = HistologyDataset(
        split='val',
        transform=get_transforms(is_training=False),
        task=args.task
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return train_loader, val_loader

def suggest_hyperparameters(trial: Trial, task: str, suggested_pos_weight: float) -> Dict:
    """Suggest hyperparameters with focused model selection and efficient ranges.
    
    Args:
        trial: Optuna trial
        task: Classification task
        suggested_pos_weight: Suggested weight based on class distribution
        
    Returns:
        Dict of hyperparameters
    """
    model = trial.suggest_categorical('model', [
        'resnet18',        # Baseline model for comparison
        'swin_v2_b',       # Modern vision transformer
        'convnext_large',  # Modern CNN architecture
        'gigapath',        # Prov-GigaPath foundation model
        'densenet121',     # DenseNet medium size
        'densenet169'      # DenseNet larger size
    ])
    
    # Optimize batch size based on model
    if model == 'convnext_large':
        batch_size = trial.suggest_int('batch_size', 32, 64, step=32)
    elif model == 'swin_v2_b':
        batch_size = trial.suggest_int('batch_size', 32, 96, step=32)
    elif model == 'gigapath':
        batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    elif model == 'densenet121':
        batch_size = trial.suggest_int('batch_size', 32, 96, step=32)
    elif model == 'densenet169':
        batch_size = trial.suggest_int('batch_size', 24, 64, step=8)
    else:  # resnet18
        batch_size = trial.suggest_int('batch_size', 64, 128, step=32)
    
    # Focused learning rate range based on empirical findings
    if model == 'gigapath':
        # GigaPath only trains the classifier head, so can use a higher learning rate
        lr = trial.suggest_float('learning_rate', 5e-4, 1e-2, log=True)
    elif model == 'densenet121' or model == 'densenet169':
        # DenseNet models typically need lower learning rates
        lr = trial.suggest_float('learning_rate', 5e-5, 2e-4, log=True)
    else:
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    
    # More focused dropout range
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    
    # Task-specific pos_weight based on class distribution
    # Use the suggested pos_weight as the center of the search range
    pos_weight_min = max(0.1, suggested_pos_weight - 0.15)
    pos_weight_max = min(0.9, suggested_pos_weight + 0.15)
    pos_weight = trial.suggest_float('pos_weight', pos_weight_min, pos_weight_max)
    
    return {
        'model': model,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'dropout': dropout,
        'pos_weight': pos_weight
    }

def objective(trial: Trial, args: argparse.Namespace, model_tracker: ModelTracker,
            base_train_loader: DataLoader, base_val_loader: DataLoader,
            suggested_pos_weight: float) -> float:
    """Optimized objective function with early pruning.
    
    Args:
        trial: Optuna trial
        args: Command line arguments
        model_tracker: Tracks best models for each architecture
        base_train_loader: Base train dataloader
        base_val_loader: Base validation dataloader
        suggested_pos_weight: Suggested pos_weight based on class distribution
        
    Returns:
        Validation loss
    """
    # Clear memory at the start of each trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        config = suggest_hyperparameters(trial, args.task, suggested_pos_weight)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add device info to logs
        logging.info(f"Trial {trial.number}: Using device {device}, model: {config['model']}")
        
        # Update dataloaders if needed
        try:
            if config['batch_size'] != base_train_loader.batch_size:
                logging.info(f"Trial {trial.number}: Creating new dataloaders with batch size {config['batch_size']}")
                train_loader, val_loader = create_dataloaders(args, config['batch_size'])
            else:
                train_loader, val_loader = base_train_loader, base_val_loader
        except Exception as loader_e:
            logging.error(f"Trial {trial.number}: Error creating dataloaders: {str(loader_e)}")
            import traceback
            logging.error(f"Dataloader traceback: {traceback.format_exc()}")
            raise
        
        # Initialize model based on selected architecture
        try:
            if config['model'] == 'gigapath':
                model = GigaPathClassifier(
                    num_classes=1,
                    dropout_rate=config['dropout']
                )
            else:
                # Only use supported models
                if config['model'] not in ['resnet18', 'convnext_large', 'swin_v2_b', 'densenet121', 'densenet169']:
                    raise ValueError(f"Unsupported model: {config['model']}")
                    
                model = HistologyClassifier(
                    model_name=config['model'],
                    num_classes=1,
                    dropout_rate=config['dropout']
                )
            
            logging.info(f"Trial {trial.number}: Model {config['model']} initialized successfully")
        except Exception as model_e:
            logging.error(f"Trial {trial.number}: Error initializing model {config['model']}: {str(model_e)}")
            import traceback
            logging.error(f"Model initialization traceback: {traceback.format_exc()}")
            raise
        
        # Move model to device with error handling
        try:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            logging.info(f"Trial {trial.number}: Model successfully moved to {device}")
        except Exception as device_e:
            logging.error(f"Trial {trial.number}: Error moving model to device: {str(device_e)}")
            import traceback
            logging.error(f"Device error traceback: {traceback.format_exc()}")
            raise
        
        # Initialize optimizer and criterion with error handling
        try:
            pos_weight = torch.tensor([config['pos_weight']]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer_type = config.get('optimizer_type', None)

            logging.info(f"Using {optimizer_type} optimizer for model: {config.get('model', 'unknown')}")

            # Create the appropriate optimizer based on type
            if optimizer_type.upper() == 'SGD':
                # Get momentum from config or use default
                momentum = config.get('momentum', 0.9)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config['learning_rate'],
                    momentum=momentum,
                    weight_decay=config['weight_decay']
                )
                logging.info(f"SGD parameters: momentum={momentum}, lr={config['learning_rate']}")
            else:
                # Default to AdamW for all other models
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
                logging.info(f"AdamW parameters: lr={config['learning_rate']}")

            logging.info(f"Trial {trial.number}: Optimizer and criterion created successfully")
        except Exception as opt_e:
            logging.error(f"Trial {trial.number}: Error creating optimizer/criterion: {str(opt_e)}")
            import traceback
            logging.error(f"Optimizer traceback: {traceback.format_exc()}")
            raise
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        min_epochs = 5  # Minimum epochs before pruning
        
        for epoch in range(args.epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory usage to detect OOM issues
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logging.info(f"Trial {trial.number}, Epoch {epoch}, GPU {i} memory: "
                              f"{torch.cuda.memory_allocated(i)/1024**2:.1f}MB allocated, "
                              f"{torch.cuda.max_memory_allocated(i)/1024**2:.1f}MB max")
            
            # Training phase with error handling
            try:
                model.train()
                train_loss = 0.0
                
                for inputs, labels, _ in train_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                logging.info(f"Trial {trial.number}, Epoch {epoch}: Train loss = {avg_train_loss:.6f}")
            except Exception as train_e:
                logging.error(f"Trial {trial.number}: Error during training epoch {epoch}: {str(train_e)}")
                import traceback
                logging.error(f"Training traceback: {traceback.format_exc()}")
                raise
            
            # Validation phase with error handling  
            try:
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_labels = []
                val_raw_preds = []
                
                with torch.no_grad():
                    for inputs, labels, _ in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        # Store predictions for metric tracking
                        preds = (outputs > 0).float()
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                        val_raw_preds.extend(outputs.cpu().numpy())
                                
                # Calculate average validation loss
                val_loss = val_loss / len(val_loader)
                logging.info(f"Trial {trial.number}, Epoch {epoch}: Validation loss = {val_loss:.6f}")

                # Create metrics dict for model tracker
                val_metrics = {
                    'preds': val_preds,
                    'labels': val_labels,
                    'raw_preds': val_raw_preds
                }

                # Update model tracker with normalized loss
                model_tracker.update(
                    trial.number, 
                    config['model'], 
                    val_loss,
                    model, 
                    config, 
                    val_metrics
                )

                # Report to Optuna ONCE
                trial.report(val_loss, epoch)

                # Update best_val_loss logic with the properly normalized loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Handle pruning and early stopping separately
                if epoch >= min_epochs:
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                        
                    if patience_counter >= args.patience:
                        break

            except Exception as val_e:
                logging.error(f"Trial {trial.number}: Error during validation epoch {epoch}: {str(val_e)}")
                import traceback
                logging.error(f"Validation traceback: {traceback.format_exc()}")
                raise
            
        logging.info(f"Trial {trial.number} completed with best_val_loss: {best_val_loss}")
        return best_val_loss
        
    except Exception as e:
        # Much improved error logging with full traceback
        import traceback
        logging.error(f"Error in objective function for trial {trial.number}: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Log CUDA memory status if available (helpful for OOM errors)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logging.error(f"GPU {i} memory at error: "
                           f"{torch.cuda.memory_allocated(i)/1024**2:.1f}MB allocated, "
                           f"{torch.cuda.max_memory_allocated(i)/1024**2:.1f}MB max")
        
        # Optuna requires us to return a value or raise TrialPruned
        raise optuna.TrialPruned()

def calculate_class_distribution(dataset: HistologyDataset) -> float:
    """Calculate class distribution and suggest optimal pos_weight.
    
    Args:
        dataset: Training dataset
        
    Returns:
        float: Suggested pos_weight based on class distribution
    """
    total = len(dataset)
    pos_count = sum(1 for _, label, _ in dataset if label == 1)
    neg_count = total - pos_count
    
    # Calculate class distribution
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_ratio:.2%})")
    logging.info(f"Negative class: {neg_count} ({neg_ratio:.2%})")
    
    # Calculate suggested pos_weight (inverse of class ratio)
    suggested_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logging.info(f"Suggested pos_weight: {suggested_weight:.3f}")
    
    return suggested_weight

def calculate_metrics(labels: List, predictions: List, raw_predictions: List) -> Dict[str, float]:
    """Calculate classification metrics with error handling."""
    metrics = {}
    
    try:
        metrics['f1'] = 100 * f1_score(labels, predictions)
    except Exception as e:
        logging.warning(f"Error calculating F1 score: {e}")
        metrics['f1'] = 0.0
    
    try:
        metrics['auc'] = 100 * roc_auc_score(labels, raw_predictions)
    except Exception as e:
        logging.warning(f"Error calculating AUC-ROC: {e}")
        metrics['auc'] = 0.0
    
    return metrics

def save_study_results(study: optuna.Study, args: argparse.Namespace, results_dir: Path) -> None:
    """Save optimization results and create visualization plots.
    
    Args:
        study: Completed Optuna study
        args: Command line arguments
        results_dir: Directory to save results
    """
    # Save best parameters
    best_params = {
        'task': args.task,
        'value': study.best_trial.value,
        'params': study.best_trial.params,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(results_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    try:
        train_final_model(study.best_trial, args, results_dir)
    except Exception as e:
        logging.error(f"Error training final model: {str(e)}")
    
    # Final summary
    logging.info("\nOptimization and final training completed!")
    
    # Create visualization plots
    try:
        for plot_func in [
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        ]:
            fig = plot_func(study)
            plot_name = plot_func.__name__.replace('plot_', '')
            fig.write_html(str(results_dir / f'{plot_name}.html'))
        
        logging.info(f"Visualization plots saved to {results_dir}")
        
    except Exception as e:
        logging.error(f"Error creating visualization plots: {e}")

def train_final_model(trial: optuna.trial.FrozenTrial, args: argparse.Namespace, results_dir: Path) -> None:
    """Train final model with best hyperparameters.
    
    Args:
        trial: Best trial from Optuna study
        args: Command line arguments
        results_dir: Directory to save results
    """
    logging.info("\nTraining final model with best hyperparameters...")
    
    # Create final model directory
    final_model_dir = results_dir / 'final_model'
    final_model_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get best hyperparameters
    config = trial.params
    logging.info(f"Best hyperparameters:\n{json.dumps(config, indent=2)}")
    
    # Create datasets and dataloaders
    train_dataset = HistologyDataset(
        split='train',
        transform=get_transforms(is_training=True),
        task=args.task
    )
    
    val_dataset = HistologyDataset(
        split='val',
        transform=get_transforms(is_training=False),
        task=args.task
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
    
    # Initialize model based on selected architecture
    if config['model'] == 'gigapath':
        model = GigaPathClassifier(
            num_classes=1,
            dropout_rate=config['dropout']
        )
    else:
        model = HistologyClassifier(
            model_name=config['model'],
            num_classes=1,
            dropout_rate=config['dropout']
        )
    
    # Multi-GPU support
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    
    # Configure training components
    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        pos_weight=config['pos_weight'],
        device=device
    )
    
    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']
    
    # Training metrics history
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
    for epoch in range(args.final_epochs):
        # Training phase
        train_metrics = training_utils.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            deterministic=True
        )
        
        # Validation phase
        val_metrics = training_utils.validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Log progress
        logging.info(
            f"Epoch {epoch+1}/{args.final_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"Val F1: {val_metrics['f1']:.2f}%, Val AUC: {val_metrics['auc']:.2f}%"
        )
        
        # Save best model using the standardized format
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # Create a save dictionary that matches the standard format
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config,
                # Add tuning-specific information
                'trial_number': trial.number,
                'val_loss': val_metrics['loss']
            }
            
            torch.save(save_dict, final_model_dir / 'best_model.pt')
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f}")
    
    # Save training history
    with open(final_model_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create training plots
    plot_training_history(history, final_model_dir)
    logging.info(f"\nFinal model training completed. Results saved to: {final_model_dir}")

def plot_training_history(history: Dict, save_dir: Path) -> None:
    """Create and save training history plots.
    
    Args:
        history (Dict): Dictionary containing training metrics
        save_dir (Path): Directory to save plots
    """
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'loss_history.png')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'metrics_history.png')
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    logging.info(f"Random seed set to {args.seed}")

    # Update base directory if provided
    global BASE_DIR, CONFIG_DIR, RESULTS_DIR, LOG_DIR, MODEL_DIR, TUNING_DIR
    if args.base_dir != BASE_DIR:
        BASE_DIR = args.base_dir
        CONFIG_DIR = BASE_DIR / 'configs'
        RESULTS_DIR = BASE_DIR / 'results'
        LOG_DIR = RESULTS_DIR / 'logs'
        MODEL_DIR = RESULTS_DIR / 'models'
        TUNING_DIR = RESULTS_DIR / 'tuning'
    
    setup_logging(args)
    logging.info(f"Starting optimization for {args.task} task")
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Setup device
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info(f"Using GPUs: {args.gpus}")
    
    # Initialize model tracker
    model_tracker = ModelTracker(
        save_dir=MODEL_DIR,
        task=args.task
    )
    
    # Create initial dataloaders with default batch size
    train_loader, val_loader = create_dataloaders(args, batch_size=32)
    
    # Calculate class distribution from training dataset
    suggested_weight = calculate_class_distribution(train_loader.dataset)

    # Create results directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create storage path for Optuna database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = args.output_dir / f"{args.task}_{args.study_name}_{timestamp}.db"
    storage = f"sqlite:///{db_path}"
    
    logging.info(f"Optuna database will be saved to: {db_path}")
    
    # Create study with more aggressive pruning
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,
            n_ei_candidates=10,  # More focused sampling
            multivariate=True  # Consider parameter relationships
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,  # Minimum epochs before pruning
            max_resource=args.epochs,
            reduction_factor=3
        )
    )

    
    # Log study information
    logging.info(f"Study name: {study.study_name}")
    logging.info(f"Storage path: {db_path}")
    logging.info(f"Number of trials: {args.n_trials}")
    
    # Print dashboard instructions
    print("\nTo view the Optuna Dashboard, run in a separate terminal:")
    print(f"optuna-dashboard {storage}")
    
    # Run optimization with proper use of suggested_weight parameter
    study.optimize(
        lambda trial: objective(trial, args, model_tracker, train_loader, val_loader, suggested_weight),
        n_trials=args.n_trials,
        timeout=args.timeout * 3600 if args.timeout else None,
        gc_after_trial=True,
        show_progress_bar=True,
        n_jobs=args.n_jobs  # Use configurable n_jobs parameter
    )
    
    # Save study results and create visualizations
    results_dir = args.output_dir / f"{args.task}_{args.study_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_study_results(study, args, results_dir)
    
    # Final summary
    logging.info("\nOptimization completed!")
    logging.info("\nBest models for each architecture:")
    for model_name, info in model_tracker.best_models.items():
        if info['trial_number'] is not None:
            logging.info(f"\n{model_name}:")
            logging.info(f"Trial: {info['trial_number']}")
            logging.info(f"Validation Loss: {info['val_loss']:.4f}")
            logging.info(f"Metrics: {info['metrics']}")
            logging.info(f"Saved as: {info['filename']}")

if __name__ == "__main__":
    main()